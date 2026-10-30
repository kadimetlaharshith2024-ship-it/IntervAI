import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.session_service import session_service
from app.services.knowledge_state_service import knowledge_state_service
from app.services.adaptive_selection_service import (
    adaptive_selection_engine,
    AdaptiveSelectionConfig,
)
from app.schemas.adaptive_selection import (
    AdaptiveStrategyType,
    NextQuestionSelectionResult,
    AdaptiveQuestionSelectionRequest,
    AdaptiveNextQuestionResponse,
)
from app.schemas.rag import QuestionGenerationRequest, QuestionGenerationResponse, QuestionOutputSchema
from app.schemas.knowledge_state import ClaimVerificationStatus, TopicCoverageStatus, GapNature, GapSeverity
from app.schemas.contradiction import ContradictionStatus, ContradictionType, ContradictionSeverity
from app.services.evaluation.contradiction_analyzer import contradiction_analyzer
from app.services.evaluation.scorecard import scorecard_aggregator
from app.services.evaluation.evaluator import answer_evaluator
from app.services.rag.rag_pipeline import rag_pipeline
from app.core.database import get_db
from app.main import app


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session: Session):
    """Provides a FastAPI TestClient with database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def multi_turn_resume(db_session: Session) -> Resume:
    """Creates a rich candidate resume fixture for multi-turn validation."""
    resume = Resume(
        id=uuid4(),
        filename="jordan_lead_systems_engineer.pdf",
        raw_text="Jordan Lead Systems Engineer. Skills: Python, FastAPI, PostgreSQL, Redis, Docker, Kafka. Project: PaymentGateway, CloudLedger."
    )
    db_session.add(resume)
    db_session.flush()

    # Skills
    skills_data = [
        (501, "Python", "Language"),
        (502, "FastAPI", "Framework"),
        (503, "PostgreSQL", "Database"),
        (504, "Redis", "Database"),
        (505, "Docker", "Tool/DevOps"),
        (506, "Kafka", "Messaging/Queue"),
    ]
    for sid, name, cat in skills_data:
        sk = Skill(id=sid, name=name, category=cat)
        db_session.add(sk)
        db_session.flush()
        rsk = ResumeSkill(resume_id=resume.id, skill_id=sid, extraction_confidence=0.92)
        db_session.add(rsk)

    # Relationships
    rel1 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="COMMUNICATES_WITH",
        target_entity="PostgreSQL",
        context_entity="PaymentGateway",
        section="Experience",
        evidence_text="Built asynchronous transactional pipeline between FastAPI and PostgreSQL database.",
        confidence=0.95
    )
    rel2 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="PaymentGateway",
        relation_type="USES_DATABASE",
        target_entity="PostgreSQL",
        context_entity="PaymentGateway",
        section="Experience",
        evidence_text="Configured PostgreSQL as the primary relational database for PaymentGateway.",
        confidence=0.98
    )
    db_session.add_all([rel1, rel2])

    # Resume Claim
    claim = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-PG-LOCKING",
        section="Experience",
        project="PaymentGateway",
        target_entity="Redis",
        relation_type="DEPLOYED_FOR",
        claim_text="Implemented distributed Redis locking with Redlock algorithm preventing double payments under 10k RPS load.",
        evidence_text="Implemented distributed Redis locking with Redlock algorithm preventing double payments under 10k RPS load."
    )
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(resume)
    return resume


def make_mock_rag_side_effect(
    question_text: str = None,
    category: str = None,
    difficulty: int = None,
    target_project: str = None,
    target_entity: str = None,
    evidence_ids: list = None,
    evidence_text: str = None
):
    def _side_effect(resume_id: UUID, req: QuestionGenerationRequest, db: Session):
        q_id = uuid4()
        parent_id = getattr(req, "parent_question_id", None)
        text = question_text or f"Technical inquiry on {req.target_entity or req.target_project or 'the system'}."
        cat = category or req.category or "Architecture"
        diff = difficulty if difficulty is not None else (req.difficulty or 3)
        t_proj = target_project or req.target_project or "PaymentGateway"
        t_ent = target_entity or req.target_entity or "FastAPI"
        ev_text = evidence_text or f"Verified architecture details for {t_ent}."

        q_rec = InterviewQuestion(
            id=q_id,
            resume_id=resume_id,
            session_id=req.session_id,
            parent_question_id=parent_id,
            question_text=text,
            category=cat,
            difficulty=diff,
            targeted_claim=getattr(req, "target_claim_id", None) or "Evidence-Grounded Question",
            target_project=t_proj,
            target_entity=t_ent,
            target_claim_id=getattr(req, "target_claim_id", None),
            target_relationship=getattr(req, "target_relationship", None),
            evidence_ids=evidence_ids or ["rel_1"],
            evidence_text=ev_text,
            rationale=f"Grounded question on {t_ent} {cat}",
            grounding_score=1.0,
            relevance_score=1.0,
            technical_quality_score=1.0,
            retrieval_trace_id=f"trc_{uuid4().hex[:8]}"
        )
        db.add(q_rec)
        db.commit()
        db.refresh(q_rec)

        mock_q = QuestionOutputSchema(
            question=text,
            category=cat,
            difficulty=diff,
            target_project=t_proj,
            target_entity=t_ent,
            evidence_ids=evidence_ids or ["rel_1"],
            evidence_text=ev_text,
            rationale=f"Grounded question on {t_ent} {cat}",
            grounding_score=1.0,
            relevance_score=1.0,
            technical_quality_score=1.0,
            trace_id=q_rec.retrieval_trace_id
        )
        return QuestionGenerationResponse(
            resume_id=resume_id,
            request_id=f"req_{uuid4().hex[:8]}",
            trace_id=mock_q.trace_id,
            question_id=q_rec.id,
            question=mock_q,
            validation_status="ACCEPTED",
            message="Question successfully generated and verified.",
            attempts_count=1,
            latencies_ms={"total_ms": 10.0}
        )
    return _side_effect


# =========================================================================
# SCENARIO A: Weak Answer Evaluation, Gap Recording, and Difficulty De-escalation
# =========================================================================
def test_scenario_a_weak_answer_adaptation_and_gap_followup(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario A:
    1. Candidate receives an Architecture question on FastAPI.
    2. Candidate provides a genuinely weak technical answer (score 35.0, confidence 0.8).
    3. Knowledge state records a demonstrated knowledge gap (HIGH severity, DEMONSTRATED_WEAKNESS).
    4. Difficulty de-escalates from 3 to 2.
    5. The next question targets the weakness (GAP_FOLLOW_UP) with parent linkage.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    # Turn 1: Generate initial question
    mock_gen1 = make_mock_rag_side_effect(
        question_text="How do you architect asynchronous database transactions in FastAPI?",
        target_entity="FastAPI",
        category="Architecture",
        difficulty=3
    )
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen1):
        q1_resp = session_service.select_and_generate_next_question(session.id, db_session)

    # Turn 1: Candidate submits weak answer
    q1_row = db_session.query(InterviewQuestion).filter(InterviewQuestion.id == q1_resp.question_id).first()
    assert q1_row is not None

    weak_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q1_resp.question_id,
        candidate_answer="I just put sync sleep calls in my router functions and it works fine.",
        status="EVALUATED",
        overall_score=35.0,
        evaluation_confidence=0.85,
        missing_concepts=["asyncio event loop", "non-blocking I/O", "connection pooling"],
        weaknesses=["Blocking the event loop breaks asynchronous scalability."],
        evaluation_feedback="Candidate demonstrated a critical misunderstanding of asynchronous I/O in FastAPI."
    )
    db_session.add(weak_ans)
    db_session.commit()

    # Update knowledge state
    k_state = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1_resp.question_id,
        answer=weak_ans,
        db=db_session
    )

    # Invariants Check: Knowledge gap recorded with HIGH severity
    active_gaps = [g for g in k_state.get("knowledge_gaps", []) if not g.get("is_addressed")]
    assert len(active_gaps) >= 1
    top_gap = active_gaps[0]
    assert top_gap["severity"] == GapSeverity.HIGH.value
    assert top_gap["gap_nature"] == GapNature.DEMONSTRATED_WEAKNESS.value
    assert top_gap["topic"] == "FastAPI"

    # Turn 2: Next question selection adapts difficulty and targets gap
    next_target = session_service.preview_next_question_selection(session.id, db_session)
    assert next_target.strategy == AdaptiveStrategyType.GAP_FOLLOW_UP
    assert next_target.difficulty == 2  # De-escalated from 3 to 2
    assert next_target.is_follow_up is True
    assert next_target.parent_question_id == q1_resp.question_id
    assert "FastAPI" in next_target.target_topic


# =========================================================================
# SCENARIO B: Strong Answer, Mastery Update, and Difficulty Escalation
# =========================================================================
def test_scenario_b_strong_answer_mastery_and_difficulty_escalation(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario B:
    1. Candidate receives an Architecture question on Python.
    2. Candidate gives a strong, well-supported answer (score 92.0, confidence 0.90).
    3. Evaluation and knowledge state update correctly: mastery rises, Python marked as demonstrated strength.
    4. Difficulty escalates from 3 to 4.
    5. Next question selection probes advanced failure-mode/scalability trade-offs.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    mock_gen1 = make_mock_rag_side_effect(
        question_text="How does Python's GIL impact multithreaded CPU-bound versus I/O-bound performance?",
        target_entity="Python",
        category="Architecture",
        difficulty=3
    )
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen1):
        q1_resp = session_service.select_and_generate_next_question(session.id, db_session)

    q1_row = db_session.query(InterviewQuestion).filter(InterviewQuestion.id == q1_resp.question_id).first()
    assert q1_row is not None

    strong_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q1_resp.question_id,
        candidate_answer=(
            "The GIL serializes thread execution within a single Python process. For CPU-bound tasks, "
            "multiprocessing or C extensions bypassing the GIL are necessary. For I/O-bound tasks, "
            "threads or asyncio release the GIL during syscalls, allowing high concurrent throughput."
        ),
        status="EVALUATED",
        overall_score=92.0,
        evaluation_confidence=0.90,
        evaluation_feedback="Exceptional technical explanation covering GIL mechanics, multiprocessing, and async I/O."
    )
    db_session.add(strong_ans)
    db_session.commit()

    k_state = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1_resp.question_id,
        answer=strong_ans,
        db=db_session
    )

    # Invariants Check: Topic mastery reflects strong performance
    python_topic = k_state["topics"]["Python"]
    assert python_topic["mastery_score"] == 92.0
    assert "Python" in k_state["summary"]["demonstrated_strengths"]

    # Turn 2: Next question selection escalates difficulty to 4
    next_target = session_service.preview_next_question_selection(session.id, db_session)
    assert next_target.difficulty == 4
    assert next_target.strategy in [AdaptiveStrategyType.DIFFICULTY_ESCALATION, AdaptiveStrategyType.UNCOVERED_TOPIC]


# =========================================================================
# SCENARIO C: Partial Answer, Missing Concepts, and Targeted Clarification
# =========================================================================
def test_scenario_c_partial_answer_and_targeted_clarification(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario C:
    1. Candidate answers part of the question correctly but omits expected concepts (score 65.0).
    2. Missing concepts are persisted in answer record.
    3. The selector chooses PARTIAL_CLARIFICATION.
    4. Parent-child linkage and follow_up_reason are consistently populated without repeating Q1 verbatim.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    mock_gen1 = make_mock_rag_side_effect(
        question_text="How do you handle Redis cache invalidation and distributed lock leases?",
        target_entity="Redis",
        category="Architecture",
        difficulty=3
    )
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen1):
        q1_resp = session_service.select_and_generate_next_question(session.id, db_session)

    partial_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q1_resp.question_id,
        candidate_answer="We set a TTL on the Redis key so it expires automatically after 60 seconds.",
        status="EVALUATED",
        overall_score=65.0,
        evaluation_confidence=0.80,
        missing_concepts=["lease renewal / lock heartbeat", "fencing tokens", "redlock clock drift"],
        evaluation_feedback="Correctly identified TTL expiration but omitted lease renewal and fencing tokens."
    )
    db_session.add(partial_ans)
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1_resp.question_id,
        answer=partial_ans,
        db=db_session
    )

    next_target = session_service.preview_next_question_selection(session.id, db_session)
    assert next_target.strategy == AdaptiveStrategyType.PARTIAL_CLARIFICATION
    assert next_target.is_follow_up is True
    assert next_target.parent_question_id == q1_resp.question_id
    assert next_target.follow_up_reason is not None
    assert "lease renewal" in next_target.follow_up_reason.lower() or "missing" in next_target.rationale.lower()


# =========================================================================
# SCENARIO D: Evidence-Backed Contradiction vs Hypothetical Statements
# =========================================================================
def test_scenario_d_evidence_backed_contradiction_vs_hypothetical(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario D:
    1. Provide controlled answer with actual factual conflict (claiming MySQL on PaymentGateway vs verified PostgreSQL).
    2. Contradiction analyzer detects conflict with severity HIGH.
    3. Adaptive selector chooses CONTRADICTION_CLARIFICATION.
    4. Confirm that hypothetical statements and unsupported additions do NOT trigger false contradiction.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    q = InterviewQuestion(
        resume_id=multi_turn_resume.id,
        session_id=session.id,
        question_text="What database did you use for PaymentGateway and how was it architected?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="PostgreSQL",
        evidence_text="Configured PostgreSQL as the primary relational database for PaymentGateway."
    )
    db_session.add(q)
    db_session.commit()

    # 1. Genuine contradiction: Candidate substitutes verified PostgreSQL with MySQL
    factual_conflict_answer = "For PaymentGateway, our primary database was MySQL instead of PostgreSQL."
    contra_res = contradiction_analyzer.analyze(
        candidate_answer=factual_conflict_answer,
        question=q,
        db=db_session
    )
    assert contra_res.has_contradiction is True
    assert contra_res.status == ContradictionStatus.CONTRADICTION
    assert contra_res.conflicts[0].conflict_type == ContradictionType.TECHNOLOGY_SUBSTITUTION

    # 2. Persist answer with contradiction metadata
    ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        candidate_answer=factual_conflict_answer,
        status="EVALUATED",
        overall_score=40.0,
        dimension_scores={"_contradiction_analysis": contra_res.model_dump()},
        evaluation_feedback="Candidate contradicted resume evidence regarding database technology."
    )
    setattr(ans, "contradiction_analysis", contra_res.model_dump())
    db_session.add(ans)
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=ans,
        db=db_session
    )

    # Next target selection must prioritize CONTRADICTION_CLARIFICATION
    next_target = session_service.preview_next_question_selection(session.id, db_session)
    assert next_target.strategy == AdaptiveStrategyType.CONTRADICTION_CLARIFICATION
    assert next_target.is_follow_up is True
    assert next_target.parent_question_id == q.id

    # 3. Hypothetical statements do NOT trigger contradiction
    hypothetical_answer = (
        "If we had considered MySQL instead of PostgreSQL, we might have had different JSON support, "
        "but PostgreSQL gave us superior ACID compliance."
    )
    hypo_res = contradiction_analyzer.analyze(
        candidate_answer=hypothetical_answer,
        question=q,
        db=db_session
    )
    assert hypo_res.has_contradiction is False


# =========================================================================
# SCENARIO E: Resume Claim Verification Lifecycle
# =========================================================================
def test_scenario_e_resume_claim_verification_lifecycle(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario E:
    1. Targets unverified claim CLM-PG-LOCKING.
    2. Weak/refusal answer marks claim as INSUFFICIENT_EVIDENCE, NOT CONTRADICTED.
    3. Strong answer marks claim as SUPPORTED.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    q = InterviewQuestion(
        resume_id=multi_turn_resume.id,
        session_id=session.id,
        question_text="Walk through your Redis Redlock implementation under 10k RPS load.",
        category="Claim-Verification",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis",
        targeted_claim="Implemented distributed Redis locking with Redlock algorithm preventing double payments under 10k RPS load.",
        target_claim_id="CLM-PG-LOCKING"
    )
    db_session.add(q)
    db_session.commit()

    # Part A: Refusal answer -> Claim status is INSUFFICIENT_EVIDENCE
    refusal_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        candidate_answer="I don't recall the specific RPS numbers, skip.",
        status="EVALUATED",
        overall_score=0.0,
        evaluator_model="fast_path_refusal",
        evaluation_feedback="Candidate declined to provide implementation details."
    )
    db_session.add(refusal_ans)
    db_session.commit()

    k_state_refusal = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=refusal_ans,
        db=db_session
    )
    claim_item = [c for c in k_state_refusal["claims"] if c["claim_id"] == "CLM-PG-LOCKING"][0]
    assert claim_item["status"] == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE.value
    assert claim_item["status"] != ClaimVerificationStatus.CONTRADICTED.value

    # Part B: Strong answer attempt 2 -> Claim status transitions to SUPPORTED
    strong_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        attempt_number=2,
        candidate_answer=(
            "We used Redlock with 5 independent Redis nodes. Each lock acquired had a 500ms validity time. "
            "If lock acquisition succeeded across a majority (3/5) of instances within drift tolerance, "
            "we executed payment processing, eliminating double payments during flash sales."
        ),
        status="EVALUATED",
        overall_score=88.0,
        evidence_alignment_score=90.0,
        evaluation_confidence=0.92,
        evaluation_feedback="Concrete, grounded technical walkthrough validating the resume claim."
    )
    db_session.add(strong_ans)
    db_session.commit()

    k_state_strong = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=strong_ans,
        db=db_session
    )
    claim_item_updated = [c for c in k_state_strong["claims"] if c["claim_id"] == "CLM-PG-LOCKING"][0]
    assert claim_item_updated["status"] == ClaimVerificationStatus.SUPPORTED.value


# =========================================================================
# SCENARIO F: Unanswered, Refused, and Timed-Out Answers
# =========================================================================
def test_scenario_f_unanswered_refused_and_timed_out_answers(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario F:
    1. Unanswered questions do not receive fabricated technical scores.
    2. Failed / timed-out evaluations (status FAILED, score None) do not corrupt knowledge state.
    3. Fast-path refusal answers de-escalate difficulty safely without corrupting unassessed topics.
    4. Session continues safely.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    q = InterviewQuestion(
        resume_id=multi_turn_resume.id,
        session_id=session.id,
        question_text="How does Kafka consumer group rebalancing work?",
        category="Architecture",
        difficulty=3,
        target_entity="Kafka"
    )
    db_session.add(q)
    db_session.commit()

    # 1. State before answer: UNANSWERED, score is None
    state1 = session_service.get_interview_state(session.id, db_session)
    assert state1.asked_questions[0].status == "UNANSWERED"
    assert state1.asked_questions[0].latest_score is None

    # 2. Failed evaluation (e.g. LLM timeout)
    failed_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        candidate_answer="My connection timed out.",
        status="FAILED",
        overall_score=None,
        evaluation_confidence=0.0
    )
    db_session.add(failed_ans)
    db_session.commit()

    # Update knowledge state skips safely
    k_state_after_failed = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=failed_ans,
        db=db_session
    )
    kafka_topic = k_state_after_failed["topics"]["Kafka"]
    assert kafka_topic["answers_evaluated_count"] == 0
    assert kafka_topic["latest_score"] is None
    assert session.current_difficulty == 3  # Unchanged

    # 3. Refusal fast-path: Candidate refuses question
    refusal_ans = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        attempt_number=2,
        candidate_answer="I don't know Kafka, please pass.",
        status="EVALUATED",
        overall_score=0.0,
        evaluator_model="fast_path_refusal",
        evaluation_feedback="Candidate declined question.",
        evaluation_confidence=1.0
    )
    db_session.add(refusal_ans)
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=refusal_ans,
        db=db_session
    )

    # Next question target de-escalates difficulty from 3 to 2 safely
    next_target = adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    assert next_target.difficulty == 2


# =========================================================================
# SCENARIO G: Repeated Attempts History and Idempotent Requests
# =========================================================================
def test_scenario_g_repeated_attempts_history_and_idempotent_requests(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario G:
    1. Multiple answer attempts on one question maintain sequential attempt_number and chronological history.
    2. Scorecard aggregation reflects latest attempt.
    3. Retrying next-question request with request_idempotency_key returns cached response
       without creating duplicate questions or burning budget.
    """
    session = session_service.create_session(resume_id=multi_turn_resume.id, db=db_session, initial_difficulty=3)

    q = InterviewQuestion(
        resume_id=multi_turn_resume.id,
        session_id=session.id,
        question_text="Explain Docker multi-stage builds for production images.",
        category="Implementation",
        difficulty=3,
        target_entity="Docker"
    )
    db_session.add(q)
    db_session.commit()

    # Attempt 1: Poor answer (score 40)
    ans1 = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        attempt_number=1,
        candidate_answer="You just copy everything into one stage.",
        status="EVALUATED",
        overall_score=40.0,
        evaluation_confidence=0.85
    )
    db_session.add(ans1)
    db_session.commit()
    scorecard_aggregator.calculate_session_scorecard(db_session, session.id)

    db_session.refresh(session)
    assert session.overall_score == 40.0

    # Attempt 2: Improved answer (score 85)
    ans2 = InterviewAnswer(
        session_id=session.id,
        question_id=q.id,
        attempt_number=2,
        candidate_answer="Multi-stage builds separate build dependencies from the minimal runtime image using FROM ... AS builder.",
        status="EVALUATED",
        overall_score=85.0,
        evaluation_confidence=0.90
    )
    db_session.add(ans2)
    db_session.commit()
    scorecard_aggregator.calculate_session_scorecard(db_session, session.id)

    # Invariants Check: Scorecard uses latest attempt
    db_session.refresh(session)
    assert session.overall_score == 85.0

    # Attempt history returns both attempts chronologically
    attempts = db_session.query(InterviewAnswer).filter(InterviewAnswer.question_id == q.id).order_by(InterviewAnswer.attempt_number.asc()).all()
    assert len(attempts) == 2
    assert attempts[0].attempt_number == 1
    assert attempts[1].attempt_number == 2

    # Idempotent Next-Question Retry Check
    idem_key = f"key_{uuid4().hex}"
    req_override = AdaptiveQuestionSelectionRequest(request_idempotency_key=idem_key)
    mock_gen = make_mock_rag_side_effect(
        question_text="How do you configure Kafka partition replication factor?",
        target_entity="Kafka",
        category="Architecture",
        difficulty=3
    )
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen):
        res1 = session_service.select_and_generate_next_question(session.id, db_session, request_override=req_override)
        # Immediate retry with same key
        res2 = session_service.select_and_generate_next_question(session.id, db_session, request_override=req_override)

    assert res1.question_id == res2.question_id
    assert res1.question_text == res2.question_text
    db_session.refresh(session)
    assert session.total_questions == 2  # q and the newly generated question, no duplicates


# =========================================================================
# SCENARIO H: Budget Exhaustion, Session Recovery, and Scorecard Finalization
# =========================================================================
def test_scenario_h_budget_exhaustion_recovery_and_final_scorecard(db_session: Session, multi_turn_resume: Resume):
    """
    Scenario H:
    1. Multi-turn interview runs until configured budget (max_questions: 3) is exhausted.
    2. Further generation requests return BUDGET_EXHAUSTED.
    3. Session state reloaded from persistence matches in-memory counters.
    4. Session ends cleanly with status COMPLETED and finalized scorecard.
    5. Finalization is idempotent.
    """
    session = session_service.create_session(
        resume_id=multi_turn_resume.id,
        db=db_session,
        initial_difficulty=3,
        session_metadata={"adaptive_config": {"max_questions": 3}}
    )

    # Run 3 turns
    turn_entities = ["FastAPI", "PostgreSQL", "Redis"]
    turn_scores = [80.0, 75.0, 85.0]

    for i in range(3):
        mock_gen = make_mock_rag_side_effect(
            question_text=f"Turn {i + 1} Question on {turn_entities[i]} architecture?",
            target_entity=turn_entities[i],
            category="Architecture",
            difficulty=3
        )
        with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen):
            q_resp = session_service.select_and_generate_next_question(session.id, db_session)

        # Submit evaluated answer
        ans = InterviewAnswer(
            session_id=session.id,
            question_id=q_resp.question_id,
            candidate_answer=f"Detailed substantive technical answer for {turn_entities[i]}.",
            status="EVALUATED",
            overall_score=turn_scores[i],
            evaluation_confidence=0.90
        )
        db_session.add(ans)
        db_session.commit()
        knowledge_state_service.update_from_evaluation(
            session_id=session.id,
            question_id=q_resp.question_id,
            answer=ans,
            db=db_session
        )

    # Turn 4: Request next question -> Budget exhausted
    budget_resp = session_service.select_and_generate_next_question(session.id, db_session)
    assert budget_resp.status == "BUDGET_EXHAUSTED"
    assert budget_resp.selection is None

    # Reload session state from DB and verify consistency
    reloaded_state = session_service.get_interview_state(session.id, db_session)
    assert reloaded_state.progress.total_questions == 3
    assert reloaded_state.progress.completed_questions == 3
    assert reloaded_state.progress.progress_percentage == 100.0

    # Finalize interview session
    finalized_session = session_service.end_interview(session.id, db_session, reason="ALL_QUESTIONS_COMPLETED")
    assert finalized_session.status == "COMPLETED"
    assert finalized_session.completed_at is not None
    assert finalized_session.overall_score == round(sum(turn_scores) / 3, 2)

    # Idempotent conclusion: Repeated end call returns completed session safely
    repeated_end = session_service.end_interview(session.id, db_session)
    assert repeated_end.status == "COMPLETED"
    assert repeated_end.completed_at == finalized_session.completed_at
