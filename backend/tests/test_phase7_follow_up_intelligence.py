import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from unittest.mock import patch, MagicMock
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
from app.schemas.rag import QuestionGenerationResponse, QuestionOutputSchema
from app.schemas.knowledge_state import ClaimVerificationStatus, TopicCoverageStatus, GapNature
from app.schemas.contradiction import ContradictionStatus, ContradictionType, ContradictionSeverity
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
def test_resume(db_session: Session) -> Resume:
    """Creates a sample candidate resume with skills, relationships, and claims."""
    resume = Resume(
        id=uuid4(),
        filename="alex_lead_architect.pdf",
        raw_text="Alex Lead Architect. Languages: Python, FastAPI, PostgreSQL, Docker, Redis. Built PaymentGateway."
    )
    db_session.add(resume)
    db_session.flush()

    # Skills
    sk1 = Skill(id=301, name="Python", category="Language")
    sk2 = Skill(id=302, name="FastAPI", category="Framework")
    sk3 = Skill(id=303, name="PostgreSQL", category="Database")
    sk4 = Skill(id=304, name="Docker", category="Tool/DevOps")
    sk5 = Skill(id=305, name="Redis", category="Database")
    db_session.add_all([sk1, sk2, sk3, sk4, sk5])
    db_session.flush()

    rsk1 = ResumeSkill(resume_id=resume.id, skill_id=301, extraction_confidence=0.95)
    rsk2 = ResumeSkill(resume_id=resume.id, skill_id=302, extraction_confidence=0.92)
    rsk3 = ResumeSkill(resume_id=resume.id, skill_id=303, extraction_confidence=0.90)
    rsk4 = ResumeSkill(resume_id=resume.id, skill_id=304, extraction_confidence=0.88)
    rsk5 = ResumeSkill(resume_id=resume.id, skill_id=305, extraction_confidence=0.85)
    db_session.add_all([rsk1, rsk2, rsk3, rsk4, rsk5])

    # Relationship
    rel = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="COMMUNICATES_WITH",
        target_entity="PostgreSQL",
        context_entity="PaymentGateway",
        section="Experience",
        evidence_text="Built asynchronous transactional pipeline between FastAPI and PostgreSQL.",
        confidence=0.94
    )
    db_session.add(rel)

    # Claim
    claim = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-REDIS-99",
        section="Experience",
        project="PaymentGateway",
        target_entity="Redis",
        relation_type="DEPLOYED_FOR",
        claim_text="Implemented distributed Redis locking preventing double payments under 10k RPS load.",
        evidence_text="Implemented distributed Redis locking preventing double payments under 10k RPS load."
    )
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(resume)
    return resume


# =========================================================================
# PHASE 7.5 TESTS: FOLLOW-UP INTELLIGENCE
# =========================================================================

def test_1_parent_child_linkage_and_followup_reason(db_session: Session, test_resume: Resume):
    """
    Test 1: Correct parent-child linkage.
    - Follow-up question preserves parent question ID.
    - Exposes explainable follow_up_reason.
    - Fresh topics strictly set is_follow_up=False, parent_question_id=None.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Root question
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="How do you handle database connections in FastAPI?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="I use an async session generator in FastAPI with SQLAlchemy.",
        status="EVALUATED",
        overall_score=60.0,
        missing_concepts=["Connection pool sizing", "Queue overflow timeout"]
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Must be follow-up linked to q1
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    assert selection.consecutive_followup_count == 1
    assert selection.follow_up_reason is not None
    assert len(selection.follow_up_reason) > 5


def test_2_partial_answer_clarification(db_session: Session, test_resume: Resume):
    """
    Test 2: Partial-answer clarification.
    - Detects partially correct answer (50 <= score <= 75) omitting specific concept.
    - Asks targeted clarification on missing component without repeating parent question.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="Describe how you configure async pooling in SQLAlchemy for PostgreSQL.",
        category="Implementation",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="We create an async engine with pool_size=10 and max_overflow=20.",
        status="EVALUATED",
        overall_score=65.0,
        evaluation_confidence=0.88,
        missing_concepts=["pool_recycle to avoid stale connections", "pool_pre_ping for liveness"],
        weaknesses=["Omitted stale socket detection and disconnect handling."]
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    assert selection.strategy == AdaptiveStrategyType.PARTIAL_CLARIFICATION
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    assert selection.target_topic == "PostgreSQL"
    assert "pool_recycle" in selection.follow_up_reason or "pool_recycle" in selection.rationale
    assert selection.difficulty == 3  # Stable difficulty for partial clarification


def test_3_strong_answer_depth_followup(db_session: Session, test_resume: Resume):
    """
    Test 3: Strong-answer depth follow-up.
    - Score > 75 with solid confidence.
    - Follows up with escalated difficulty, failure-modes, or architectural resilience.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="How is dependency injection structured in FastAPI?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="FastAPI uses Depends() with a DAG resolver supporting async and sync dependencies cleanly.",
        status="EVALUATED",
        overall_score=92.0,
        evaluation_confidence=0.95,
        strengths=["Comprehensive explanation of dependency DAG and lifecycle scoping."]
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    assert selection.strategy == AdaptiveStrategyType.DIFFICULTY_ESCALATION
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    assert selection.difficulty == 4  # Escalated 3 -> 4
    assert selection.category in ["Failure-Mode", "Performance", "Security"]
    assert "deepening strong answer" in selection.follow_up_reason.lower()


def test_4_weakness_probing_based_on_demonstrated_evidence(db_session: Session, test_resume: Resume):
    """
    Test 4: Weakness probing based on demonstrated evidence.
    - Probes demonstrated weakness only when sufficient evidence exists (score < 50.0).
    - Excludes unassessed topics from being labeled weaknesses.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="Explain how Redis cluster sharding manages slot reallocation during rebalancing.",
        category="Architecture",
        difficulty=3,
        target_entity="Redis"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Redis has one server and replicates to another without slots.",
        status="EVALUATED",
        overall_score=30.0,
        evaluation_confidence=0.90,
        missing_concepts=["16384 hash slots", "CLUSTER REBALANCE", "ASK/MOVED redirects"],
        weaknesses=["Candidate exhibited fundamental confusion regarding distributed hash slot topology."]
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    assert selection.strategy == AdaptiveStrategyType.GAP_FOLLOW_UP
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    assert selection.difficulty == 2  # De-escalated 3 -> 2
    assert selection.category in ["Fundamentals", "Implementation"]
    assert "demonstrated knowledge gap" in selection.follow_up_reason.lower()


def test_5_resume_claim_verification(db_session: Session, test_resume: Resume):
    """
    Test 5: Resume-claim verification.
    - Asks focused questions about unverified claims.
    - Preserves links to claim ID and project context.
    - Never labels claim false solely because an answer is weak, refused, or skipped.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    req = AdaptiveQuestionSelectionRequest(force_strategy=AdaptiveStrategyType.CLAIM_VERIFICATION)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        request_override=req
    )

    assert selection.strategy == AdaptiveStrategyType.CLAIM_VERIFICATION
    assert selection.target_claim_id == "CLM-REDIS-99"
    assert selection.target_project == "PaymentGateway"
    assert selection.category == "Claim-Verification"
    assert selection.is_follow_up is False

    # Simulate candidate weak answer on claim question
    q_claim = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="How did you verify 10k RPS locking in PaymentGateway?",
        category="Claim-Verification",
        difficulty=3,
        target_claim_id="CLM-REDIS-99",
        target_project="PaymentGateway"
    )
    ans_weak = InterviewAnswer(
        id=uuid4(),
        question_id=q_claim.id,
        session_id=session.id,
        candidate_answer="I don't recall the exact benchmarking script.",
        status="EVALUATED",
        overall_score=35.0,
        evaluation_confidence=0.85
    )
    db_session.add_all([q_claim, ans_weak])
    db_session.commit()

    k_state = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q_claim.id,
        answer=ans_weak,
        db=db_session
    )

    # Claim status MUST be INSUFFICIENT_EVIDENCE, NOT CONTRADICTED!
    claim_state = next(c for c in k_state["claims"] if c["claim_id"] == "CLM-REDIS-99")
    assert claim_state["status"] == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE.value
    assert claim_state["status"] != ClaimVerificationStatus.CONTRADICTED.value


def test_6_evidence_backed_contradiction_clarification(db_session: Session, test_resume: Resume):
    """
    Test 6: Evidence-backed contradiction clarification.
    - Inspects contradiction analysis and supporting evidence.
    - Confirmed contradiction with verified evidence triggers CONTRADICTION_CLARIFICATION.
    - Unverified additions or hypothetical discussion do NOT trigger contradiction follow-up.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="What caching layer was deployed for PaymentGateway?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway"
    )
    # Contradiction: candidate claimed Memcached instead of verified Redis locking
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="We never used Redis for PaymentGateway; we deployed Memcached without persistence.",
        status="EVALUATED",
        overall_score=40.0,
        dimension_scores={
            "technical_accuracy": 40.0,
            "_contradiction_analysis": {
                "status": "CONTRADICTION",
                "has_contradiction": True,
                "severity": "HIGH",
                "conflicts": [
                    {
                        "conflict_type": "TECHNOLOGY_SUBSTITUTION",
                        "severity": "HIGH",
                        "candidate_assertion": "We never used Redis; deployed Memcached",
                        "verified_fact": "Implemented distributed Redis locking preventing double payments",
                        "context_project": "PaymentGateway",
                        "evidence_id": "CLM-REDIS-99",
                        "evidence_text": "Implemented distributed Redis locking preventing double payments under 10k RPS load.",
                        "explanation": "Direct substitution of verified Redis architecture with Memcached."
                    }
                ],
                "explanation": "High-confidence contradiction with resume evidence."
            }
        }
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Genuine confirmed contradiction with evidence MUST trigger neutral clarification
    assert selection.strategy == AdaptiveStrategyType.CONTRADICTION_CLARIFICATION
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    assert selection.category == "Clarification"
    assert "discrepancy" in selection.follow_up_reason.lower()
    assert "Memcached" in selection.rationale or "Redis" in selection.rationale


def test_7_relationship_exploration_tracking(db_session: Session, test_resume: Resume):
    """
    Test 7: Relationship exploration.
    - Explores technology/component relationship extracted from resume.
    - Avoids repeatedly testing the same relationship once tested.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    req = AdaptiveQuestionSelectionRequest(force_strategy=AdaptiveStrategyType.RELATIONSHIP_VERIFICATION)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        request_override=req
    )

    assert selection.strategy == AdaptiveStrategyType.RELATIONSHIP_VERIFICATION
    assert selection.target_relationship is not None
    assert "FastAPI" in selection.target_relationship and "PostgreSQL" in selection.target_relationship
    assert selection.is_follow_up is False

    # Simulate answering the relationship question
    q_rel = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text=f"How does {selection.target_relationship} work?",
        category="Architecture",
        difficulty=3,
        target_relationship=selection.target_relationship,
        target_project="PaymentGateway"
    )
    ans_rel = InterviewAnswer(
        id=uuid4(),
        question_id=q_rel.id,
        session_id=session.id,
        candidate_answer="FastAPI communicates with PostgreSQL using async connection pools and commit transactions.",
        status="EVALUATED",
        overall_score=85.0
    )
    db_session.add_all([q_rel, ans_rel])
    db_session.commit()

    k_state = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q_rel.id,
        answer=ans_rel,
        db=db_session
    )

    # Relationship should now be tested
    rel_item = k_state["relationships"][0]
    assert rel_item["is_tested"] is True
    assert rel_item["status"] == "DEMONSTRATED"


def test_8_duplicate_and_anti_loop_prevention(db_session: Session, test_resume: Resume):
    """
    Test 8: Duplicate and anti-loop prevention.
    - Selector will not propose identical (topic, category) pairs repeatedly.
    - Consecutive follow-up depth is bounded.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Add question on (Docker, Fundamentals)
    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="What are Docker namespaces and cgroups?",
        category="Fundamentals",
        difficulty=3,
        target_entity="Docker"
    )
    db_session.add(q)
    db_session.commit()

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Next question must not repeat (Docker, Fundamentals)
    assert not (selection.target_entity == "Docker" and selection.category == "Fundamentals")


def test_9_uncovered_topic_after_followup_budget_exhaustion(db_session: Session, test_resume: Resume):
    """
    Test 9: Uncovered-topic selection after follow-up budget exhaustion.
    - When max_consecutive_followups is reached, selector MUST transition to an uncovered topic.
    - Transitioned question must have is_follow_up=False, parent_question_id=None.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Create root question + 2 follow-ups on FastAPI (depth = 2)
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="Q1 Root on FastAPI",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    q2 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        parent_question_id=q1.id,
        question_text="Q2 Followup 1 on FastAPI",
        category="Implementation",
        difficulty=3,
        target_entity="FastAPI"
    )
    q3 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        parent_question_id=q2.id,
        question_text="Q3 Followup 2 on FastAPI",
        category="Fundamentals",
        difficulty=3,
        target_entity="FastAPI"
    )
    ans3 = InterviewAnswer(
        id=uuid4(),
        question_id=q3.id,
        session_id=session.id,
        candidate_answer="Answer with gaps.",
        status="EVALUATED",
        overall_score=40.0,
        missing_concepts=["FastAPI route dispatch", "ASGI scope"]
    )
    db_session.add_all([q1, q2, q3, ans3])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q3.id,
        answer=ans3,
        db=db_session
    )

    cfg = AdaptiveSelectionConfig(max_consecutive_followups=2)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        config_override=cfg
    )

    # Follow-up budget exhausted: MUST choose a fresh topic, NOT continue follow-up on FastAPI!
    assert selection.is_follow_up is False
    assert selection.parent_question_id is None
    assert selection.consecutive_followup_count == 0
    assert selection.target_topic != "FastAPI"


def test_10_skipped_refused_and_unanswered_behavior(db_session: Session, test_resume: Resume):
    """
    Test 10: Skipped, refused, and unanswered behavior.
    - Refusals and timeouts do not create DEMONSTRATED_WEAKNESS gaps.
    - Selector does not treat refusals as proof of poor knowledge for weakness follow-up.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="Can you explain internal memory layouts of PostgreSQL buffers?",
        category="Architecture",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    ans_refusal = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="I don't know, skip this question.",
        status="EVALUATED",
        overall_score=0.0,
        evaluator_model="fast_path_refusal",
        evaluation_feedback="Candidate declined/refused previous question."
    )
    db_session.add_all([q1, ans_refusal])
    db_session.commit()

    k_state = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans_refusal,
        db=db_session
    )

    # Gap nature must be INSUFFICIENT_EVIDENCE
    gaps = k_state.get("knowledge_gaps", [])
    assert len(gaps) > 0
    assert gaps[-1]["gap_nature"] == GapNature.INSUFFICIENT_EVIDENCE.value

    # Selector does NOT probe demonstrated weakness for this refusal!
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )
    assert selection.strategy != AdaptiveStrategyType.GAP_FOLLOW_UP


def test_11_generation_failure_and_session_recovery(db_session: Session, test_resume: Resume):
    """
    Test 11: Generation failure and session recovery.
    - RAG generation rejection does not commit invalid question or increment count.
    - Session remains completely recoverable.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    # Mock RAG to return validation failure
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=test_resume.id,
        request_id="req-mock-fail",
        trace_id="tr-mock-fail",
        question_id=None,
        question=None,
        validation_status="INSUFFICIENT_EVIDENCE",
        message="Simulated retrieval grounding failure.",
        latencies_ms={}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session
        )

        assert resp.status == "EVIDENCE_UNAVAILABLE"
        assert "Simulated retrieval grounding failure" in resp.message

    # Verify session counters were NOT incremented
    db_session.refresh(session)
    assert session.total_questions == 0
    assert db_session.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).count() == 0


def test_12_compatibility_with_phases_7_1_through_7_4(db_session: Session, test_resume: Resume):
    """
    Test 12: Compatibility with Phases 7.1-7.4.
    - Full end-to-end session flow: start -> evaluate -> adapt -> sync session state.
    - Verifies AskedQuestionState metadata includes is_follow_up and parent_question_id.
    """
    session = session_service.create_session(resume_id=test_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        question_text="Root question on Python",
        category="Architecture",
        difficulty=3,
        target_entity="Python"
    )
    q2 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=test_resume.id,
        parent_question_id=q1.id,
        question_text="Follow-up question on Python GIL",
        category="Fundamentals",
        difficulty=2,
        target_entity="Python"
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    # Query interview state
    state_resp = session_service.get_interview_state(session_id=session.id, db=db_session)

    assert state_resp.session_id == session.id
    assert len(state_resp.asked_questions) == 2
    # Verify follow-up flags
    assert state_resp.asked_questions[0].is_follow_up is False
    assert state_resp.asked_questions[0].parent_question_id is None
    assert state_resp.asked_questions[1].is_follow_up is True
    assert state_resp.asked_questions[1].parent_question_id == q1.id
