import time
from uuid import uuid4, UUID
from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException
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
    AdaptiveNextQuestionResponse,
)
from app.schemas.rag import QuestionGenerationRequest, QuestionGenerationResponse, QuestionOutputSchema
from app.schemas.knowledge_state import TopicCoverageStatus
from app.services.rag.rag_pipeline import rag_pipeline
from app.services.rag.validator import grounding_validator


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
def multi_topic_resume(db_session: Session) -> Resume:
    """Creates a resume with multiple distinct technical skills, relationships, and claims."""
    resume = Resume(
        id=uuid4(),
        filename="engineer_resume.pdf",
        raw_text="Full-stack engineer. Python, FastAPI, PostgreSQL, Docker, Redis. Built PaymentGateway."
    )
    db_session.add(resume)
    db_session.flush()

    skills = [
        Skill(id=501, name="Python", category="Language"),
        Skill(id=502, name="FastAPI", category="Framework"),
        Skill(id=503, name="PostgreSQL", category="Database"),
        Skill(id=504, name="Docker", category="Tool/DevOps"),
        Skill(id=505, name="Redis", category="Database"),
    ]
    db_session.add_all(skills)
    db_session.flush()

    for sk in skills:
        db_session.add(ResumeSkill(resume_id=resume.id, skill_id=sk.id, extraction_confidence=0.92))

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


def make_accepted_rag_response(
    question_text: str,
    target_entity: str,
    target_project: str = "PaymentGateway",
    category: str = "Architecture",
    difficulty: int = 3,
    resume_id: UUID = None,
    session_id: UUID = None,
    db: Session = None
) -> QuestionGenerationResponse:
    q_id = uuid4()
    q_out = QuestionOutputSchema(
        question=question_text,
        category=category,
        difficulty=difficulty,
        target_project=target_project,
        target_entity=target_entity,
        evidence_ids=["CLM-REDIS-99"],
        evidence_text="Evidence text verifying technology implementation.",
        rationale="Technical rationale.",
        grounding_score=0.95,
        relevance_score=0.90,
        technical_quality_score=0.88,
        trace_id="trace_test"
    )

    if db is not None and session_id is not None:
        q_rec = InterviewQuestion(
            id=q_id,
            resume_id=resume_id or uuid4(),
            session_id=session_id,
            question_text=question_text,
            category=category,
            difficulty=difficulty,
            target_project=target_project,
            target_entity=target_entity,
            evidence_ids=["CLM-REDIS-99"],
            evidence_text="Evidence text verifying technology implementation.",
            rationale="Technical rationale.",
            grounding_score=0.95,
            relevance_score=0.90,
            technical_quality_score=0.88
        )
        db.add(q_rec)
        db.commit()
        db.refresh(q_rec)

    return QuestionGenerationResponse(
        resume_id=resume_id or uuid4(),
        request_id=str(uuid4()),
        trace_id="trace_test",
        question_id=q_id,
        question=q_out,
        validation_status="ACCEPTED",
        message="Question successfully generated and verified.",
        attempts_count=1,
        latencies_ms={"total_ms": 120.0},
        assessed_difficulty=difficulty,
        rubric_version="v2.0-canonical-5level"
    )


def make_duplicate_rejected_rag_response(
    rejected_question_text: str,
    resume_id: UUID = None
) -> QuestionGenerationResponse:
    return QuestionGenerationResponse(
        resume_id=resume_id or uuid4(),
        request_id=str(uuid4()),
        trace_id="trace_dup",
        question=None,
        validation_status="INSUFFICIENT_EVIDENCE",
        error_code="redundancy_rejection",
        message=f"Candidate question was rejected as semantic duplicate: '{rejected_question_text}'",
        attempts_count=2,
        latencies_ms={"total_ms": 95.0},
        rejected_candidates=[rejected_question_text]
    )


# =========================================================================
# TEST 1: Automatic skip of single duplicate candidate & return valid alternative
# =========================================================================
def test_duplicate_candidate_automatically_skipped_and_different_valid_question_returned(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Core Requirement:
    When a generated question fails semantic duplicate validation, automatically
    discard that candidate, exclude the rejected concept, select a different
    technical dimension, and generate and return a valid alternative question.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)

    # Pre-existing question in session
    existing_q = InterviewQuestion(
        resume_id=multi_topic_resume.id,
        session_id=session.id,
        question_text="How did you configure Redis distributed locking in PaymentGateway to handle high concurrency?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis",
        evidence_text="Implemented distributed Redis locking preventing double payments under 10k RPS load."
    )
    db_session.add(existing_q)
    db_session.commit()

    duplicate_candidate_text = "How did you set up distributed Redis locks in PaymentGateway under high concurrency?"
    valid_candidate_text = "How would you structure database indexing and query plans in PostgreSQL for PaymentGateway?"

    call_count = 0
    generated_targets = []

    def mock_rag_generate(resume_id, req, db):
        nonlocal call_count, generated_targets
        call_count += 1
        generated_targets.append((req.target_entity, req.category))

        # First candidate targets Redis / PaymentGateway -> duplicate rejection
        if call_count == 1:
            return make_duplicate_rejected_rag_response(
                rejected_question_text=duplicate_candidate_text,
                resume_id=resume_id
            )
        # Second candidate targets different technical dimension (e.g. PostgreSQL) -> valid accepted
        return make_accepted_rag_response(
            question_text=valid_candidate_text,
            target_entity=req.target_entity or "PostgreSQL",
            target_project=req.target_project or "PaymentGateway",
            category=req.category,
            difficulty=req.difficulty,
            resume_id=resume_id,
            session_id=req.session_id,
            db=db
        )

    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_rag_generate):
        resp = session_service.select_and_generate_next_question(session.id, db_session)

    # 1. Returned question is the valid alternative, NOT the duplicate candidate
    assert resp.status == "GENERATED"
    assert resp.question_text == valid_candidate_text
    assert resp.target_entity != "Redis"
    assert call_count == 2

    # 2. Duplicate candidate was discarded and NEVER persisted in DB
    all_questions = db_session.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).all()
    all_texts = [q.question_text for q in all_questions]
    assert duplicate_candidate_text not in all_texts
    assert valid_candidate_text in all_texts
    assert len(all_questions) == 2  # existing_q + valid alternative

    # 3. Question budget was NOT consumed by the rejected candidate
    assert session.total_questions == 2


# =========================================================================
# TEST 2: Multiple duplicate candidates skipped until valid candidate found
# =========================================================================
def test_multiple_duplicate_candidates_skipped_until_valid_candidate_found(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Validates that if multiple candidates are rejected sequentially as duplicates,
    the engine discards each rejected candidate, accumulates exclusions, and continues
    until a valid alternative candidate passes validation.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)

    # Pre-existing questions
    q1 = InterviewQuestion(
        resume_id=multi_topic_resume.id,
        session_id=session.id,
        question_text="How did you configure Redis distributed locking in PaymentGateway?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="Redis"
    )
    q2 = InterviewQuestion(
        resume_id=multi_topic_resume.id,
        session_id=session.id,
        question_text="How did you design asynchronous communication between FastAPI and PostgreSQL?",
        category="Architecture",
        difficulty=3,
        target_project="PaymentGateway",
        target_entity="FastAPI"
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    dup_text_1 = "How did you implement distributed Redis locks in PaymentGateway?"
    dup_text_2 = "How do FastAPI and PostgreSQL communicate asynchronously in PaymentGateway?"
    valid_text_3 = "How would you design container isolation and resource constraints in Docker for PaymentGateway?"

    call_count = 0
    candidate_requests = []

    def mock_multi_rag(resume_id, req, db):
        nonlocal call_count, candidate_requests
        call_count += 1
        candidate_requests.append(req)

        if call_count == 1:
            return make_duplicate_rejected_rag_response(dup_text_1, resume_id=resume_id)
        elif call_count == 2:
            return make_duplicate_rejected_rag_response(dup_text_2, resume_id=resume_id)
        else:
            return make_accepted_rag_response(
                question_text=valid_text_3,
                target_entity="Docker",
                target_project="PaymentGateway",
                category=req.category,
                difficulty=req.difficulty,
                resume_id=resume_id,
                session_id=req.session_id,
                db=db
            )

    cfg = AdaptiveSelectionConfig(max_candidate_retries=4)
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_multi_rag):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session,
            config_override=cfg
        )

    # 1. Succeeded on candidate 3
    assert resp.status == "GENERATED"
    assert resp.question_text == valid_text_3
    assert call_count == 3

    # 2. Only candidate 3 was persisted
    persisted = db_session.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).all()
    persisted_texts = {q.question_text for q in persisted}
    assert dup_text_1 not in persisted_texts
    assert dup_text_2 not in persisted_texts
    assert valid_text_3 in persisted_texts
    assert len(persisted) == 3  # q1 + q2 + candidate 3

    # 3. Budget accurately counts exactly 3 questions
    assert session.total_questions == 3


# =========================================================================
# TEST 3: Rejected candidates do not consume budget or mark topics covered
# =========================================================================
def test_rejected_candidates_not_persisted_or_counted_as_asked_questions(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Proves that rejected candidates never consume the question budget, never
    create ghost InterviewQuestion records, and never mark knowledge state topics
    as QUESTION_ASKED prematurely.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)
    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)

    # Verify initial coverage status of Docker and Redis
    assert k_state["topics"]["Redis"]["questions_asked_count"] == 0
    assert k_state["topics"]["Redis"]["coverage_status"] == TopicCoverageStatus.NOT_CONSIDERED.value

    dup_text = "How did you handle Redis distributed locking?"
    valid_text = "How would you manage schema migrations in PostgreSQL?"

    attempt = 0
    def mock_gen(resume_id, req, db):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return make_duplicate_rejected_rag_response(dup_text, resume_id=resume_id)
        return make_accepted_rag_response(
            question_text=valid_text,
            target_entity="PostgreSQL",
            category="Implementation",
            difficulty=3,
            resume_id=resume_id,
            session_id=req.session_id,
            db=db
        )

    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_gen):
        resp = session_service.select_and_generate_next_question(session.id, db_session)

    assert resp.status == "GENERATED"

    # Reload session and knowledge state
    db_session.refresh(session)
    reloaded_k_state = session.session_state["knowledge_state"]

    # Rejected target "Redis" was NOT marked asked
    assert reloaded_k_state["topics"]["Redis"]["questions_asked_count"] == 0

    # Accepted target "PostgreSQL" was properly marked asked
    assert reloaded_k_state["topics"]["PostgreSQL"]["questions_asked_count"] == 1
    assert reloaded_k_state["topics"]["PostgreSQL"]["coverage_status"] == TopicCoverageStatus.QUESTION_ASKED.value

    # Exactly 1 question in DB and session counter
    q_count = db_session.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).count()
    assert q_count == 1
    assert session.total_questions == 1


# =========================================================================
# TEST 4: Retry attempts exclude previously rejected concepts & texts
# =========================================================================
def test_retry_attempts_exclude_previously_rejected_concepts(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Proves that subsequent candidate attempts receive excluded_topics and
    excluded_question_texts from prior rejected candidate attempts.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)

    rejected_text_1 = "How did you set up Redis caching?"
    captured_reqs = []

    def mock_rag_capture(resume_id, req, db):
        captured_reqs.append(req)
        if len(captured_reqs) == 1:
            return make_duplicate_rejected_rag_response(rejected_text_1, resume_id=resume_id)
        return make_accepted_rag_response(
            question_text="How would you design container networks in Docker?",
            target_entity="Docker",
            resume_id=resume_id,
            session_id=req.session_id,
            db=db
        )

    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_rag_capture):
        resp = session_service.select_and_generate_next_question(session.id, db_session)

    assert resp.status == "GENERATED"
    assert len(captured_reqs) == 2

    # Attempt 2 must explicitly receive the rejected question text in excluded_question_texts
    assert rejected_text_1 in captured_reqs[1].excluded_question_texts

    # Direct validator check: extra_excluded_texts flags duplicates of rejected candidate
    is_dup, msg = grounding_validator.check_semantic_duplicate(
        new_question_text="How did you set up Redis caching?",
        existing_questions=[],
        extra_excluded_texts=[rejected_text_1]
    )
    assert is_dup
    assert "rejected candidate" in msg.lower()


# =========================================================================
# TEST 5: Timeout or exhausted retry budget produces recoverable failure
# =========================================================================
def test_timeout_or_exhausted_retry_budget_produces_recoverable_failure(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Proves that when all candidate attempts are rejected as semantic duplicates,
    or the retry budget is exhausted:
    1. Returns structured recoverable failure (status='EVIDENCE_UNAVAILABLE', error_code='redundancy_rejection').
    2. Session status remains 'IN_PROGRESS' (does not end or corrupt session).
    3. User can retry safely.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)

    # Insert 1 evaluated question/answer to ensure history exists
    existing_q = InterviewQuestion(
        resume_id=multi_topic_resume.id,
        session_id=session.id,
        question_text="Explain Python GIL and concurrency trade-offs.",
        category="Fundamentals",
        difficulty=3
    )
    db_session.add(existing_q)
    db_session.commit()

    ans = InterviewAnswer(
        session_id=session.id,
        question_id=existing_q.id,
        candidate_answer="The GIL ensures thread safety for reference counting in CPython.",
        status="EVALUATED",
        overall_score=85.0
    )
    db_session.add(ans)
    session.total_questions = 1
    session.completed_questions = 1
    db_session.commit()

    # All candidate attempts fail with redundancy_rejection
    def mock_all_duplicate(resume_id, req, db):
        return make_duplicate_rejected_rag_response("Duplicate question attempt", resume_id=resume_id)

    cfg = AdaptiveSelectionConfig(max_candidate_retries=3, candidate_timeout_seconds=25.0)
    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=mock_all_duplicate):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session,
            config_override=cfg
        )

    # 1. Structured recoverable failure response
    assert resp.status == "EVIDENCE_UNAVAILABLE"
    assert resp.error_code == "redundancy_rejection"
    assert "semantic duplicates" in resp.message.lower()

    # 2. Session state is NOT corrupted or completed
    db_session.refresh(session)
    assert session.status == "IN_PROGRESS"
    assert session.completed_at is None
    assert session.total_questions == 1
    assert session.completed_questions == 1

    # 3. Existing evaluated answers remain completely intact
    persisted_answers = db_session.query(InterviewAnswer).filter(InterviewAnswer.session_id == session.id).all()
    assert len(persisted_answers) == 1
    assert persisted_answers[0].overall_score == 85.0


def test_candidate_timeout_produces_recoverable_failure(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Proves that when total candidate elapsed time exceeds the candidate timeout budget:
    1. Loop terminates cleanly without hanging.
    2. Returns status='EVIDENCE_UNAVAILABLE' with error_code='timeout'.
    3. Session status remains 'IN_PROGRESS'.
    """
    session = session_service.create_session(resume_id=multi_topic_resume.id, db=db_session)

    # Set timeout budget to tiny fraction of second
    cfg = AdaptiveSelectionConfig(max_candidate_retries=4, candidate_timeout_seconds=0.001)

    def slow_rag_generator(resume_id, req, db):
        time.sleep(0.01)
        return make_duplicate_rejected_rag_response("Slow candidate", resume_id=resume_id)

    with patch.object(rag_pipeline, "generate_grounded_question", side_effect=slow_rag_generator):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session,
            config_override=cfg
        )

    assert resp.status == "EVIDENCE_UNAVAILABLE"
    assert resp.error_code == "timeout"
    assert "timed out" in resp.message.lower()

    db_session.refresh(session)
    assert session.status == "IN_PROGRESS"


# =========================================================================
# TEST 6: Genuine question-budget exhaustion completes session correctly
# =========================================================================
def test_genuine_question_budget_exhaustion_completes_session_correctly(
    db_session: Session, multi_topic_resume: Resume
):
    """
    Proves that when the configured session question budget is genuinely exhausted,
    the pipeline returns BUDGET_EXHAUSTED and allows completing the session
    via the intended lifecycle without generating spurious candidates.
    """
    session = session_service.create_session(
        resume_id=multi_topic_resume.id,
        db=db_session,
        session_metadata={"adaptive_config": {"max_questions": 2}}
    )

    # Add 2 questions and evaluated answers to reach budget
    for i in range(2):
        q = InterviewQuestion(
            resume_id=multi_topic_resume.id,
            session_id=session.id,
            question_text=f"Question {i + 1} on system design?",
            category="Architecture",
            difficulty=3
        )
        db_session.add(q)
        db_session.flush()

        ans = InterviewAnswer(
            session_id=session.id,
            question_id=q.id,
            candidate_answer=f"Technical answer {i + 1}.",
            status="EVALUATED",
            overall_score=80.0
        )
        db_session.add(ans)

    session.total_questions = 2
    session.completed_questions = 2
    db_session.commit()

    # Calling select_and_generate_next_question returns BUDGET_EXHAUSTED
    resp = adaptive_selection_engine.select_and_generate_next_question(session.id, db_session)
    assert resp.status == "BUDGET_EXHAUSTED"
    assert resp.selection is None
    assert "maximum question budget" in resp.message

    # Finalize session through existing intended lifecycle
    final_session = session_service.end_interview(session.id, db_session, reason="ALL_QUESTIONS_COMPLETED")
    assert final_session.status == "COMPLETED"
    assert final_session.completed_at is not None
