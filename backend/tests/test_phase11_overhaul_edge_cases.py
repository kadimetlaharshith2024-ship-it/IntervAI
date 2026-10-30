import pytest
from uuid import uuid4
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.user import User
from app.models.resume import Resume
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.session_service import session_service
from app.services.adaptive_selection_service import adaptive_selection_engine
from app.schemas.adaptive_selection import (
    AdaptiveQuestionSelectionRequest,
    NextQuestionSelectionResult,
    AdaptiveStrategyType
)
from app.schemas.rag import QuestionGenerationResponse, QuestionOutputSchema
from app.core.database import get_db
from app.main import app


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db_session: Session):
    user = User(
        id=uuid4(),
        email="test.candidate@example.com",
        password_hash="hashed_pwd_placeholder",
        full_name="Test Candidate"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session: Session):
    user = User(
        id=uuid4(),
        email="other.candidate@example.com",
        password_hash="hashed_pwd_placeholder",
        full_name="Other Candidate"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_resume(db_session: Session, test_user: User):
    resume = Resume(
        id=uuid4(),
        user_id=test_user.id,
        filename="test_resume.pdf",
        raw_text="Experienced in Python, FastAPI, and PostgreSQL architecture."
    )
    db_session.add(resume)
    db_session.commit()
    db_session.refresh(resume)
    return resume


@pytest.fixture
def test_session(db_session: Session, test_user: User, test_resume: Resume):
    session = InterviewSession(
        id=uuid4(),
        user_id=test_user.id,
        resume_id=test_resume.id,
        status="IN_PROGRESS",
        current_difficulty=3,
        total_questions=0,
        completed_questions=0,
        session_state={"knowledge_state": {"topics": {}, "projects": {}, "relationships": [], "claims": [], "knowledge_gaps": []}}
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


def test_next_question_idempotency_prevents_duplicates(db_session: Session, test_session: InterviewSession, monkeypatch):
    """
    Verifies that calling select_and_generate_next_question with the exact same
    idempotency key returns the cached question and does not generate duplicates
    or increment the session's question counter twice.
    """
    call_count = {"count": 0}

    # Mock RAG generation to return deterministic question
    def mock_generate_grounded_question(resume_id, req, db):
        call_count["count"] += 1
        return QuestionGenerationResponse(
            resume_id=resume_id,
            request_id=f"req_{call_count['count']}",
            trace_id="trc_test",
            validation_status="ACCEPTED",
            question_id=uuid4(),
            question=QuestionOutputSchema(
                question="Explain connection pooling trade-offs in PostgreSQL with async SQLAlchemy.",
                category="Architecture",
                difficulty=3,
                target_project="Backend Engine",
                target_entity="PostgreSQL",
                evidence_text="Designed scalable PostgreSQL data access layer.",
                grounding_score=0.92,
                evidence_ids=["ev_claim_1"],
                rationale="Probing database architecture trade-offs based on candidate project.",
                relevance_score=0.95,
                technical_quality_score=0.90,
                trace_id="trc_test_1"
            ),
            attempts_count=1,
            latencies_ms={"total_ms": 120.0}
        )

    from app.services.rag.rag_pipeline import rag_pipeline
    monkeypatch.setattr(rag_pipeline, "generate_grounded_question", mock_generate_grounded_question)

    idempotency_key = "session_idem_key_turn_0"
    req = AdaptiveQuestionSelectionRequest(request_idempotency_key=idempotency_key)

    # First attempt: Generates and caches
    resp1 = session_service.select_and_generate_next_question(
        session_id=test_session.id,
        db=db_session,
        request_override=req
    )

    assert resp1.status == "GENERATED"
    assert resp1.question_text == "Explain connection pooling trade-offs in PostgreSQL with async SQLAlchemy."
    assert call_count["count"] == 1
    assert test_session.total_questions == 1

    # Second attempt: Same idempotency key (simulating network timeout retry)
    resp2 = session_service.select_and_generate_next_question(
        session_id=test_session.id,
        db=db_session,
        request_override=req
    )

    # Must return cached result without invoking generator again or incrementing count
    assert resp2.status == "GENERATED"
    assert resp2.question_id == resp1.question_id
    assert resp2.question_text == resp1.question_text
    assert call_count["count"] == 1, "RAG generator must NOT be invoked again for identical idempotency key"
    assert test_session.total_questions == 1, "Total questions must remain 1 after idempotent replay"


def test_next_question_on_ended_session_raises_400(db_session: Session, test_session: InterviewSession):
    """
    Verifies that calling next question on a session that has been concluded
    with an end_reason raises 400 Bad Request to prevent corrupting completed sessions.
    """
    session_service.end_interview(session_id=test_session.id, db=db_session, reason="USER_COMPLETED")

    with pytest.raises(HTTPException) as exc_info:
        session_service.select_and_generate_next_question(
            session_id=test_session.id,
            db=db_session,
            request_override=AdaptiveQuestionSelectionRequest()
        )

    assert exc_info.value.status_code == 400
    assert "already" in exc_info.value.detail.lower()


def test_session_ownership_enforcement(db_session: Session, test_session: InterviewSession, other_user: User):
    """
    Verifies that verifying access for a user who does not own the session raises 403 Forbidden.
    """
    from app.api.v1.endpoints.sessions import verify_session_access

    with pytest.raises(HTTPException) as exc_info:
        verify_session_access(
            session_id=test_session.id,
            db=db_session,
            current_user=other_user
        )

    assert exc_info.value.status_code == 403
    assert "Forbidden" in exc_info.value.detail or "not authorized" in exc_info.value.detail


def test_session_not_found_returns_404(db_session: Session, test_user: User):
    """
    Verifies that verifying access for a non-existent session UUID raises 404 Not Found.
    """
    from app.api.v1.endpoints.sessions import verify_session_access

    missing_id = uuid4()
    with pytest.raises(HTTPException) as exc_info:
        verify_session_access(
            session_id=missing_id,
            db=db_session,
            current_user=test_user
        )

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


def test_duplicate_rejection_falls_back_to_next_eligible_topic_cleanly(db_session: Session, test_session: InterviewSession, monkeypatch):
    """
    Verifies that when RAG generation on the primary candidate target fails due to semantic duplicate,
    the adaptive selection loop automatically excludes that topic and seamlessly falls back
    to the next eligible candidate target without raising 500 or crashing.
    """
    call_log = []

    def mock_generate_grounded_question(resume_id, req, db):
        call_log.append(req.target_entity or req.category)
        if len(call_log) == 1:
            # First attempt fails with duplicate rejection
            return QuestionGenerationResponse(
                resume_id=resume_id,
                request_id="req_dup",
                trace_id="trc_dup",
                validation_status="REJECTED",
                question=None,
                message="REJECTED: Question is semantically duplicate with existing Question ID 'abc' (Cosine: 0.88 >= 0.82).",
                attempts_count=3,
                latencies_ms={"total_ms": 150.0}
            )
        else:
            # Second attempt on fallback topic succeeds
            return QuestionGenerationResponse(
                resume_id=resume_id,
                request_id="req_success",
                trace_id="trc_success",
                validation_status="ACCEPTED",
                question_id=uuid4(),
                question=QuestionOutputSchema(
                    question="How would you design database indexing strategy for high throughput queries in PostgreSQL?",
                    category="Architecture",
                    difficulty=3,
                    target_project="Backend Engine",
                    target_entity="PostgreSQL",
                    evidence_text="Designed PostgreSQL database layer.",
                    grounding_score=0.91,
                    evidence_ids=["ev_1"],
                    rationale="Testing database architecture.",
                    relevance_score=0.90,
                    technical_quality_score=0.90,
                    trace_id="trc_2"
                ),
                attempts_count=1,
                latencies_ms={"total_ms": 120.0}
            )

    from app.services.rag.rag_pipeline import rag_pipeline
    monkeypatch.setattr(rag_pipeline, "generate_grounded_question", mock_generate_grounded_question)

    resp = session_service.select_and_generate_next_question(
        session_id=test_session.id,
        db=db_session,
        request_override=AdaptiveQuestionSelectionRequest()
    )

    assert resp.status == "GENERATED"
    assert resp.question_text == "How would you design database indexing strategy for high throughput queries in PostgreSQL?"
    assert len(call_log) == 2, "Must have invoked fallback target after first target was rejected as duplicate"
    assert test_session.total_questions == 1
