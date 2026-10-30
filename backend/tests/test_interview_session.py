import pytest
from datetime import datetime, timezone
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

from app.models.resume import Resume
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.session_service import session_service
from app.services.evaluation.evaluator import answer_evaluator
from app.services.evaluation.scorecard import scorecard_aggregator
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult
from app.core.database import get_db
from app.main import app


class MockEvaluationProvider(BaseLLMProvider):
    """Deterministic mock provider for answer evaluation testing."""
    def __init__(self, scores=None):
        self.provider_name = "mock_session_evaluator"
        self.scores = scores or {
            "technical_accuracy": 85.0,
            "conceptual_understanding": 80.0,
            "completeness": 85.0,
            "depth_and_specificity": 80.0,
            "practical_reasoning": 85.0,
            "evidence_alignment": 80.0,
        }

    def generate_text(self, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> LLMGenerationResult:
        return LLMGenerationResult(text="Mock", provider=self.provider_name, model="mock-model", latency_ms=10.0)

    def generate_structured(self, prompt: str, system_instruction: str = None, schema=None, temperature: float = 0.1) -> LLMGenerationResult:
        payload = {
            "scores": self.scores,
            "feedback": {
                "summary": "Solid technical explanation demonstrating clear understanding.",
                "strengths": ["Clear architectural explanation"],
                "weaknesses": ["Could provide deeper trade-off discussion"],
                "missing_concepts": [],
                "key_takeaways": "Focus on system trade-offs."
            }
        }
        return LLMGenerationResult(
            raw_content="{}",
            parsed_json=payload,
            provider=self.provider_name,
            model="mock-model",
            latency_ms=15.0
        )


@pytest.fixture(scope="function")
def db_session():
    """Provides a clean in-memory SQLite database session for each test function."""
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
    """Provides a FastAPI TestClient with the in-memory database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def resume_fixture(db_session: Session):
    """Creates a candidate resume."""
    resume = Resume(
        id=uuid4(),
        filename="session_candidate.pdf",
        raw_text="Experienced backend engineer with Python, PostgreSQL, and distributed systems."
    )
    db_session.add(resume)
    db_session.commit()
    db_session.refresh(resume)
    return resume


# --------------------------------------------------------------------------
# Test 1: Successful session creation via service and API
# --------------------------------------------------------------------------
def test_create_session_service_success(db_session: Session, resume_fixture: Resume):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    assert session.id is not None
    assert session.resume_id == resume_fixture.id
    assert session.status == "IN_PROGRESS"
    assert session.completed_questions == 0
    assert session.total_questions == 0
    assert session.overall_score is None
    assert session.category_scores == {}
    assert session.started_at is not None
    assert session.completed_at is None


def test_create_session_api_endpoint(client: TestClient, resume_fixture: Resume):
    # Top-level POST /api/v1/sessions
    response = client.post("/api/v1/sessions", json={"resume_id": str(resume_fixture.id)})
    assert response.status_code == 201
    data = response.json()
    assert data["resume_id"] == str(resume_fixture.id)
    assert data["status"] == "IN_PROGRESS"
    assert data["completed_questions"] == 0
    assert data["total_questions"] == 0
    assert "id" in data

    # Resume-scoped convenience POST /api/v1/resumes/{resume_id}/sessions
    resp2 = client.post(f"/api/v1/resumes/{resume_fixture.id}/sessions")
    assert resp2.status_code == 201
    data2 = resp2.json()
    assert data2["resume_id"] == str(resume_fixture.id)


def test_get_session_api_endpoint(client: TestClient, resume_fixture: Resume, db_session: Session):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    response = client.get(f"/api/v1/sessions/{session.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(session.id)
    assert data["resume_id"] == str(resume_fixture.id)
    assert data["status"] == "IN_PROGRESS"


# --------------------------------------------------------------------------
# Test 2: Nonexistent resume error handling
# --------------------------------------------------------------------------
def test_create_session_nonexistent_resume(db_session: Session, client: TestClient):
    missing_id = uuid4()
    with pytest.raises(HTTPException) as exc:
        session_service.create_session(resume_id=missing_id, db=db_session)
    assert exc.value.status_code == 404

    response = client.post("/api/v1/sessions", json={"resume_id": str(missing_id)})
    assert response.status_code == 404

    resp2 = client.post(f"/api/v1/resumes/{missing_id}/sessions")
    assert resp2.status_code == 404


def test_get_nonexistent_session(client: TestClient):
    missing_id = uuid4()
    response = client.get(f"/api/v1/sessions/{missing_id}")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Test 3: Question association with a valid session
# --------------------------------------------------------------------------
def test_associate_question_success(db_session: Session, resume_fixture: Resume):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        question_text="How do you handle database connection pooling in PostgreSQL?",
        category="Architecture",
        difficulty=3
    )
    db_session.add(question)
    db_session.commit()

    # Associate question
    updated_q = session_service.associate_question(session.id, question.id, db_session)
    assert updated_q.session_id == session.id

    # Verify session total_questions counter is updated
    db_session.refresh(session)
    assert session.total_questions == 1
    assert session.completed_questions == 0
    assert session.status == "IN_PROGRESS"

    # Idempotent re-association does not error
    reassociated_q = session_service.associate_question(session.id, question.id, db_session)
    assert reassociated_q.session_id == session.id


def test_associate_question_api_endpoints(client: TestClient, db_session: Session, resume_fixture: Resume):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        question_text="Explain ACID transactions.",
        category="Fundamentals",
        difficulty=2
    )
    db_session.add(question)
    db_session.commit()

    # Associate via POST /api/v1/sessions/{session_id}/questions (body)
    resp = client.post(
        f"/api/v1/sessions/{session.id}/questions",
        json={"question_id": str(question.id)}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ASSOCIATED"

    # List questions via GET /api/v1/sessions/{session_id}/questions
    list_resp = client.get(f"/api/v1/sessions/{session.id}/questions")
    assert list_resp.status_code == 200
    q_data = list_resp.json()
    assert q_data["total_questions"] == 1
    assert q_data["questions"][0]["id"] == str(question.id)


# --------------------------------------------------------------------------
# Test 4: Resume mismatch rejection
# --------------------------------------------------------------------------
def test_associate_question_resume_mismatch_rejected(db_session: Session, resume_fixture: Resume, client: TestClient):
    # Create second resume
    resume2 = Resume(id=uuid4(), filename="other.pdf", raw_text="Frontend engineer.")
    db_session.add(resume2)
    db_session.commit()

    session1 = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    # Question belongs to resume2
    q_other = InterviewQuestion(
        id=uuid4(),
        resume_id=resume2.id,
        question_text="Explain React hooks.",
        category="Application"
    )
    db_session.add(q_other)
    db_session.commit()

    with pytest.raises(HTTPException) as exc:
        session_service.associate_question(session1.id, q_other.id, db_session)
    assert exc.value.status_code == 400
    assert "does not belong" in exc.value.detail

    # Via API
    resp = client.post(
        f"/api/v1/sessions/{session1.id}/questions",
        json={"question_id": str(q_other.id)}
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# Test 5: Attempted reassignment across sessions rejected
# --------------------------------------------------------------------------
def test_reassignment_to_different_session_rejected(db_session: Session, resume_fixture: Resume, client: TestClient):
    session1 = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    session2 = session_service.create_session(resume_id=resume_fixture.id, db=db_session)

    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        session_id=session1.id,
        question_text="Explain WAL logging.",
        category="Architecture"
    )
    db_session.add(question)
    db_session.commit()

    # Attempting to assign to session2 should be rejected
    with pytest.raises(HTTPException) as exc:
        session_service.associate_question(session2.id, question.id, db_session)
    assert exc.value.status_code == 400
    assert "already assigned to another interview session" in exc.value.detail

    resp = client.post(
        f"/api/v1/sessions/{session2.id}/questions",
        json={"question_id": str(question.id)}
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# Test 6: Session-linked answer evaluation and scorecard updates
# --------------------------------------------------------------------------
def test_session_linked_answer_evaluation_updates_scorecard(db_session: Session, resume_fixture: Resume, client: TestClient):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        session_id=session.id,
        question_text="How do you handle sharding in distributed databases?",
        category="Architecture",
        difficulty=4
    )
    db_session.add(question)
    db_session.commit()

    # Recalculate session to reflect total_questions = 1
    scorecard_aggregator.calculate_session_scorecard(db_session, session.id)
    db_session.refresh(session)
    assert session.total_questions == 1
    assert session.completed_questions == 0
    assert session.status == "IN_PROGRESS"

    # Submit answer via answer_evaluator with session_id
    mock_provider = MockEvaluationProvider(scores={
        "technical_accuracy": 90.0,
        "conceptual_understanding": 85.0,
        "completeness": 90.0,
        "depth_and_specificity": 85.0,
        "practical_reasoning": 90.0,
        "evidence_alignment": 80.0,
    })
    answer = answer_evaluator.evaluate_answer(
        db=db_session,
        question_id=question.id,
        candidate_answer="Consistent hashing with virtual nodes ensures even key distribution across shards.",
        session_id=session.id,
        provider_override=mock_provider
    )

    assert answer.status == "EVALUATED"
    assert answer.session_id == session.id
    assert answer.overall_score is not None

    # Verify session scorecard was automatically recalculated
    db_session.refresh(session)
    assert session.completed_questions == 1
    assert session.total_questions == 1
    assert session.status == "COMPLETED"
    assert session.overall_score == answer.overall_score
    assert session.completed_at is not None

    # Verify scorecard endpoint returns full result
    scorecard_resp = client.get(f"/api/v1/sessions/{session.id}/scorecard")
    assert scorecard_resp.status_code == 200
    sc_data = scorecard_resp.json()
    assert sc_data["session_id"] == str(session.id)
    assert sc_data["status"] == "COMPLETED"
    assert sc_data["overall_score"] == answer.overall_score
    assert len(sc_data["answers"]) == 1


# --------------------------------------------------------------------------
# Test 7: Auto-association when unlinked question receives session-linked answer
# --------------------------------------------------------------------------
def test_unlinked_question_auto_associates_on_session_answer_submission(db_session: Session, resume_fixture: Resume):
    session = session_service.create_session(resume_id=resume_fixture.id, db=db_session)
    # Question created with session_id = None
    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        session_id=None,
        question_text="What is write-ahead logging?",
        category="Architecture"
    )
    db_session.add(question)
    db_session.commit()

    # Fast-path refusal evaluation submitted with session_id
    answer = answer_evaluator.evaluate_answer(
        db=db_session,
        question_id=question.id,
        candidate_answer="I don't know.",
        session_id=session.id
    )

    # Question is now bound to session
    db_session.refresh(question)
    assert question.session_id == session.id

    # Session completed with refusal score 0.0
    db_session.refresh(session)
    assert session.completed_questions == 1
    assert session.total_questions == 1
    assert session.overall_score == 0.0
    assert session.status == "COMPLETED"


# --------------------------------------------------------------------------
# Test 8: Sessionless answer evaluation remains completely backward-compatible
# --------------------------------------------------------------------------
def test_sessionless_answer_evaluation_remains_functional(db_session: Session, resume_fixture: Resume):
    question = InterviewQuestion(
        id=uuid4(),
        resume_id=resume_fixture.id,
        session_id=None,
        question_text="What is a B-tree index?",
        category="Fundamentals"
    )
    db_session.add(question)
    db_session.commit()

    # Submit without session_id
    answer = answer_evaluator.evaluate_answer(
        db=db_session,
        question_id=question.id,
        candidate_answer="I don't know the answer.",
        session_id=None
    )

    assert answer.status == "EVALUATED"
    assert answer.session_id is None
    assert answer.overall_score == 0.0
    assert question.session_id is None
