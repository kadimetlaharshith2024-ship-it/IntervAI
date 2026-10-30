import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
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
from app.services.evaluation.evaluator import answer_evaluator, AnswerEvaluatorService
from app.services.evaluation.scorecard import scorecard_aggregator
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult
from app.core.database import get_db
from app.main import app


class MockEvaluationProvider(BaseLLMProvider):
    """Deterministic mock provider for answer evaluation testing."""
    def __init__(self, scores=None, feedback=None):
        self.provider_name = "mock_phase7_evaluator"
        self.scores = scores or {
            "technical_accuracy": 85.0,
            "conceptual_understanding": 80.0,
            "completeness": 85.0,
            "depth_and_specificity": 80.0,
            "practical_reasoning": 85.0,
            "evidence_alignment": 80.0,
        }
        self.feedback = feedback or {
            "summary": "Solid technical explanation demonstrating clear understanding.",
            "strengths": ["Clear architectural explanation"],
            "weaknesses": ["Could provide deeper trade-off discussion"],
            "missing_concepts": [],
            "key_takeaways": "Focus on system trade-offs."
        }

    def generate_structured(self, prompt: str, system_instruction: str = None, schema=None, temperature: float = 0.1) -> LLMGenerationResult:
        payload = {
            "scores": self.scores,
            "feedback": self.feedback
        }
        return LLMGenerationResult(
            raw_content="{}",
            parsed_json=payload,
            provider=self.provider_name,
            model="mock-v1",
            latency_ms=12.0
        )


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
    """Provides a FastAPI TestClient with the in-memory database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_resume(db_session: Session):
    """Creates a sample candidate resume."""
    resume = Resume(
        id=uuid4(),
        filename="alice_backend_engineer.pdf",
        raw_text="Experienced engineer with Python, PostgreSQL, Redis, and Kafka."
    )
    db_session.add(resume)
    db_session.commit()
    db_session.refresh(resume)
    return resume


# ==============================================================================
# PHASE 7.1: SESSION CREATION, STARTING, AND IDEMPOTENCY
# ==============================================================================

def test_start_interview_creates_new_session(db_session: Session, sample_resume: Resume):
    """Starting an interview with resume_id initializes active session and timestamps."""
    session = session_service.start_interview(
        db=db_session,
        resume_id=sample_resume.id,
        initial_difficulty=4,
        session_metadata={"role": "Senior Backend"}
    )
    assert session.id is not None
    assert session.resume_id == sample_resume.id
    assert session.status == "IN_PROGRESS"
    assert session.current_difficulty == 4
    assert session.started_at is not None
    assert session.completed_at is None
    assert session.session_state is not None
    assert session.session_state.get("metadata", {}).get("role") == "Senior Backend"


def test_start_interview_idempotent_for_existing_active_session(db_session: Session, sample_resume: Resume):
    """Calling start_interview with resume_id returns existing active session without creating duplicates."""
    session1 = session_service.start_interview(db=db_session, resume_id=sample_resume.id)
    session2 = session_service.start_interview(db=db_session, resume_id=sample_resume.id)

    assert session1.id == session2.id
    # Ensure only 1 session exists in database
    count = db_session.query(InterviewSession).filter(InterviewSession.resume_id == sample_resume.id).count()
    assert count == 1


def test_start_interview_by_session_id(db_session: Session, sample_resume: Resume):
    """Starting an interview by existing session_id returns the session and reopens if completed."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)
    session.status = "ENDED"
    db_session.commit()

    reopened = session_service.start_interview(db=db_session, session_id=session.id)
    assert reopened.id == session.id
    assert reopened.status == "IN_PROGRESS"
    assert reopened.completed_at is None


def test_start_interview_validation_errors(db_session: Session):
    """Invalid resume ID or missing parameters trigger clean HTTP exceptions."""
    nonexistent_id = uuid4()
    with pytest.raises(HTTPException) as exc1:
        session_service.start_interview(db=db_session, resume_id=nonexistent_id)
    assert exc1.value.status_code == 404

    with pytest.raises(HTTPException) as exc2:
        session_service.start_interview(db=db_session, session_id=nonexistent_id)
    assert exc2.value.status_code == 404

    with pytest.raises(HTTPException) as exc3:
        session_service.start_interview(db=db_session)
    assert exc3.value.status_code == 400


# ==============================================================================
# PHASE 7.1: END INTERVIEW AND STATUS TRANSITIONS
# ==============================================================================

def test_end_interview_incomplete_transitions_to_ended(db_session: Session, sample_resume: Resume):
    """Ending an incomplete interview sets status to ENDED and records completion timestamp and reason."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)
    ended = session_service.end_interview(
        session_id=session.id,
        db=db_session,
        reason="CANDIDATE_REQUEST",
        feedback_notes="Candidate requested early wrap-up."
    )
    assert ended.status == "ENDED"
    assert ended.completed_at is not None
    assert ended.session_state.get("end_reason") == "CANDIDATE_REQUEST"
    assert ended.session_state.get("feedback_notes") == "Candidate requested early wrap-up."


def test_end_interview_idempotent_repeated_calls(db_session: Session, sample_resume: Resume):
    """Repeated calls to end_interview are safe, return the already ended session, and do not crash."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)
    ended1 = session_service.end_interview(session_id=session.id, db=db_session, reason="USER_COMPLETED")
    ended2 = session_service.end_interview(session_id=session.id, db=db_session, reason="USER_COMPLETED")
    assert ended1.id == ended2.id
    assert ended2.status == "ENDED"


def test_end_interview_fully_evaluated_transitions_to_completed(db_session: Session, sample_resume: Resume):
    """If all questions have evaluated answers, ending the interview records status COMPLETED."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)
    q = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_resume.id,
        session_id=session.id,
        question_text="Explain Redis TTL caching.",
        category="Architecture",
        difficulty=3
    )
    db_session.add(q)
    db_session.commit()

    evaluator = AnswerEvaluatorService(default_provider=MockEvaluationProvider())
    evaluator.evaluate_answer(db=db_session, question_id=q.id, candidate_answer="Redis TTL caches expire keys automatically.", session_id=session.id)

    concluded = session_service.end_interview(session_id=session.id, db=db_session)
    assert concluded.status == "COMPLETED"
    assert concluded.completed_at is not None
    assert concluded.completed_questions == 1
    assert concluded.total_questions == 1


# ==============================================================================
# PHASE 7.1: ASKED-QUESTION TRACKING AND ATTEMPT HISTORY INTEGRATION
# ==============================================================================

def test_asked_question_tracking_and_attempt_integration(db_session: Session, sample_resume: Resume):
    """Interview state accurately tracks asked questions, attempt counts, and latest scores."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)

    # 1. Associate 2 questions
    q1 = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_resume.id,
        session_id=session.id,
        question_text="Q1 text",
        category="Architecture",
        difficulty=3,
        target_project="HighVolumeApp",
        target_entity="Redis"
    )
    q2 = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_resume.id,
        session_id=session.id,
        question_text="Q2 text",
        category="Implementation",
        difficulty=4,
        target_project="HighVolumeApp",
        target_entity="Kafka"
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    state = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert len(state.asked_questions) == 2
    assert state.asked_questions[0].status == "UNANSWERED"
    assert state.asked_questions[0].attempt_count == 0
    assert state.asked_questions[1].status == "UNANSWERED"

    # 2. Answer Question 1 (Attempt 1)
    evaluator = AnswerEvaluatorService(default_provider=MockEvaluationProvider(scores={"technical_accuracy": 70.0, "conceptual_understanding": 70.0, "completeness": 70.0, "depth_and_specificity": 70.0, "practical_reasoning": 70.0, "evidence_alignment": 70.0}))
    evaluator.evaluate_answer(db=db_session, question_id=q1.id, candidate_answer="Attempt 1 on Q1", session_id=session.id)

    state_after_ans1 = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert state_after_ans1.asked_questions[0].status == "EVALUATED"
    assert state_after_ans1.asked_questions[0].attempt_count == 1
    assert state_after_ans1.asked_questions[0].latest_score == 70.0

    # 3. Answer Question 1 again (Attempt 2 - corrected)
    evaluator_high = AnswerEvaluatorService(default_provider=MockEvaluationProvider(scores={"technical_accuracy": 90.0, "conceptual_understanding": 90.0, "completeness": 90.0, "depth_and_specificity": 90.0, "practical_reasoning": 90.0, "evidence_alignment": 90.0}))
    evaluator_high.evaluate_answer(db=db_session, question_id=q1.id, candidate_answer="Attempt 2 on Q1 with deep details", session_id=session.id)

    state_after_ans2 = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert state_after_ans2.asked_questions[0].attempt_count == 2
    assert state_after_ans2.asked_questions[0].latest_score == 90.0


# ==============================================================================
# PHASE 7.1: TESTED TOPICS AND KNOWLEDGE GAP TRACKING
# ==============================================================================

def test_tested_skills_technologies_and_relationships_tracking(db_session: Session, sample_resume: Resume):
    """Interview state aggregates tested technologies, skills, projects, and relationships."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_resume.id,
        session_id=session.id,
        question_text="How did you implement Kafka streaming?",
        category="Architecture",
        difficulty=3,
        target_project="DataPipeline",
        target_entity="Kafka",
        target_relationship="TECH_STACK: DataPipeline -> Kafka",
        targeted_claim="Built real-time streaming pipeline",
        expected_concepts=["Consumer groups", "Partition rebalance"]
    )
    db_session.add(q)
    db_session.commit()

    state = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert "Kafka" in state.tested_technologies
    assert "Kafka" in state.tested_skills
    assert "Consumer groups" in state.tested_skills
    assert "DataPipeline" in state.tested_projects
    assert "TECH_STACK: DataPipeline -> Kafka" in state.tested_relationships
    assert "Built real-time streaming pipeline" in state.tested_claims


def test_knowledge_gap_tracking_structure(db_session: Session, sample_resume: Resume):
    """Evaluated answers with omissions or low scores generate structured knowledge gaps with severity."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        resume_id=sample_resume.id,
        session_id=session.id,
        question_text="Explain distributed locking in Redis.",
        category="Architecture",
        difficulty=4,
        target_entity="Redis"
    )
    db_session.add(q)
    db_session.commit()

    low_score_feedback = {
        "summary": "Candidate answer is incomplete.",
        "strengths": [],
        "weaknesses": ["Did not mention Redlock algorithm"],
        "missing_concepts": ["Redlock algorithm", "Clock drift"],
        "key_takeaways": "Study distributed consensus."
    }
    low_scores = {dim: 45.0 for dim in ["technical_accuracy", "conceptual_understanding", "completeness", "depth_and_specificity", "practical_reasoning", "evidence_alignment"]}

    evaluator = AnswerEvaluatorService(default_provider=MockEvaluationProvider(scores=low_scores, feedback=low_score_feedback))
    evaluator.evaluate_answer(db=db_session, question_id=q.id, candidate_answer="Just set key with expiration.", session_id=session.id)

    state = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert len(state.knowledge_gaps) == 1
    gap = state.knowledge_gaps[0]
    assert gap.question_id == q.id
    assert gap.topic == "Redis"
    assert gap.severity == "HIGH"  # score 45.0 < 50.0
    assert "Redlock algorithm" in gap.missing_concepts
    assert "Did not mention Redlock algorithm" in gap.weaknesses


# ==============================================================================
# PHASE 7.1: DIFFICULTY MANAGEMENT AND PROGRESS TRACKING
# ==============================================================================

def test_update_difficulty_bounds(db_session: Session, sample_resume: Resume):
    """Updating difficulty modifies current_difficulty and rejects out-of-range bounds."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session, initial_difficulty=3)
    assert session.current_difficulty == 3

    updated = session_service.update_difficulty(session.id, 5, db_session)
    assert updated.current_difficulty == 5

    with pytest.raises(HTTPException) as exc_low:
        session_service.update_difficulty(session.id, 0, db_session)
    assert exc_low.value.status_code == 400

    with pytest.raises(HTTPException) as exc_high:
        session_service.update_difficulty(session.id, 6, db_session)
    assert exc_high.value.status_code == 400


def test_progress_tracking_and_duration_calculation(db_session: Session, sample_resume: Resume):
    """Interview state calculates accurate progress percentages and duration."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session)

    # Empty session
    state_empty = session_service.get_interview_state(session_id=session.id, db=db_session)
    assert state_empty.progress.progress_percentage == 0.0
    assert state_empty.progress.completed_questions == 0
    assert state_empty.progress.total_questions == 0
    assert state_empty.progress.duration_seconds is not None
    assert state_empty.progress.duration_seconds >= 0.0


def test_session_recovery_after_reloading_persisted_state(db_session: Session, sample_resume: Resume):
    """State stored in PostgreSQL survives session close and reload."""
    session = session_service.create_session(resume_id=sample_resume.id, db=db_session, initial_difficulty=4)
    session_id = session.id

    # Update difficulty and conclude
    session_service.update_difficulty(session_id, 5, db_session)
    session_service.end_interview(session_id, db_session, reason="FINISHED_ALL")

    # Clear ORM session cache
    db_session.expunge_all()

    # Re-fetch fresh from DB
    reloaded = db_session.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    assert reloaded is not None
    assert reloaded.current_difficulty == 5
    assert reloaded.status == "ENDED"
    assert reloaded.completed_at is not None
    assert reloaded.session_state.get("end_reason") == "FINISHED_ALL"


# ==============================================================================
# PHASE 7.1: API ENDPOINTS END-TO-END
# ==============================================================================

def test_api_session_lifecycle_flow(client: TestClient, sample_resume: Resume):
    """End-to-end API test exercising POST /start, GET /state, PATCH /difficulty, POST /end."""
    # 1. Start interview via POST /api/v1/sessions/start
    start_resp = client.post(
        "/api/v1/sessions/start",
        json={"resume_id": str(sample_resume.id), "initial_difficulty": 3}
    )
    assert start_resp.status_code == 200
    sess_data = start_resp.json()
    session_id = sess_data["id"]
    assert sess_data["status"] == "IN_PROGRESS"
    assert sess_data["current_difficulty"] == 3

    # 2. Get initial interview state via GET /api/v1/sessions/{id}/state
    state_resp1 = client.get(f"/api/v1/sessions/{session_id}/state")
    assert state_resp1.status_code == 200
    state1 = state_resp1.json()
    assert state1["session_id"] == session_id
    assert state1["progress"]["total_questions"] == 0

    # 3. Update difficulty via PATCH /api/v1/sessions/{id}/difficulty
    patch_resp = client.patch(
        f"/api/v1/sessions/{session_id}/difficulty",
        json={"difficulty": 4}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["current_difficulty"] == 4

    # 4. End interview via POST /api/v1/sessions/{id}/end
    end_resp = client.post(
        f"/api/v1/sessions/{session_id}/end",
        json={"reason": "NORMAL_COMPLETION", "feedback_notes": "Well done."}
    )
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "ENDED"
    assert end_resp.json()["completed_at"] is not None

    # 5. Verify concluded state
    state_resp2 = client.get(f"/api/v1/sessions/{session_id}/state")
    assert state_resp2.status_code == 200
    state2 = state_resp2.json()
    assert state2["status"] == "ENDED"
    assert state2["completed_at"] is not None
