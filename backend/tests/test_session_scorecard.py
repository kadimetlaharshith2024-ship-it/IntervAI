from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4
import pytest
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
from app.services.evaluation.scorecard import ScorecardAggregatorService
from app.main import app


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
def base_session_setup(db_session: Session):
    """Creates a base candidate resume and session."""
    resume = Resume(
        id=uuid4(),
        filename="scorecard_test_candidate.pdf",
        raw_text="Candidate with extensive backend architecture experience."
    )
    db_session.add(resume)
    db_session.flush()

    session = InterviewSession(
        id=uuid4(),
        resume_id=resume.id,
        status="IN_PROGRESS"
    )
    db_session.add(session)
    db_session.commit()

    return {
        "resume": resume,
        "session": session,
    }


# ==============================================================================
# Step G: Focused Automated Tests for Session Scorecard Aggregation
# ==============================================================================

def test_empty_session_aggregation(db_session: Session, base_session_setup):
    """Test 1: An empty session with zero questions has None overall score and stays IN_PROGRESS."""
    session = base_session_setup["session"]
    aggregator = ScorecardAggregatorService()

    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 0
    assert result.completed_questions == 0
    assert result.overall_score is None
    assert result.category_scores == {}
    assert result.status == "IN_PROGRESS"
    assert result.completed_at is None


def test_one_evaluated_answer(db_session: Session, base_session_setup):
    """Test 2: A session with 1 question and 1 valid evaluation calculates overall score and completes."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=resume.id,
        question_text="Explain database connection pooling in high-throughput services.",
        category="Architecture",
        difficulty=3
    )
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Connection pooling reuses open TCP connections to eliminate handshake overhead.",
        status="EVALUATED",
        overall_score=85.0
    )
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 1
    assert result.completed_questions == 1
    assert result.overall_score == 85.0
    assert result.category_scores == {"Architecture": 85.0}
    assert result.status == "COMPLETED"
    assert result.completed_at is not None


def test_multiple_evaluated_answers(db_session: Session, base_session_setup):
    """Test 3: Multiple evaluated answers across questions produce correct arithmetic mean."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    q3 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q3", category="Implementation")
    db_session.add_all([q1, q2, q3])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=90.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="EVALUATED", overall_score=80.0)
    ans3 = InterviewAnswer(id=uuid4(), question_id=q3.id, session_id=session.id, candidate_answer="A3", status="EVALUATED", overall_score=70.0)
    db_session.add_all([ans1, ans2, ans3])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 3
    assert result.completed_questions == 3
    # (90 + 80 + 70) / 3 = 80.0
    assert result.overall_score == 80.0
    assert result.category_scores == {
        "Architecture": 85.0,  # (90 + 80) / 2
        "Implementation": 70.0
    }
    assert result.status == "COMPLETED"


def test_overall_score_calculation_and_rounding(db_session: Session, base_session_setup):
    """Test 4: Overall score rounds precisely to two decimal places."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    q3 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q3", category="Architecture")
    db_session.add_all([q1, q2, q3])
    db_session.flush()

    # (85.0 + 85.0 + 86.0) / 3 = 85.33333333... -> 85.33
    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=85.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="EVALUATED", overall_score=85.0)
    ans3 = InterviewAnswer(id=uuid4(), question_id=q3.id, session_id=session.id, candidate_answer="A3", status="EVALUATED", overall_score=86.0)
    db_session.add_all([ans1, ans2, ans3])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.overall_score == 85.33


def test_category_score_aggregation_and_normalization(db_session: Session, base_session_setup):
    """Test 5: Calculates distinct category averages and strips category whitespace."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category=" Security ")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Security")
    q3 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q3", category="Performance")
    db_session.add_all([q1, q2, q3])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=90.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="EVALUATED", overall_score=94.0)
    ans3 = InterviewAnswer(id=uuid4(), question_id=q3.id, session_id=session.id, candidate_answer="A3", status="EVALUATED", overall_score=88.0)
    db_session.add_all([ans1, ans2, ans3])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert "Security" in result.category_scores
    assert result.category_scores["Security"] == 92.0  # (90 + 94) / 2
    assert result.category_scores["Performance"] == 88.0


def test_mixed_categories_with_some_missing_evaluations(db_session: Session, base_session_setup):
    """Test 6: Categories without evaluations are omitted from category_scores."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q_arch = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Arch Q", category="Architecture")
    q_sec = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Sec Q", category="Security")
    db_session.add_all([q_arch, q_sec])
    db_session.flush()

    # Only answer Architecture question; Security remains unanswered
    ans_arch = InterviewAnswer(id=uuid4(), question_id=q_arch.id, session_id=session.id, candidate_answer="A", status="EVALUATED", overall_score=80.0)
    db_session.add(ans_arch)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.category_scores == {"Architecture": 80.0}
    assert "Security" not in result.category_scores
    assert result.status == "IN_PROGRESS"


def test_pending_answers_excluded_from_scores(db_session: Session, base_session_setup):
    """Test 7: PENDING answers are excluded from overall score and progress counts."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    db_session.add_all([q1, q2])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=90.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="PENDING", overall_score=None)
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 2
    assert result.completed_questions == 1
    assert result.overall_score == 90.0
    assert result.status == "IN_PROGRESS"


def test_failed_evaluations_excluded_from_scores(db_session: Session, base_session_setup):
    """Test 8: FAILED evaluations are excluded from overall score and do not cause completion."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    db_session.add_all([q1, q2])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=80.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="FAILED", overall_score=None, error_message="LLM timeout")
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 2
    assert result.completed_questions == 1
    assert result.overall_score == 80.0  # NOT (80 + 0)/2
    assert result.status == "IN_PROGRESS"


def test_invalid_or_missing_scores_excluded_safely(db_session: Session, base_session_setup):
    """Test 9: Evaluated answers with out-of-range or missing scores are excluded from calculations."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    db_session.add_all([q1, q2])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=80.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="EVALUATED", overall_score=150.0)  # Invalid score
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.completed_questions == 1
    assert result.overall_score == 80.0


def test_no_valid_evaluations_results_in_null_score(db_session: Session, base_session_setup):
    """Test 10: If a session has only failed or pending answers, overall_score is None, not 0.0."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="FAILED", overall_score=None)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.overall_score is None
    assert result.completed_questions == 0
    assert result.status == "IN_PROGRESS"


def test_duplicate_answer_records_do_not_double_count(db_session: Session, base_session_setup):
    """Test 11: If multiple answer rows exist for a question, only the latest is counted."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    # Earlier answer scored 50.0
    ans_old = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Old answer",
        status="EVALUATED",
        overall_score=50.0,
        created_at=datetime.now(timezone.utc) - timedelta(hours=1)
    )
    # Later answer scored 90.0
    ans_new = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="New answer",
        status="EVALUATED",
        overall_score=90.0,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([ans_old, ans_new])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.completed_questions == 1
    assert result.overall_score == 90.0  # Uses latest answer score


def test_repeated_aggregation_is_idempotent(db_session: Session, base_session_setup):
    """Test 12: Repeatedly calling calculate_session_scorecard produces identical state and does not alter completed_at."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=88.0)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    run1 = aggregator.calculate_session_scorecard(db_session, session.id)
    first_completed_at = run1.completed_at

    run2 = aggregator.calculate_session_scorecard(db_session, session.id)

    assert run1.overall_score == run2.overall_score == 88.0
    assert run1.completed_questions == run2.completed_questions == 1
    assert run1.status == run2.status == "COMPLETED"
    assert run2.completed_at == first_completed_at


def test_unanswered_questions_prevent_premature_completion(db_session: Session, base_session_setup):
    """Test 13: A session with partially evaluated questions stays IN_PROGRESS."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    db_session.add_all([q1, q2])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=95.0)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.status == "IN_PROGRESS"
    assert result.completed_at is None
    assert result.completed_questions == 1
    assert result.total_questions == 2


def test_failed_evaluations_do_not_silently_count_as_zero(db_session: Session, base_session_setup):
    """Test 14: Failed evaluations are not averaged as zero into the overall score."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Architecture")
    db_session.add_all([q1, q2])
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=100.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=q2.id, session_id=session.id, candidate_answer="A2", status="FAILED", overall_score=None)
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.overall_score == 100.0  # NOT 50.0
    assert result.completed_questions == 1


def test_fully_evaluated_session_completion(db_session: Session, base_session_setup):
    """Test 15: Session with all questions evaluated completes and records completed_at timestamp."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=92.0)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.status == "COMPLETED"
    assert isinstance(result.completed_at, datetime)


def test_empty_session_never_becomes_completed(db_session: Session, base_session_setup):
    """Test 16: Zero-question sessions never become COMPLETED."""
    session = base_session_setup["session"]
    aggregator = ScorecardAggregatorService()

    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.status == "IN_PROGRESS"
    assert result.completed_at is None


def test_questions_belonging_to_another_session_excluded(db_session: Session, base_session_setup):
    """Test 17: Answers for questions belonging to Session B do not affect Session A's scorecard."""
    session_a = base_session_setup["session"]
    resume = base_session_setup["resume"]

    session_b = InterviewSession(id=uuid4(), resume_id=resume.id, status="IN_PROGRESS")
    db_session.add(session_b)
    db_session.flush()

    q_a = InterviewQuestion(id=uuid4(), session_id=session_a.id, resume_id=resume.id, question_text="QA", category="Architecture")
    q_b = InterviewQuestion(id=uuid4(), session_id=session_b.id, resume_id=resume.id, question_text="QB", category="Architecture")
    db_session.add_all([q_a, q_b])
    db_session.flush()

    ans_a = InterviewAnswer(id=uuid4(), question_id=q_a.id, session_id=session_a.id, candidate_answer="AA", status="EVALUATED", overall_score=60.0)
    ans_b = InterviewAnswer(id=uuid4(), question_id=q_b.id, session_id=session_b.id, candidate_answer="AB", status="EVALUATED", overall_score=100.0)
    db_session.add_all([ans_a, ans_b])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    scorecard_a = aggregator.calculate_session_scorecard(db_session, session_a.id)

    assert scorecard_a.total_questions == 1
    assert scorecard_a.completed_questions == 1
    assert scorecard_a.overall_score == 60.0  # Does not include ans_b


def test_session_isolation(db_session: Session, base_session_setup):
    """Test 18: Recalculating Session A does not mutate Session B."""
    session_a = base_session_setup["session"]
    resume = base_session_setup["resume"]

    session_b = InterviewSession(id=uuid4(), resume_id=resume.id, status="IN_PROGRESS")
    db_session.add(session_b)
    db_session.flush()

    q_a = InterviewQuestion(id=uuid4(), session_id=session_a.id, resume_id=resume.id, question_text="QA", category="Architecture")
    q_b = InterviewQuestion(id=uuid4(), session_id=session_b.id, resume_id=resume.id, question_text="QB", category="Security")
    db_session.add_all([q_a, q_b])
    db_session.flush()

    ans_a = InterviewAnswer(id=uuid4(), question_id=q_a.id, session_id=session_a.id, candidate_answer="AA", status="EVALUATED", overall_score=75.0)
    ans_b = InterviewAnswer(id=uuid4(), question_id=q_b.id, session_id=session_b.id, candidate_answer="AB", status="EVALUATED", overall_score=95.0)
    db_session.add_all([ans_a, ans_b])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    res_a = aggregator.calculate_session_scorecard(db_session, session_a.id)
    res_b = aggregator.calculate_session_scorecard(db_session, session_b.id)

    assert res_a.overall_score == 75.0
    assert res_a.category_scores == {"Architecture": 75.0}
    assert res_b.overall_score == 95.0
    assert res_b.category_scores == {"Security": 95.0}


def test_progress_counters_accuracy(db_session: Session, base_session_setup):
    """Test 19: Progress counters accurately reflect total, completed, and in-progress questions."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    questions = [
        InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text=f"Q{i}", category="Architecture")
        for i in range(5)
    ]
    db_session.add_all(questions)
    db_session.flush()

    # 3 evaluated, 1 failed, 1 unanswered
    ans1 = InterviewAnswer(id=uuid4(), question_id=questions[0].id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=80.0)
    ans2 = InterviewAnswer(id=uuid4(), question_id=questions[1].id, session_id=session.id, candidate_answer="A2", status="EVALUATED", overall_score=85.0)
    ans3 = InterviewAnswer(id=uuid4(), question_id=questions[2].id, session_id=session.id, candidate_answer="A3", status="EVALUATED", overall_score=90.0)
    ans4 = InterviewAnswer(id=uuid4(), question_id=questions[3].id, session_id=session.id, candidate_answer="A4", status="FAILED", overall_score=None)
    db_session.add_all([ans1, ans2, ans3, ans4])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    assert result.total_questions == 5
    assert result.completed_questions == 3
    assert result.status == "IN_PROGRESS"


def test_database_persistence_and_transaction_handling(db_session: Session, base_session_setup):
    """Test 20: Aggregation commits to the database and reloads cleanly."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=84.0)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    aggregator.calculate_session_scorecard(db_session, session.id)

    # Query afresh from DB
    reloaded = db_session.query(InterviewSession).filter(InterviewSession.id == session.id).first()
    assert reloaded.overall_score == 84.0
    assert reloaded.completed_questions == 1
    assert reloaded.status == "COMPLETED"


def test_scorecard_api_endpoints(db_session: Session, base_session_setup):
    """Test 21: FastAPI scorecard API endpoint integration (GET and POST recalculate)."""
    client = TestClient(app)
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="A1",
        status="EVALUATED",
        overall_score=88.5,
        technical_accuracy_score=90.0,
        conceptual_understanding_score=85.0,
        completeness_score=88.0,
        depth_score=90.0,
        reasoning_score=90.0,
        evidence_alignment_score=85.0,
        evaluation_feedback="Solid answer."
    )
    db_session.add(ans1)
    db_session.commit()

    # Override get_db dependency
    from app.core.database import get_db
    app.dependency_overrides[get_db] = lambda: db_session

    try:
        # 1. GET scorecard
        response = client.get(f"/api/v1/evaluation/session/{session.id}/scorecard")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == str(session.id)
        assert data["overall_score"] == 88.5
        assert data["completed_questions"] == 1
        assert data["status"] == "COMPLETED"
        assert len(data["answers"]) == 1
        assert data["category_scores"]["Architecture"] == 88.5

        # 2. POST recalculate
        post_response = client.post(f"/api/v1/evaluation/session/{session.id}/recalculate")
        assert post_response.status_code == 200
        post_data = post_response.json()
        assert post_data["session_id"] == str(session.id)
        assert post_data["overall_score"] == 88.5
    finally:
        app.dependency_overrides.clear()


def test_scorecard_nonexistent_session_returns_404(db_session: Session):
    """Test 22: Nonexistent session returns 404."""
    aggregator = ScorecardAggregatorService()
    with pytest.raises(HTTPException) as exc_info:
        aggregator.calculate_session_scorecard(db_session, uuid4())
    assert exc_info.value.status_code == 404


def test_evaluated_answer_followed_by_newer_failed_attempt(db_session: Session, base_session_setup):
    """Test 23: An evaluated answer followed by a newer failed attempt makes the question FAILED, preventing completion."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    # Earlier answer EVALUATED (90.0)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="A1",
        status="EVALUATED",
        overall_score=90.0,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=10)
    )
    # Later resubmission FAILED
    ans2 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="A2",
        status="FAILED",
        overall_score=None,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    # Current state of Q1 is FAILED -> 0 completed questions, overall_score is None, status IN_PROGRESS
    assert result.completed_questions == 0
    assert result.overall_score is None
    assert result.status == "IN_PROGRESS"


def test_evaluated_answer_followed_by_newer_pending_attempt(db_session: Session, base_session_setup):
    """Test 24: An evaluated answer followed by a newer pending attempt makes the question PENDING, preventing completion."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    # Earlier answer EVALUATED (90.0)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="A1",
        status="EVALUATED",
        overall_score=90.0,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5)
    )
    # Later resubmission PENDING
    ans2 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="A2",
        status="PENDING",
        overall_score=None,
        created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([ans1, ans2])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    # Current state of Q1 is PENDING -> 0 completed questions, overall_score is None, status IN_PROGRESS
    assert result.completed_questions == 0
    assert result.overall_score is None
    assert result.status == "IN_PROGRESS"


def test_two_answers_with_identical_timestamps_deterministic_tie_breaker(db_session: Session, base_session_setup):
    """Test 25: Two answers with exact same timestamp resolve deterministically via secondary ID tie-breaker."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    now = datetime.now(timezone.utc)
    id1 = uuid4()
    id2 = uuid4()
    higher_id, lower_id = (id1, id2) if str(id1) > str(id2) else (id2, id1)

    ans_higher = InterviewAnswer(
        id=higher_id,
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Higher ID answer",
        status="EVALUATED",
        overall_score=95.0,
        created_at=now
    )
    ans_lower = InterviewAnswer(
        id=lower_id,
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Lower ID answer",
        status="EVALUATED",
        overall_score=60.0,
        created_at=now
    )
    db_session.add_all([ans_higher, ans_lower])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    result = aggregator.calculate_session_scorecard(db_session, session.id)

    # Deterministic tie-breaker selects higher ID string consistently
    assert result.completed_questions == 1
    assert result.overall_score == 95.0


def test_completed_session_reopens_when_new_question_added(db_session: Session, base_session_setup):
    """Test 26: A completed session transitions back to IN_PROGRESS and completed_at is cleared when a new question is added."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    db_session.add(q1)
    db_session.flush()

    ans1 = InterviewAnswer(id=uuid4(), question_id=q1.id, session_id=session.id, candidate_answer="A1", status="EVALUATED", overall_score=85.0)
    db_session.add(ans1)
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    res1 = aggregator.calculate_session_scorecard(db_session, session.id)
    assert res1.status == "COMPLETED"
    assert res1.completed_at is not None

    # Add a second unanswered question to the session
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Security")
    db_session.add(q2)
    db_session.commit()

    res2 = aggregator.calculate_session_scorecard(db_session, session.id)
    assert res2.total_questions == 2
    assert res2.completed_questions == 1
    assert res2.status == "IN_PROGRESS"
    assert res2.completed_at is None


def test_format_scorecard_response_deduplicates_multiple_attempts(db_session: Session, base_session_setup):
    """Test 27: format_scorecard_response returns at most one answer entry per question, using the latest attempt."""
    session = base_session_setup["session"]
    resume = base_session_setup["resume"]

    q1 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q1", category="Architecture")
    q2 = InterviewQuestion(id=uuid4(), session_id=session.id, resume_id=resume.id, question_text="Q2", category="Security")
    db_session.add_all([q1, q2])
    db_session.flush()

    # Q1: 3 attempts
    ans_q1_a1 = InterviewAnswer(
        id=uuid4(), question_id=q1.id, session_id=session.id,
        candidate_answer="A1", status="EVALUATED", overall_score=50.0,
        attempt_number=1, created_at=datetime.now(timezone.utc) - timedelta(minutes=15)
    )
    ans_q1_a2 = InterviewAnswer(
        id=uuid4(), question_id=q1.id, session_id=session.id,
        candidate_answer="A2", status="EVALUATED", overall_score=75.0,
        attempt_number=2, created_at=datetime.now(timezone.utc) - timedelta(minutes=10)
    )
    ans_q1_a3 = InterviewAnswer(
        id=uuid4(), question_id=q1.id, session_id=session.id,
        candidate_answer="A3", status="EVALUATED", overall_score=95.0,
        attempt_number=3, created_at=datetime.now(timezone.utc) - timedelta(minutes=5)
    )

    # Q2: 1 attempt
    ans_q2_a1 = InterviewAnswer(
        id=uuid4(), question_id=q2.id, session_id=session.id,
        candidate_answer="B1", status="EVALUATED", overall_score=88.0,
        attempt_number=1, created_at=datetime.now(timezone.utc)
    )
    db_session.add_all([ans_q1_a1, ans_q1_a2, ans_q1_a3, ans_q2_a1])
    db_session.commit()

    aggregator = ScorecardAggregatorService()
    session_result = aggregator.calculate_session_scorecard(db_session, session.id)
    scorecard_schema = aggregator.format_scorecard_response(session_result)

    # 4 total answer rows in DB, but scorecard must contain exactly 2 entries (one per question)
    assert len(scorecard_schema.answers) == 2
    answers_by_qid = {ans.question_id: ans for ans in scorecard_schema.answers}

    assert q1.id in answers_by_qid
    assert answers_by_qid[q1.id].attempt_number == 3
    assert answers_by_qid[q1.id].overall_score == 95.0
    assert answers_by_qid[q1.id].candidate_answer == "A3"

    assert q2.id in answers_by_qid
    assert answers_by_qid[q2.id].attempt_number == 1
    assert answers_by_qid[q2.id].overall_score == 88.0
    assert answers_by_qid[q2.id].candidate_answer == "B1"

