import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from unittest.mock import patch, MagicMock
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
    AdaptiveQuestionSelectionRequest,
)
from app.schemas.rag import QuestionGenerationResponse, QuestionOutputSchema
from app.services.rag.rag_pipeline import rag_pipeline


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
def populated_resume(db_session: Session) -> Resume:
    """Creates a realistic candidate resume with diverse skills, claims, and relationships."""
    resume = Resume(
        id=uuid4(),
        filename="jane_doe_senior_backend.pdf",
        raw_text=(
            "Jane Doe - Senior Backend Engineer\n"
            "Languages & Frameworks: Python, FastAPI, Docker, PostgreSQL\n"
            "Experience: Built distributed event processing with FastAPI and PostgreSQL.\n"
            "Architected Redis pub/sub broker reducing latency by 45%.\n"
        )
    )
    db_session.add(resume)
    db_session.commit()

    # Add core skills
    sk1 = Skill(id=201, name="Python", category="Language")
    sk2 = Skill(id=202, name="FastAPI", category="Framework")
    sk3 = Skill(id=203, name="PostgreSQL", category="Database")
    sk4 = Skill(id=204, name="Docker", category="Tool/DevOps")
    db_session.add_all([sk1, sk2, sk3, sk4])
    db_session.flush()

    rsk1 = ResumeSkill(resume_id=resume.id, skill_id=201, extraction_confidence=0.95)
    rsk2 = ResumeSkill(resume_id=resume.id, skill_id=202, extraction_confidence=0.90)
    rsk3 = ResumeSkill(resume_id=resume.id, skill_id=203, extraction_confidence=0.92)
    rsk4 = ResumeSkill(resume_id=resume.id, skill_id=204, extraction_confidence=0.88)
    db_session.add_all([rsk1, rsk2, rsk3, rsk4])

    # Add relationship
    rel = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="USED_WITH",
        target_entity="PostgreSQL",
        context_entity="Analytics Engine",
        section="Experience",
        evidence_text="Built distributed event processing with FastAPI and PostgreSQL.",
        confidence=0.92
    )
    db_session.add(rel)

    # Add claim
    claim = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id=f"CLM-{uuid4().hex[:6]}",
        section="Experience",
        project="Redis PubSub",
        target_entity="Redis",
        relation_type="OPTIMIZED",
        claim_text="Architected Redis pub/sub broker reducing latency by 45%.",
        evidence_text="Architected Redis pub/sub broker reducing latency by 45%."
    )
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(resume)
    return resume


# =========================================================================
# PHASE 7.4 FOCUSED TESTS: SCORE-BASED ADAPTATION
# =========================================================================

def test_score_below_5_targeted_weakness_probing(db_session: Session, populated_resume: Resume):
    """
    Requirement A: Score below 5 (mapped to < 50.0 on 0-100 scale).
    - Identifies weak concept using evaluation evidence and knowledge state.
    - Selects targeted follow-up on foundational mechanics (Fundamentals/Implementation).
    - Reduces difficulty from 3 to 2.
    - Rationale explains weakness de-escalation and remediation.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How does PostgreSQL MVCC handle transaction isolation and dead tuple vacuuming?",
        category="Architecture",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    # Score 38.0 / 100 (< 50.0, i.e. < 5.0) with solid evaluation confidence 0.85
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="PostgreSQL locks rows so other transactions wait.",
        status="EVALUATED",
        overall_score=38.0,
        evaluation_confidence=0.85,
        missing_concepts=["xmin/xmax headers", "autovacuum", "visibility maps"],
        weaknesses=["Candidate confused multiversion concurrency control with pessimistic table-level locking."]
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
    assert selection.target_topic == "PostgreSQL"
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    # Difficulty de-escalated from 3 to 2
    assert selection.difficulty == 2
    assert selection.category in ["Fundamentals", "Implementation"]
    assert "knowledge weakness" in selection.rationale.lower() or "knowledge gap" in selection.rationale.lower()


def test_score_5_through_7_5_appropriate_intent_and_stable_difficulty(db_session: Session, populated_resume: Resume):
    """
    Requirement B: Score from 5 through 7.5 (mapped to 50.0 to 75.0 on 0-100 scale).
    - Candidate demonstrated solid baseline competence.
    - Difficulty remains stable (e.g. difficulty stays at 3).
    - Question intent focuses on practical implementation trade-offs and reasoning.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How would you design a rate limiter in FastAPI?",
        category="Implementation",
        difficulty=3,
        target_entity="FastAPI"
    )
    # Score 68.0 / 100 (in [50.0, 75.0]) with confidence 0.88
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="I would use Redis with token bucket middleware in FastAPI, tracking IP or user tokens.",
        status="EVALUATED",
        overall_score=68.0,
        evaluation_confidence=0.88,
        strengths=["Clear token bucket concept", "Realistic Redis integration"],
        weaknesses=["Did not discuss distributed race conditions or sliding window trade-offs."]
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

    # Difficulty remains stable at 3
    assert selection.difficulty == 3
    # Rationale explains solid baseline competence and maintaining difficulty
    assert "baseline competence" in selection.rationale.lower() or "maintaining difficulty" in selection.rationale.lower()
    # Broad topic selected or trade-off category probed
    assert selection.category in ["Implementation", "Trade-Offs", "Architecture", "Fundamentals"]


def test_score_above_7_5_justified_difficulty_escalation(db_session: Session, populated_resume: Resume):
    """
    Requirement C: Score above 7.5 (mapped to > 75.0 on 0-100 scale).
    - Score is high (e.g. 88.0) and evaluation confidence is high (0.92 >= 0.65).
    - Increases difficulty from 3 to 4.
    - Selects architecture, failure-mode, or advanced-depth questions.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Describe how Docker container networking bridges communicate with host iptables.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer=(
            "Docker daemon creates a bridge network (docker0). It uses veth pairs with one interface inside "
            "the container network namespace and the peer bound to docker0. Packets routed externally undergo "
            "source NAT via iptables MASQUERADE rules."
        ),
        status="EVALUATED",
        overall_score=88.0,
        evaluation_confidence=0.92,
        strengths=["Precise veth pair explanation", "Accurate iptables MASQUERADE knowledge"]
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
    assert selection.target_topic == "Docker"
    assert selection.is_follow_up is True
    assert selection.parent_question_id == q1.id
    # Difficulty escalated from 3 to 4
    assert selection.difficulty == 4
    assert selection.category in ["Failure-Mode", "Performance", "Security"]
    assert "escalating difficulty" in selection.rationale.lower()


def test_confidence_aware_adaptation_high_score_low_confidence(db_session: Session, populated_resume: Resume):
    """
    Edge Case: High score with low evaluation confidence.
    Candidate answered briefly or evaluation was noisy; confidence is 0.40 (< 0.65).
    Difficulty must NOT escalate; it must remain stable at 3.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain Python GIL.",
        category="Fundamentals",
        difficulty=3,
        target_entity="Python"
    )
    # Score 90.0, but confidence is only 0.40 (below min_confidence_for_escalation of 0.65)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Mutex preventing multi-core python bytecodes.",
        status="EVALUATED",
        overall_score=90.0,
        evaluation_confidence=0.40
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

    # Difficulty must NOT escalate because evaluation confidence is insufficient
    assert selection.difficulty == 3
    assert "below threshold" in selection.rationale.lower() or "maintaining difficulty" in selection.rationale.lower()


def test_confidence_aware_adaptation_low_score_low_confidence(db_session: Session, populated_resume: Resume):
    """
    Edge Case: Low score with low evaluation confidence.
    Candidate answer was ambiguous or evaluator was uncertain (confidence 0.35 < 0.50).
    Difficulty must NOT de-escalate; it remains stable to avoid noisy drops.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain PostgreSQL vacuum.",
        category="Architecture",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    # Score 40.0, but confidence is only 0.35 (below min_confidence_for_deescalation of 0.50)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="It cleans things.",
        status="EVALUATED",
        overall_score=40.0,
        evaluation_confidence=0.35
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

    # Difficulty remains stable at 3 because evaluation confidence is too low
    assert selection.difficulty == 3


def test_difficulty_remaining_stable_when_evidence_insufficient(db_session: Session, populated_resume: Resume):
    """
    Edge Case: No evaluated answers available yet (e.g. initial question or pending evaluations).
    Difficulty remains stable at initial difficulty, no fake scores fabricated.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    # Question asked, but answer is PENDING
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Describe your experience with FastAPI.",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Submitted answer awaiting evaluation...",
        status="PENDING",
        overall_score=None
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Still at initial difficulty 3, no score fabricated
    assert selection.difficulty == 3
    assert "no evaluated answers" in selection.rationale.lower() or "retaining difficulty" in selection.rationale.lower()


def test_topic_selection_independent_from_difficulty_selection(db_session: Session, populated_resume: Resume):
    """
    Requirement 6: Topic selection remains independent from difficulty selection while cooperating.
    Demonstrates that difficulty escalation decision is applied to an uncovered topic
    when breadth is chosen after a strong performance.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Deep dive into Docker kernel mechanics.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Thorough explanation of cgroups v2 and user namespaces.",
        status="EVALUATED",
        overall_score=94.0,
        evaluation_confidence=0.95
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    # Force strategy to UNCOVERED_TOPIC to test independent topic selection
    req = AdaptiveQuestionSelectionRequest(force_strategy=AdaptiveStrategyType.UNCOVERED_TOPIC)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        request_override=req
    )

    assert selection.strategy == AdaptiveStrategyType.UNCOVERED_TOPIC
    assert selection.target_topic != "Docker"  # Selected an unassessed topic for breadth
    # But difficulty was escalated to 4 based on candidate's strong score!
    assert selection.difficulty == 4


def test_choosing_uncovered_topic_after_strong_answer_exhausted_budget(db_session: Session, populated_resume: Resume):
    """
    Requirement 7: Choosing an uncovered topic after a strong answer.
    Candidate already had a follow-up on Docker and scored 90+ on both.
    Selector does not ask a 3rd question on Docker; it pivots to an important uncovered
    topic (e.g. FastAPI/PostgreSQL) at the newly escalated difficulty.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    # Q1 on Docker
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Q1 on Docker namespaces",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Detailed namespaces explanation.",
        status="EVALUATED",
        overall_score=90.0,
        evaluation_confidence=0.90
    )
    # Q2 follow-up on Docker (consecutive_followup_count reaches 1)
    q2 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        parent_question_id=q1.id,
        question_text="Q2 on Docker cgroups and OOM killer",
        category="Failure-Mode",
        difficulty=4,
        target_entity="Docker"
    )
    ans2 = InterviewAnswer(
        id=uuid4(),
        question_id=q2.id,
        session_id=session.id,
        candidate_answer="Detailed OOM killer and memory cgroup explanation.",
        status="EVALUATED",
        overall_score=92.0,
        evaluation_confidence=0.90
    )
    db_session.add_all([q1, ans1, q2, ans2])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )
    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q2.id,
        answer=ans2,
        db=db_session
    )

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Candidate already had follow-up on Docker: MUST switch to an uncovered topic, NOT Docker again!
    assert selection.strategy in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION]
    assert selection.target_topic != "Docker"
    assert selection.is_follow_up is False
    # Tested at senior difficulty (difficulty 4 or 5)
    assert selection.difficulty >= 4


def test_difficulty_boundaries_min_and_max(db_session: Session, populated_resume: Resume):
    """
    Requirement 9: Difficulty minimum and maximum boundaries (1 to 5).
    - Session at difficulty 5 with another strong score (95.0) remains bounded at 5.
    - Session at difficulty 1 with a weak score (30.0) remains bounded at 1.
    """
    # Max bound test
    session_max = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=5, db=db_session)
    q_max = InterviewQuestion(
        id=uuid4(),
        session_id=session_max.id,
        resume_id=populated_resume.id,
        question_text="High performance distributed systems design.",
        category="Architecture",
        difficulty=5,
        target_entity="Python"
    )
    ans_max = InterviewAnswer(
        id=uuid4(),
        question_id=q_max.id,
        session_id=session_max.id,
        candidate_answer="Exceptional senior answer.",
        status="EVALUATED",
        overall_score=95.0,
        evaluation_confidence=0.95
    )
    db_session.add_all([q_max, ans_max])
    db_session.commit()
    knowledge_state_service.update_from_evaluation(session_id=session_max.id, question_id=q_max.id, answer=ans_max, db=db_session)

    sel_max = adaptive_selection_engine.determine_next_question_target(session_id=session_max.id, db=db_session)
    assert sel_max.difficulty == 5  # Cannot exceed 5

    # Min bound test
    session_min = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=1, db=db_session)
    q_min = InterviewQuestion(
        id=uuid4(),
        session_id=session_min.id,
        resume_id=populated_resume.id,
        question_text="Basic variable assignment.",
        category="Fundamentals",
        difficulty=1,
        target_entity="Python"
    )
    ans_min = InterviewAnswer(
        id=uuid4(),
        question_id=q_min.id,
        session_id=session_min.id,
        candidate_answer="I don't know.",
        status="EVALUATED",
        overall_score=20.0,
        evaluation_confidence=0.90
    )
    db_session.add_all([q_min, ans_min])
    db_session.commit()
    knowledge_state_service.update_from_evaluation(session_id=session_min.id, question_id=q_min.id, answer=ans_min, db=db_session)

    sel_min = adaptive_selection_engine.determine_next_question_target(session_id=session_min.id, db=db_session)
    assert sel_min.difficulty == 1  # Cannot drop below 1


def test_refusal_fast_path_handling_without_fabrication(db_session: Session, populated_resume: Resume):
    """
    Requirement 9: Refusal and fast-path refusal handling.
    Candidate refused ("I don't know"), evaluator identified fast_path_refusal.
    Difficulty is reduced if > 1 to provide approachable question, without fabricating false scores.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain internal LSM-tree storage engines in PostgreSQL.",
        category="Architecture",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="I don't know anything about LSM-trees.",
        status="EVALUATED",
        overall_score=0.0,
        evaluation_confidence=1.0,
        evaluator_model="fast_path_refusal",
        evaluation_feedback="Candidate explicitly refused or stated inability to answer question."
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

    # De-escalates difficulty from 3 to 2 so candidate receives an approachable question
    assert selection.difficulty == 2
    assert "refused" in selection.rationale.lower() or "declined" in selection.rationale.lower()


def test_difficulty_persistence_and_session_recovery(db_session: Session, populated_resume: Resume):
    """
    Requirement 8: Persistence and recovery of adaptive difficulty across requests.
    When a question is selected and generated, session.current_difficulty is committed to PostgreSQL.
    Upon reloading the session from DB, the difficulty is restored.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    # Add strong answer to Docker (92.0)
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain Docker networking.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Solid docker answer with cgroups and iptables.",
        status="EVALUATED",
        overall_score=92.0,
        evaluation_confidence=0.90
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q1.id,
        answer=ans1,
        db=db_session
    )

    mock_q = QuestionOutputSchema(
        question="Advanced failure recovery in Docker containers.",
        category="Failure-Mode",
        difficulty=4,
        target_entity="Docker",
        evidence_ids=["ev_1"],
        evidence_text="Docker cgroup OOM killer",
        rationale="Probe advanced failure modes",
        grounding_score=0.95,
        relevance_score=0.95,
        technical_quality_score=1.0,
        trace_id="trc_test"
    )
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=populated_resume.id,
        request_id="req_test",
        trace_id="trc_test",
        question_id=uuid4(),
        question=mock_q,
        validation_status="ACCEPTED",
        message="Generated successfully",
        attempts_count=1,
        latencies_ms={"total_ms": 15.0}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = session_service.select_and_generate_next_question(
            session_id=session.id,
            db=db_session
        )

    assert resp.status == "GENERATED"
    assert resp.difficulty == 4

    # Expire and reload session from DB to verify persistence
    db_session.expire_all()
    reloaded_session = db_session.query(InterviewSession).filter(InterviewSession.id == session.id).first()
    assert reloaded_session is not None
    # Persisted current_difficulty in DB is now 4!
    assert reloaded_session.current_difficulty == 4


def test_scale_normalization_supports_10_point_scale_inputs(db_session: Session, populated_resume: Resume):
    """
    Requirement Step 2: Explicit mapping between 10-point and 100-point scales.
    Verifies that a score passed as 8.5 / 10 maps to 85.0 / 100, correctly escalating difficulty.
    """
    session = session_service.create_session(resume_id=populated_resume.id, initial_difficulty=3, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain FastAPI dependency injection.",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    # Passed as 8.8 (on a 0-10 scale)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="FastAPI solves the dependency graph using Depends.",
        status="EVALUATED",
        overall_score=8.8,
        evaluation_confidence=0.90
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

    # 8.8 normalized to 88.0 (> 75.0) -> escalates difficulty to 4!
    assert selection.difficulty == 4
    assert selection.strategy == AdaptiveStrategyType.DIFFICULTY_ESCALATION
