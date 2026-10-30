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
from app.schemas.knowledge_state import ClaimVerificationStatus, TopicCoverageStatus
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
def populated_resume(db_session: Session):
    """Creates a sample resume with extracted skills, relationships, and claims."""
    resume = Resume(
        id=uuid4(),
        filename="alice_cloud_engineer.pdf",
        raw_text="Cloud Engineer with Python, FastAPI, PostgreSQL, and Docker. Built TalentBridge."
    )
    db_session.add(resume)
    db_session.flush()

    # Add skills
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

    # Add relationships
    rel1 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="TalentBridge",
        relation_type="built_with",
        target_entity="FastAPI",
        section="Projects",
        context_entity="TalentBridge",
        evidence_text="Architected TalentBridge backend using FastAPI and PostgreSQL.",
        confidence=0.95
    )
    rel2 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="uses",
        target_entity="PostgreSQL",
        section="Projects",
        context_entity="TalentBridge",
        evidence_text="Integrated async SQLAlchemy ORM connecting FastAPI with PostgreSQL.",
        confidence=0.90
    )
    db_session.add_all([rel1, rel2])

    # Add claims
    claim1 = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLAIM-FASTAPI-01",
        claim_text="Scaled TalentBridge API throughput to 10,000 requests/sec with uvloop and connection pooling.",
        section="Projects",
        project="TalentBridge",
        target_entity="FastAPI",
        relation_type="scaled_with",
        evidence_text="Scaled API throughput to 10k RPS.",
        confidence=0.95,
        status="UNTESTED"
    )
    db_session.add(claim1)

    db_session.commit()
    db_session.refresh(resume)
    return resume


# ============================================================================
# PHASE 7.3: ADAPTIVE NEXT-QUESTION SELECTION TESTS
# ============================================================================

def test_choosing_uncovered_important_topic(db_session: Session, populated_resume: Resume):
    """
    Test 1: Choosing an uncovered important topic for interview breadth.
    At session start, selector identifies an unassessed high-priority topic (e.g. FastAPI / Database / Language).
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    assert selection.strategy == AdaptiveStrategyType.UNCOVERED_TOPIC
    assert selection.is_follow_up is False
    assert selection.parent_question_id is None
    assert selection.target_topic in ["FastAPI", "PostgreSQL", "Python", "Docker"]
    assert selection.difficulty == 3
    assert "breadth coverage" in selection.rationale


def test_following_up_on_demonstrated_weakness(db_session: Session, populated_resume: Resume):
    """
    Test 2: Following up on a demonstrated weakness.
    When candidate has an unresolved high-severity knowledge gap, selector prioritizes
    GAP_FOLLOW_UP on that topic with adjusted difficulty to probe core understanding.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Add an asked question on PostgreSQL
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How does PostgreSQL handle WAL logging and checkpointing under load?",
        category="Architecture",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    # Add an evaluated answer with low score (40.0)
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="PostgreSQL writes logs.",
        status="EVALUATED",
        overall_score=40.0,
        missing_concepts=["WAL buffers", "checkpoint spikes"],
        weaknesses=["Candidate struggled with write amplification and buffer pool flush mechanics."]
    )
    db_session.add_all([q1, ans1])
    db_session.commit()

    # Update knowledge state with the answer
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
    # Difficulty reduced to probe fundamentals
    assert selection.difficulty <= 3
    assert selection.category in ["Fundamentals", "Implementation"]
    assert "knowledge gap" in selection.rationale.lower()


def test_escalating_after_strong_answer(db_session: Session, populated_resume: Resume):
    """
    Test 3: Escalating difficulty after a strong answer.
    Candidate scores 92.0 on Docker; selector chooses DIFFICULTY_ESCALATION, increases
    difficulty (e.g. 3 -> 4) and probes Failure-Mode or Performance.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain Docker container lifecycle and cgroup isolation.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q1.id,
        session_id=session.id,
        candidate_answer="Docker leverages Linux namespaces for PID/network isolation and cgroups for CPU/memory limits.",
        status="EVALUATED",
        overall_score=92.0,
        strengths=["Deep understanding of Linux kernel primitives"]
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
    assert selection.difficulty == 4
    assert selection.category in ["Failure-Mode", "Performance", "Security"]
    assert "escalating" in selection.rationale.lower()


def test_choosing_between_weakness_followup_and_uncovered_topic_budget(db_session: Session, populated_resume: Resume):
    """
    Test 4: Choosing between a weakness follow-up and an uncovered topic.
    When consecutive follow-ups reach the maximum limit (e.g. 2), the selector refuses to get
    trapped in a loop and forces a switch to an uncovered topic for breadth.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Question 1 (Root)
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Q1 on Python",
        category="Fundamentals",
        difficulty=3,
        target_entity="Python"
    )
    # Question 2 (Follow-up 1)
    q2 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        parent_question_id=q1.id,
        question_text="Q2 follow-up on Python",
        category="Implementation",
        difficulty=2,
        target_entity="Python"
    )
    # Question 3 (Follow-up 2 -> consecutive_followup_count reaches 2)
    q3 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        parent_question_id=q2.id,
        question_text="Q3 follow-up on Python",
        category="Fundamentals",
        difficulty=2,
        target_entity="Python"
    )
    ans3 = InterviewAnswer(
        id=uuid4(),
        question_id=q3.id,
        session_id=session.id,
        candidate_answer="Answer with gaps.",
        status="EVALUATED",
        overall_score=45.0,
        missing_concepts=["GIL", "thread synchronization"]
    )
    db_session.add_all([q1, q2, q3, ans3])
    db_session.commit()

    knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q3.id,
        answer=ans3,
        db=db_session
    )

    # Max consecutive follow-ups is 2
    cfg = AdaptiveSelectionConfig(max_consecutive_followups=2)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        config_override=cfg
    )

    # Budget reached: MUST switch to an uncovered topic, NOT follow up on Python again!
    assert selection.strategy in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION]
    assert selection.target_topic != "Python"
    assert selection.is_follow_up is False
    assert selection.consecutive_followup_count == 0


def test_relationship_and_resume_claim_coverage(db_session: Session, populated_resume: Resume):
    """
    Test 5: Relationship and resume-claim coverage.
    Selector identifies unverified resume claims and untested relationships.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Explicitly test CLAIM_VERIFICATION strategy
    req = AdaptiveQuestionSelectionRequest(force_strategy=AdaptiveStrategyType.CLAIM_VERIFICATION)
    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        request_override=req
    )

    assert selection.strategy == AdaptiveStrategyType.CLAIM_VERIFICATION
    assert selection.target_claim_id == "CLAIM-FASTAPI-01"
    assert selection.category == "Claim-Verification"
    assert selection.target_project == "TalentBridge"

    # Explicitly test RELATIONSHIP_VERIFICATION strategy
    req_rel = AdaptiveQuestionSelectionRequest(force_strategy=AdaptiveStrategyType.RELATIONSHIP_VERIFICATION)
    sel_rel = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session,
        request_override=req_rel
    )

    assert sel_rel.strategy == AdaptiveStrategyType.RELATIONSHIP_VERIFICATION
    assert sel_rel.target_relationship is not None
    assert "FastAPI" in sel_rel.target_relationship or "TalentBridge" in sel_rel.target_relationship


def test_duplicate_question_prevention(db_session: Session, populated_resume: Resume):
    """
    Test 6: Duplicate question prevention.
    If a question on (FastAPI, Architecture) already exists, selector prevents repeating
    the identical category on the same topic.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How is dependency injection configured in FastAPI?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI"
    )
    db_session.add(q)
    db_session.commit()

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    # Next question must not be a duplicate (FastAPI, Architecture)
    is_exact_dup = (selection.target_entity == "FastAPI" and selection.category == "Architecture")
    assert not is_exact_dup


def test_follow_up_budget_exhaustion(db_session: Session, populated_resume: Resume):
    """
    Test 7: Session question budget exhaustion.
    When question count hits configured limit (e.g. 5), select_and_generate_next_question
    returns status 'BUDGET_EXHAUSTED' safely.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Add 5 questions
    for i in range(5):
        q = InterviewQuestion(
            id=uuid4(),
            session_id=session.id,
            resume_id=populated_resume.id,
            question_text=f"Question {i}",
            category="Architecture",
            difficulty=3
        )
        db_session.add(q)
    db_session.commit()

    cfg = AdaptiveSelectionConfig(default_max_session_questions=5)
    resp = adaptive_selection_engine.select_and_generate_next_question(
        session_id=session.id,
        db=db_session,
        config_override=cfg
    )

    assert resp.status == "BUDGET_EXHAUSTED"
    assert "budget" in resp.message.lower()


def test_empty_or_incomplete_knowledge_state(db_session: Session, populated_resume: Resume):
    """
    Test 8: Empty or incomplete knowledge state.
    Session with empty session_state safely initializes knowledge state and selects next target.
    """
    session = InterviewSession(
        id=uuid4(),
        resume_id=populated_resume.id,
        status="IN_PROGRESS",
        current_difficulty=3,
        session_state={}  # completely empty
    )
    db_session.add(session)
    db_session.commit()

    selection = adaptive_selection_engine.determine_next_question_target(
        session_id=session.id,
        db=db_session
    )

    assert selection is not None
    assert selection.strategy == AdaptiveStrategyType.UNCOVERED_TOPIC
    assert selection.target_topic is not None


def test_missing_retrieval_evidence_handling(db_session: Session, populated_resume: Resume):
    """
    Test 9: Missing retrieval evidence handling.
    If RAG generation returns INSUFFICIENT_EVIDENCE, select_and_generate_next_question
    returns status EVIDENCE_UNAVAILABLE without corrupting session state or raising unhandled errors.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Mock RAG pipeline returning INSUFFICIENT_EVIDENCE
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=populated_resume.id,
        request_id="req_test",
        trace_id="trc_test",
        question=None,
        validation_status="INSUFFICIENT_EVIDENCE",
        message="No verified evidence found for query.",
        attempts_count=1,
        latencies_ms={"total_ms": 10.0}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session.id,
            db=db_session
        )

        assert resp.status == "EVIDENCE_UNAVAILABLE"
        assert "could not generate grounded question" in resp.message.lower()
        # Session state should remain IN_PROGRESS
        assert session.status == "IN_PROGRESS"


def test_ended_or_invalid_sessions(db_session: Session, populated_resume: Resume):
    """
    Test 10: Ended or invalid session handling.
    Calling selector on ENDED or COMPLETED session raises HTTPException 400.
    Invalid session ID raises HTTPException 404.
    """
    # Invalid session ID -> 404
    with pytest.raises(HTTPException) as exc404:
        adaptive_selection_engine.determine_next_question_target(
            session_id=uuid4(),
            db=db_session
        )
    assert exc404.value.status_code == 404

    # Ended session -> 400
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)
    session_service.end_interview(session_id=session.id, db=db_session)

    with pytest.raises(HTTPException) as exc400:
        adaptive_selection_engine.determine_next_question_target(
            session_id=session.id,
            db=db_session
        )
    assert exc400.value.status_code == 400
    assert "already" in exc400.value.detail.lower()


def test_deterministic_selection_for_equivalent_state(db_session: Session, populated_resume: Resume):
    """
    Test 11: Deterministic selection for equivalent state.
    Multiple invocations of determine_next_question_target on identical state yield identical results.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    res1 = adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    res2 = adaptive_selection_engine.determine_next_question_target(session.id, db_session)
    res3 = adaptive_selection_engine.determine_next_question_target(session.id, db_session)

    assert res1.strategy == res2.strategy == res3.strategy
    assert res1.target_topic == res2.target_topic == res3.target_topic
    assert res1.category == res2.category == res3.category
    assert res1.difficulty == res2.difficulty == res3.difficulty
    assert res1.rationale == res2.rationale == res3.rationale


def test_persistence_and_recovery(db_session: Session, populated_resume: Resume):
    """
    Test 12: Persistence and recovery of adaptive selection.
    select_and_generate_next_question generates a question, updates session difficulty,
    marks coverage_status in session_state, and commits to DB.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)
    session_id = session.id

    mock_q = QuestionOutputSchema(
        question="How does FastAPI utilize starlette routing and pydantic models?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI",
        target_project="TalentBridge",
        evidence_ids=["ev_1"],
        evidence_text="Architected TalentBridge backend using FastAPI and PostgreSQL.",
        rationale="Testing FastAPI architecture",
        grounding_score=1.0,
        relevance_score=1.0,
        technical_quality_score=1.0,
        trace_id="trc_mock"
    )

    created_q_id = uuid4()
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=populated_resume.id,
        request_id="req_test",
        trace_id="trc_test",
        question_id=created_q_id,
        question=mock_q,
        validation_status="ACCEPTED",
        message="Generated successfully",
        attempts_count=1,
        latencies_ms={"total_ms": 15.0}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        resp = adaptive_selection_engine.select_and_generate_next_question(
            session_id=session_id,
            db=db_session
        )

        assert resp.status == "GENERATED"
        assert resp.question_text == mock_q.question
        assert resp.target_entity == "FastAPI"

    # Reload from DB
    reloaded_session = db_session.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    assert reloaded_session.current_difficulty == 3
    k_state = reloaded_session.session_state.get("knowledge_state", {})
    fastapi_topic = k_state.get("topics", {}).get("FastAPI", {})
    assert fastapi_topic.get("coverage_status") == TopicCoverageStatus.QUESTION_ASKED.value


def test_api_endpoints_adaptive_selection(client: TestClient, db_session: Session, populated_resume: Resume):
    """
    Test 13: REST API endpoints for previewing and generating next adaptive question.
    """
    res_start = client.post("/api/v1/sessions", json={"resume_id": str(populated_resume.id), "initial_difficulty": 3})
    assert res_start.status_code == 201
    session_id = res_start.json()["id"]

    # 1. GET /api/v1/sessions/{id}/next-question/preview
    res_prev = client.get(f"/api/v1/sessions/{session_id}/next-question/preview")
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert "strategy" in prev_data
    assert "target_topic" in prev_data
    assert "rationale" in prev_data

    # 2. POST /api/v1/sessions/{id}/next-question
    mock_q = QuestionOutputSchema(
        question="How does FastAPI configure dependency injection?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI",
        evidence_ids=["ev_1"],
        evidence_text="Backend with FastAPI.",
        rationale="Testing DI",
        grounding_score=1.0,
        relevance_score=1.0,
        technical_quality_score=1.0,
        trace_id="trc_api_mock"
    )
    mock_rag_resp = QuestionGenerationResponse(
        resume_id=populated_resume.id,
        request_id="req_api",
        trace_id="trc_api",
        question_id=uuid4(),
        question=mock_q,
        validation_status="ACCEPTED",
        message="Generated successfully",
        attempts_count=1,
        latencies_ms={"total_ms": 20.0}
    )

    with patch.object(rag_pipeline, "generate_grounded_question", return_value=mock_rag_resp):
        res_next = client.post(f"/api/v1/sessions/{session_id}/next-question")
        assert res_next.status_code == 200
        next_data = res_next.json()
        assert next_data["status"] == "GENERATED"
        assert next_data["session_id"] == session_id
        assert next_data["question_text"] == mock_q.question
        assert next_data["selection"] is not None
