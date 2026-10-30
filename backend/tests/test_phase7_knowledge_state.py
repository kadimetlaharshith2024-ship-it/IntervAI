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
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.services.session_service import session_service
from app.services.evaluation.evaluator import AnswerEvaluatorService
from app.services.knowledge_state_service import knowledge_state_service
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult
from app.schemas.knowledge_state import (
    TopicAssessmentStatus,
    TopicCoverageStatus,
    ClaimVerificationStatus,
    GapSeverity,
    GapNature,
)
from app.core.database import get_db
from app.main import app


class MockKnowledgeEvaluationProvider(BaseLLMProvider):
    """Configurable mock provider for knowledge state evaluation tests."""
    def __init__(self, scores=None, feedback=None):
        self.provider_name = "mock_knowledge_evaluator"
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
            "strengths": ["Clear explanation of core principles"],
            "weaknesses": [],
            "missing_concepts": [],
            "key_takeaways": "Good performance."
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
            latency_ms=10.0
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
    """Provides a FastAPI TestClient with database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def populated_resume(db_session: Session):
    """Creates a resume with extracted skills, relationships, and claims."""
    resume = Resume(
        id=uuid4(),
        filename="test_engineer_resume.pdf",
        raw_text="Full-Stack Engineer with Python, FastAPI, PostgreSQL, and Docker. Built TalentBridge."
    )
    db_session.add(resume)
    db_session.flush()

    # Add skills
    sk1 = Skill(id=101, name="Python", category="Language")
    sk2 = Skill(id=102, name="FastAPI", category="Framework")
    sk3 = Skill(id=103, name="PostgreSQL", category="Database")
    sk4 = Skill(id=104, name="Docker", category="Tool/DevOps")
    db_session.add_all([sk1, sk2, sk3, sk4])
    db_session.flush()

    rsk1 = ResumeSkill(resume_id=resume.id, skill_id=101, extraction_confidence=0.95)
    rsk2 = ResumeSkill(resume_id=resume.id, skill_id=102, extraction_confidence=0.90)
    rsk3 = ResumeSkill(resume_id=resume.id, skill_id=103, extraction_confidence=0.92)
    rsk4 = ResumeSkill(resume_id=resume.id, skill_id=104, extraction_confidence=0.88)
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
        claim_id="CLAIM-001",
        claim_text="Implemented async connection pool handling 5,000 queries/sec with zero drops.",
        section="Projects",
        project="TalentBridge",
        target_entity="PostgreSQL",
        relation_type="scaled_with",
        evidence_text="Engineered custom connection pool achieving 5k RPS.",
        confidence=0.92,
        status="UNTESTED"
    )
    db_session.add(claim1)

    db_session.commit()
    db_session.refresh(resume)
    return resume


# ============================================================================
# UNIT & INTEGRATION TESTS: CANDIDATE KNOWLEDGE STATE ENGINE (PHASE 7.2)
# ============================================================================

def test_knowledge_state_initialization(db_session: Session, populated_resume: Resume):
    """
    Test 1: Pre-registers skills, relationships, claims, and projects into knowledge state
    as UNASSESSED / UNTESTED. No topic is falsely marked as a weakness.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)
    state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)

    assert "topics" in state
    topics = state["topics"]
    assert "Python" in topics
    assert "FastAPI" in topics
    assert "PostgreSQL" in topics
    assert "Docker" in topics

    for t_name, t_data in topics.items():
        assert t_data["assessment_status"] == TopicAssessmentStatus.UNASSESSED.value
        assert t_data["coverage_status"] == TopicCoverageStatus.NOT_CONSIDERED.value
        assert t_data["mastery_score"] == 0.0
        assert t_data["confidence"] == 0.0
        assert t_data["questions_asked_count"] == 0
        assert t_data["answers_evaluated_count"] == 0

    assert "TalentBridge" in state["projects"]
    proj = state["projects"]["TalentBridge"]
    assert proj["is_assessed"] is False
    assert proj["architecture_understanding"] == "UNASSESSED"

    claims = state["claims"]
    assert len(claims) >= 1
    assert claims[0]["status"] == ClaimVerificationStatus.UNTESTED.value

    relationships = state["relationships"]
    assert len(relationships) >= 2
    for r in relationships:
        assert r["status"] == "UNTESTED"
        assert r["is_tested"] is False

    summary = state["summary"]
    assert len(summary["demonstrated_strengths"]) == 0
    assert len(summary["demonstrated_weaknesses"]) == 0
    assert len(summary["partially_understood"]) == 0
    assert len(summary["unassessed_topics"]) >= 4


def test_first_answer_mastery_and_confidence(db_session: Session, populated_resume: Resume):
    """
    Test 2: First answer for a topic bounds confidence to <= 0.50 and sets status PARTIALLY_ASSESSED.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How do dependency injection and routing work in FastAPI?",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI",
        expected_concepts=["Depends", "APIRouter"]
    )
    db_session.add(q)
    db_session.commit()

    provider = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 88.0, "conceptual_understanding": 86.0, "completeness": 85.0,
                "depth_and_specificity": 84.0, "practical_reasoning": 85.0, "evidence_alignment": 88.0}
    )
    evaluator = AnswerEvaluatorService(default_provider=provider)
    answer = evaluator.evaluate_answer(
        db=db_session,
        question_id=q.id,
        candidate_answer="FastAPI uses the Depends callable for dependency injection and APIRouter for modular routing.",
        session_id=session.id
    )

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    fastapi_topic = k_state["topics"]["FastAPI"]

    assert fastapi_topic["answers_evaluated_count"] == 1
    assert fastapi_topic["latest_score"] == pytest.approx(86.0, rel=1e-1)
    assert fastapi_topic["mastery_score"] == pytest.approx(86.0, rel=1e-1)
    # 1 observation cannot yield high confidence
    assert fastapi_topic["confidence"] <= 0.50
    assert fastapi_topic["confidence"] > 0.0
    assert fastapi_topic["assessment_status"] == TopicAssessmentStatus.PARTIALLY_ASSESSED.value


def test_multiple_answers_same_topic_recency_mastery(db_session: Session, populated_resume: Resume):
    """
    Test 3: Multiple answers for the same topic apply recency-weighted mastery
    (0.6 * latest + 0.4 * prior) and increase confidence.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Question 1 on PostgreSQL
    q1 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How does PostgreSQL MVCC handle concurrent row writes?",
        category="Fundamentals",
        difficulty=3,
        target_entity="PostgreSQL"
    )
    # Question 2 on PostgreSQL
    q2 = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How do you tune VACUUM and autovacuum in high-write PostgreSQL databases?",
        category="Architecture",
        difficulty=4,
        target_entity="PostgreSQL"
    )
    db_session.add_all([q1, q2])
    db_session.commit()

    # Answer 1 scores 70
    p1 = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 70.0, "conceptual_understanding": 70.0, "completeness": 70.0,
                "depth_and_specificity": 70.0, "practical_reasoning": 70.0, "evidence_alignment": 70.0}
    )
    evaluator1 = AnswerEvaluatorService(default_provider=p1)
    evaluator1.evaluate_answer(db=db_session, question_id=q1.id, candidate_answer="PostgreSQL uses MVCC with xmin and xmax.", session_id=session.id)

    # Answer 2 scores 90
    p2 = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 90.0, "conceptual_understanding": 90.0, "completeness": 90.0,
                "depth_and_specificity": 90.0, "practical_reasoning": 90.0, "evidence_alignment": 90.0}
    )
    evaluator2 = AnswerEvaluatorService(default_provider=p2)
    evaluator2.evaluate_answer(db=db_session, question_id=q2.id, candidate_answer="Autovacuum frees dead tuples by tuning scale factors.", session_id=session.id)

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    pg_topic = k_state["topics"]["PostgreSQL"]

    assert pg_topic["answers_evaluated_count"] == 2
    assert pg_topic["historical_scores"] == [pytest.approx(70.0, abs=1.0), pytest.approx(90.0, abs=1.0)]
    # Expected mastery: 0.6 * 90 + 0.4 * 70 = 54 + 28 = 82
    expected_mastery = 0.6 * pg_topic["historical_scores"][1] + 0.4 * pg_topic["historical_scores"][0]
    assert pg_topic["mastery_score"] == pytest.approx(expected_mastery, abs=1.0)
    # 2 observations increase confidence to ~0.75
    assert pg_topic["confidence"] >= 0.70
    assert pg_topic["assessment_status"] == TopicAssessmentStatus.ADEQUATELY_ASSESSED.value


def test_improved_performance_later_attempt(db_session: Session, populated_resume: Resume):
    """
    Test 4: Candidate re-attempts a question with improved score; knowledge gap is resolved.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain Docker multi-stage builds and layer caching.",
        category="Architecture",
        difficulty=3,
        target_entity="Docker"
    )
    db_session.add(q)
    db_session.commit()

    # Attempt 1: Poor performance (score 40)
    p1 = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 40.0, "conceptual_understanding": 40.0, "completeness": 40.0,
                "depth_and_specificity": 40.0, "practical_reasoning": 40.0, "evidence_alignment": 40.0},
        feedback={"summary": "Incomplete answer.", "strengths": [], "weaknesses": ["Missed caching mechanisms"], "missing_concepts": ["multi-stage"], "key_takeaways": "Study caching."}
    )
    evaluator1 = AnswerEvaluatorService(default_provider=p1)
    ans1 = evaluator1.evaluate_answer(db=db_session, question_id=q.id, candidate_answer="Docker builds images.", session_id=session.id)

    k_state_mid = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    assert len(k_state_mid["knowledge_gaps"]) == 1
    assert k_state_mid["knowledge_gaps"][0]["is_addressed"] is False
    assert k_state_mid["topics"]["Docker"]["mastery_score"] <= 45.0

    # Attempt 2: Improved performance (score 88)
    p2 = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 88.0, "conceptual_understanding": 88.0, "completeness": 88.0,
                "depth_and_specificity": 88.0, "practical_reasoning": 88.0, "evidence_alignment": 88.0},
        feedback={"summary": "Detailed and accurate explanation.", "strengths": ["Clear breakdown of build stages"], "weaknesses": [], "missing_concepts": [], "key_takeaways": "Great."}
    )
    evaluator2 = AnswerEvaluatorService(default_provider=p2)
    ans2 = evaluator2.evaluate_answer(
        db=db_session,
        question_id=q.id,
        candidate_answer="Multi-stage builds use separate FROM statements to keep the final artifact minimal, leveraging build layer caching.",
        session_id=session.id
    )

    k_state_final = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    docker_topic = k_state_final["topics"]["Docker"]

    assert docker_topic["answers_evaluated_count"] == 2
    # Mastery increased
    assert docker_topic["mastery_score"] > 65.0
    # The previous gap is now addressed
    gaps = k_state_final["knowledge_gaps"]
    first_gap = gaps[0]
    assert first_gap["is_addressed"] is True
    assert "Addressed in attempt 2" in (first_gap.get("resolution_notes") or "")


def test_unanswered_skipped_question_handling(db_session: Session, populated_resume: Resume):
    """
    Test 5: An unanswered or skipped question does not fabricate scores or get falsely marked as a weakness.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How do Python generators optimize memory usage with iterators?",
        category="Fundamentals",
        difficulty=3,
        target_entity="Python"
    )
    db_session.add(q)
    db_session.commit()

    # No answer submitted
    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    py_topic = k_state["topics"]["Python"]

    assert py_topic["answers_evaluated_count"] == 0
    assert py_topic["mastery_score"] == 0.0
    assert py_topic["confidence"] == 0.0
    assert py_topic["assessment_status"] == TopicAssessmentStatus.UNASSESSED.value
    # Must NOT be marked as a demonstrated weakness
    assert "Python" not in k_state["summary"]["demonstrated_weaknesses"]
    assert "Python" in k_state["summary"]["unassessed_topics"]


def test_refusal_fast_path_answer_claim_not_contradicted(db_session: Session, populated_resume: Resume):
    """
    Test 6: Candidate refusal produces 0 score, but targeted resume claim is set to
    INSUFFICIENT_EVIDENCE, NOT CONTRADICTED.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How did your connection pool handle 5,000 queries/sec in TalentBridge?",
        category="Architecture",
        difficulty=4,
        target_project="TalentBridge",
        targeted_claim="Implemented async connection pool handling 5,000 queries/sec with zero drops."
    )
    db_session.add(q)
    db_session.commit()

    evaluator = AnswerEvaluatorService()
    # Candidate skips / refuses
    answer = evaluator.evaluate_answer(
        db=db_session,
        question_id=q.id,
        candidate_answer="I don't know the exact details, please skip this question.",
        session_id=session.id
    )

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    claims = k_state["claims"]
    matching_claim = [c for c in claims if "5,000" in c["claim_text"]][0]

    # Refusal must be INSUFFICIENT_EVIDENCE, never false CONTRADICTED
    assert matching_claim["status"] == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE.value
    assert matching_claim["status"] != ClaimVerificationStatus.CONTRADICTED.value


def test_multi_topic_question_updates_all_topics(db_session: Session, populated_resume: Resume):
    """
    Test 7: A question targeting FastAPI with expected concepts Pydantic and AsyncIO
    updates all three topics properly.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain how FastAPI uses Pydantic models with Python AsyncIO coroutines.",
        category="Architecture",
        difficulty=3,
        target_entity="FastAPI",
        expected_concepts=["Pydantic", "AsyncIO"]
    )
    db_session.add(q)
    db_session.commit()

    provider = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 85.0, "conceptual_understanding": 85.0, "completeness": 85.0,
                "depth_and_specificity": 85.0, "practical_reasoning": 85.0, "evidence_alignment": 85.0}
    )
    evaluator = AnswerEvaluatorService(default_provider=provider)
    evaluator.evaluate_answer(
        db=db_session,
        question_id=q.id,
        candidate_answer="FastAPI serializes models through Pydantic while event loops manage AsyncIO non-blocking coroutines.",
        session_id=session.id
    )

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    topics = k_state["topics"]

    assert "FastAPI" in topics
    assert "Pydantic" in topics
    assert "AsyncIO" in topics

    assert topics["FastAPI"]["answers_evaluated_count"] == 1
    assert topics["Pydantic"]["answers_evaluated_count"] == 1
    assert topics["AsyncIO"]["answers_evaluated_count"] == 1


def test_question_without_identifiable_topic_fallback(db_session: Session, populated_resume: Resume):
    """
    Test 8: Question with no target entity or expected concepts safely falls back to category.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="How do you handle distributed consensus in microservices?",
        category="Architecture",
        difficulty=4,
        target_entity=None,
        expected_concepts=[]
    )
    db_session.add(q)
    db_session.commit()

    provider = MockKnowledgeEvaluationProvider()
    evaluator = AnswerEvaluatorService(default_provider=provider)
    evaluator.evaluate_answer(
        db=db_session,
        question_id=q.id,
        candidate_answer="We can use Raft or Paxos protocols to elect leaders.",
        session_id=session.id
    )

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    assert "Architecture" in k_state["topics"]
    assert k_state["topics"]["Architecture"]["answers_evaluated_count"] == 1


def test_project_understanding_architecture_and_implementation(db_session: Session, populated_resume: Resume):
    """
    Test 9: Project architecture and implementation understanding are assessed independently.
    Evidence is isolated to the correct project.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    # Architecture question on TalentBridge
    q_arch = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Walk me through the overall system architecture of TalentBridge.",
        category="Architecture",
        difficulty=3,
        target_project="TalentBridge"
    )
    db_session.add(q_arch)
    db_session.commit()

    provider = MockKnowledgeEvaluationProvider(
        scores={"technical_accuracy": 85.0, "conceptual_understanding": 85.0, "completeness": 85.0,
                "depth_and_specificity": 85.0, "practical_reasoning": 85.0, "evidence_alignment": 85.0}
    )
    evaluator = AnswerEvaluatorService(default_provider=provider)
    evaluator.evaluate_answer(
        db=db_session,
        question_id=q_arch.id,
        candidate_answer="TalentBridge uses a layered architecture with FastAPI, async workers, and PostgreSQL.",
        session_id=session.id
    )

    k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db_session)
    proj = k_state["projects"]["TalentBridge"]

    assert proj["is_assessed"] is True
    assert proj["architecture_understanding"] == "STRONG"
    # Implementation not yet assessed on this project
    assert proj["implementation_understanding"] == "UNASSESSED"


def test_cross_resume_cross_session_isolation(db_session: Session, populated_resume: Resume):
    """
    Test 10: Validates that questions and answers cannot cross resume or session boundaries.
    """
    session1 = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    other_resume = Resume(
        id=uuid4(),
        filename="other.pdf",
        raw_text="Other resume content."
    )
    db_session.add(other_resume)
    db_session.commit()

    session2 = session_service.create_session(resume_id=other_resume.id, db=db_session)

    # Question belongs to session1 / populated_resume
    q = InterviewQuestion(
        id=uuid4(),
        session_id=session1.id,
        resume_id=populated_resume.id,
        question_text="Question text",
        category="Fundamentals"
    )
    db_session.add(q)
    db_session.commit()

    evaluator = AnswerEvaluatorService()
    # Attempting to submit for session2 must raise HTTPException(400)
    with pytest.raises(HTTPException) as exc:
        evaluator.evaluate_answer(
            db=db_session,
            question_id=q.id,
            candidate_answer="Some answer",
            session_id=session2.id
        )
    assert exc.value.status_code == 400


def test_idempotent_evaluation_processing(db_session: Session, populated_resume: Resume):
    """
    Test 11: Duplicate processing of the same answer attempt does not double-count or corrupt state.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain Python list comprehensions vs generators.",
        category="Fundamentals",
        target_entity="Python"
    )
    db_session.add(q)
    db_session.commit()

    ans = InterviewAnswer(
        id=uuid4(),
        question_id=q.id,
        session_id=session.id,
        candidate_answer="List comprehensions build lists in memory while generators yield lazily.",
        status="EVALUATED",
        attempt_number=1,
        overall_score=85.0,
        evaluation_confidence=0.9
    )
    db_session.add(ans)
    db_session.commit()

    # First call
    s1 = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=ans,
        db=db_session
    )
    py_count_1 = s1["topics"]["Python"]["answers_evaluated_count"]

    # Second identical call
    s2 = knowledge_state_service.update_from_evaluation(
        session_id=session.id,
        question_id=q.id,
        answer=ans,
        db=db_session
    )
    py_count_2 = s2["topics"]["Python"]["answers_evaluated_count"]

    assert py_count_1 == 1
    assert py_count_2 == 1
    assert len(s2["topics"]["Python"]["historical_scores"]) == 1


def test_session_reload_and_persistence_recovery(db_session: Session, populated_resume: Resume):
    """
    Test 12: Knowledge state persists in PostgreSQL session_state JSON column and reloads cleanly.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)
    session_id = session.id

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session_id,
        resume_id=populated_resume.id,
        question_text="Explain FastAPI dependency injection.",
        category="Architecture",
        target_entity="FastAPI"
    )
    db_session.add(q)
    db_session.commit()

    provider = MockKnowledgeEvaluationProvider(scores={"technical_accuracy": 92.0, "conceptual_understanding": 90.0, "completeness": 90.0, "depth_and_specificity": 90.0, "practical_reasoning": 90.0, "evidence_alignment": 90.0})
    evaluator = AnswerEvaluatorService(default_provider=provider)
    evaluator.evaluate_answer(db=db_session, question_id=q.id, candidate_answer="FastAPI uses Depends callable.", session_id=session_id)

    # Simulate server restart / fresh session lookup
    reloaded_session = db_session.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    reloaded_state = knowledge_state_service.get_or_initialize_knowledge_state(reloaded_session, db_session)

    assert "FastAPI" in reloaded_state["topics"]
    assert reloaded_state["topics"]["FastAPI"]["answers_evaluated_count"] == 1
    assert reloaded_state["topics"]["FastAPI"]["mastery_score"] >= 85.0


def test_reconcile_knowledge_state_recovery(db_session: Session, populated_resume: Resume):
    """
    Test 13: Reconcile re-constructs knowledge state from scratch by cleanly replaying historical answers.
    """
    session = session_service.create_session(resume_id=populated_resume.id, db=db_session)

    q = InterviewQuestion(
        id=uuid4(),
        session_id=session.id,
        resume_id=populated_resume.id,
        question_text="Explain PostgreSQL indexing strategies.",
        category="Architecture",
        target_entity="PostgreSQL"
    )
    db_session.add(q)
    db_session.commit()

    # Insert two answers
    ans1 = InterviewAnswer(
        id=uuid4(),
        question_id=q.id,
        session_id=session.id,
        candidate_answer="B-tree indexes are default.",
        status="EVALUATED",
        attempt_number=1,
        overall_score=70.0,
        evaluation_confidence=0.9
    )
    ans2 = InterviewAnswer(
        id=uuid4(),
        question_id=q.id,
        session_id=session.id,
        candidate_answer="B-Tree for equality and range, GIN for inverted full-text and jsonb.",
        status="EVALUATED",
        attempt_number=2,
        overall_score=90.0,
        evaluation_confidence=0.9
    )
    db_session.add_all([ans1, ans2])
    db_session.commit()

    # Reconcile state
    reconciled = knowledge_state_service.reconcile_knowledge_state(session_id=session.id, db=db_session)
    pg = reconciled["topics"]["PostgreSQL"]

    assert pg["answers_evaluated_count"] == 2
    assert len(pg["historical_scores"]) == 2
    assert pg["latest_score"] == 90.0
    assert pg["confidence"] >= 0.70


def test_api_endpoints_knowledge_state_routes(client: TestClient, db_session: Session, populated_resume: Resume):
    """
    Test 14: API endpoints for retrieving and reconciling Candidate Knowledge State.
    """
    # Create session via API
    res = client.post("/api/v1/sessions", json={"resume_id": str(populated_resume.id), "initial_difficulty": 3})
    assert res.status_code == 201
    session_id = res.json()["id"]

    # 1. GET /api/v1/sessions/{id}/knowledge-state
    res_ks = client.get(f"/api/v1/sessions/{session_id}/knowledge-state")
    assert res_ks.status_code == 200
    ks_data = res_ks.json()
    assert ks_data["session_id"] == session_id
    assert "topics" in ks_data
    assert "summary" in ks_data
    assert "coverage" in ks_data

    # 2. GET /api/v1/sessions/{id}/knowledge-state/topics
    res_topics = client.get(f"/api/v1/sessions/{session_id}/knowledge-state/topics")
    assert res_topics.status_code == 200
    assert "Python" in res_topics.json()

    # 3. GET /api/v1/sessions/{id}/knowledge-state/projects
    res_proj = client.get(f"/api/v1/sessions/{session_id}/knowledge-state/projects")
    assert res_proj.status_code == 200
    assert "TalentBridge" in res_proj.json()

    # 4. GET /api/v1/sessions/{id}/knowledge-state/relationships
    res_rel = client.get(f"/api/v1/sessions/{session_id}/knowledge-state/relationships")
    assert res_rel.status_code == 200
    assert len(res_rel.json()) >= 2

    # 5. GET /api/v1/sessions/{id}/knowledge-state/claims
    res_claims = client.get(f"/api/v1/sessions/{session_id}/knowledge-state/claims")
    assert res_claims.status_code == 200
    assert len(res_claims.json()) >= 1

    # 6. GET /api/v1/sessions/{id}/knowledge-state/gaps
    res_gaps = client.get(f"/api/v1/sessions/{session_id}/knowledge-state/gaps")
    assert res_gaps.status_code == 200
    assert isinstance(res_gaps.json(), list)

    # 7. POST /api/v1/sessions/{id}/knowledge-state/reconcile
    res_rec = client.post(f"/api/v1/sessions/{session_id}/knowledge-state/reconcile")
    assert res_rec.status_code == 200
    assert res_rec.json()["session_id"] == session_id
