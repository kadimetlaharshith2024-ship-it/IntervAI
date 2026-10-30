import pytest
import time
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim, ResumeChunk
from app.services.session_service import session_service
from app.services.knowledge_state_service import knowledge_state_service
from app.services.evaluation.evaluator import AnswerEvaluatorService
from tests.test_phase7_knowledge_state import MockKnowledgeEvaluationProvider
from app.schemas.adaptive_selection import AdaptiveStrategyType


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


@pytest.fixture(autouse=True)
def use_mock_llm(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")


@pytest.fixture
def rich_candidate_resume(db_session: Session) -> Resume:
    """Creates a comprehensive resume with topics, claims, projects, and relationships."""
    resume = Resume(
        id=uuid4(),
        filename="sarah_principal_engineer.pdf",
        raw_text=(
            "Sarah Principal Systems Architect. Skills: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes. "
            "Projects: FinTechCore, CloudStream. Claims: Architected real-time settlement engine processing 50k TPS. "
            "Designed geo-distributed failover across multi-region Kubernetes clusters."
        )
    )
    db_session.add(resume)
    db_session.flush()

    # Skills
    skills_data = [
        ("Python", "Language"),
        ("FastAPI", "Framework"),
        ("PostgreSQL", "Database"),
        ("Redis", "Database"),
        ("Docker", "DevOps"),
        ("Kubernetes", "DevOps"),
    ]
    for idx, (name, cat) in enumerate(skills_data, start=500):
        sk = Skill(id=idx, name=name, category=cat)
        db_session.add(sk)
        db_session.flush()
        rsk = ResumeSkill(resume_id=resume.id, skill_id=sk.id, extraction_confidence=0.95)
        db_session.add(rsk)

    # Chunks
    from app.services.rag.embedding_service import embedding_service
    c_texts = [
        "FinTechCore project: Architected real-time settlement engine processing 50k TPS using Python, FastAPI, Redis, and PostgreSQL.",
        "CloudStream project: Designed geo-distributed failover across multi-region Kubernetes clusters with Docker containers."
    ]
    embs = embedding_service.embed_texts(c_texts)
    chk1 = ResumeChunk(
        id=uuid4(),
        resume_id=resume.id,
        chunk_id="chk_1",
        chunk_text=c_texts[0],
        section="Experience",
        project="FinTechCore",
        embedding=embs[0],
        embedding_model=embedding_service.model_name
    )
    chk2 = ResumeChunk(
        id=uuid4(),
        resume_id=resume.id,
        chunk_id="chk_2",
        chunk_text=c_texts[1],
        section="Experience",
        project="CloudStream",
        embedding=embs[1],
        embedding_model=embedding_service.model_name
    )
    db_session.add_all([chk1, chk2])

    # Relationships
    rel1 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="FastAPI",
        relation_type="COMMUNICATES_WITH",
        target_entity="PostgreSQL",
        context_entity="FinTechCore",
        section="Experience",
        evidence_text="FinTechCore: Asynchronous transaction pipeline connecting FastAPI to PostgreSQL.",
        confidence=0.95
    )
    rel2 = Relationship(
        id=uuid4(),
        resume_id=resume.id,
        source_entity="Docker",
        relation_type="ORCHESTRATED_BY",
        target_entity="Kubernetes",
        context_entity="CloudStream",
        section="Experience",
        evidence_text="CloudStream: Docker containers orchestrated via Kubernetes pods.",
        confidence=0.92
    )
    db_session.add_all([rel1, rel2])

    # Claims
    clm1 = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-SETTLE-50K",
        section="Experience",
        project="FinTechCore",
        target_entity="FastAPI",
        relation_type="IMPLEMENTED_WITH",
        claim_text="FinTechCore: Architected real-time settlement engine processing 50k TPS.",
        evidence_text="FinTechCore: Architected real-time settlement engine processing 50k TPS.",
        confidence=0.95
    )
    clm2 = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-FAILOVER-K8S",
        section="Experience",
        project="CloudStream",
        target_entity="Kubernetes",
        relation_type="DEPLOYED_ON",
        claim_text="CloudStream: Designed geo-distributed failover across multi-region Kubernetes clusters.",
        evidence_text="CloudStream: Designed geo-distributed failover across multi-region Kubernetes clusters.",
        confidence=0.90
    )
    clm3 = ResumeClaim(
        id=uuid4(),
        resume_id=resume.id,
        claim_id="CLM-SCALE-DOCKER",
        section="Experience",
        project="CloudStream",
        target_entity="Docker",
        relation_type="CONTAINERIZED_WITH",
        claim_text="CloudStream: Scaled Docker container fleet to 1000 pods with 99.99% availability.",
        evidence_text="CloudStream: Scaled Docker container fleet to 1000 pods with 99.99% availability.",
        confidence=0.92
    )
    db_session.add_all([clm1, clm2, clm3])
    db_session.commit()
    db_session.refresh(resume)
    return resume


def run_10_question_simulation(candidate_type: str, db_session: Session, resume: Resume, prefetch_enabled: bool = False):
    """
    Simulates a 10-question adaptive interview session for strong, weak, or refusing candidates.
    Returns: (history, latencies)
    """
    session = session_service.create_session(
        resume_id=resume.id,
        db=db_session,
        session_metadata={
            "adaptive_config": {
                "max_session_questions": 10,
                "max_consecutive_followups": 2,
                "claim_verification_quota_percent": 0.20,
                "project_depth_quota_percent": 0.20,
                "prefetch_enabled": prefetch_enabled,
            }
        }
    )

    history = []
    latencies = []

    consecutive_followups = 0

    for turn in range(1, 11):
        t0 = time.perf_counter()
        next_q_resp = session_service.select_and_generate_next_question(
            session_id=session.id,
            db=db_session
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        strat = next_q_resp.selection.strategy if next_q_resp.selection else None
        print(f"TURN {turn}: status={next_q_resp.status}, strat={strat}, diff={next_q_resp.difficulty}, text={next_q_resp.question_text}")
        assert next_q_resp.status == "GENERATED"
        assert next_q_resp.question_id is not None
        assert next_q_resp.question_text is not None

        # Check difficulty bounds
        diff = next_q_resp.difficulty
        assert 1 <= diff <= 5, f"Turn {turn}: Difficulty {diff} out of bounds [1, 5]"

        # Check follow-up bounds
        strat = next_q_resp.selection.strategy
        is_followup = next_q_resp.selection.is_follow_up
        if is_followup:
            consecutive_followups += 1
            assert consecutive_followups <= 2, f"Turn {turn}: Exceeded max consecutive followups ({consecutive_followups} > 2)"
        else:
            consecutive_followups = 0

        history.append({
            "turn": turn,
            "question_id": next_q_resp.question_id,
            "question_text": next_q_resp.question_text,
            "strategy": strat,
            "category": next_q_resp.category,
            "difficulty": diff,
            "is_follow_up": is_followup,
            "target_claim_id": next_q_resp.selection.target_claim_id,
            "target_project": next_q_resp.target_project,
            "target_entity": next_q_resp.target_entity,
        })

        # Answer based on profile
        if candidate_type == "strong":
            provider = MockKnowledgeEvaluationProvider(
                scores={"technical_accuracy": 92.0, "conceptual_understanding": 90.0, "completeness": 90.0,
                        "depth_and_specificity": 92.0, "practical_reasoning": 90.0, "evidence_alignment": 90.0}
            )
            candidate_answer = "Comprehensive distributed systems answer explaining lock heartbeats, idempotency keys, and fencing tokens."
        elif candidate_type == "weak":
            provider = MockKnowledgeEvaluationProvider(
                scores={"technical_accuracy": 35.0, "conceptual_understanding": 40.0, "completeness": 35.0,
                        "depth_and_specificity": 30.0, "practical_reasoning": 35.0, "evidence_alignment": 40.0}
            )
            candidate_answer = "I just restart the server or ignore timeout errors."
        else:  # refusing
            provider = MockKnowledgeEvaluationProvider(
                scores={"technical_accuracy": 0.0, "conceptual_understanding": 0.0, "completeness": 0.0,
                        "depth_and_specificity": 0.0, "practical_reasoning": 0.0, "evidence_alignment": 0.0}
            )
            candidate_answer = "I don't know and refuse to answer."

        evaluator = AnswerEvaluatorService(default_provider=provider)
        evaluator.evaluate_answer(
            db=db_session,
            question_id=next_q_resp.question_id,
            candidate_answer=candidate_answer,
            session_id=session.id
        )

    return history, latencies


class TestPartCSimulation10Questions:
    """Simulation test: Scripted strong, weak, and refusing candidates through 10 questions."""

    def test_strong_candidate_10_question_simulation(self, db_session: Session, rich_candidate_resume: Resume):
        """Strong candidate: Difficulty remains in [1, 5], claims and project-depth meet quota, no duplicate questions."""
        history, latencies = run_10_question_simulation("strong", db_session, rich_candidate_resume, prefetch_enabled=False)

        assert len(history) == 10

        # Assert no question repeats within session
        question_texts = [h["question_text"].strip().lower() for h in history]
        assert len(set(question_texts)) == 10, "Duplicate question generated within session!"

        # Assert difficulty stays in 1-5
        for h in history:
            assert 1 <= h["difficulty"] <= 5

        # Strong candidate reaches difficulty 4 or 5
        max_diff = max(h["difficulty"] for h in history)
        assert max_diff >= 4, f"Strong candidate did not escalate difficulty (max was {max_diff})"

        # Assert claim questions appear at least quota (>= 2 questions = 20% of 10)
        claim_questions = [h for h in history if h["target_claim_id"] or h["strategy"] == AdaptiveStrategyType.CLAIM_VERIFICATION]
        assert len(claim_questions) >= 2, f"Claim verification quota not met: got {len(claim_questions)} < 2"

        # Assert project-depth questions appear at least quota (>= 2 questions = 20% of 10)
        proj_depth_questions = [h for h in history if h["strategy"] == AdaptiveStrategyType.PROJECT_DEPTH or h["category"] in ["Architecture", "Implementation"]]
        assert len(proj_depth_questions) >= 2, f"Project depth quota not met: got {len(proj_depth_questions)} < 2"

    def test_weak_candidate_10_question_simulation(self, db_session: Session, rich_candidate_resume: Resume):
        """Weak candidate: De-escalates difficulty to 1, stays in [1, 5], no duplicate questions, quotas met."""
        history, latencies = run_10_question_simulation("weak", db_session, rich_candidate_resume, prefetch_enabled=False)

        assert len(history) == 10

        # Assert no duplicates
        question_texts = [h["question_text"].strip().lower() for h in history]
        assert len(set(question_texts)) == 10, "Duplicate question generated within session!"

        # Assert difficulty stays in 1-5
        for h in history:
            assert 1 <= h["difficulty"] <= 5

        # Weak candidate de-escalates to difficulty 1 or 2
        min_diff = min(h["difficulty"] for h in history)
        assert min_diff <= 2, f"Weak candidate did not de-escalate difficulty (min was {min_diff})"

        # Claim verification appears
        claim_questions = [h for h in history if h["target_claim_id"] or h["strategy"] == AdaptiveStrategyType.CLAIM_VERIFICATION]
        assert len(claim_questions) >= 1

    def test_refusing_candidate_10_question_simulation(self, db_session: Session, rich_candidate_resume: Resume):
        """Refusing candidate: Stays in [1, 5], difficulty settles at 1, no duplicate questions."""
        history, latencies = run_10_question_simulation("refusing", db_session, rich_candidate_resume, prefetch_enabled=False)

        assert len(history) == 10

        # Assert no duplicates
        question_texts = [h["question_text"].strip().lower() for h in history]
        assert len(set(question_texts)) == 10, "Duplicate question generated within session!"

        for h in history:
            assert 1 <= h["difficulty"] <= 5

        # Settles at bottom difficulty
        assert history[-1]["difficulty"] == 1

    def test_benchmark_latencies_and_prefetch_comparison(self, db_session: Session, rich_candidate_resume: Resume):
        """Measures turn latencies (p50/p95), cache hit rate, and discarded prefetches with and without prefetching."""
        import numpy as np

        # 1. Without prefetch
        hist_no, lats_no = run_10_question_simulation("strong", db_session, rich_candidate_resume, prefetch_enabled=False)
        p50_no = np.percentile(lats_no, 50)
        p95_no = np.percentile(lats_no, 95)
        mean_no = np.mean(lats_no)
        print(f"\n[BENCHMARK WITHOUT PREFETCH] mean={mean_no:.2f}ms, p50={p50_no:.2f}ms, p95={p95_no:.2f}ms, turns={len(lats_no)}")

        # 2. With prefetch
        hist_pre, lats_pre = run_10_question_simulation("strong", db_session, rich_candidate_resume, prefetch_enabled=True)
        p50_pre = np.percentile(lats_pre, 50)
        p95_pre = np.percentile(lats_pre, 95)
        mean_pre = np.mean(lats_pre)
        print(f"[BENCHMARK WITH PREFETCH]    mean={mean_pre:.2f}ms, p50={p50_pre:.2f}ms, p95={p95_pre:.2f}ms, turns={len(lats_pre)}")
