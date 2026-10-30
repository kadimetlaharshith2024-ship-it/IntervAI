import pytest
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.rag import ResumeChunk
from app.services.knowledge_state_service import knowledge_state_service
from app.services.adaptive_selection_service import adaptive_selection_service
from app.schemas.adaptive_selection import AdaptiveStrategyType


@pytest.fixture(scope="function")
def db_session():
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


class TestTask4GapLogic:
    def test_reproduce_score_95_does_not_register_gap(self, db_session: Session):
        """
        TASK 4 Requirement:
        Evaluate answer with score 95 -> assert NO gap is registered,
        even if missing_concepts has non-substantive entries like 'None significant'.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(id=resume_id, filename="resume.pdf", raw_text="Python Backend Engineer")
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        sk = Skill(name="Python", category="Language")
        db_session.add_all([res, sess, sk])
        db_session.flush()

        db_session.add(ResumeSkill(resume_id=resume_id, skill_id=sk.id, extraction_confidence=0.9))
        db_session.commit()

        knowledge_state_service.initialize_knowledge_state(sess, db_session)

        # Question on Python
        q = InterviewQuestion(
            session_id=session_id,
            resume_id=resume_id,
            question_text="Explain Python GIL and concurrency trade-offs.",
            category="Implementation",
            difficulty=3,
            target_entity="Python",
            expected_concepts=["threading", "multiprocessing"]
        )
        db_session.add(q)
        db_session.commit()

        # Answer with score 95 and non-substantive missing concept
        ans = InterviewAnswer(
            session_id=session_id,
            question_id=q.id,
            candidate_answer="Detailed deep explanation of GIL, multiprocessing, asyncio, etc.",
            status="EVALUATED",
            overall_score=95.0,
            evaluation_confidence=0.95,
            evaluation_feedback="Excellent deep answer.",
            missing_concepts=["None significant", "Could briefly mention subinterpreters"],
            weaknesses=[]
        )
        db_session.add(ans)
        db_session.commit()

        updated_kstate = knowledge_state_service.update_knowledge_state_with_answer(
            session=sess,
            question=q,
            answer=ans,
            db=db_session
        )

        gaps = updated_kstate.get("knowledge_gaps", [])
        assert len(gaps) == 0, f"Expected 0 gaps for score 95, but found: {gaps}"

    def test_reproduce_mastery_ge_85_no_gap_followup(self, db_session: Session):
        """
        TASK 4 Requirement:
        Never trigger a gap follow-up on a topic whose mastery score is >= 85.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(id=resume_id, filename="resume.pdf", raw_text="PostgreSQL DBA")
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        sk_pg = Skill(name="PostgreSQL", category="Database")
        sk_redis = Skill(name="Redis", category="Database")
        db_session.add_all([res, sess, sk_pg, sk_redis])
        db_session.flush()

        db_session.add_all([
            ResumeSkill(resume_id=resume_id, skill_id=sk_pg.id, extraction_confidence=0.95),
            ResumeSkill(resume_id=resume_id, skill_id=sk_redis.id, extraction_confidence=0.95),
        ])
        c_proj = ResumeChunk(
            resume_id=resume_id,
            chunk_id="c1",
            section="Projects",
            project="DBSystem",
            chunk_text="DBSystem used PostgreSQL and Redis.",
            embedding_model="text-embedding-3-small"
        )
        db_session.add(c_proj)
        db_session.commit()

        knowledge_state_service.initialize_knowledge_state(sess, db_session)

        # Question on PostgreSQL evaluated with score 90 (mastery >= 85)
        q = InterviewQuestion(
            session_id=session_id,
            resume_id=resume_id,
            question_text="How do PostgreSQL MVCC and vacuuming work?",
            category="Implementation",
            difficulty=3,
            target_entity="PostgreSQL",
        )
        db_session.add(q)
        db_session.flush()

        ans = InterviewAnswer(
            session_id=session_id,
            question_id=q.id,
            candidate_answer="Thorough explanation of MVCC transaction isolation and autovacuum.",
            status="EVALUATED",
            overall_score=90.0,
            evaluation_confidence=0.95,
            evaluation_feedback="Strong answer.",
            missing_concepts=[]
        )
        db_session.add(ans)
        db_session.commit()

        kstate = knowledge_state_service.update_knowledge_state_with_answer(
            session=sess,
            question=q,
            answer=ans,
            db=db_session
        )
        # Manually plant an older gap on PostgreSQL if any existed
        kstate.setdefault("knowledge_gaps", []).append({
            "gap_id": "old_pg_gap",
            "topic": "PostgreSQL",
            "severity": "HIGH",
            "gap_nature": "INCOMPLETE_UNDERSTANDING",
            "is_addressed": False,
            "created_at": "2026-01-01T00:00:00"
        })
        sess.session_state = {"knowledge_state": kstate}
        db_session.commit()

        # Selector must NOT select GAP_FOLLOW_UP on PostgreSQL because mastery is >= 85
        next_sel = adaptive_selection_service.determine_next_question_target(session_id, db_session)
        assert next_sel.strategy != AdaptiveStrategyType.GAP_FOLLOW_UP or next_sel.target_topic != "PostgreSQL", (
            f"Expected no GAP_FOLLOW_UP on PostgreSQL (mastery >= 85), but got: {next_sel.strategy} on {next_sel.target_topic}"
        )

    def test_reproduce_max_2_followups_forces_breadth_question(self, db_session: Session):
        """
        TASK 4 Requirement:
        Max 2 follow-ups per topic per session; after a follow-up, force one breadth
        (uncovered topic) question before returning to gaps.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(id=resume_id, filename="resume.pdf", raw_text="Engineer: Docker, Go, Kubernetes")
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        sk_dk = Skill(name="Docker", category="DevOps")
        sk_go = Skill(name="Go", category="Language")
        sk_k8s = Skill(name="Kubernetes", category="DevOps")
        db_session.add_all([res, sess, sk_dk, sk_go, sk_k8s])
        db_session.flush()

        db_session.add_all([
            ResumeSkill(resume_id=resume_id, skill_id=sk_dk.id, extraction_confidence=0.9),
            ResumeSkill(resume_id=resume_id, skill_id=sk_go.id, extraction_confidence=0.9),
            ResumeSkill(resume_id=resume_id, skill_id=sk_k8s.id, extraction_confidence=0.9),
        ])
        c_proj = ResumeChunk(
            resume_id=resume_id,
            chunk_id="c_dk",
            section="Projects",
            project="InfraProj",
            chunk_text="InfraProj used Docker, Go, and Kubernetes.",
            embedding_model="text-embedding-3-small"
        )
        db_session.add(c_proj)
        db_session.commit()

        knowledge_state_service.initialize_knowledge_state(sess, db_session)

        # Q1 on Docker (initial question)
        q1 = InterviewQuestion(
            session_id=session_id,
            resume_id=resume_id,
            question_text="Docker question 1",
            category="Implementation",
            difficulty=3,
            target_entity="Docker",
            target_project="InfraProj"
        )
        db_session.add(q1)
        db_session.flush()

        ans1 = InterviewAnswer(
            session_id=session_id,
            question_id=q1.id,
            candidate_answer="Docker uses cgroups.",
            status="EVALUATED",
            overall_score=60.0,
            missing_concepts=["cgroups isolation"]
        )
        db_session.add(ans1)
        db_session.commit()
        knowledge_state_service.update_knowledge_state_with_answer(sess, q1, ans1, db_session)

        # Q2 is Follow-up 1 on Docker (has parent_question_id=q1.id)
        q2 = InterviewQuestion(
            session_id=session_id,
            resume_id=resume_id,
            parent_question_id=q1.id,
            question_text="Docker follow-up 1 on cgroups",
            category="Implementation",
            difficulty=3,
            target_entity="Docker",
            target_project="InfraProj"
        )
        db_session.add(q2)
        db_session.flush()

        ans2 = InterviewAnswer(
            session_id=session_id,
            question_id=q2.id,
            candidate_answer="cgroups limit CPU and memory.",
            status="EVALUATED",
            overall_score=65.0,
            missing_concepts=["namespaces"]
        )
        db_session.add(ans2)
        db_session.commit()
        knowledge_state_service.update_knowledge_state_with_answer(sess, q2, ans2, db_session)

        # Immediately after Follow-up 1:
        # Rule: "after a follow-up, force one breadth (uncovered topic) question before returning to gaps"
        sel_after_fu1 = adaptive_selection_service.determine_next_question_target(session_id, db_session)
        assert sel_after_fu1.strategy == AdaptiveStrategyType.UNCOVERED_TOPIC, (
            f"Expected breadth (UNCOVERED_TOPIC) immediately after follow-up, got: {sel_after_fu1.strategy}"
        )
        assert sel_after_fu1.target_topic in ["Go", "Kubernetes"], (
            f"Expected uncovered breadth topic (Go or Kubernetes), got: {sel_after_fu1.target_topic}"
        )
