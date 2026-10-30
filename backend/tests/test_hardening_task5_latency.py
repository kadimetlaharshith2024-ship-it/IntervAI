import time
import pytest
from uuid import uuid4
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.interview import Base, InterviewSession, InterviewQuestion
from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.rag import ResumeChunk
from app.core.config import settings
from app.services.session_service import session_service
from app.services.rag.rag_pipeline import rag_pipeline
from app.schemas.rag import QuestionGenerationRequest
from app.services.adaptive_selection_service import adaptive_selection_service


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


class TestTask5LatencyHardening:
    def test_reproduce_session_creation_pregenerates_q1(self, db_session: Session):
        """
        TASK 5 Requirement:
        Pre-generate Question 1 when the session is created, so the candidate's
        first 'next question' call is an instant cache hit.
        """
        resume_id = uuid4()
        res = Resume(
            id=resume_id,
            filename="backend_dev.pdf",
            raw_text="Fullstack Developer: Python, Django, PostgreSQL"
        )
        db_session.add(res)
        sk = Skill(name="Python", category="Language")
        db_session.add(sk)
        db_session.flush()

        db_session.add(ResumeSkill(resume_id=resume_id, skill_id=sk.id, extraction_confidence=0.9))
        c_proj = ResumeChunk(
            resume_id=resume_id,
            chunk_id="chk_py",
            section="Projects",
            project="ECommerce",
            chunk_text="ECommerce built with Python and Django.",
            embedding_model="BAAI/bge-small-en-v1.5"
        )
        db_session.add(c_proj)
        db_session.commit()

        # Create session via session_service
        sess = session_service.create_session(
            resume_id=resume_id,
            db=db_session,
            initial_difficulty=3
        )

        # Assert Question 1 was pre-generated and stored in prefetched_question
        prefetched = (sess.session_state or {}).get("prefetched_question")
        assert prefetched is not None, "Expected Question 1 to be pre-generated upon session creation!"
        assert prefetched.get("question") is not None
        assert prefetched.get("question_id") is not None

    def test_reproduce_slow_llm_aborts_second_attempt_when_budget_insufficient(self, db_session: Session):
        """
        TASK 5 Requirement:
        Mock LLM with 7s response -> second attempt aborted because remaining budget < timeout.
        Max 2 attempts, bounded by 12s total deadline.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(id=resume_id, filename="resume.pdf", raw_text="Python Engineer")
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        c_proj = ResumeChunk(
            resume_id=resume_id,
            chunk_id="c_py_1",
            section="Projects",
            project="ProjectA",
            chunk_text="ProjectA backend built with Python.",
            embedding_model="BAAI/bge-small-en-v1.5"
        )
        db_session.add_all([res, sess, c_proj])
        db_session.commit()

        req = QuestionGenerationRequest(
            category="Implementation",
            difficulty=3,
            target_project="ProjectA",
            target_entity="Python",
            session_id=session_id
        )

        llm_call_count = 0

        current_time = [0.0]

        def slow_failing_llm(*args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            # Simulate a 7.5-second LLM call
            current_time[0] += 7.5
            raise ValueError("Candidate rejected: Model returned malformed output.")

        def mock_perf_counter():
            current_time[0] += 0.01
            return current_time[0]

        with patch("app.services.rag.generator.question_generator.generate", side_effect=slow_failing_llm):
            with patch("time.perf_counter", side_effect=mock_perf_counter):
                resp = rag_pipeline.generate_grounded_question(
                    resume_id=resume_id,
                    req=req,
                    db=db_session
                )

        # Because attempt 1 took 7.5s, remaining time from 12s is 4.5s.
        # If minimum budget or single attempt timeout requires more (or deadline expired),
        # second attempt must be aborted, resulting in exactly 1 call!
        assert llm_call_count == 1, (
            f"Expected 2nd attempt to be aborted due to latency deadline, but LLM was called {llm_call_count} times!"
        )

    def test_reproduce_startup_warms_fastembed_and_spacy(self):
        """
        TASK 5 Requirement:
        Warm models at app startup (FastAPI lifespan): load FastEmbed and spaCy once during startup,
        remove any runtime model download from the request path.
        """
        from app.main import lifespan, app
        import asyncio

        # Run lifespan context manager to verify warmup executes without error
        async def run_lifespan():
            async with lifespan(app):
                pass

        asyncio.run(run_lifespan())
