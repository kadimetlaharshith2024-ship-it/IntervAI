import pytest
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.interview import Base, InterviewSession
from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeChunk, ResumeClaim
from app.schemas.rag import QuestionOutputSchema
from app.services.knowledge_state_service import knowledge_state_service
from app.services.adaptive_selection_service import adaptive_selection_service
from app.services.rag.validator import grounding_validator


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


class TestTask3EvidenceGate:
    """Tests for Task 3: Evidence gate before LLM call, skills-list-only conceptual handling, and entity grounding."""

    def test_reproduce_skill_without_project_evidence_not_tied_to_project(self, db_session: Session):
        """
        Reproduce failure:
        A resume includes C++ in the skills list, but only Python and Docker are used in project OmniCommerce.
        Target selection must:
        1. Select project-grounded topics (Python, Docker, OmniCommerce) before C++.
        2. When C++ is selected, target_project must be None, category must be Fundamentals (conceptual).
        3. C++ must NEVER be tied to OmniCommerce.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(
            id=resume_id,
            filename="test_resume.pdf",
            raw_text="SKILLS: Python, Docker, C++\nPROJECTS:\nOmniCommerce: Built high-throughput backend in Python and Docker."
        )
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        db_session.add_all([res, sess])

        # Skills
        sk_py = Skill(name="Python", category="Programming Language")
        sk_doc = Skill(name="Docker", category="DevOps")
        sk_cpp = Skill(name="C++", category="Programming Language")
        db_session.add_all([sk_py, sk_doc, sk_cpp])
        db_session.flush()

        db_session.add_all([
            ResumeSkill(resume_id=resume_id, skill_id=sk_py.id, extraction_confidence=0.95),
            ResumeSkill(resume_id=resume_id, skill_id=sk_doc.id, extraction_confidence=0.95),
            ResumeSkill(resume_id=resume_id, skill_id=sk_cpp.id, extraction_confidence=0.95),
        ])

        # Only Python and Docker appear in project relationships
        r1 = Relationship(
            resume_id=resume_id,
            source_entity="OmniCommerce",
            relation_type="BUILT_WITH",
            target_entity="Python",
            section="Projects",
            context_entity="OmniCommerce",
            evidence_text="OmniCommerce built with Python"
        )
        r2 = Relationship(
            resume_id=resume_id,
            source_entity="OmniCommerce",
            relation_type="CONTAINERIZED_WITH",
            target_entity="Docker",
            section="Projects",
            context_entity="OmniCommerce",
            evidence_text="Containerized OmniCommerce using Docker"
        )
        # Chunks: Only Python and Docker are in Projects chunk. C++ is only in Skills chunk.
        c_proj = ResumeChunk(
            resume_id=resume_id,
            chunk_id="chk_proj_1",
            section="Projects",
            project="OmniCommerce",
            chunk_text="OmniCommerce: Distributed backend built with Python and containerized with Docker.",
            embedding_model="text-embedding-3-small"
        )
        c_skills = ResumeChunk(
            resume_id=resume_id,
            chunk_id="chk_skills_1",
            section="Skills",
            chunk_text="Languages: Python, Docker, C++",
            embedding_model="text-embedding-3-small"
        )
        db_session.add_all([r1, r2, c_proj, c_skills])
        db_session.commit()

        # Initialize knowledge state
        knowledge_state_service.initialize_knowledge_state(sess, db_session)

        # 1. First selection must pick a project-grounded topic, NOT C++
        sel1 = adaptive_selection_service.determine_next_question_target(session_id, db_session)
        assert sel1.target_topic in ["Python", "Docker", "OmniCommerce"], (
            f"Expected project-backed topic first, but got: {sel1.target_topic}"
        )

        # 2. If C++ is evaluated (e.g. when project topics are excluded or exhausted),
        # it must be conceptual and NOT tied to OmniCommerce
        sel_cpp = adaptive_selection_service.determine_next_question_target(
            session_id,
            db_session,
            excluded_topics={"Python", "Docker", "OmniCommerce"}
        )
        assert sel_cpp.target_topic == "C++" or sel_cpp.target_entity == "C++"
        assert sel_cpp.target_project is None, (
            f"Skills-list-only topic C++ was incorrectly tied to project: {sel_cpp.target_project}"
        )
        assert sel_cpp.category in ["Fundamentals", "Concepts"], (
            f"Skills-list-only topic C++ should have conceptual category, got: {sel_cpp.category}"
        )

    def test_reproduce_validator_rejects_target_entity_absent_from_cited_evidence(self):
        """
        Assert that the grounding validator strictly rejects any question where
        the declared target_entity is absent from the cited evidence blocks.
        """
        question = QuestionOutputSchema(
            question="In OmniCommerce, how did you handle memory allocation with C++?",
            category="Implementation",
            difficulty=3,
            target_project="OmniCommerce",
            target_entity="C++",
            evidence_ids=["ev_1"],
            evidence_text="OmniCommerce backend was implemented in Python using asyncio.",
            rationale="Tests C++ memory mechanics",
            difficulty_rationale="Tests C++ memory mechanics",
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="tr_1"
        )
        # Evidence only mentions Python, not C++
        resolved_evidence = {
            "ev_1": {
                "text": "OmniCommerce backend was implemented in Python using asyncio.",
                "project": "OmniCommerce",
                "source_type": "chunk"
            }
        }
        allowed_projects = ["OmniCommerce"]

        is_valid, reason, details = grounding_validator.validate(
            question=question,
            allowed_projects=allowed_projects,
            allowed_entities=["Python"],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )

        assert not is_valid, "Validator should have rejected question when target_entity C++ is absent from evidence!"
        assert details.get("code") == "entity_not_in_cited_evidence"

    def test_reproduce_validator_rejects_conceptual_question_referencing_project(self):
        """
        Assert that when target_project is None (conceptual question on a skills-list topic),
        the question text must NOT reference any project the skill does not appear in.
        """
        question = QuestionOutputSchema(
            question="In OmniCommerce, what are the core principles of C++ memory management?",
            category="Fundamentals",
            difficulty=3,
            target_project=None,
            target_entity="C++",
            evidence_ids=["ev_skills"],
            evidence_text="Languages: Python, Docker, C++",
            rationale="Tests C++ memory mechanics",
            difficulty_rationale="Tests C++ memory mechanics",
            grounding_score=0.9,
            relevance_score=0.9,
            technical_quality_score=0.9,
            trace_id="tr_2"
        )
        resolved_evidence = {
            "ev_skills": {
                "text": "Languages: Python, Docker, C++",
                "project": None,
                "source_type": "chunk"
            }
        }
        allowed_projects = ["OmniCommerce"]

        is_valid, reason, details = grounding_validator.validate(
            question=question,
            allowed_projects=allowed_projects,
            allowed_entities=["C++"],
            resolved_evidence_items=resolved_evidence,
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )

        assert not is_valid, "Validator should have rejected conceptual question that references OmniCommerce!"
        assert details.get("code") == "ungrounded_project_reference"
