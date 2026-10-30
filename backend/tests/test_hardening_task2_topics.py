import pytest
import re
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
from app.models.rag import ResumeClaim
from app.services.knowledge_state_service import (
    knowledge_state_service,
    JUNK_TOPIC_BLOCKLIST,
    is_assessable_topic
)
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


class TestTask2CleanTopicExtraction:
    """Tests for Task 2: Junk topic elimination, is_assessable flag, and assessable-only coverage."""

    def test_reproduce_junk_topics_in_knowledge_state(self, db_session: Session):
        """
        Reproduce failure: Currently, relationships like:
        Candidate -> ATTENDED -> SRM Institute of Science and Technology
        Candidate -> WON -> 1st Place at Hackathon
        Candidate -> SPEARS -> English
        cause 'Candidate', 'SRM Institute of Science and Technology', '1st Place at Hackathon',
        and 'English' to be added as assessable topics in knowledge_state.
        """
        resume_id = uuid4()
        session_id = uuid4()

        res = Resume(id=resume_id, filename="resume1.pdf", raw_text="Sample resume text")
        sess = InterviewSession(
            id=session_id,
            resume_id=resume_id,
            status="IN_PROGRESS",
            current_difficulty=3,
            session_state={}
        )
        db_session.add_all([res, sess])

        # Technical skills
        sk1 = Skill(name="Python", category="Programming Language")
        sk2 = Skill(name="Docker", category="DevOps")
        db_session.add_all([sk1, sk2])
        db_session.flush()

        rs1 = ResumeSkill(resume_id=resume_id, skill_id=sk1.id, extraction_confidence=0.95)
        rs2 = ResumeSkill(resume_id=resume_id, skill_id=sk2.id, extraction_confidence=0.90)
        db_session.add_all([rs1, rs2])

        # Relationships containing education, ranks, achievements, languages, and Candidate
        r1 = Relationship(
            resume_id=resume_id,
            source_entity="Candidate",
            relation_type="ATTENDED",
            target_entity="SRM Institute of Science and Technology",
            section="Education",
            evidence_text="B.Tech in CSE at SRM Institute of Science and Technology"
        )
        r2 = Relationship(
            resume_id=resume_id,
            source_entity="Candidate",
            relation_type="WON",
            target_entity="1st Place at National Hackathon",
            section="Achievements",
            evidence_text="Won 1st Place at National Hackathon"
        )
        r3 = Relationship(
            resume_id=resume_id,
            source_entity="Candidate",
            relation_type="ACHIEVED",
            target_entity="A Rank of 299 in AP EAPCET",
            section="Achievements",
            evidence_text="Secured a rank of 299 in AP EAPCET"
        )
        r4 = Relationship(
            resume_id=resume_id,
            source_entity="Candidate",
            relation_type="SPEAKS",
            target_entity="English",
            section="Languages",
            evidence_text="Fluent in English and Telugu"
        )
        r5 = Relationship(
            resume_id=resume_id,
            source_entity="CampusBridge",
            relation_type="BUILT_WITH",
            target_entity="Python",
            section="Projects",
            context_entity="CampusBridge",
            evidence_text="Built CampusBridge backend using Python"
        )
        db_session.add_all([r1, r2, r3, r4, r5])
        db_session.commit()

        # Initialize knowledge state
        k_state = knowledge_state_service.initialize_knowledge_state(sess, db_session)
        topics = k_state.get("topics", {})

        # Assert: assessable topics must ONLY contain canonical skills and projects
        assessable_topics = {
            name: data for name, data in topics.items()
            if data.get("is_assessable", False)
        }

        # Junk topics must NOT be assessable
        assert "Candidate" not in assessable_topics
        assert "SRM Institute of Science and Technology" not in assessable_topics
        assert "1st Place at National Hackathon" not in assessable_topics
        assert "A Rank of 299 in AP EAPCET" not in assessable_topics
        assert "English" not in assessable_topics

        # Technical skills and projects MUST be assessable
        assert "Python" in assessable_topics
        assert "Docker" in assessable_topics
        assert assessable_topics["Python"]["is_assessable"] is True

        # Coverage must be computed ONLY over assessable topics
        cov = k_state.get("coverage", {})
        assert cov["total_resume_topics"] == len(assessable_topics), (
            f"Coverage denominator ({cov['total_resume_topics']}) was not restricted to assessable topics ({len(assessable_topics)})"
        )

    def test_three_sample_resumes_zero_junk_topics(self, db_session: Session):
        """
        Test 3 sample resumes end-to-end to assert zero junk topics are marked assessable.
        Resume 1: Student (SRM Institute, EAPCET Rank, Candidate, Telugu)
        Resume 2: Experienced (IIT Madras, Hackathon Winner, College, Team Lead)
        Resume 3: Graduate (Stanford University, GPA 3.9, ACM ICPC, English)
        """
        sample_resumes = [
            {
                "skills": ["Python", "PostgreSQL", "FastAPI"],
                "projects": ["OmniCommerce"],
                "junk_entities": ["Candidate", "SRM Institute", "Rank 299 EAPCET", "Telugu", "College"]
            },
            {
                "skills": ["React", "TypeScript", "Node.js"],
                "projects": ["HealthTracker"],
                "junk_entities": ["IIT Madras", "Smart India Hackathon Winner", "University", "Bachelor of Technology"]
            },
            {
                "skills": ["Go", "Kubernetes", "Redis"],
                "projects": ["DistributedCache"],
                "junk_entities": ["Stanford University", "GPA 3.9", "1st Place ACM ICPC", "Candidate"]
            }
        ]

        for idx, s_data in enumerate(sample_resumes):
            res_id = uuid4()
            sess_id = uuid4()

            res = Resume(id=res_id, filename=f"sample_{idx}.pdf", raw_text=f"Sample text {idx}")
            sess = InterviewSession(id=sess_id, resume_id=res_id, status="IN_PROGRESS", current_difficulty=3, session_state={})
            db_session.add_all([res, sess])

            for s_name in s_data["skills"]:
                sk = Skill(name=s_name, category="Software")
                db_session.add(sk)
                db_session.flush()
                db_session.add(ResumeSkill(resume_id=res_id, skill_id=sk.id, extraction_confidence=0.9))

            # Add junk relationships
            for junk in s_data["junk_entities"]:
                db_session.add(Relationship(
                    resume_id=res_id,
                    source_entity="Candidate",
                    relation_type="ATTENDED" if "Institute" in junk or "University" in junk else "ACHIEVED",
                    target_entity=junk,
                    section="Education" if "Institute" in junk or "University" in junk else "Achievements",
                    evidence_text=f"Evidence for {junk}"
                ))

            # Add project relationship
            for proj in s_data["projects"]:
                db_session.add(Relationship(
                    resume_id=res_id,
                    source_entity="Candidate",
                    relation_type="DEVELOPED",
                    target_entity=proj,
                    section="Projects",
                    context_entity=proj,
                    evidence_text=f"Developed {proj}"
                ))

            db_session.commit()

            k_state = knowledge_state_service.initialize_knowledge_state(sess, db_session)
            topics = k_state.get("topics", {})

            assessable_topics = [
                name for name, data in topics.items()
                if data.get("is_assessable", False)
            ]

            # Assert ZERO junk topics in assessable list
            for junk in s_data["junk_entities"]:
                assert junk not in assessable_topics, f"Junk entity '{junk}' was marked assessable in Resume {idx+1}!"
                assert not any(junk.lower() in t.lower() for t in assessable_topics), (
                    f"Junk entity '{junk}' leaked into assessable topic in Resume {idx+1}: {assessable_topics}"
                )

            # Assert legitimate skills and projects ARE assessable
            for s_name in s_data["skills"]:
                assert s_name in assessable_topics, f"Skill '{s_name}' missing from assessable topics in Resume {idx+1}!"

            # Assert selection excludes non-assessable topics
            selection = adaptive_selection_service.determine_next_question_target(sess_id, db_session)
            assert selection.target_topic not in s_data["junk_entities"], (
                f"Selected target topic was a junk entity: {selection.target_topic}"
            )
