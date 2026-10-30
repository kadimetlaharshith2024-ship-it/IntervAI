import pytest
import re
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.interview import Base, InterviewQuestion, InterviewSession
from app.models.resume import Resume
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.relationship import Relationship
from app.services.privacy_guard import privacy_guard
from app.services.resume_service import segment_resume_sections
from app.services.rag.chunking_service import chunking_service


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


class TestTask1PIIHardening:
    """Tests for Task 1: Extended PII detection, header dropping, and data re-sanitization."""

    def test_reproduce_labelled_contact_and_handles_not_redacted(self):
        """
        Reproduce failure: Current privacy_guard fails to redact:
        1. 'T: 9876543210'
        2. 'E: user @ domain . com' (odd-spaced email)
        3. 'GitHub: mydevhandle'
        4. 'LinkedIn: myhandle'
        """
        line = "T: 9876543210 E: user @ domain . com GitHub: mydevhandle LinkedIn: myhandle"
        sanitized, meta = privacy_guard.sanitize_text(line)

        # Phone must be redacted
        assert "9876543210" not in sanitized, f"Phone was not redacted in: {sanitized}"
        # Email with odd spaces must be redacted
        assert "user @ domain . com" not in sanitized and "domain" not in sanitized, f"Odd-spaced email was not redacted in: {sanitized}"
        # GitHub and LinkedIn handles must be redacted
        assert "mydevhandle" not in sanitized, f"GitHub handle was not redacted in: {sanitized}"
        assert "myhandle" not in sanitized, f"LinkedIn handle was not redacted in: {sanitized}"

    def test_reproduce_extended_mobile_formats(self):
        """
        Tests 10-digit Indian mobiles with/without +91 and labels:
        'Mob: 9876543210', 'Ph: +91 9876543210', 'Mobile: +91-9876543210', 'T:9876543210'
        """
        samples = [
            "Mob: 9876543210",
            "Ph: +91 9876543210",
            "Mobile: +91-9876543210",
            "T:9876543210",
            "Call: 9876543210"
        ]
        for s in samples:
            sanitized, _ = privacy_guard.sanitize_text(s)
            assert "9876543210" not in sanitized, f"Failed to redact number in: {s} -> {sanitized}"

    def test_reproduce_header_block_dropped_from_chunking(self):
        """
        Reproduce failure: Current chunking_service generates chunks for the 'header' section,
        which contains candidate names, emails, phones, and addresses.
        The header/contact block must be dropped from chunk extraction entirely.
        """
        raw_resume = (
            "Alex Morgan\n"
            "T: 9876543210 E: alex.morgan@example.com GitHub: alexm LinkedIn: in/alexm\n"
            "Bangalore, Karnataka, India\n\n"
            "TECHNICAL SKILLS\n"
            "Languages: Python, Java, SQL\n\n"
            "PROJECTS\n"
            "OmniCommerce | Java, Spring Boot, PostgreSQL\n"
            "Built distributed checkout engine handling 5k TPS.\n"
        )
        chunks = chunking_service.chunk_resume("res_123", raw_resume)
        chunk_sections = [c.section.lower() for c in chunks]

        # Header section must be dropped entirely
        assert "header" not in chunk_sections, f"Header section was chunked: {chunk_sections}"
        assert not any("9876543210" in c.text for c in chunks), "Phone found in chunks!"
        assert not any("alex.morgan@example.com" in c.text for c in chunks), "Email found in chunks!"
        assert not any("alexm" in c.text for c in chunks), "Social handle found in chunks!"

    def test_database_resanitization_script(self, db_session: Session):
        """
        Test that existing rows in resume_chunks, resume_claims, relationships,
        and interview_questions.evidence_text containing unredacted PII are cleaned.
        """
        from scripts.resanitize_database import resanitize_existing_records

        resume_id = uuid4()
        session_id = uuid4()

        header_chunk = ResumeChunk(
            resume_id=resume_id,
            chunk_id="chk_test_1",
            section="Header",
            chunk_text="Contact: T: 9876543210 E: test@example.com GitHub: hacker123",
            embedding_model="text-embedding-3-small"
        )
        project_chunk = ResumeChunk(
            resume_id=resume_id,
            chunk_id="chk_test_2",
            section="Projects",
            chunk_text="Built with Python. Contact: T: 9876543210 E: test@example.com",
            embedding_model="text-embedding-3-small"
        )
        claim = ResumeClaim(
            resume_id=resume_id,
            claim_id="clm_test_1",
            section="Header",
            claim_text="Contacted candidate at 9876543210",
            evidence_text="Header: T: 9876543210 E: test@example.com"
        )
        rel = Relationship(
            resume_id=resume_id,
            source_entity="Candidate",
            relation_type="HAS_CONTACT",
            target_entity="9876543210",
            evidence_text="T: 9876543210 E: test@example.com"
        )
        q = InterviewQuestion(
            session_id=session_id,
            resume_id=resume_id,
            question_text="How would you design connection pooling?",
            evidence_text="In project OmniCommerce. Contact: test@example.com Ph: 9876543210",
            category="Architecture"
        )
        db_session.add_all([header_chunk, project_chunk, claim, rel, q])
        db_session.commit()

        # Run resanitization
        stats = resanitize_existing_records(db_session)

        # Assert header chunk was dropped
        assert db_session.query(ResumeChunk).filter_by(chunk_id="chk_test_1").first() is None

        # Assert project chunk is sanitized
        db_session.refresh(project_chunk)
        db_session.refresh(claim)
        db_session.refresh(rel)
        db_session.refresh(q)

        assert "9876543210" not in project_chunk.chunk_text
        assert "test@example.com" not in project_chunk.chunk_text
        assert "9876543210" not in claim.claim_text
        assert "test@example.com" not in claim.evidence_text
        assert "9876543210" not in rel.evidence_text
        assert "9876543210" not in q.evidence_text
        assert "test@example.com" not in q.evidence_text
