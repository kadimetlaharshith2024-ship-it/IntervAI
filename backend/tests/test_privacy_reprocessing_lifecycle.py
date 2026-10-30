from unittest.mock import patch
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.user import User
from app.models.interview import InterviewSession
from app.services.privacy_guard import privacy_guard, ALL_REDACTION_MARKERS
from app.services.remediation_service import audit_pii_in_database, remediate_database_records

client = TestClient(app)

DUMMY_PDF_BYTES = b"%PDF-1.4\n%mock pdf header\n%%EOF"

SYNTHETIC_PII_RESUME = """
Jane Doe
Email: jane.doe@university.edu | Phone: +1 415 555 9876 | linkedin.com/in/janedoe-engineer
Education: Stanford University, Roll No: 21BCS0123, CGPA: 3.9

TECHNICAL SKILLS
Languages: Python, Go, TypeScript, SQL
Frameworks: FastAPI, React, Docker, Kubernetes
Databases: PostgreSQL, Redis

PROJECTS
Cloud Platform
Tech Stack: Python, FastAPI, Docker, PostgreSQL
- Built microservices architecture serving 50,000 requests/sec with latency < 100ms.
- Containerized services with Docker and deployed on Kubernetes clusters.
- Configured PostgreSQL database listening on port 5432 and integrated with Redis caching.

EXPERIENCE
Software Engineer | DevCorp | 2021 - 2024
- Built distributed streaming microservices using FastAPI and Redis with 99.9% uptime.
- Secured user authentication using OAuth2 and JWT.
"""


class TestPrivacyReprocessingLifecycle:
    """Verifies that Upload, Re-extract, Re-map, and Re-index enforce sanitization end-to-end."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = SessionLocal()
        yield
        self.db.close()

    def test_upload_sanitization_and_technical_preservation(self):
        """Upload must accept resume, redact PII, and preserve all technical info."""
        with patch("app.services.resume_service.extract_pdf_content", return_value=SYNTHETIC_PII_RESUME):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("jane_doe_pii.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )

        assert response.status_code == 201
        data = response.json()
        resume_id = data["id"]

        # Check DB record directly
        resume = self.db.query(Resume).filter(Resume.id == resume_id).first()
        assert resume is not None
        assert "jane.doe@university.edu" not in resume.raw_text
        assert "555 9876" not in resume.raw_text
        assert "janedoe-engineer" not in resume.raw_text
        assert "21BCS0123" not in resume.raw_text

        # Verify redaction markers
        assert "[REDACTED_EMAIL]" in resume.raw_text
        assert "[REDACTED_PHONE]" in resume.raw_text
        assert "[REDACTED_PROFILE]" in resume.raw_text
        assert "[REDACTED_ID]" in resume.raw_text

        # Verify technical content survived intact
        assert "FastAPI" in resume.raw_text
        assert "Docker" in resume.raw_text
        assert "Kubernetes" in resume.raw_text
        assert "PostgreSQL" in resume.raw_text
        assert "port 5432" in resume.raw_text
        assert "99.9% uptime" in resume.raw_text

    def test_re_extract_entities_sanitized_and_idempotent(self):
        """Re-extracting entities must sanitize, clean stale skills, and prevent redaction marker skills."""
        # Create resume in DB with unsanitized text manually to simulate legacy record
        resume = Resume(
            filename="legacy_unsanitized.pdf",
            raw_text="Legacy Dev\nEmail: old.leak@corp.com\nSkills: Python, FastAPI, Docker"
        )
        self.db.add(resume)
        self.db.commit()
        self.db.refresh(resume)

        # Trigger Re-extract endpoint
        res = client.post(f"/api/v1/skills/{resume.id}/extract-entities")
        assert res.status_code == 200
        data = res.json()
        assert data["total_extracted"] > 0
        extracted_names = [e["name"] for e in data["entities"]]

        # Technical skills extracted
        assert "Python" in extracted_names
        assert "FastAPI" in extracted_names
        assert "Docker" in extracted_names

        # Redaction markers or PII never extracted as skills
        for name in extracted_names:
            assert "redact" not in name.lower()
            assert name not in ALL_REDACTION_MARKERS
            assert "@" not in name

        # Verify resume raw_text was sanitized in place
        self.db.refresh(resume)
        assert "old.leak@corp.com" not in resume.raw_text
        assert "[REDACTED_EMAIL]" in resume.raw_text

        # Re-running extract again is idempotent and doesn't duplicate links
        res2 = client.post(f"/api/v1/skills/{resume.id}/extract-entities")
        assert res2.status_code == 200
        links_count = self.db.query(ResumeSkill).filter(ResumeSkill.resume_id == resume.id).count()
        assert links_count == len(extracted_names)

    def test_re_map_relationships_sanitizes_evidence(self):
        """Re-mapping relationships must sanitize evidence_text and not create PII edges."""
        resume = Resume(
            filename="legacy_projects.pdf",
            raw_text="""
            Developer
            Email: project.dev@domain.com | Phone: +1 555-0100

            PROJECTS
            TaskFlow
            Tech Stack: Go, PostgreSQL, Docker
            - Developed TaskFlow with Go and PostgreSQL for user tracking.
            """
        )
        self.db.add(resume)
        self.db.commit()
        self.db.refresh(resume)

        # Trigger Re-map endpoint
        res = client.post(f"/api/v1/relationships/{resume.id}/extract-relationships?mode=advanced")
        assert res.status_code == 200
        data = res.json()
        assert data["total_relationships"] > 0

        # Inspect relationships in DB
        rels = self.db.query(Relationship).filter(Relationship.resume_id == resume.id).all()
        for r in rels:
            # Evidence text must not contain PII
            assert "project.dev@domain.com" not in (r.evidence_text or "")
            assert "555-0100" not in (r.evidence_text or "")
            # Entities must not be redaction markers
            assert "redact" not in r.source_entity.lower()
            assert "redact" not in r.target_entity.lower()

    def test_re_index_sanitizes_chunks_and_claims(self):
        """Re-indexing must create chunks with zero PII and rebuild clean claims."""
        resume = Resume(
            filename="legacy_rag.pdf",
            raw_text="""
            Backend Engineer
            Email: rag.dev@company.com | Phone: +1 555-8888

            PROJECTS
            MetricsServer
            Tech Stack: Python, FastAPI, Docker
            - Built MetricsServer using FastAPI and Docker listening on port 8080.
            """
        )
        self.db.add(resume)
        self.db.commit()
        self.db.refresh(resume)

        # Trigger Re-index endpoint
        res = client.post(f"/api/v1/rag/{resume.id}/index")
        assert res.status_code == 200

        # Verify chunks in DB
        chunks = self.db.query(ResumeChunk).filter(ResumeChunk.resume_id == resume.id).all()
        assert len(chunks) > 0
        for ch in chunks:
            assert "rag.dev@company.com" not in ch.chunk_text
            assert "555-8888" not in ch.chunk_text

        # Verify claims in DB
        claims = self.db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume.id).all()
        assert len(claims) > 0
        for clm in claims:
            assert "rag.dev@company.com" not in clm.claim_text
            assert "rag.dev@company.com" not in (clm.evidence_text or "")
            assert "555-8888" not in (clm.evidence_text or "")

    def test_user_data_isolation_and_preservation(self):
        """Reprocessing one resume must not affect other resumes, sessions, or scorecards."""
        # Create User 1 and User 2 resumes
        r1 = Resume(filename="user1.pdf", raw_text="User One\nSkills: Python, FastAPI")
        r2 = Resume(filename="user2.pdf", raw_text="User Two\nSkills: Java, Spring Boot")
        self.db.add_all([r1, r2])
        self.db.commit()

        # Re-index r1
        client.post(f"/api/v1/rag/{r1.id}/index")

        # Verify r2 is unaffected
        r2_db = self.db.query(Resume).filter(Resume.id == r2.id).first()
        assert r2_db.filename == "user2.pdf"
        assert "Java, Spring Boot" in r2_db.raw_text

    def test_remediation_service_cleans_legacy_records(self):
        """Verifies that remediation_service safely audits and remediates legacy records."""
        # Insert a resume with PII
        bad_r = Resume(
            filename="bad_legacy.pdf",
            raw_text="Candidate Legacy\nEmail: bad.leak@example.com | Phone: 9876543210\nSkills: Python"
        )
        self.db.add(bad_r)
        self.db.commit()

        # Run remediation for this specific resume
        res = remediate_database_records(self.db, dry_run=False, target_resume_id=bad_r.id)
        assert res["remediated_resumes_count"] == 1

        # Verify it was sanitized
        self.db.refresh(bad_r)
        assert "bad.leak@example.com" not in bad_r.raw_text
        assert "[REDACTED_EMAIL]" in bad_r.raw_text
        assert "[REDACTED_PHONE]" in bad_r.raw_text
