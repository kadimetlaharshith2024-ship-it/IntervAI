from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.privacy_guard import (
    PrivacyScreeningService,
    PrivacyScreeningConfig,
    PrivacyCategory,
    privacy_guard
)

client = TestClient(app)

DUMMY_PDF_BYTES = b"%PDF-1.4\n%mock pdf header\n%%EOF"


class TestPrivacyGateUnit:
    """Unit tests for multi-category privacy screening and false-positive prevention."""

    def test_personal_email_detection(self):
        text = "Candidate Name\nEmail: alex.developer@gmail.com\nSoftware Engineer"
        res = privacy_guard.screen_text(text)
        assert not res.is_clean
        assert PrivacyCategory.EMAIL.value in res.violation_categories

    def test_institutional_email_detection(self):
        text = "Graduate Student\nContact: student.researcher@stanford.edu\nComputer Science"
        res = privacy_guard.screen_text(text)
        assert not res.is_clean
        assert PrivacyCategory.EMAIL.value in res.violation_categories

        text_in = "VIT Vellore Student\nEmail: roll20bce@vit.ac.in\nB.Tech CSE"
        res_in = privacy_guard.screen_text(text_in)
        assert not res_in.is_clean
        assert PrivacyCategory.EMAIL.value in res_in.violation_categories

    def test_obfuscated_email_detection(self):
        text = "Reach me at alex.dev [at] domain [dot] com for queries"
        res = privacy_guard.screen_text(text)
        assert not res.is_clean
        assert PrivacyCategory.EMAIL.value in res.violation_categories

    def test_linkedin_url_detection(self):
        text = "Portfolio: https://www.linkedin.com/in/alex-dev-senior/ | Github: github.com/alex"
        res = privacy_guard.screen_text(text)
        assert not res.is_clean
        assert PrivacyCategory.LINKEDIN.value in res.violation_categories

        text_short = "Profile: linkedin.com/in/alexdev"
        res_short = privacy_guard.screen_text(text_short)
        assert not res_short.is_clean
        assert PrivacyCategory.LINKEDIN.value in res_short.violation_categories

    def test_phone_number_detection_various_formats(self):
        # US Domestic
        res_us = privacy_guard.screen_text("Contact: (415) 555-2671\nLocation: San Francisco")
        assert not res_us.is_clean
        assert PrivacyCategory.PHONE.value in res_us.violation_categories

        # International with +
        res_intl = privacy_guard.screen_text("Alex Developer | +91 98765 43210 | Bangalore")
        assert not res_intl.is_clean
        assert PrivacyCategory.PHONE.value in res_intl.violation_categories

        # Labeled phone
        res_lbl = privacy_guard.screen_text("Tel: +44 20 7946 0919\nLondon, UK")
        assert not res_lbl.is_clean
        assert PrivacyCategory.PHONE.value in res_lbl.violation_categories

    def test_campus_and_student_id_detection(self):
        # Context-labeled Roll Number
        res_roll = privacy_guard.screen_text("Education: VIT University\nRoll No: 20BCE1042\nCGPA: 8.9")
        assert not res_roll.is_clean
        assert PrivacyCategory.CAMPUS_ID.value in res_roll.violation_categories

        # Student ID labeled
        res_sid = privacy_guard.screen_text("Student ID: 88472910\nStanford University")
        assert not res_sid.is_clean
        assert PrivacyCategory.CAMPUS_ID.value in res_sid.violation_categories

        # Structured university ID format (20BCE1042)
        res_fmt = privacy_guard.screen_text("VIT Vellore - 20BCE1042 - B.Tech CSE (2020 - 2024)")
        assert not res_fmt.is_clean
        assert PrivacyCategory.CAMPUS_ID.value in res_fmt.violation_categories

    def test_safe_technical_resume_no_false_positives(self):
        """
        Critical test: Verifies that technical terms, ports, versions, dates,
        percentages, and metrics are NOT falsely flagged as PII.
        """
        safe_resume_text = """
        Alex Rivera
        Senior Backend Engineer

        SUMMARY
        Backend specialist with 5 years experience designing distributed microservices with 99.9% uptime.

        SKILLS
        Languages: Python 3.12, Go 1.21, TypeScript, SQL
        Frameworks: FastAPI, Django 4.2, React 19, Node.js 20
        Databases: PostgreSQL 16, Redis 7, MongoDB 6.0
        Infrastructure: Docker, Kubernetes, AWS EC2, AWS S3, OAuth2, SHA256

        EXPERIENCE
        Distributed Systems Engineer | TechCorp | 2021 - 2024
        - Architected event-driven streaming pipeline handling 100,000 requests/sec with latency < 50ms.
        - Deployed microservices listening on port 8080 and 5432 in containerized VPC.
        - Resolved HTTP 500 error propagation and reduced 404 occurrences by 40%.
        - Designed zero-downtime database migration strategy on AWS EC2 clusters.

        EDUCATION
        Bachelor of Science in Computer Science | 2017 - 2021
        State University | GPA 3.8 / 4.0
        """
        res = privacy_guard.screen_text(safe_resume_text)
        assert res.is_clean, f"False positives detected: {[v.description for v in res.violations]}"
        assert len(res.violations) == 0

    def test_pii_not_echoed_in_descriptions(self):
        text = "Contact: sensitive_user_secret@private-domain.org, Phone: +1 555-0199"
        res = privacy_guard.screen_text(text)
        for v in res.violations:
            assert "sensitive_user_secret" not in v.description
            assert "+1 555-0199" not in v.description

    def test_sanitize_email(self):
        text = "Name: John Doe\nEmail: alex.developer@gmail.com\nSkills: Python, FastAPI"
        sanitized, meta = privacy_guard.sanitize_text(text)
        assert "[REDACTED_EMAIL]" in sanitized
        assert "alex.developer@gmail.com" not in sanitized
        assert "Python, FastAPI" in sanitized
        assert meta["is_sanitized"] is True
        assert "EMAIL" in meta["categories_redacted"]

    def test_sanitize_phone(self):
        text = "Mobile: +1 (415) 555-2671\nLocation: San Francisco\nSkills: Docker"
        sanitized, meta = privacy_guard.sanitize_text(text)
        assert "[REDACTED_PHONE]" in sanitized
        assert "555-2671" not in sanitized
        assert "Skills: Docker" in sanitized
        assert meta["is_sanitized"] is True
        assert "PHONE" in meta["categories_redacted"]

    def test_sanitize_linkedin(self):
        text = "Profile: https://www.linkedin.com/in/alex-dev-senior/ | Projects: API"
        sanitized, meta = privacy_guard.sanitize_text(text)
        assert "[REDACTED_PROFILE]" in sanitized
        assert "alex-dev-senior" not in sanitized
        assert "Projects: API" in sanitized
        assert meta["is_sanitized"] is True
        assert "LINKEDIN" in meta["categories_redacted"]

    def test_sanitize_campus_id(self):
        text = "Education: State University\nRoll No: 20BCE1042\nCGPA: 8.9"
        sanitized, meta = privacy_guard.sanitize_text(text)
        assert "[REDACTED_ID]" in sanitized
        assert "20BCE1042" not in sanitized
        assert "CGPA: 8.9" in sanitized
        assert meta["is_sanitized"] is True
        assert "CAMPUS_ID" in meta["categories_redacted"]

    def test_sanitize_multiple_pii_categories_and_rescreen(self):
        text = """
        John Doe
        Email: john.doe@example.com | Phone: +91 98765 43210 | linkedin.com/in/johndoe
        Education: VIT Vellore, Roll No: 20BCE1042, CGPA: 9.1
        Skills: Python 3.12, FastAPI, PostgreSQL listening on port 5432
        Experience: Designed microservices with 99.9% uptime and SHA256 checksums.
        """
        sanitized, meta = privacy_guard.sanitize_text(text)
        assert "john.doe@example.com" not in sanitized
        assert "98765 43210" not in sanitized
        assert "johndoe" not in sanitized
        assert "20BCE1042" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "[REDACTED_PROFILE]" in sanitized
        assert "[REDACTED_ID]" in sanitized

        # Technical context completely preserved
        assert "Python 3.12" in sanitized
        assert "FastAPI" in sanitized
        assert "PostgreSQL" in sanitized
        assert "port 5432" in sanitized
        assert "99.9% uptime" in sanitized
        assert "SHA256" in sanitized

        # Re-screening sanitized text returns is_clean=True with 0 violations
        res = privacy_guard.screen_text(sanitized)
        assert res.is_clean is True
        assert len(res.violations) == 0


class TestPrivacyGateAPIIntegration:
    """Integration tests verifying HTTP 201 acceptance, automatic sanitization, and DB safety."""

    def test_upload_accepted_and_sanitizes_email(self):
        leaky_text = "Candidate With Email\nEmail: dev.leak@gmail.com\nPython, FastAPI Developer"
        with patch("app.services.resume_service.extract_pdf_content", return_value=leaky_text):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("email_resume.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        # Raw text preview in response must contain the redaction marker and NOT the raw email
        assert "[REDACTED_EMAIL]" in data["raw_text_preview"]
        assert "dev.leak@gmail.com" not in data["raw_text_preview"]

    def test_upload_accepted_and_sanitizes_phone_number(self):
        phone_text = "Candidate With Phone\nMobile: +91 99887 76655\nSkills: Java, Spring Boot"
        with patch("app.services.resume_service.extract_pdf_content", return_value=phone_text):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("phone_resume.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )
        assert response.status_code == 201
        data = response.json()
        assert "[REDACTED_PHONE]" in data["raw_text_preview"]
        assert "99887" not in data["raw_text_preview"]
        assert "Java, Spring Boot" in data["raw_text_preview"]

    def test_upload_accepted_and_sanitizes_linkedin(self):
        linkedin_text = "Senior Engineer\nProfile: linkedin.com/in/alex-dev-senior\nSkills: Go, Kubernetes"
        with patch("app.services.resume_service.extract_pdf_content", return_value=linkedin_text):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("linkedin_resume.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )
        assert response.status_code == 201
        data = response.json()
        assert "[REDACTED_PROFILE]" in data["raw_text_preview"]
        assert "alex-dev-senior" not in data["raw_text_preview"]
        assert "Go, Kubernetes" in data["raw_text_preview"]

    def test_upload_accepted_and_sanitizes_campus_roll_number(self):
        roll_text = "Student Candidate\nEducation: VIT University\nRoll No: 20BCE1042\nProjects: Task Manager"
        with patch("app.services.resume_service.extract_pdf_content", return_value=roll_text):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("student_resume.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )
        assert response.status_code == 201
        data = response.json()
        assert "[REDACTED_ID]" in data["raw_text_preview"]
        assert "20BCE1042" not in data["raw_text_preview"]
        assert "VIT University" in data["raw_text_preview"]

    def test_upload_accepted_with_multiple_pii_categories(self):
        full_text = """
        Alex Developer
        Email: alex.developer@example.com | Phone: +1 415 555 2671 | linkedin.com/in/alexdeveloper
        Education: State University, Roll No: 20BCE1042, CGPA: 9.1

        SKILLS
        Python, FastAPI, Docker, PostgreSQL

        EXPERIENCE
        Backend Engineer | TechCorp | 2022 - 2024
        - Built API platform on port 8080 with 99.9% uptime.
        """
        with patch("app.services.resume_service.extract_pdf_content", return_value=full_text):
            response = client.post(
                "/api/v1/resumes/upload",
                files={"file": ("full_pii_resume.pdf", DUMMY_PDF_BYTES, "application/pdf")}
            )
        assert response.status_code == 201
        data = response.json()
        # Ensure all PII markers are present and all raw PII is absent
        assert "[REDACTED_EMAIL]" in data["raw_text_preview"]
        assert "[REDACTED_PHONE]" in data["raw_text_preview"]
        assert "alex.developer@example.com" not in data["raw_text_preview"]
        assert "555 2671" not in data["raw_text_preview"]

    def test_upload_rejects_invalid_file_type(self):
        response = client.post(
            "/api/v1/resumes/upload",
            files={"file": ("invalid_doc.docx", b"dummy docx bytes", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]

    def test_upload_rejects_empty_file(self):
        response = client.post(
            "/api/v1/resumes/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
