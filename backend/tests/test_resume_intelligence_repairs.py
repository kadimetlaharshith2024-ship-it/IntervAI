import pytest
import re
from uuid import uuid4

from app.core.database import SessionLocal
from app.models.resume import Resume
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.core.canonicalizer import (
    clean_project_title,
    is_tech_stack_or_tool_list,
    is_patent_or_metadata,
    canonicalize_entity,
    PROJECT_TITLE_BLACKLIST
)
from app.services.resume_service import detect_sections, segment_resume_sections
from app.services.relationship_service import (
    chunk_project_blocks,
    parse_certifications_section,
    parse_achievements_section,
    extract_and_store_relationships_advanced
)
from app.services.rag.claim_service import claim_service
from app.services.rag.rag_pipeline import rag_pipeline


# -------------------------------------------------------------------------
# Test Suite 1: Section Heading Recognition & Combined Headings
# -------------------------------------------------------------------------
class TestSectionIntelligence:
    def test_combined_heading_detection_achievements_and_certifications(self):
        raw_text = """
JOHN DOE
john.doe@example.com

EDUCATION
B.Tech Computer Science, 2024

PROJECTS
CloudPortal
Built cloud dashboard with React.

LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS
● Switchuation Hackathon (Sept 2024): Winner - Best Freshers Award; led a team of 4 to build a working prototype in 12 hours, ranking 1st among 32 teams.
● Hackathon Participant (SDG-focused MVP) - Celestia
● AWS Certified Solutions Architect - Associate (In Progress) | Mastering DSA in C++ (Udemy)
"""
        sections = detect_sections(raw_text)
        assert sections["projects"] is True
        assert sections["education"] is True
        assert sections["achievements"] is True
        assert sections["certifications"] is True

        segmented = segment_resume_sections(raw_text)
        assert "achievements" in segmented
        assert "certifications" in segmented
        assert "Switchuation Hackathon" in segmented["achievements"]
        assert "AWS Certified Solutions Architect" in segmented["certifications"]
        assert "LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS" not in segmented["projects"]

    def test_honors_awards_and_certifications_combined(self):
        raw_text = """
HONORS, AWARDS & CERTIFICATIONS
- 1st Place at National Hackathon 2025
- Google Cloud Certified Associate Cloud Engineer
"""
        sections = detect_sections(raw_text)
        assert sections["achievements"] is True
        assert sections["certifications"] is True


# -------------------------------------------------------------------------
# Test Suite 2: Anti-Hijacking, Anti-Tech-Stack & Anti-Patent Protection
# -------------------------------------------------------------------------
class TestProjectTitleExtractionAndValidation:
    def test_section_heading_rejection(self):
        assert clean_project_title("Leadership, Achievements & Certifications") is None
        assert clean_project_title("LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS") is None
        assert clean_project_title("Achievements & Certifications") is None
        assert clean_project_title("Honors & Awards") is None
        assert clean_project_title("Professional Experience") is None
        assert clean_project_title("Technical Skills") is None

    def test_comma_separated_tech_stack_rejection(self):
        assert clean_project_title("Python, FastAPI, Pydantic v2, Gemini, React.js") is None
        assert clean_project_title("Java, Spring Boot, PostgreSQL, Docker") is None
        assert clean_project_title("C++, Embedded C, ESP32-CAM, MPU-6050, IoT Edge Computing") is None
        assert clean_project_title("React, Redux, Tailwind CSS, TypeScript") is None

    def test_patent_number_and_metadata_rejection(self):
        assert clean_project_title("202641102290 (Published)") is None
        assert clean_project_title("Indian Patent Filed") is None
        assert clean_project_title("Patent Application 2026") is None
        assert clean_project_title("12 hours, ranking 1st among 32 teams") is None
        assert clean_project_title("99.98% Detection Accuracy") is None

    def test_action_verb_normalization(self):
        title = clean_project_title("Developed the Real-Time Conversational Telephony AI Voicebot")
        assert title == "Real-Time Conversational Telephony AI Voicebot"

        title2 = clean_project_title("Architected an Event-Triggered Road Surface Monitoring System")
        assert title2 == "Event-Triggered Road Surface Monitoring System"

    def test_valid_genuine_project_titles(self):
        assert clean_project_title("Telecom Geofencing & Fraud Compliance System") == "Telecom Geofencing & Fraud Compliance System"
        assert clean_project_title("CampusBridge") == "CampusBridge"
        assert clean_project_title("TalentBridge") == "TalentBridge"


# -------------------------------------------------------------------------
# Test Suite 3: Certifications & Courses Discrimination
# -------------------------------------------------------------------------
class TestCertificationsAndCourses:
    def test_in_progress_certification_not_marked_earned(self):
        text = "AWS Certified Solutions Architect - Associate (In Progress)"
        triplets = parse_certifications_section(text)
        assert len(triplets) == 1
        t = triplets[0]
        assert t.relation_type == "PURSUING_CERTIFICATION"
        assert "AWS Certified Solutions Architect" in t.target_entity
        assert t.relation_type != "EARNED_CERTIFICATION"

    def test_online_courses_not_marked_certifications(self):
        text = "Mastering DSA in C++ (Udemy) | Java Masterclass (Udemy)"
        triplets = parse_certifications_section(text)
        assert len(triplets) == 2
        for t in triplets:
            assert t.relation_type == "COMPLETED_COURSE"
            assert t.relation_type != "EARNED_CERTIFICATION"

    def test_earned_certification_marked_correctly(self):
        text = "Certified Kubernetes Administrator (CKA)"
        triplets = parse_certifications_section(text)
        assert len(triplets) == 1
        assert triplets[0].relation_type == "EARNED_CERTIFICATION"


# -------------------------------------------------------------------------
# Test Suite 4: Hackathons & Achievements Extraction
# -------------------------------------------------------------------------
class TestAchievementsAndHackathons:
    def test_hackathon_award_and_participation_extraction(self):
        text = "Switchuation Hackathon (Sept 2024): Winner - Best Freshers Award; led a team of 4 to build a working prototype in 12 hours, ranking 1st among 32 teams."
        triplets = parse_achievements_section(text)
        rel_types = [t.relation_type for t in triplets]
        assert "WON" in rel_types
        assert "PARTICIPATED_IN" in rel_types

        won_triplet = next(t for t in triplets if t.relation_type == "WON")
        assert "Best Freshers Award" in won_triplet.target_entity
        assert won_triplet.context_entity == "Switchuation Hackathon"

    def test_hackathon_participation_extraction(self):
        text = "Hackathon Participant (SDG-focused MVP) - Celestia"
        triplets = parse_achievements_section(text)
        assert len(triplets) >= 1
        part_triplet = next(t for t in triplets if t.relation_type == "PARTICIPATED_IN")
        assert "Celestia" in part_triplet.target_entity


# -------------------------------------------------------------------------
# Test Suite 5: Anti-Skill Bleed & Project Header Tech Extraction
# -------------------------------------------------------------------------
class TestProjectRelationshipGrounding:
    def test_project_header_technologies_extracted_and_associated(self):
        db = SessionLocal()
        resume_id = uuid4()
        raw_text = """
JOHN DOE
john@example.com

TECHNICAL SKILLS
Languages: Java, C++, Python, Ruby, Rust, Go, Scala
Frameworks: Spring Boot, Angular, Django, Flutter

PROJECTS
Telecom Geofencing & Fraud Compliance System | FastAPI, React.js, QGIS, Shapely, R-Tree Indexing, GeoJSON, Python
● Built high-throughput geospatial backend cross-checking device GPS coordinates.
● Replaced legacy scan with R-Tree spatial index over GeoJSON MultiPolygons in FastAPI.
"""
        resume = Resume(
            id=resume_id,
            filename="Test_Grounding.pdf",
            raw_text=raw_text
        )
        db.add(resume)
        db.commit()

        try:
            triplets = extract_and_store_relationships_advanced(resume_id, db)
            proj_rels = [t for t in triplets if t.context_entity == "Telecom Geofencing & Fraud Compliance System"]
            target_entities = {t.target_entity for t in proj_rels}

            # Proved project technologies are present
            assert "FastAPI" in target_entities
            assert "QGIS" in target_entities
            assert "Shapely" in target_entities
            assert "R-Tree Indexing" in target_entities
            assert "GeoJSON" in target_entities

            # Provenance: Global skills that were NOT mentioned in the project MUST NOT bleed into the project
            assert "Ruby" not in target_entities
            assert "Rust" not in target_entities
            assert "Scala" not in target_entities
            assert "Angular" not in target_entities
            assert "Spring Boot" not in target_entities
            assert "Flutter" not in target_entities

            # Verify no tool-to-tool BUILT_WITH relationship between R-Tree and FastAPI
            r_tree_fastapi = [t for t in triplets if t.source_entity == "R-Tree Indexing" and t.target_entity == "FastAPI" and t.relation_type == "BUILT_WITH"]
            assert len(r_tree_fastapi) == 0

        finally:
            db.query(Relationship).filter(Relationship.resume_id == resume_id).delete()
            db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).delete()
            db.query(Resume).filter(Resume.id == resume_id).delete()
            db.commit()
            db.close()

    def test_reprocessing_idempotency(self):
        db = SessionLocal()
        resume_id = uuid4()
        raw_text = """
ALEX CARTER
alex@example.com

PROJECTS
TaskEngine | Python, Redis
● Built asynchronous task runner with Redis caching.
"""
        resume = Resume(id=resume_id, filename="Idempotency_Test.pdf", raw_text=raw_text)
        db.add(resume)
        db.commit()

        try:
            # First pass
            t1 = extract_and_store_relationships_advanced(resume_id, db)
            c1 = claim_service.extract_and_store_claims(resume_id, db)

            count_rel_1 = db.query(Relationship).filter(Relationship.resume_id == resume_id).count()
            count_clm_1 = db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).count()

            # Second pass (reprocessing)
            t2 = extract_and_store_relationships_advanced(resume_id, db)
            c2 = claim_service.extract_and_store_claims(resume_id, db)

            count_rel_2 = db.query(Relationship).filter(Relationship.resume_id == resume_id).count()
            count_clm_2 = db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).count()

            assert count_rel_1 == count_rel_2, "Relationships accumulated duplicates upon reprocessing"
            assert count_clm_1 == count_clm_2, "Claims accumulated duplicates upon reprocessing"

        finally:
            db.query(Relationship).filter(Relationship.resume_id == resume_id).delete()
            db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).delete()
            db.query(Resume).filter(Resume.id == resume_id).delete()
            db.commit()
            db.close()
