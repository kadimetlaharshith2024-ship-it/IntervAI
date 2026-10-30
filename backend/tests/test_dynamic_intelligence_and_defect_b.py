import uuid
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.resume import Resume
from app.models.relationship import Relationship
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.interview import InterviewSession, InterviewQuestion
from app.schemas.rag import QuestionGenerationRequest, QuestionOutputSchema, QuestionGenerationResponse
from app.services.rag.rag_pipeline import RAGPipeline
from app.services.rag.validator import grounding_validator
from app.services.relationship_service import (
    parse_linguistic_sentence,
    parse_tech_stack_line,
    deduplicate_and_resolve_conflicts,
    RelationshipTriplet
)
from app.core.canonicalizer import ALIAS_INDEX, canonicalize_entity


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TestDynamicUnseenTechnologyExtraction:
    """Tests Phase 3 & 4: Open entity vocabulary with closed-world relationships."""

    def test_unseen_technology_in_engineering_sentence(self):
        """Verifies that technologies absent from ALIAS_INDEX (DuckDB, NATS) are extracted and typed."""
        sentence = "Engineered real-time analytics pipeline using DuckDB and integrated with NATS messaging."
        known_skills = dict(ALIAS_INDEX)

        # Confirm DuckDB and NATS are unseen in initial taxonomy
        assert "duckdb" not in known_skills
        assert "nats" not in known_skills

        triplets = parse_linguistic_sentence(sentence, context_name="AnalyticsPlatform", section="Projects", known_skills=known_skills)

        targets = {t.target_entity for t in triplets}
        assert "Duckdb" in targets or "DuckDB" in targets or any("duckdb" in t.lower() for t in targets)
        assert any("nats" in t.lower() for t in targets)

        # Check relationship types from closed-world ontology
        rel_types = {t.relation_type for t in triplets}
        assert "BUILT_WITH" in rel_types or "INTEGRATES_WITH" in rel_types

        # Verify evidence provenance is preserved
        for t in triplets:
            assert sentence in t.evidence_text
            assert t.section == "Projects"

    def test_unseen_technology_in_tech_stack_enumeration(self):
        """Verifies that unseen technologies in tech stack lines are extracted dynamically."""
        line = "Core Stack: Svelte, TailwindCSS, DuckDB, ChromaDB"
        triplets = parse_tech_stack_line(line, context_name="InteractiveDashboard", section="Projects")

        targets = {t.target_entity for t in triplets}
        assert any("duckdb" in t.lower() for t in targets)
        assert any("chromadb" in t.lower() for t in targets)
        assert any("svelte" in t.lower() for t in targets)

        for t in triplets:
            assert t.relation_type == "HAS_TECHNOLOGY"
            assert t.context_entity == "InteractiveDashboard"

    def test_evidence_validation_rejects_hallucinated_triplet(self):
        """Verifies that triplets with no textual backing fail evidence validation."""
        fake_triplet = RelationshipTriplet(
            source_entity="Candidate",
            relation_type="BUILT_WITH",
            target_entity="Rust",
            section="Projects",
            context_entity="MyProject",
            evidence_text="Built project with Python and Django.",
            confidence=0.9,
            extraction_method="dependency_parsed"
        )
        resolved = deduplicate_and_resolve_conflicts([fake_triplet])
        assert len(resolved) == 0


class TestDefectBTargetEvidenceAlignment:
    """
    Tests Phase 0 Defect B: Generic Target-to-Evidence Retrieval Alignment.
    Verifies that when a target entity (CSS, Architecture, DuckDB, Docker) is requested,
    retrieval guarantees that evidence containing the target entity is included in the context,
    preventing 'target entity absent from cited evidence blocks' rejections.
    """

    def test_target_entity_css_alignment_and_grounding(self, db_session: Session):
        """Regression test for Defect B with CSS."""
        pipeline = RAGPipeline()
        resume_id = uuid.uuid4()

        resume = Resume(
            id=resume_id,
            filename="frontend_engineer_resume.pdf",
            raw_text="Frontend Developer.\n\nPROJECTS\nPortfolio Website\nBuilt responsive accessible UI using HTML5, CSS3, and JavaScript.\nImplemented CSS Grid and Flexbox layouts for mobile compatibility.\n\nEXPERIENCE\nWeb Developer | Digital Agency\nMaintained client style sheets and optimized CSS bundle size.",
            user_id=None
        )
        db_session.add(resume)
        db_session.flush()

        # Index resume
        idx_res = pipeline.index_resume(resume_id, db_session)
        assert idx_res["status"] == "INDEXED"

        # Request grounded question for target entity CSS
        req = QuestionGenerationRequest(
            category="Frontend Architecture",
            difficulty=3,
            target_project="Portfolio Website",
            target_entity="CSS"
        )

        def mock_generate_css(ctx, **kwargs):
            ev_items = ctx.get("selected_evidence_items", [])
            eid = ev_items[0]["evidence_id"] if ev_items else "chk_0"
            ev_text = ev_items[0]["text"] if ev_items else "Built responsive accessible UI using HTML5, CSS3, and JavaScript."
            return QuestionOutputSchema(
                question="In your Portfolio Website, how did you approach structuring your CSS Grid and Flexbox layouts for cross-device compatibility?",
                category="Frontend Architecture",
                difficulty=3,
                target_project="Portfolio Website",
                target_entity="CSS",
                evidence_ids=[eid],
                evidence_text=ev_text,
                rationale="Grounded in Portfolio Website CSS implementation.",
                grounding_score=1.0,
                relevance_score=1.0,
                technical_quality_score=1.0,
                trace_id="trc_test"
            )

        with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_generate_css):
            resp: QuestionGenerationResponse = pipeline.generate_grounded_question(resume_id, req, db_session)

        assert resp.validation_status == "ACCEPTED"
        assert resp.question is not None
        assert resp.question.target_entity == "CSS"
        assert "css" in resp.question.question.lower()

    def test_target_entity_unseen_technology_alignment(self, db_session: Session):
        """Verifies that an unseen technology (DuckDB) is retrieved, grounded, and validated."""
        pipeline = RAGPipeline()
        resume_id = uuid.uuid4()

        resume = Resume(
            id=resume_id,
            filename="data_engineer_resume.pdf",
            raw_text="Data Engineer.\n\nPROJECTS\nAnalytics Core\nBuilt columnar aggregation engine using DuckDB for local analytical queries.\nIntegrated DuckDB with Apache Arrow for zero-copy memory transfers.",
            user_id=None
        )
        db_session.add(resume)
        db_session.flush()

        pipeline.index_resume(resume_id, db_session)

        req = QuestionGenerationRequest(
            category="Data Engineering",
            difficulty=4,
            target_project="Analytics Core",
            target_entity="DuckDB"
        )

        def mock_generate_duckdb(ctx, **kwargs):
            ev_items = ctx.get("selected_evidence_items", [])
            eid = ev_items[0]["evidence_id"] if ev_items else "chk_0"
            ev_text = ev_items[0]["text"] if ev_items else "Built columnar aggregation engine using DuckDB for local analytical queries."
            return QuestionOutputSchema(
                question="In Analytics Core, what architectural trade-offs did you evaluate when integrating DuckDB with Apache Arrow for analytical queries?",
                category="Data Engineering",
                difficulty=4,
                target_project="Analytics Core",
                target_entity="DuckDB",
                evidence_ids=[eid],
                evidence_text=ev_text,
                rationale="Grounded in Analytics Core DuckDB usage.",
                grounding_score=1.0,
                relevance_score=1.0,
                technical_quality_score=1.0,
                trace_id="trc_test"
            )

        with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_generate_duckdb):
            resp: QuestionGenerationResponse = pipeline.generate_grounded_question(resume_id, req, db_session)

        assert resp.validation_status == "ACCEPTED"
        assert resp.question is not None
        assert resp.question.target_entity == "DuckDB"

    def test_unsupported_target_entity_graceful_rotation(self, db_session: Session):
        """
        Verifies that if a target entity is requested that is NOT supported by the candidate's resume,
        the system rotates to an eligible supported target rather than repeating identical failing attempts.
        """
        pipeline = RAGPipeline()
        resume_id = uuid.uuid4()

        resume = Resume(
            id=resume_id,
            filename="python_backend_resume.pdf",
            raw_text="Backend Engineer.\n\nPROJECTS\nAPI Platform\nDeveloped asynchronous REST APIs using FastAPI and PostgreSQL.\nDeployed services on Docker containers.",
            user_id=None
        )
        db_session.add(resume)
        db_session.flush()

        pipeline.index_resume(resume_id, db_session)

        # Request an entity completely absent from the resume: Kubernetes
        req = QuestionGenerationRequest(
            category="Backend Architecture",
            difficulty=3,
            target_project="API Platform",
            target_entity="Kubernetes"  # Absent!
        )

        def mock_generate_rotation(ctx, **kwargs):
            t_ent = kwargs.get("target_entity", "FastAPI")
            ev_items = ctx.get("selected_evidence_items", [])
            eid = ev_items[0]["evidence_id"] if ev_items else "chk_0"
            ev_text = ev_items[0]["text"] if ev_items else "Developed asynchronous REST APIs using FastAPI and PostgreSQL."
            return QuestionOutputSchema(
                question=f"In API Platform, how would you approach designing the architecture when working with {t_ent}?",
                category="Backend Architecture",
                difficulty=3,
                target_project="API Platform",
                target_entity=t_ent,
                evidence_ids=[eid],
                evidence_text=ev_text,
                rationale=f"Grounded in API Platform {t_ent} usage.",
                grounding_score=1.0,
                relevance_score=1.0,
                technical_quality_score=1.0,
                trace_id="trc_test"
            )

        with patch("app.services.rag.generator.question_generator.generate", side_effect=mock_generate_rotation):
            resp: QuestionGenerationResponse = pipeline.generate_grounded_question(resume_id, req, db_session)

        assert resp.validation_status == "ACCEPTED"
        assert resp.question is not None
        assert resp.question.target_entity in ["FastAPI", "PostgreSQL", "Docker", "API Platform"]
