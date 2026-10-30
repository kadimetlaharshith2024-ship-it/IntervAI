"""
Comprehensive Regression & Integration Test Suite for:
1. Five-Level Canonical Difficulty Calibration (Levels 1-5).
2. Strict Resume Grounding & Cross-Candidate Isolation.
3. Structured Question Contract & Rejection Handling.
4. Adaptive Interview State Preservation.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY
from fastapi.testclient import TestClient

# Register SQLite ARRAY compatibility for in-memory testing
compiles(ARRAY, "sqlite")(lambda type_, compiler, **kw: "JSON")

from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.models.relationship import Relationship
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.interview import Base, InterviewSession, InterviewQuestion, InterviewAnswer
from app.schemas.rag import QuestionOutputSchema, QuestionGenerationRequest, QuestionGenerationResponse
from app.services.rag.validator import IndependentGroundingValidator
from app.services.rag.rag_pipeline import RAGPipeline
from app.services.adaptive_selection_service import (
    adaptive_selection_engine,
    AdaptiveSelectionConfig,
)
from app.schemas.adaptive_selection import (
    AdaptiveQuestionSelectionRequest,
    AdaptiveNextQuestionResponse,
)
from app.core.database import get_db
from app.main import app


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


@pytest.fixture
def client(db_session: Session):
    """Provides a FastAPI TestClient with database override."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


from app.services.rag.embedding_service import embedding_service


@pytest.fixture
def candidate_resumes(db_session: Session):
    """Creates two distinct candidates to test provenance, grounding, and cross-candidate isolation."""
    # Candidate A: Geofencing & Spatial Systems Engineer
    resume_a = Resume(
        id=uuid4(),
        filename="alice_spatial_engineer.pdf",
        raw_text="Alice Spatial Systems Engineer. Skills: Python, R-tree, PostgreSQL, Redis, FastAPI. Project: GeofencingService."
    )
    db_session.add(resume_a)
    db_session.flush()

    chunk_a1_text = "Implemented R-tree spatial indexing for real-time geofence lookup in GeofencingService, reducing query latency by 45%."
    chunk_a1 = ResumeChunk(
        resume_id=resume_a.id,
        chunk_id="chk_a1",
        chunk_text=chunk_a1_text,
        section="Experience",
        project="GeofencingService",
        embedding=embedding_service.embed_query(chunk_a1_text),
        embedding_model=embedding_service.model_name
    )
    chunk_a2_text = "Engineered PostgreSQL PostGIS spatial queries and Redis caching for concurrent mobile location updates."
    chunk_a2 = ResumeChunk(
        resume_id=resume_a.id,
        chunk_id="chk_a2",
        chunk_text=chunk_a2_text,
        section="Experience",
        project="GeofencingService",
        embedding=embedding_service.embed_query(chunk_a2_text),
        embedding_model=embedding_service.model_name
    )
    claim_a1 = ResumeClaim(
        resume_id=resume_a.id,
        claim_id="clm_a1",
        claim_text="Implemented R-tree spatial indexing for real-time geofence lookup in GeofencingService.",
        section="Experience",
        project="GeofencingService",
        target_entity="R-tree",
        evidence_text="Implemented R-tree spatial indexing for real-time geofence lookup in GeofencingService.",
        confidence=0.95
    )
    db_session.add_all([chunk_a1, chunk_a2, claim_a1])

    # Candidate B: Distributed ML Pipeline Engineer
    resume_b = Resume(
        id=uuid4(),
        filename="bob_ml_engineer.pdf",
        raw_text="Bob ML Engineer. Skills: PyTorch, Ray, Kubernetes, Apache Spark. Project: DistributedTrainer."
    )
    db_session.add(resume_b)
    db_session.flush()

    chunk_b1_text = "Trained distributed transformers using Ray Train and PyTorch DDP on Kubernetes clusters for DistributedTrainer."
    chunk_b1 = ResumeChunk(
        resume_id=resume_b.id,
        chunk_id="chk_b1",
        chunk_text=chunk_b1_text,
        section="Experience",
        project="DistributedTrainer",
        embedding=embedding_service.embed_query(chunk_b1_text),
        embedding_model=embedding_service.model_name
    )
    claim_b1 = ResumeClaim(
        resume_id=resume_b.id,
        claim_id="clm_b1",
        claim_text="Trained distributed transformers using Ray Train and PyTorch DDP on Kubernetes.",
        section="Experience",
        project="DistributedTrainer",
        target_entity="Ray",
        evidence_text="Trained distributed transformers using Ray Train and PyTorch DDP on Kubernetes.",
        confidence=0.92
    )
    db_session.add_all([chunk_b1, claim_b1])
    db_session.commit()

    return {"candidate_a": resume_a, "candidate_b": resume_b}


# ==============================================================================
# SECTION A: DIFFICULTY CALIBRATION UNIT TESTS (CANONICAL 5-LEVEL RUBRIC)
# ==============================================================================

class TestDifficultyCalibrationFiveLevels:
    """Rigorous evaluation of the canonical 5-level difficulty rubric estimator and validator."""

    def setup_method(self):
        self.validator = IndependentGroundingValidator()

    def test_canonical_level_1_fundamentals_understand(self):
        """Level 1 tests understanding of core concepts, definitions, principles, and mechanisms."""
        q1 = "What is an R-tree, and why is it useful for spatial data?"
        assessed = self.validator.estimate_difficulty_level(q1)
        assert assessed == 1
        expl = self.validator.explain_difficulty_level(q1, assessed)
        assert "Level 1" in expl or "Fundamentals" in expl

    def test_canonical_level_2_application_apply(self):
        """Level 2 requires applying a known concept to a concrete situation in documented experience."""
        q2 = "How would an R-tree help determine whether a user's location falls within a geofenced region?"
        assessed = self.validator.estimate_difficulty_level(q2)
        assert assessed == 2
        expl = self.validator.explain_difficulty_level(q2, assessed)
        assert "Level 2" in expl or "Application" in expl

    def test_canonical_level_3_implementation_build_and_debug(self):
        """Level 3 requires reasoning about implementation details, algorithms, debugging, testing, or code decisions."""
        q3 = "How would you implement a point-in-region lookup in the geofencing application, and how would you test boundary cases?"
        assessed = self.validator.estimate_difficulty_level(q3)
        assert assessed == 3
        expl = self.validator.explain_difficulty_level(q3, assessed)
        assert "Level 3" in expl or "Implementation" in expl

    def test_canonical_level_4_architecture_design_and_evaluate(self):
        """Level 4 requires system-level reasoning about component interactions, scalability, reliability, and trade-offs."""
        q4 = "How would you design the geofencing service to handle concurrent location updates, spatial queries, and reliable API responses?"
        assessed = self.validator.estimate_difficulty_level(q4)
        assert assessed == 4
        expl = self.validator.explain_difficulty_level(q4, assessed)
        assert "Level 4" in expl or "Architecture" in expl

    def test_canonical_level_5_advanced_depth_analyze_and_defend(self):
        """Level 5 requires reasoning under uncertainty, competing constraints, alternative approaches, and trade-off defense."""
        q5 = "If spatial density becomes uneven and updates are frequent, how would you compare indexing strategies and determine when the existing design should change?"
        assessed = self.validator.estimate_difficulty_level(q5)
        assert assessed == 5
        expl = self.validator.explain_difficulty_level(q5, assessed)
        assert "Level 5" in expl or "Advanced Depth" in expl

    def test_adjacent_boundary_mislabeling_detection(self):
        """A simple definition incorrectly requested as Level 4 or 5 must be assessed as Level 1 and rejected."""
        q_simple = "What is the primary difference between a clustered index and a non-clustered index in PostgreSQL?"
        assessed = self.validator.estimate_difficulty_level(q_simple)
        assert assessed == 1

        cand_q = QuestionOutputSchema(
            question=q_simple,
            category="Database",
            difficulty=5,  # Mislabeled as Level 5!
            grounding_score=0.95,
            relevance_score=0.90,
            technical_quality_score=0.90,
            trace_id="trc_test_1",
            evidence_ids=["chk_a1"],
            evidence_text="PostgreSQL spatial queries",
            rationale="Test mislabeling"
        )
        is_valid, reason, details = self.validator.validate(
            question=cand_q,
            allowed_projects=["GeofencingService"],
            allowed_entities=["PostgreSQL", "R-tree"],
            resolved_evidence_items={"chk_a1": {"text": "PostgreSQL spatial queries"}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        assert is_valid is False
        assert details["code"] == "difficulty_mismatch"
        assert details["assessed_difficulty"] == 1
        assert details["requested_difficulty"] == 5

    def test_application_confused_with_architecture_rejection(self):
        """A straightforward application question (Level 2) requested at Level 4 must fail difficulty calibration."""
        q_apply = "How does your service use Redis caching to store mobile device coordinates?"
        assessed = self.validator.estimate_difficulty_level(q_apply)
        assert assessed == 2

        cand_q = QuestionOutputSchema(
            question=q_apply,
            category="Architecture",
            difficulty=4,  # Requested 4, actual 2 -> |4 - 2| = 2 >= 2 triggers difficulty_mismatch
            grounding_score=0.95,
            relevance_score=0.90,
            technical_quality_score=0.90,
            trace_id="trc_test_2",
            evidence_ids=["chk_a2"],
            evidence_text="Redis caching for mobile location",
            rationale="Test application vs architecture"
        )
        is_valid, reason, details = self.validator.validate(
            question=cand_q,
            allowed_projects=["GeofencingService"],
            allowed_entities=["Redis"],
            resolved_evidence_items={"chk_a2": {"text": "Redis caching for mobile location"}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        assert is_valid is False
        assert details["code"] == "difficulty_mismatch"
        assert details["assessed_difficulty"] == 2

    def test_complex_wording_without_reasoning_not_level_5(self):
        """A question with verbose jargon and clauses that only asks for a definition is not Level 5."""
        q_verbose_def = "In the domain of high-performance spatial computation engines, can you elucidate and define what an R-tree spatial partitioning index actually represents?"
        assessed = self.validator.estimate_difficulty_level(q_verbose_def)
        # Even though it contains complex words, its cognitive demand is definitional (Level 1)
        assert assessed == 1


# ==============================================================================
# SECTION B: RESUME GROUNDING & EVIDENCE PROVENANCE TESTS
# ==============================================================================

class TestResumeGroundingAndIsolation:
    """Verifies that questions are strictly grounded in candidate-owned evidence and prevents cross-candidate leakage."""

    def setup_method(self):
        self.validator = IndependentGroundingValidator()

    def test_evidence_must_belong_to_candidate_resolved_items(self):
        """Question citing evidence IDs not present in resolved candidate items must be rejected."""
        cand_q = QuestionOutputSchema(
            question="How would you design the GeofencingService to scale across regions?",
            category="Architecture",
            difficulty=4,
            grounding_score=0.95,
            relevance_score=0.90,
            technical_quality_score=0.90,
            trace_id="trc_ground_1",
            evidence_ids=["chk_unowned_999"],  # Bogus/unowned ID
            evidence_text="Some unverified text",
            rationale="Test unowned evidence"
        )
        is_valid, reason, details = self.validator.validate(
            question=cand_q,
            allowed_projects=["GeofencingService"],
            allowed_entities=["R-tree", "PostgreSQL"],
            resolved_evidence_items={"chk_a1": {"text": "R-tree geofence lookup"}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        assert is_valid is False
        assert details["code"] == "invalid_citation_id"

    def test_unsupported_candidate_facts_rejected(self):
        """Presupposing an unverified implementation mechanism (e.g. distributed Raft consensus) is rejected."""
        cand_q = QuestionOutputSchema(
            question="Walk me through how your custom Raft consensus quorum handles leader elections in GeofencingService.",
            category="Architecture",
            difficulty=4,
            grounding_score=0.95,
            relevance_score=0.90,
            technical_quality_score=0.90,
            trace_id="trc_ground_2",
            evidence_ids=["chk_a1"],
            evidence_text="Implemented R-tree spatial indexing for GeofencingService.",
            rationale="Presupposes Raft not in resume"
        )
        is_valid, reason, details = self.validator.validate(
            question=cand_q,
            allowed_projects=["GeofencingService"],
            allowed_entities=["R-tree", "Raft"],
            resolved_evidence_items={"chk_a1": {"text": "Implemented R-tree spatial indexing for GeofencingService."}},
            existing_questions=[],
            available_claims={},
            available_relationships={}
        )
        # Raft is not in allowed entities nor in evidence text
        assert is_valid is False


# ==============================================================================
# SECTION C: INTEGRATED END-TO-END RAG & ADAPTIVE STATE TESTS
# ==============================================================================

class TestEndToEndPipelineIntegration:
    """Verifies that RAG generation, question contract persistence, and adaptive selection operate seamlessly across all 5 levels."""

    def test_rag_pipeline_generates_and_persists_contract_metadata(self, db_session, candidate_resumes):
        """Verify that an accepted question persists assessed_difficulty, rubric_version, and validation_details."""
        resume_a = candidate_resumes["candidate_a"]
        pipeline = RAGPipeline()

        # Generate Level 3 question for candidate A
        req = QuestionGenerationRequest(
            category="Implementation",
            difficulty=3,
            target_project="GeofencingService"
        )
        resp = pipeline.generate_grounded_question(resume_a.id, req, db_session)

        assert resp.validation_status == "ACCEPTED"
        assert resp.question is not None
        assert resp.question_id is not None
        assert resp.rubric_version == "v2.0-canonical-5level"
        assert resp.assessed_difficulty is not None

        # Verify database record
        q_rec = db_session.query(InterviewQuestion).filter(InterviewQuestion.id == resp.question_id).first()
        assert q_rec is not None
        assert q_rec.rubric_version == "v2.0-canonical-5level"
        assert q_rec.assessed_difficulty in [1, 2, 3, 4, 5]
        assert q_rec.validation_details is not None
        assert isinstance(q_rec.validation_details, dict)
        assert q_rec.validation_details.get("status") in ["PASSED", "ACCEPTED"]

    def test_cross_candidate_isolation_prevents_leakage(self, db_session, candidate_resumes):
        """Candidate B's claims cannot be requested or used in Candidate A's interview."""
        resume_a = candidate_resumes["candidate_a"]
        pipeline = RAGPipeline()

        # Attempt to target Candidate B's claim (clm_b1: Ray Train) on Candidate A's resume
        req = QuestionGenerationRequest(
            category="Architecture",
            difficulty=4,
            target_claim_id="clm_b1"
        )
        resp = pipeline.generate_grounded_question(resume_a.id, req, db_session)

        # Must be rejected immediately with structured error code
        assert resp.validation_status == "REJECTED"
        assert resp.error_code == "invalid_claim_reference"
        assert "not supported by candidate resume" in resp.message

    def test_unsupported_project_rejected_with_typed_error(self, db_session, candidate_resumes):
        """Requesting an unsupported project returns typed error without corrupting state."""
        resume_a = candidate_resumes["candidate_a"]
        pipeline = RAGPipeline()

        req = QuestionGenerationRequest(
            category="Architecture",
            difficulty=4,
            target_project="AutonomousDrivingPlatform"  # Not on resume
        )
        resp = pipeline.generate_grounded_question(resume_a.id, req, db_session)

        assert resp.validation_status == "REJECTED"
        assert resp.error_code == "invalid_project_reference"
        assert "AutonomousDrivingPlatform" in resp.message

    def test_all_five_difficulty_levels_via_rag_generation(self, db_session, candidate_resumes):
        """Verify question generation requests across all five canonical difficulty levels."""
        resume_a = candidate_resumes["candidate_a"]
        pipeline = RAGPipeline()

        for requested_lvl in [1, 2, 3, 4, 5]:
            session = InterviewSession(
                id=uuid4(),
                resume_id=resume_a.id,
                user_id=uuid4(),
                status="IN_PROGRESS",
                current_difficulty=requested_lvl
            )
            db_session.add(session)
            db_session.flush()

            req = QuestionGenerationRequest(
                category="Technical",
                difficulty=requested_lvl,
                target_project="GeofencingService",
                session_id=session.id
            )
            resp = pipeline.generate_grounded_question(resume_a.id, req, db_session)
            assert resp.validation_status == "ACCEPTED", f"Failed for level {requested_lvl}: {resp.message} | Attempts: {getattr(resp, 'attempts_log', None)}"
            assert resp.question.difficulty == requested_lvl
            assert resp.assessed_difficulty is not None
            # Assessed difficulty should be within 1 level of requested level (calibrated)
            assert abs(resp.assessed_difficulty - requested_lvl) <= 1

    def test_interview_state_preservation_on_generation_failure(self, db_session, candidate_resumes):
        """When question generation encounters an error, existing answers and session state remain untouched."""
        resume_a = candidate_resumes["candidate_a"]

        # Create session with 1 evaluated question and answer
        session = InterviewSession(
            id=uuid4(),
            resume_id=resume_a.id,
            user_id=uuid4(),
            status="IN_PROGRESS",
            current_difficulty=3,
            total_questions=1,
            completed_questions=1,
            session_state={
                "knowledge_state": {"topics": {"R-tree": {"coverage_status": "EVALUATED"}}},
                "answers_count": 1
            }
        )
        db_session.add(session)
        db_session.flush()

        q1 = InterviewQuestion(
            id=uuid4(),
            resume_id=resume_a.id,
            session_id=session.id,
            question_text="Describe how R-tree indexes spatial boundaries.",
            category="Technical",
            difficulty=3,
            assessed_difficulty=3,
            rubric_version="v2.0-canonical-5level"
        )
        ans1 = InterviewAnswer(
            id=uuid4(),
            session_id=session.id,
            question_id=q1.id,
            candidate_answer="R-trees group nearby objects using bounding boxes.",
            overall_score=85.0,
            status="EVALUATED"
        )
        db_session.add_all([q1, ans1])
        db_session.commit()

        # Simulate adaptive selection attempt targeting an unsupported entity that fails
        req_override = AdaptiveQuestionSelectionRequest(
            preferred_category="UnsupportedTopicXYZ"
        )

        from app.schemas.adaptive_selection import NextQuestionSelectionResult, AdaptiveStrategyType

        with patch.object(adaptive_selection_engine, "determine_next_question_target") as mock_target:
            mock_target.return_value = NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.UNCOVERED_TOPIC,
                target_topic="NonExistentTopic",
                target_project="NonExistentProject",
                target_entity="NonExistentEntity",
                category="NonExistentCategory",
                difficulty=3,
                rationale="Test failure state preservation",
                parent_question_id=None,
                target_claim_id=None,
                target_relationship=None,
                follow_up_reason=None
            )
            # Call adaptive selection service
            resp = adaptive_selection_engine.select_and_generate_next_question(
                session_id=session.id,
                db=db_session,
                request_override=req_override
            )

        # Session must NOT be completed or corrupted
        db_session.refresh(session)
        assert session.status == "IN_PROGRESS"
        assert session.completed_questions == 1

        # Previous question and answer must remain intact
        saved_q = db_session.query(InterviewQuestion).filter(InterviewQuestion.id == q1.id).first()
        saved_ans = db_session.query(InterviewAnswer).filter(InterviewAnswer.id == ans1.id).first()
        assert saved_q is not None
        assert saved_ans is not None
        assert saved_ans.overall_score == 85.0
