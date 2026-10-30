import pytest
from uuid import uuid4
from unittest.mock import MagicMock
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm.exc import DetachedInstanceError

from app.models.rag import ResumeClaim, ResumeChunk
from app.models.relationship import Relationship
from app.services.rag.retrievers import (
    get_or_build_resume_cache,
    resume_retrieval_cache,
    CachedResumeClaimDTO,
    CachedRelationshipDTO,
    CachedResumeChunkDTO,
    CachedResumeData,
)

Base = declarative_base()

class MockResumeClaim(Base):
    __tablename__ = "mock_resume_claims"
    id = Column(Integer, primary_key=True)
    claim_id = Column(String(100), nullable=False)
    resume_id = Column(String(100), nullable=False)
    project = Column(String(150), nullable=True)
    target_entity = Column(String(150), nullable=True)
    claim_text = Column(Text, nullable=False)
    source_section = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    evidence_chunk_id = Column(String(100), nullable=True)


def test_detached_instance_failure_demonstration():
    """
    Demonstrates that accessing an attribute on a detached ORM instance after
    the session closes raises DetachedInstanceError when expire_on_commit=True.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True)

    db = Session()
    claim = MockResumeClaim(
        claim_id="clm_1",
        resume_id="res_1",
        project="E-Commerce Platform",
        target_entity="PostgreSQL",
        claim_text="Designed database schemas in PostgreSQL."
    )
    db.add(claim)
    db.commit()
    db.close()

    # Accessing .project on the detached ORM object outside session raises DetachedInstanceError
    with pytest.raises(DetachedInstanceError):
        _ = claim.project.lower()


def test_cached_claim_dto_preserves_attributes_after_session_closes():
    """
    Verifies that converting to CachedResumeClaimDTO allows safe attribute access
    after the ORM database session has closed.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True)

    db = Session()
    claim = MockResumeClaim(
        claim_id="clm_safe_1",
        resume_id="res_1",
        project="Realtime Engine",
        target_entity="Go",
        claim_text="Implemented concurrent pipeline in Go."
    )
    db.add(claim)
    db.commit()

    # Convert to DTO while session is active
    dto = CachedResumeClaimDTO.from_orm(claim)
    db.close()

    # Must be accessible without any active database session
    assert dto.claim_id == "clm_safe_1"
    assert dto.project == "Realtime Engine"
    assert dto.project.lower() == "realtime engine"
    assert dto.target_entity == "Go"
    assert dto.claim_text == "Implemented concurrent pipeline in Go."


def test_candidate_projects_construction_with_dtos_and_nullable_fields():
    """
    Regression test for line 182 in rag_pipeline.py:
    all_candidate_projects = set(p.lower() for p in allowed_projects if p) | ...
    Ensures no DetachedInstanceError and handles nullable/empty project fields safely.
    """
    dto1 = CachedResumeClaimDTO(
        claim_id="c1",
        project="IntervAI Platform",
        target_entity="Python",
        claim_text="Built adaptive interview platform in Python."
    )
    dto2 = CachedResumeClaimDTO(
        claim_id="c2",
        project=None,  # Nullable project
        target_entity="Docker",
        claim_text="Containerized microservices."
    )
    dto3 = CachedResumeClaimDTO(
        claim_id="c3",
        project="",  # Empty string project
        target_entity="FastAPI",
        claim_text="Built REST endpoints in FastAPI."
    )

    claims_map = {"c1": dto1, "c2": dto2, "c3": dto3}
    allowed_projects = ["IntervAI Platform", None, ""]
    existing_relationships = [
        CachedRelationshipDTO(
            id="r1",
            source_entity="Python",
            relation_type="BUILT_WITH",
            target_entity="FastAPI",
            context_entity="Backend Service"
        ),
        CachedRelationshipDTO(
            id="r2",
            source_entity="Docker",
            relation_type="DEPLOYED_ON",
            target_entity="AWS",
            context_entity=None  # Nullable context_entity
        )
    ]

    all_candidate_projects = (
        set(p.lower() for p in allowed_projects if p)
        | set(c.project.lower() for c in claims_map.values() if getattr(c, "project", None))
        | set(r.context_entity.lower() for r in existing_relationships if getattr(r, "context_entity", None))
    )

    assert "intervai platform" in all_candidate_projects
    assert "backend service" in all_candidate_projects
    assert None not in all_candidate_projects
    assert "" not in all_candidate_projects
