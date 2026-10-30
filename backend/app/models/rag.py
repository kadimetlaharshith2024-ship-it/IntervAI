import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, Integer, ForeignKey, DateTime, Index, JSON, func
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base

# Re-export canonical InterviewQuestion so all RAG services and endpoints import cleanly
from app.models.interview import InterviewQuestion

class ResumeChunk(Base):
    __tablename__ = "resume_chunks"
    __table_args__ = (
        Index("idx_chunks_resume", "resume_id"),
        Index("idx_chunks_section", "section"),
        Index("idx_chunks_project", "project"),
        {"extend_existing": True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String(100), nullable=False, unique=True, index=True)
    
    section = Column(String(100), nullable=False)
    project = Column(String(150), nullable=True)
    chunk_text = Column(Text, nullable=False)
    
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=False)
    embedding_version = Column(String(50), nullable=False, default="1.0")
    
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ResumeClaim(Base):
    __tablename__ = "resume_claims"
    __table_args__ = (
        Index("idx_claims_resume", "resume_id"),
        Index("idx_claims_status", "status"),
        Index("idx_claims_project", "project"),
        {"extend_existing": True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    claim_id = Column(String(100), nullable=False, unique=True, index=True)
    
    claim_text = Column(Text, nullable=False)
    section = Column(String(100), nullable=False)
    project = Column(String(150), nullable=True)
    
    target_entity = Column(String(150), nullable=True)
    relation_type = Column(String(100), nullable=True)
    evidence_text = Column(Text, nullable=False)
    
    confidence = Column(Float, nullable=False, default=1.0)
    status = Column(String(50), nullable=False, default="UNTESTED")
    provenance = Column(String(100), nullable=False, default="relationship_extraction")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"
    __table_args__ = (
        Index("idx_traces_resume", "resume_id"),
        {"extend_existing": True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    trace_id = Column(String(100), nullable=False, unique=True, index=True)
    request_id = Column(String(100), nullable=False, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    
    query_intent = Column(Text, nullable=False)
    retrieval_config = Column(JSON, nullable=False)
    retrieval_scores = Column(JSON, nullable=False)
    selected_evidence = Column(JSON, nullable=False)
    discarded_evidence = Column(JSON, nullable=True)
    
    final_context = Column(JSON, nullable=False)
    latencies_ms = Column(JSON, nullable=False)
    validation_result = Column(JSON, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)