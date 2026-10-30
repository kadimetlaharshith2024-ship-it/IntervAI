import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, ForeignKey, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    
    source_entity = Column(Text, nullable=False)
    relation_type = Column(String(100), nullable=False)
    target_entity = Column(Text, nullable=False)
    
    section = Column(String(100), nullable=False, default="General")
    context_entity = Column(String(150), nullable=True)  # Project, Role, or Institution name
    evidence_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    extraction_method = Column(String(50), nullable=False, default="dependency_parsed")  # baseline_rule, dependency_parsed, hybrid
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    resume = relationship("Resume", back_populates="relationships")

    __table_args__ = (
        Index("idx_relationships_resume", "resume_id"),
        Index("idx_relationships_section", "section"),
        Index("idx_relationships_method", "extraction_method"),
    )