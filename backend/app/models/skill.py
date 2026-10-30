from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=True)

    resumes = relationship("ResumeSkill", back_populates="skill")

class ResumeSkill(Base):
    __tablename__ = "resume_skills"

    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), primary_key=True)
    skill_id = Column(Integer, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)
    extraction_confidence = Column(Float, nullable=False, default=1.0)

    resume = relationship("Resume", back_populates="skills")
    skill = relationship("Skill", back_populates="resumes")