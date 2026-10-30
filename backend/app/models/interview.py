import uuid
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Index, ARRAY, JSON, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    __table_args__ = {"extend_existing": True}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(30), default="IN_PROGRESS")  # 'IN_PROGRESS', 'COMPLETED', 'ENDED'
    current_difficulty = Column(Integer, nullable=False, default=3)
    overall_score = Column(Float, nullable=True)
    completed_questions = Column(Integer, nullable=False, default=0)
    total_questions = Column(Integer, nullable=False, default=0)
    category_scores = Column(JSON, nullable=True, default=dict)
    session_state = Column(JSON, nullable=True, default=dict)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="interview_sessions")
    resume = relationship("Resume", back_populates="interview_sessions")
    questions = relationship("InterviewQuestion", back_populates="session", cascade="all, delete-orphan")
    answers = relationship("InterviewAnswer", back_populates="session", cascade="all, delete-orphan")


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"
    __table_args__ = (
        Index("idx_questions_session", "session_id"),
        Index("idx_questions_resume", "resume_id"),
        Index("idx_questions_category", "category"),
        Index("idx_questions_difficulty", "difficulty"),
        {"extend_existing": True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False)
    parent_question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id"), nullable=True)

    question_text = Column(Text, nullable=False)
    difficulty = Column(Integer, nullable=False, default=3)  # 1 to 5
    category = Column(String(100), nullable=False, default="Architecture")
    targeted_claim = Column(Text, nullable=True, default="")

    # Phase 5 RAG Provenance & Grounding Fields
    target_project = Column(String(150), nullable=True)
    target_entity = Column(String(150), nullable=True)
    target_claim_id = Column(String(100), nullable=True)
    target_relationship = Column(String(150), nullable=True)

    evidence_ids = Column(JSON, nullable=False, default=list)
    evidence_text = Column(Text, nullable=False, default="")
    rationale = Column(Text, nullable=False, default="")
    expected_concepts = Column(JSON, nullable=True, default=list)

    grounding_score = Column(Float, nullable=False, default=1.0)
    relevance_score = Column(Float, nullable=False, default=1.0)
    technical_quality_score = Column(Float, nullable=False, default=1.0)

    retrieval_trace_id = Column(String(100), nullable=True, index=True)
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    assessed_difficulty = Column(Integer, nullable=True)
    rubric_version = Column(String(50), nullable=True, default="v2.0-canonical-5level")
    validation_details = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("InterviewSession", back_populates="questions")
    resume = relationship("Resume", back_populates="interview_questions")
    answers = relationship("InterviewAnswer", back_populates="question", cascade="all, delete-orphan")
    parent_question = relationship("InterviewQuestion", remote_side=[id], backref="follow_ups")


class InterviewAnswer(Base):
    __tablename__ = "interview_answers"
    __table_args__ = (
        Index("idx_answers_question", "question_id"),
        Index("idx_answers_session", "session_id"),
        Index("idx_answers_question_attempt", "question_id", "attempt_number"),
        Index("idx_answers_session_question_attempt", "session_id", "question_id", "attempt_number"),
        {"extend_existing": True},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=True)
    candidate_answer = Column(Text, nullable=False)
    attempt_number = Column(Integer, nullable=False, default=1)

    # Phase 6 Evaluation & Scoring Fields
    status = Column(String(30), nullable=False, default="PENDING")  # 'PENDING', 'EVALUATED', 'FAILED'
    overall_score = Column(Float, nullable=True)  # Composite 0-100 score

    # Reused existing field: technical_accuracy_score (0-100)
    technical_accuracy_score = Column(Float, nullable=True)
    conceptual_understanding_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    depth_score = Column(Float, nullable=True)
    reasoning_score = Column(Float, nullable=True)
    evidence_alignment_score = Column(Float, nullable=True)
    evaluation_confidence = Column(Float, nullable=True, default=1.0)

    # Reused existing field: evaluation_feedback (Text summary)
    evaluation_feedback = Column(Text, nullable=True)
    # Reused existing field: identified_gaps (ARRAY(Text) / JSON)
    identified_gaps = Column(ARRAY(Text), nullable=True)

    strengths = Column(JSON, nullable=True, default=list)
    weaknesses = Column(JSON, nullable=True, default=list)
    missing_concepts = Column(JSON, nullable=True, default=list)
    key_takeaways = Column(Text, nullable=True)
    dimension_scores = Column(JSON, nullable=True, default=dict)

    # Metadata & Auditing
    evaluator_provider = Column(String(50), nullable=True)
    evaluator_model = Column(String(100), nullable=True)
    rubric_version = Column(String(20), nullable=False, default="v1.0")
    evaluation_latency_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    question = relationship("InterviewQuestion", back_populates="answers")
    session = relationship("InterviewSession", back_populates="answers")

    def __init__(self, **kwargs):
        if "status" not in kwargs:
            kwargs["status"] = "PENDING"
        if "evaluation_confidence" not in kwargs:
            kwargs["evaluation_confidence"] = 1.0
        if "rubric_version" not in kwargs:
            kwargs["rubric_version"] = "v1.0"
        if "attempt_number" not in kwargs:
            kwargs["attempt_number"] = 1
        super().__init__(**kwargs)