from uuid import UUID
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from app.schemas.contradiction import ContradictionAnalysisResult

class AnswerSubmissionRequest(BaseModel):
    question_id: UUID = Field(..., description="ID of the interview question being answered")
    session_id: Optional[UUID] = Field(None, description="Optional interview session ID linking this answer")
    candidate_answer: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Candidate's raw answer text (maximum 4000 characters)"
    )

    @field_validator("candidate_answer")
    @classmethod
    def validate_answer_text(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("Candidate answer cannot contain null bytes (\\x00).")
        s = v.strip()
        import re
        if not s or not re.sub(r"[\u200b-\u200d\ufeff\s]", "", s):
            raise ValueError("Candidate answer cannot be empty or whitespace only.")
        return s

class DimensionalScoresSchema(BaseModel):
    technical_accuracy: float = Field(..., ge=0.0, le=100.0, description="Correctness of technical facts and mechanisms (0-100)")
    conceptual_understanding: float = Field(..., ge=0.0, le=100.0, description="Clarity of underlying principles and why systems behave as described (0-100)")
    completeness: float = Field(..., ge=0.0, le=100.0, description="Extent to which all question aspects were addressed (0-100)")
    depth_and_specificity: float = Field(..., ge=0.0, le=100.0, description="Presence of concrete architectural/code details vs vague generalities (0-100)")
    practical_reasoning: float = Field(..., ge=0.0, le=100.0, description="Step-by-step problem-solving and trade-off analysis (0-100)")
    evidence_alignment: float = Field(..., ge=0.0, le=100.0, description="Consistency with verified resume experience (0-100)")

class FeedbackBreakdownSchema(BaseModel):
    summary: str = Field(..., description="High-level synthesized evaluation summary")
    strengths: List[str] = Field(default_factory=list, description="Specific aspects of the answer that demonstrated competence")
    weaknesses: List[str] = Field(default_factory=list, description="Specific technical inaccuracies, shallow explanations, or omissions")
    missing_concepts: List[str] = Field(default_factory=list, description="Key concepts or mechanisms expected but omitted")
    key_takeaways: Optional[str] = Field(None, description="Actionable recommendation for candidate improvement")

class EvaluationMetadataSchema(BaseModel):
    provider: Optional[str] = Field(None, description="LLM provider name (e.g. gemini, openai, mock)")
    model: Optional[str] = Field(None, description="LLM model identifier used for evaluation")
    rubric_version: str = Field(default="v1.0", description="Rubric version applied")
    latency_ms: Optional[float] = Field(None, description="Total evaluation latency in milliseconds")
    stage_latencies_ms: Optional[Dict[str, float]] = Field(None, description="Fine-grained stage latencies in milliseconds")
    contradiction_analysis: Optional[ContradictionAnalysisResult] = Field(
        None, description="Deterministic contradiction analysis results if performed"
    )

class SessionProgressSchema(BaseModel):
    completed_questions: int = Field(default=0, ge=0)
    total_questions: int = Field(default=0, ge=0)
    session_overall_score: Optional[float] = Field(None, ge=0.0, le=100.0)

class EvaluationResponseSchema(BaseModel):
    answer_id: UUID
    question_id: UUID
    session_id: Optional[UUID] = None
    attempt_number: int = Field(default=1, ge=1, description="Sequential attempt number for this question")
    candidate_answer: Optional[str] = Field(None, description="The candidate's submitted answer text")
    status: str = Field(..., description="'PENDING', 'EVALUATED', or 'FAILED'")
    overall_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Deterministic composite score (0-100)")
    scores: Optional[DimensionalScoresSchema] = None
    evaluation_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Reliability score of the evaluation itself (0.0 - 1.0)")
    feedback: Optional[FeedbackBreakdownSchema] = None
    evaluation_metadata: Optional[EvaluationMetadataSchema] = None
    session_progress: Optional[SessionProgressSchema] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class SessionScorecardSchema(BaseModel):
    session_id: UUID
    resume_id: UUID
    status: str
    overall_score: Optional[float] = None
    completed_questions: int = 0
    total_questions: int = 0
    category_scores: Dict[str, float] = Field(default_factory=dict)
    answers: List[EvaluationResponseSchema] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
