from uuid import UUID
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

def clean_placeholder_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ["string", "<string>", "null", "none", "undefined", ""]:
        return None
    return s

class EvidenceItem(BaseModel):
    evidence_id: str
    text: str
    section: str
    project: Optional[str] = None
    source_type: str  # "chunk", "relationship", "claim"
    score: float
    interview_relevance_score: float = 0.0
    provenance: str

class CandidateClaimSchema(BaseModel):
    claim_id: str
    claim_text: str
    section: str
    project: Optional[str] = None
    target_entity: Optional[str] = None
    relation_type: Optional[str] = None
    evidence_text: str
    confidence: float
    status: str

    model_config = ConfigDict(from_attributes=True)

class QuestionGenerationRequest(BaseModel):
    category: Optional[str] = Field(
        default="Architecture",
        description="Fundamentals, Application, Implementation, Architecture, Performance, Security, Failure-Mode, Claim-Verification"
    )
    difficulty: int = Field(default=3, ge=1, le=5)
    target_project: Optional[str] = Field(default=None, description="Target project name (optional)")
    target_entity: Optional[str] = Field(default=None, description="Target technology/entity (optional)")
    session_id: Optional[UUID] = Field(default=None, description="Optional interview session ID to link the question")
    parent_question_id: Optional[UUID] = Field(default=None, description="Optional parent question ID for follow-up questions")
    target_claim_id: Optional[str] = Field(default=None, description="Optional target claim ID for claim verification")
    target_relationship: Optional[str] = Field(default=None, description="Optional target relationship description or ID")
    follow_up_reason: Optional[str] = Field(default=None, description="Optional explainable reason/objective for follow-up")
    excluded_question_texts: Optional[List[str]] = Field(default_factory=list, description="Questions/concepts to explicitly avoid duplicating")
    retrieval_settings: Optional[Dict[str, Any]] = None

    @field_validator("target_project", "target_entity", "target_claim_id", "target_relationship", mode="before")
    @classmethod
    def sanitize_placeholders(cls, v):
        return clean_placeholder_text(v)

class QuestionOutputSchema(BaseModel):
    question: str
    category: str
    difficulty: int
    target_project: Optional[str] = None
    target_entity: Optional[str] = None
    target_claim_id: Optional[str] = None
    target_relationship: Optional[str] = None
    evidence_ids: List[str]
    evidence_text: str
    rationale: str
    grounding_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    technical_quality_score: float = Field(ge=0.0, le=1.0)
    trace_id: str
    # Structured Question Contract fields
    assessed_difficulty: Optional[int] = None
    difficulty_rationale: Optional[str] = None
    grounding_status: Optional[str] = "PASSED"
    grounding_rationale: Optional[str] = None
    hypothetical_assumptions: Optional[List[str]] = Field(default_factory=list)
    technical_relevance_status: Optional[str] = "PASSED"
    clarity_status: Optional[str] = "PASSED"
    redundancy_status: Optional[str] = "PASSED"
    validation_status: Optional[str] = "ACCEPTED"
    rubric_version: Optional[str] = "v2.0-canonical-5level"

class QuestionGenerationResponse(BaseModel):
    resume_id: UUID
    request_id: str
    trace_id: str
    question_id: Optional[UUID] = None
    question: Optional[QuestionOutputSchema] = None
    validation_status: str  # "ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE"
    message: Optional[str] = None
    attempts_count: int = 1
    latencies_ms: Dict[str, float]
    # Structured Question Contract metadata
    error_code: Optional[str] = None
    assessed_difficulty: Optional[int] = None
    rubric_version: Optional[str] = "v2.0-canonical-5level"
    rejected_candidates: Optional[List[str]] = Field(default_factory=list)

class ResumeIndexResponse(BaseModel):
    resume_id: UUID
    total_chunks: int
    total_claims: int
    embedding_model: str
    status: str