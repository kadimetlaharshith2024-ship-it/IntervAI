from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class AdaptiveStrategyType(str, Enum):
    UNCOVERED_TOPIC = "UNCOVERED_TOPIC"
    GAP_FOLLOW_UP = "GAP_FOLLOW_UP"
    DIFFICULTY_ESCALATION = "DIFFICULTY_ESCALATION"
    DIFFICULTY_DEESCALATION = "DIFFICULTY_DEESCALATION"
    CLAIM_VERIFICATION = "CLAIM_VERIFICATION"
    RELATIONSHIP_VERIFICATION = "RELATIONSHIP_VERIFICATION"
    PROJECT_DEPTH = "PROJECT_DEPTH"
    COVERAGE_EXPANSION = "COVERAGE_EXPANSION"
    PARTIAL_CLARIFICATION = "PARTIAL_CLARIFICATION"
    CONTRADICTION_CLARIFICATION = "CONTRADICTION_CLARIFICATION"


class NextQuestionSelectionResult(BaseModel):
    strategy: AdaptiveStrategyType = Field(..., description="The adaptive selection strategy applied")
    target_topic: Optional[str] = Field(default=None, description="Primary canonical topic/entity selected")
    target_entity: Optional[str] = Field(default=None, description="Specific technology/entity for RAG targeting")
    target_project: Optional[str] = Field(default=None, description="Resume project name if question is anchored to a project")
    target_claim_id: Optional[str] = Field(default=None, description="Specific claim ID if strategy is CLAIM_VERIFICATION")
    target_relationship: Optional[str] = Field(default=None, description="Relationship text/ID if strategy is RELATIONSHIP_VERIFICATION")
    category: str = Field(default="Architecture", description="Category for question (Architecture, Implementation, Fundamentals, etc.)")
    difficulty: int = Field(default=3, ge=1, le=5, description="Target difficulty level (1 to 5)")
    rationale: str = Field(..., description="Explainable justification for why this target, category, and difficulty were selected")
    is_follow_up: bool = Field(default=False, description="True if this is a follow-up on a previous question")
    parent_question_id: Optional[UUID] = Field(default=None, description="UUID of the parent question if following up")
    consecutive_followup_count: int = Field(default=0, ge=0, description="Number of consecutive follow-ups preceding this question")
    follow_up_reason: Optional[str] = Field(default=None, description="Explainable justification or objective if question is a follow-up")

    model_config = ConfigDict(from_attributes=True)


class AdaptiveQuestionSelectionRequest(BaseModel):
    preferred_category: Optional[str] = Field(default=None, description="Optional override for question category")
    force_strategy: Optional[AdaptiveStrategyType] = Field(default=None, description="Optional override to force a specific strategy")
    max_consecutive_followups: Optional[int] = Field(default=None, ge=0, le=5, description="Optional override for follow-up budget")
    request_idempotency_key: Optional[str] = Field(default=None, description="Client idempotency key to prevent duplicate questions on retries")


class AdaptiveNextQuestionResponse(BaseModel):
    session_id: UUID
    resume_id: UUID
    status: str = Field(default="GENERATED", description="'GENERATED', 'SESSION_ENDED', 'BUDGET_EXHAUSTED', 'EVIDENCE_UNAVAILABLE', 'ALL_TOPICS_COVERED'")
    selection: Optional[NextQuestionSelectionResult] = None
    question_id: Optional[UUID] = None
    question_text: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[int] = None
    target_project: Optional[str] = None
    target_entity: Optional[str] = None
    evidence_text: Optional[str] = None
    grounding_score: Optional[float] = None
    assessed_difficulty: Optional[int] = None
    difficulty_rationale: Optional[str] = None
    evidence_ids: Optional[List[str]] = Field(default_factory=list)
    rubric_version: Optional[str] = "v2.0-canonical-5level"
    grounding_status: Optional[str] = "PASSED"
    error_code: Optional[str] = Field(default=None, description="Explicit error code if status is EVIDENCE_UNAVAILABLE or RECOVERABLE_ERROR")
    message: str = ""
    latencies_ms: Optional[Dict[str, float]] = Field(default=None, description="Execution stage latencies in milliseconds")

    model_config = ConfigDict(from_attributes=True)
