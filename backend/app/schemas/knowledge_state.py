from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class TopicAssessmentStatus(str, Enum):
    UNASSESSED = "UNASSESSED"
    PARTIALLY_ASSESSED = "PARTIALLY_ASSESSED"
    ASSESSED = "ASSESSED"
    ADEQUATELY_ASSESSED = "ADEQUATELY_ASSESSED"


class TopicCoverageStatus(str, Enum):
    NOT_CONSIDERED = "NOT_CONSIDERED"
    SELECTED = "SELECTED"
    QUESTION_ASKED = "QUESTION_ASKED"
    PARTIALLY_ASSESSED = "PARTIALLY_ASSESSED"
    ADEQUATELY_ASSESSED = "ADEQUATELY_ASSESSED"
    REQUIRES_FOLLOW_UP = "REQUIRES_FOLLOW_UP"


class ClaimVerificationStatus(str, Enum):
    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GapSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GapNature(str, Enum):
    DEMONSTRATED_WEAKNESS = "DEMONSTRATED_WEAKNESS"
    INCOMPLETE_UNDERSTANDING = "INCOMPLETE_UNDERSTANDING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TopicMasteryState(BaseModel):
    topic_id: str = Field(..., description="Stable canonical topic identifier")
    topic_name: str = Field(..., description="Display name of the topic")
    topic_type: str = Field(default="Skill", description="skill, technology, framework, database, tool, language, etc.")
    assessment_status: str = Field(default="UNASSESSED", description="UNASSESSED, PARTIALLY_ASSESSED, ASSESSED, ADEQUATELY_ASSESSED")
    coverage_status: str = Field(default="NOT_CONSIDERED", description="NOT_CONSIDERED, SELECTED, QUESTION_ASKED, PARTIALLY_ASSESSED, ADEQUATELY_ASSESSED, REQUIRES_FOLLOW_UP")
    questions_asked_count: int = Field(default=0, ge=0)
    answers_evaluated_count: int = Field(default=0, ge=0)
    latest_score: Optional[float] = None
    historical_scores: List[float] = Field(default_factory=list)
    mastery_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Recency-weighted estimated mastery score (0-100)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in mastery estimate (0-1)")
    last_assessed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ProjectUnderstandingState(BaseModel):
    project_id: str = Field(..., description="Sanitized project identifier")
    project_name: str = Field(..., description="Display name of the project")
    is_assessed: bool = False
    coverage_status: str = Field(default="NOT_CONSIDERED")
    questions_asked_count: int = Field(default=0, ge=0)
    answers_evaluated_count: int = Field(default=0, ge=0)
    average_score: Optional[float] = None
    latest_score: Optional[float] = None
    architecture_understanding: str = Field(default="UNASSESSED", description="STRONG, ADEQUATE, WEAK, UNASSESSED")
    implementation_understanding: str = Field(default="UNASSESSED", description="STRONG, ADEQUATE, WEAK, UNASSESSED")
    known_knowledge_gaps: List[str] = Field(default_factory=list)
    remaining_areas_to_assess: List[str] = Field(default_factory=list)
    tested_question_ids: List[UUID] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class RelationshipUnderstandingState(BaseModel):
    relationship_id: str = Field(..., description="Relationship ID or composite key")
    source_entity: str
    relation_type: str
    target_entity: str
    context_entity: Optional[str] = None
    is_tested: bool = False
    status: str = Field(default="UNTESTED", description="UNTESTED, DEMONSTRATED, PARTIAL, MISUNDERSTOOD")
    demonstrated_understanding: str = Field(default="UNTESTED", description="DEMONSTRATED, WEAK_EXPLANATION, UNCLEAR, UNTESTED")
    score: Optional[float] = None
    evidence: Optional[str] = None
    supporting_question_id: Optional[UUID] = None
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ResumeClaimVerificationState(BaseModel):
    claim_id: str
    claim_text: str
    project: Optional[str] = None
    target_entity: Optional[str] = None
    relation_type: Optional[str] = None
    status: str = Field(default="UNTESTED", description="UNTESTED, SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE")
    evidence_text: Optional[str] = None
    verification_source_question_id: Optional[UUID] = None
    confidence: float = 0.0
    last_updated_at: Optional[datetime] = None
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DetailedKnowledgeGap(BaseModel):
    gap_id: str
    target_type: str = Field(..., description="skill, technology, project, relationship, or claim")
    topic: str
    description: str
    question_id: UUID
    answer_id: Optional[UUID] = None
    evidence_or_feedback: str = ""
    severity: str = Field(default="MEDIUM", description="HIGH, MEDIUM, or LOW")
    gap_nature: str = Field(default="DEMONSTRATED_WEAKNESS", description="DEMONSTRATED_WEAKNESS, INCOMPLETE_UNDERSTANDING, INSUFFICIENT_EVIDENCE")
    confidence: float = 1.0
    is_addressed: bool = False
    addressed_by_answer_id: Optional[UUID] = None
    resolution_notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class StrongWeakSummary(BaseModel):
    demonstrated_strengths: List[str] = Field(default_factory=list, description="Mastery >= 75 and confidence >= 0.40")
    demonstrated_weaknesses: List[str] = Field(default_factory=list, description="Mastery < 50 and confidence >= 0.40 and evaluated")
    partially_understood: List[str] = Field(default_factory=list, description="50 <= Mastery < 75 and evaluated")
    unassessed_topics: List[str] = Field(default_factory=list, description="Resume topics not yet evaluated")
    topics_requiring_followup: List[str] = Field(default_factory=list, description="Topics with active gaps or requiring follow-up")

    model_config = ConfigDict(from_attributes=True)


class TopicCoverageSummary(BaseModel):
    total_resume_topics: int = 0
    not_considered_count: int = 0
    question_asked_count: int = 0
    partially_assessed_count: int = 0
    adequately_assessed_count: int = 0
    requires_follow_up_count: int = 0
    coverage_percentage: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class CandidateKnowledgeStateResponse(BaseModel):
    session_id: UUID
    resume_id: UUID
    topics: Dict[str, TopicMasteryState] = Field(default_factory=dict)
    projects: Dict[str, ProjectUnderstandingState] = Field(default_factory=dict)
    relationships: List[RelationshipUnderstandingState] = Field(default_factory=list)
    claims: List[ResumeClaimVerificationState] = Field(default_factory=list)
    knowledge_gaps: List[DetailedKnowledgeGap] = Field(default_factory=list)
    summary: StrongWeakSummary = Field(default_factory=StrongWeakSummary)
    coverage: TopicCoverageSummary = Field(default_factory=TopicCoverageSummary)
    last_updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
