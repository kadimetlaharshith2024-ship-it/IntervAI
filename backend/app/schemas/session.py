from uuid import UUID
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SessionCreateRequest(BaseModel):
    resume_id: UUID = Field(..., description="ID of the resume for which to start an interview session")
    initial_difficulty: Optional[int] = Field(default=3, ge=1, le=5, description="Initial adaptive difficulty level (1 to 5)")
    session_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom metadata or configuration for the session")


class StartInterviewRequest(BaseModel):
    resume_id: Optional[UUID] = Field(None, description="ID of the resume to start a session for (if creating a new session)")
    session_id: Optional[UUID] = Field(None, description="Existing session ID to start/resume")
    initial_difficulty: Optional[int] = Field(default=3, ge=1, le=5, description="Initial difficulty level (1 to 5)")
    session_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom session metadata")


class EndInterviewRequest(BaseModel):
    reason: Optional[str] = Field(default="USER_COMPLETED", description="Reason for ending session ('USER_COMPLETED', 'EARLY_EXIT', 'TIMEOUT', etc.)")
    feedback_notes: Optional[str] = Field(default=None, description="Optional candidate notes or debrief comments")


class UpdateDifficultyRequest(BaseModel):
    difficulty: int = Field(..., ge=1, le=5, description="Target difficulty level (1 to 5)")


class SessionQuestionAssociateRequest(BaseModel):
    question_id: UUID = Field(..., description="ID of the interview question to associate with this session")


class SessionResponse(BaseModel):
    id: UUID
    resume_id: UUID
    status: str = Field(default="IN_PROGRESS", description="'IN_PROGRESS', 'COMPLETED', or 'ENDED'")
    current_difficulty: int = Field(default=3, ge=1, le=5, description="Current difficulty level")
    overall_score: Optional[float] = None
    completed_questions: int = 0
    total_questions: int = 0
    category_scores: Dict[str, float] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    session_state: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class SessionQuestionItem(BaseModel):
    id: UUID
    question_text: str
    category: str
    difficulty: int
    target_project: Optional[str] = None
    target_entity: Optional[str] = None
    evidence_text: Optional[str] = None
    grounding_score: float = 1.0
    is_follow_up: bool = False
    parent_question_id: Optional[UUID] = None
    assessed_difficulty: Optional[int] = None
    rubric_version: Optional[str] = "v2.0-canonical-5level"
    difficulty_rationale: Optional[str] = None
    grounding_status: Optional[str] = "PASSED"
    evidence_ids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionQuestionsResponse(BaseModel):
    session_id: UUID
    total_questions: int
    questions: List[SessionQuestionItem]


class SessionQuestionAssociateResponse(BaseModel):
    session_id: UUID
    question_id: UUID
    resume_id: UUID
    status: str
    message: str


class AskedQuestionState(BaseModel):
    question_id: UUID
    question_text: str
    category: str
    difficulty: int
    target_project: Optional[str] = None
    target_entity: Optional[str] = None
    target_relationship: Optional[str] = None
    targeted_claim: Optional[str] = None
    expected_concepts: List[str] = Field(default_factory=list)
    status: str = Field(default="UNANSWERED", description="'UNANSWERED', 'PENDING', 'EVALUATED', or 'FAILED'")
    attempt_count: int = 0
    latest_score: Optional[float] = None
    latest_feedback_summary: Optional[str] = None
    is_follow_up: bool = False
    parent_question_id: Optional[UUID] = None
    assessed_difficulty: Optional[int] = None
    rubric_version: Optional[str] = "v2.0-canonical-5level"
    difficulty_rationale: Optional[str] = None
    grounding_status: Optional[str] = "PASSED"
    evidence_ids: List[str] = Field(default_factory=list)
    asked_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class KnowledgeGapItem(BaseModel):
    question_id: UUID
    category: str
    topic: Optional[str] = None
    missing_concepts: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    severity: str = Field(default="MEDIUM", description="'HIGH', 'MEDIUM', or 'LOW'")
    score: Optional[float] = None


class InterviewProgressState(BaseModel):
    completed_questions: int = 0
    total_questions: int = 0
    progress_percentage: float = 0.0
    overall_score: Optional[float] = None
    category_scores: Dict[str, float] = Field(default_factory=dict)
    status: str = "IN_PROGRESS"
    duration_seconds: Optional[float] = None


class InterviewStateResponse(BaseModel):
    session_id: UUID
    resume_id: UUID
    status: str
    current_difficulty: int = 3
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: InterviewProgressState
    asked_questions: List[AskedQuestionState] = Field(default_factory=list)
    tested_skills: List[str] = Field(default_factory=list)
    tested_technologies: List[str] = Field(default_factory=list)
    tested_relationships: List[str] = Field(default_factory=list)
    tested_claims: List[str] = Field(default_factory=list)
    tested_projects: List[str] = Field(default_factory=list)
    knowledge_gaps: List[KnowledgeGapItem] = Field(default_factory=list)
    knowledge_state: Optional[Dict[str, Any]] = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)
