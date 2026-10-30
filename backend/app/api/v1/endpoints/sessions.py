from uuid import UUID
from typing import Optional, Dict, List
from fastapi import APIRouter, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.database import get_db
from app.core.deps import (
    get_current_user,
    get_current_user_optional,
    verify_resume_access,
    verify_session_access,
)
from app.models.user import User
from app.models.interview import InterviewSession
from app.schemas.session import (
    SessionCreateRequest,
    StartInterviewRequest,
    EndInterviewRequest,
    UpdateDifficultyRequest,
    SessionResponse,
    InterviewStateResponse,
    SessionQuestionAssociateRequest,
    SessionQuestionAssociateResponse,
    SessionQuestionsResponse,
    SessionQuestionItem,
)
from app.schemas.evaluation import SessionScorecardSchema
from app.schemas.knowledge_state import (
    CandidateKnowledgeStateResponse,
    TopicMasteryState,
    ProjectUnderstandingState,
    RelationshipUnderstandingState,
    ResumeClaimVerificationState,
    DetailedKnowledgeGap,
)
from app.schemas.adaptive_selection import (
    NextQuestionSelectionResult,
    AdaptiveQuestionSelectionRequest,
    AdaptiveNextQuestionResponse,
)
from app.services.session_service import session_service
from app.services.evaluation.scorecard import scorecard_aggregator

router = APIRouter()


@router.get("", response_model=List[SessionResponse])
@router.get("/", response_model=List[SessionResponse], include_in_schema=False)
def list_user_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Lists all interview sessions belonging to the currently authenticated user.
    """
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )
    return [session_service.get_session(s.id, db) for s in sessions]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
def create_session(
    request: SessionCreateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Creates a new interview session for an existing resume record.
    Enforces that caller has access to the resume and associates session ownership.
    """
    verify_resume_access(resume_id=request.resume_id, db=db, current_user=current_user)
    session = session_service.create_session(
        resume_id=request.resume_id,
        db=db,
        initial_difficulty=request.initial_difficulty or 3,
        session_metadata=request.session_metadata,
        user_id=current_user.id if current_user else None
    )
    return session


@router.post("/start", response_model=SessionResponse, status_code=status.HTTP_200_OK)
def start_interview(
    request: StartInterviewRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Starts or resumes an interview session.
    Idempotent: If an active session already exists for the resume, returns the existing active session.
    """
    if request.resume_id:
        verify_resume_access(resume_id=request.resume_id, db=db, current_user=current_user)
    if request.session_id:
        verify_session_access(session_id=request.session_id, db=db, current_user=current_user)

    session = session_service.start_interview(
        db=db,
        resume_id=request.resume_id,
        session_id=request.session_id,
        initial_difficulty=request.initial_difficulty or 3,
        session_metadata=request.session_metadata,
        user_id=current_user.id if current_user else None
    )
    return session


@router.post("/{session_id}/start", response_model=SessionResponse, status_code=status.HTTP_200_OK)
def start_interview_by_id(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Explicitly starts or reopens an interview session by session ID.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    session = session_service.start_interview(
        db=db,
        session_id=session_id,
        user_id=current_user.id if current_user else None
    )
    return session


@router.post("/{session_id}/end", response_model=SessionResponse, status_code=status.HTTP_200_OK)
def end_interview(
    session_id: UUID,
    request: Optional[EndInterviewRequest] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Ends an interview session, calculates final scorecard aggregation,
    and sets status to COMPLETED (if all questions evaluated) or ENDED.
    Idempotent: Repeated calls return the concluded session safely.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    reason = request.reason if request and request.reason else "USER_COMPLETED"
    notes = request.feedback_notes if request else None
    session = session_service.end_interview(
        session_id=session_id,
        db=db,
        reason=reason,
        feedback_notes=notes
    )
    return session


@router.get("/{session_id}/state", response_model=InterviewStateResponse)
def get_interview_state(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves the complete production interview state:
    - Current difficulty and session status
    - Chronological list of asked questions with evaluation statuses and attempts
    - Tested skills, technologies, relationships, and claims
    - Structured knowledge gaps with severity ratings
    - Comprehensive progress counters and duration
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    return session_service.get_interview_state(session_id=session_id, db=db)


@router.patch("/{session_id}/difficulty", response_model=SessionResponse)
def update_session_difficulty(
    session_id: UUID,
    request: UpdateDifficultyRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Updates the adaptive difficulty level (1 to 5) for the interview session.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    session = session_service.update_difficulty(
        session_id=session_id,
        new_difficulty=request.difficulty,
        db=db
    )
    return session


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves interview session metadata and progress status.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    return session_service.get_session(session_id=session_id, db=db)


@router.post("/{session_id}/questions", response_model=SessionQuestionAssociateResponse)
def associate_question_to_session(
    session_id: UUID,
    request: SessionQuestionAssociateRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Associates an interview question with the specified interview session.
    Enforces resume alignment and prevents reassignment across sessions.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    question = session_service.associate_question(
        session_id=session_id,
        question_id=request.question_id,
        db=db
    )
    return SessionQuestionAssociateResponse(
        session_id=session_id,
        question_id=question.id,
        resume_id=question.resume_id,
        status="ASSOCIATED",
        message="Question successfully associated with interview session."
    )


@router.post("/{session_id}/questions/{question_id}", response_model=SessionQuestionAssociateResponse)
def associate_question_by_path(
    session_id: UUID,
    question_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Convenience endpoint to associate an interview question with a session via path parameter.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    question = session_service.associate_question(
        session_id=session_id,
        question_id=question_id,
        db=db
    )
    return SessionQuestionAssociateResponse(
        session_id=session_id,
        question_id=question.id,
        resume_id=question.resume_id,
        status="ASSOCIATED",
        message="Question successfully associated with interview session."
    )


@router.get("/{session_id}/questions", response_model=SessionQuestionsResponse)
def get_session_questions(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Lists all interview questions linked to the specified session.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    questions = session_service.get_session_questions(session_id=session_id, db=db)
    return SessionQuestionsResponse(
        session_id=session_id,
        total_questions=len(questions),
        questions=[
            SessionQuestionItem(
                id=q.id,
                question_text=q.question_text,
                category=q.category,
                difficulty=q.difficulty,
                target_project=q.target_project,
                target_entity=q.target_entity,
                evidence_text=q.evidence_text,
                grounding_score=q.grounding_score,
                is_follow_up=bool(q.parent_question_id),
                parent_question_id=q.parent_question_id,
                assessed_difficulty=q.assessed_difficulty or q.difficulty,
                rubric_version=q.rubric_version or "v2.0-canonical-5level",
                difficulty_rationale=(q.validation_details or {}).get("difficulty_rationale") if isinstance(q.validation_details, dict) else None,
                grounding_status=(q.validation_details or {}).get("grounding_status") if isinstance(q.validation_details, dict) else "VERIFIED",
                evidence_ids=q.evidence_ids or [],
                created_at=q.created_at,
            )
            for q in questions
        ]
    )


@router.get("/{session_id}/scorecard", response_model=SessionScorecardSchema)
def get_session_scorecard(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves the aggregated session scorecard, category breakdowns, and answer evaluations.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    session = scorecard_aggregator.calculate_session_scorecard(db=db, session_id=session_id)
    return scorecard_aggregator.format_scorecard_response(session)


@router.get("/{session_id}/knowledge-state", response_model=CandidateKnowledgeStateResponse)
def get_session_knowledge_state(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves the complete Candidate Knowledge State (Phase 7.2):
    - Skill and technology mastery with recency-weighted scores and bounded confidence
    - Project architecture and implementation mastery
    - Extracted relationship testing statuses
    - Resume claim verification tracking (SUPPORTED, CONTRADICTED, etc.)
    - Detailed knowledge gaps with severity ratings and resolution statuses
    - Derived strengths, weaknesses, and unassessed topics summary
    - Separate topic coverage tracking
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    return session_service.get_candidate_knowledge_state(session_id=session_id, db=db)


@router.get("/{session_id}/knowledge-state/topics", response_model=Dict[str, TopicMasteryState])
def get_session_knowledge_state_topics(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves topic mastery and coverage states for all skills and technologies.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    state = session_service.get_candidate_knowledge_state(session_id=session_id, db=db)
    return state.topics


@router.get("/{session_id}/knowledge-state/projects", response_model=Dict[str, ProjectUnderstandingState])
def get_session_knowledge_state_projects(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves project understanding states and demonstrated architectural/implementation mastery.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    state = session_service.get_candidate_knowledge_state(session_id=session_id, db=db)
    return state.projects


@router.get("/{session_id}/knowledge-state/relationships", response_model=List[RelationshipUnderstandingState])
def get_session_knowledge_state_relationships(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves relationship understanding states (tested, untested, demonstrated, misunderstood).
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    state = session_service.get_candidate_knowledge_state(session_id=session_id, db=db)
    return state.relationships


@router.get("/{session_id}/knowledge-state/claims", response_model=List[ResumeClaimVerificationState])
def get_session_knowledge_state_claims(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves resume claim verification states (UNTESTED, SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE).
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    state = session_service.get_candidate_knowledge_state(session_id=session_id, db=db)
    return state.claims


@router.get("/{session_id}/knowledge-state/gaps", response_model=List[DetailedKnowledgeGap])
def get_session_knowledge_state_gaps(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves structured knowledge gaps with severity ratings and resolution statuses.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    state = session_service.get_candidate_knowledge_state(session_id=session_id, db=db)
    return state.knowledge_gaps


@router.post("/{session_id}/knowledge-state/reconcile", response_model=CandidateKnowledgeStateResponse)
def reconcile_session_knowledge_state(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Reconciles and reconstructs the Candidate Knowledge State from scratch by cleanly replaying
    all evaluated answers and attempts in chronological order.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    return session_service.reconcile_candidate_knowledge_state(session_id=session_id, db=db)


@router.post(
    "/{session_id}/next-question",
    response_model=AdaptiveNextQuestionResponse,
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_NEXT_QUESTION_PER_MIN, 60))]
)
def select_and_generate_next_question(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    request: Optional[AdaptiveQuestionSelectionRequest] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Selects and generates the next adaptive interview question (Phase 7.3).
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    return session_service.select_and_generate_next_question(
        session_id=session_id,
        db=db,
        request_override=request,
        background_tasks=background_tasks
    )


@router.get("/{session_id}/next-question/preview", response_model=NextQuestionSelectionResult)
def preview_next_question_selection(
    session_id: UUID,
    preferred_category: Optional[str] = None,
    max_consecutive_followups: Optional[int] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Previews the deterministic target, strategy, and rationale for the next question.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    req_override = None
    if preferred_category or max_consecutive_followups is not None:
        req_override = AdaptiveQuestionSelectionRequest(
            preferred_category=preferred_category,
            max_consecutive_followups=max_consecutive_followups
        )
    return session_service.preview_next_question_selection(
        session_id=session_id,
        db=db,
        request_override=req_override
    )
