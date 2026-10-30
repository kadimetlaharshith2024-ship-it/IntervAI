import time
import logging
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.database import get_db
from app.core.deps import get_current_user_optional, verify_resume_access, verify_session_access
from app.models.user import User
from app.models.interview import InterviewAnswer, InterviewSession, InterviewQuestion
from app.schemas.evaluation import (
    AnswerSubmissionRequest,
    EvaluationResponseSchema,
    DimensionalScoresSchema,
    FeedbackBreakdownSchema,
    EvaluationMetadataSchema,
    SessionScorecardSchema,
    SessionProgressSchema,
)
from app.services.evaluation import answer_evaluator, scorecard_aggregator

router = APIRouter()


def format_evaluation_response(answer: InterviewAnswer) -> EvaluationResponseSchema:
    scores = None
    feedback = None

    if answer.status == "EVALUATED":
        scores = DimensionalScoresSchema(
            technical_accuracy=answer.technical_accuracy_score if answer.technical_accuracy_score is not None else 0.0,
            conceptual_understanding=answer.conceptual_understanding_score if answer.conceptual_understanding_score is not None else 0.0,
            completeness=answer.completeness_score if answer.completeness_score is not None else 0.0,
            depth_and_specificity=answer.depth_score if answer.depth_score is not None else 0.0,
            practical_reasoning=answer.reasoning_score if answer.reasoning_score is not None else 0.0,
            evidence_alignment=answer.evidence_alignment_score if answer.evidence_alignment_score is not None else 0.0,
        )
        feedback = FeedbackBreakdownSchema(
            summary=answer.evaluation_feedback or "",
            strengths=answer.strengths or [],
            weaknesses=answer.weaknesses or [],
            missing_concepts=answer.missing_concepts or [],
            key_takeaways=answer.key_takeaways,
        )

    contra_data = getattr(answer, "contradiction_analysis", None)
    if contra_data is None and isinstance(answer.dimension_scores, dict):
        contra_data = answer.dimension_scores.get("_contradiction_analysis") or answer.dimension_scores.get("contradiction_analysis")

    stage_latencies = None
    if isinstance(answer.dimension_scores, dict):
        stage_latencies = answer.dimension_scores.get("_stage_latencies_ms")

    metadata = EvaluationMetadataSchema(
        provider=answer.evaluator_provider,
        model=answer.evaluator_model,
        rubric_version=answer.rubric_version or "v1.0",
        latency_ms=answer.evaluation_latency_ms,
        stage_latencies_ms=stage_latencies,
        contradiction_analysis=contra_data,
    )

    progress = None
    if answer.session:
        progress = SessionProgressSchema(
            completed_questions=answer.session.completed_questions,
            total_questions=answer.session.total_questions,
            session_overall_score=answer.session.overall_score,
        )

    return EvaluationResponseSchema(
        answer_id=answer.id,
        question_id=answer.question_id,
        session_id=answer.session_id,
        attempt_number=getattr(answer, "attempt_number", 1) or 1,
        candidate_answer=answer.candidate_answer,
        status=answer.status,
        overall_score=answer.overall_score,
        scores=scores,
        evaluation_confidence=answer.evaluation_confidence if answer.evaluation_confidence is not None else 1.0,
        feedback=feedback,
        evaluation_metadata=metadata,
        session_progress=progress,
        error_message=answer.error_message,
        created_at=answer.created_at or datetime.now(timezone.utc),
    )


@router.post(
    "/submit",
    response_model=EvaluationResponseSchema,
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_EVALUATION_PER_MIN, 60))]
)
def submit_and_evaluate_answer(
    request: AnswerSubmissionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Submits a candidate answer for an interview question, runs rubric-based
    evaluation with prompt injection containment and structured output validation,
    and returns the persisted evaluation.
    """
    if request.session_id:
        verify_session_access(session_id=request.session_id, db=db, current_user=current_user)

    question = db.query(InterviewQuestion).filter(InterviewQuestion.id == request.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Interview question not found.")
    verify_resume_access(resume_id=question.resume_id, db=db, current_user=current_user)
    if question.session_id and (not request.session_id or question.session_id != request.session_id):
        verify_session_access(session_id=question.session_id, db=db, current_user=current_user)

    t_eval_start = time.perf_counter()
    answer = answer_evaluator.evaluate_answer(
        db=db,
        question_id=request.question_id,
        candidate_answer=request.candidate_answer,
        session_id=request.session_id
    )
    t_eval_ms = round((time.perf_counter() - t_eval_start) * 1000, 2)
    logger.info(
        f"[TIMING] submit_and_evaluate_answer: q_id={request.question_id} "
        f"session={request.session_id} elapsed_ms={t_eval_ms} score={answer.overall_score}"
    )
    return format_evaluation_response(answer)


@router.get("/question/{question_id}/history", response_model=List[EvaluationResponseSchema])
def get_question_attempt_history(
    question_id: UUID,
    session_id: Optional[UUID] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves the chronological attempt history for a specific interview question and session.
    Returns attempts ordered chronologically by attempt_number and creation time.
    """
    question = db.query(InterviewQuestion).filter(InterviewQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Interview question not found.")
    verify_resume_access(resume_id=question.resume_id, db=db, current_user=current_user)

    if session_id is not None:
        verify_session_access(session_id=session_id, db=db, current_user=current_user)
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if question.session_id is not None and question.session_id != session_id:
            raise HTTPException(
                status_code=400,
                detail="Interview question belongs to a different interview session."
            )
        if session.resume_id != question.resume_id:
            raise HTTPException(
                status_code=400,
                detail="Question does not belong to the candidate resume linked to this session."
            )
    elif question.session_id is not None:
        verify_session_access(session_id=question.session_id, db=db, current_user=current_user)

    query = db.query(InterviewAnswer).filter(InterviewAnswer.question_id == question_id)
    if session_id is not None:
        query = query.filter(InterviewAnswer.session_id == session_id)
    elif question.session_id is not None:
        query = query.filter(InterviewAnswer.session_id == question.session_id)
    else:
        query = query.filter(InterviewAnswer.session_id.is_(None))

    attempts = query.order_by(
        InterviewAnswer.attempt_number.asc(),
        InterviewAnswer.created_at.asc(),
        InterviewAnswer.id.asc()
    ).all()

    return [format_evaluation_response(ans) for ans in attempts]


@router.get("/session/{session_id}/scorecard", response_model=SessionScorecardSchema)
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


@router.post("/session/{session_id}/recalculate", response_model=SessionScorecardSchema)
def recalculate_session_scorecard(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Recalculates progress counters, category averages, and overall score for an interview session.
    """
    verify_session_access(session_id=session_id, db=db, current_user=current_user)
    session = scorecard_aggregator.calculate_session_scorecard(db=db, session_id=session_id)
    return scorecard_aggregator.format_scorecard_response(session)


@router.get("/{answer_id}", response_model=EvaluationResponseSchema)
def get_evaluation(
    answer_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Retrieves an existing evaluation by its answer_id.
    """
    answer = db.query(InterviewAnswer).filter(InterviewAnswer.id == answer_id).first()
    if not answer:
        raise HTTPException(status_code=404, detail="Interview answer not found.")
    if answer.session_id:
        verify_session_access(session_id=answer.session_id, db=db, current_user=current_user)
    elif answer.question:
        verify_resume_access(resume_id=answer.question.resume_id, db=db, current_user=current_user)
    return format_evaluation_response(answer)

