import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.schemas.evaluation import (
    SessionScorecardSchema,
    EvaluationResponseSchema,
    DimensionalScoresSchema,
    FeedbackBreakdownSchema,
    EvaluationMetadataSchema,
    SessionProgressSchema,
)

logger = logging.getLogger(__name__)


class ScorecardAggregatorService:
    """
    Aggregates evaluated answer scores for an interview session into a deterministic,
    explainable session-level scorecard with category breakdowns and lifecycle tracking.
    """

    def calculate_session_scorecard(
        self,
        db: Session,
        session_id: UUID,
    ) -> InterviewSession:
        """
        Calculates or recalculates the scorecard for an interview session.
        Idempotent: Re-running over unchanged evaluated answers produces the exact same result.
        """
        # 1. Retrieve session
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found.")

        # 2. Retrieve questions belonging to this session
        questions = db.query(InterviewQuestion).filter(
            InterviewQuestion.session_id == session_id
        ).all()
        question_map = {q.id: q for q in questions}
        total_questions = len(questions)

        # 3. Retrieve answers linked to this session
        answers = db.query(InterviewAnswer).filter(
            InterviewAnswer.session_id == session_id
        ).all()

        # Group answers by question_id to handle duplicate submissions deterministically
        answers_by_question: Dict[UUID, List[InterviewAnswer]] = defaultdict(list)
        for ans in answers:
            if ans.question_id in question_map:
                answers_by_question[ans.question_id].append(ans)

        eligible_evaluations: Dict[UUID, InterviewAnswer] = {}

        for q_id, q_answers in answers_by_question.items():
            # Select the most recent answer record for the question (by attempt_number, created_at, with id as deterministic tie-breaker)
            sorted_ans = sorted(
                q_answers,
                key=lambda a: (
                    getattr(a, "attempt_number", 0) or 0,
                    a.created_at or datetime.min,
                    str(a.id)
                ),
                reverse=True
            )
            latest_ans = sorted_ans[0]

            # Eligible evaluations must be genuinely EVALUATED with score in [0.0, 100.0]
            if (
                latest_ans.status == "EVALUATED"
                and latest_ans.overall_score is not None
                and 0.0 <= latest_ans.overall_score <= 100.0
            ):
                eligible_evaluations[q_id] = latest_ans

        # 4. Calculate session overall score: Arithmetic mean of eligible evaluations
        if eligible_evaluations:
            scores = [ans.overall_score for ans in eligible_evaluations.values()]
            session_overall_score = round(sum(scores) / len(scores), 2)
        else:
            session_overall_score = None

        # 5. Calculate category-level scores
        category_scores: Dict[str, float] = {}
        category_groups: Dict[str, List[float]] = defaultdict(list)

        for q_id, ans in eligible_evaluations.items():
            q = question_map[q_id]
            category = (q.category or "Default").strip()
            if ans.overall_score is not None:
                category_groups[category].append(ans.overall_score)

        for cat, cat_scores in category_groups.items():
            if cat_scores:
                category_scores[cat] = round(sum(cat_scores) / len(cat_scores), 2)

        # 6. Update progress counters
        completed_questions = len(eligible_evaluations)
        session.total_questions = total_questions
        session.completed_questions = completed_questions
        session.overall_score = session_overall_score
        session.category_scores = category_scores

        # 7. Session completion rules
        # Completion criteria:
        # - total_questions > 0 (an empty session must NOT be completed)
        # - completed_questions == total_questions (all questions have valid evaluations)
        if total_questions > 0 and completed_questions == total_questions:
            session.status = "COMPLETED"
            if not session.completed_at:
                session.completed_at = datetime.now(timezone.utc)
        elif session.status == "ENDED":
            # Preserve explicit ENDED state and timestamp
            if not session.completed_at:
                session.completed_at = datetime.now(timezone.utc)
        else:
            session.status = "IN_PROGRESS"
            session.completed_at = None

        # 8. Commit and refresh
        db.commit()
        db.refresh(session)
        return session

    def format_scorecard_response(
        self,
        session: InterviewSession,
    ) -> SessionScorecardSchema:
        """
        Builds a Pydantic SessionScorecardSchema payload from an InterviewSession record.
        Only the authoritative latest attempt per question is included in the answers list.
        """
        formatted_answers: List[EvaluationResponseSchema] = []

        # Deduplicate answers to only include the authoritative latest attempt per question
        answers = session.answers or []
        answers_by_question: Dict[UUID, List[InterviewAnswer]] = defaultdict(list)
        for ans in answers:
            answers_by_question[ans.question_id].append(ans)

        latest_answers: List[InterviewAnswer] = []
        for q_id, q_answers in answers_by_question.items():
            sorted_ans = sorted(
                q_answers,
                key=lambda a: (
                    getattr(a, "attempt_number", 0) or 0,
                    a.created_at or datetime.min,
                    str(a.id)
                ),
                reverse=True
            )
            latest_answers.append(sorted_ans[0])

        # Sort the latest answers chronologically
        latest_answers.sort(key=lambda a: (a.created_at or datetime.min, str(a.id)))

        for ans in latest_answers:
            scores = None
            feedback = None

            if ans.status == "EVALUATED":
                scores = DimensionalScoresSchema(
                    technical_accuracy=ans.technical_accuracy_score if ans.technical_accuracy_score is not None else 0.0,
                    conceptual_understanding=ans.conceptual_understanding_score if ans.conceptual_understanding_score is not None else 0.0,
                    completeness=ans.completeness_score if ans.completeness_score is not None else 0.0,
                    depth_and_specificity=ans.depth_score if ans.depth_score is not None else 0.0,
                    practical_reasoning=ans.reasoning_score if ans.reasoning_score is not None else 0.0,
                    evidence_alignment=ans.evidence_alignment_score if ans.evidence_alignment_score is not None else 0.0,
                )
                feedback = FeedbackBreakdownSchema(
                    summary=ans.evaluation_feedback or "",
                    strengths=ans.strengths or [],
                    weaknesses=ans.weaknesses or [],
                    missing_concepts=ans.missing_concepts or [],
                    key_takeaways=ans.key_takeaways,
                )

            contra_data = getattr(ans, "contradiction_analysis", None)
            if contra_data is None and isinstance(ans.dimension_scores, dict):
                contra_data = ans.dimension_scores.get("_contradiction_analysis") or ans.dimension_scores.get("contradiction_analysis")

            metadata = EvaluationMetadataSchema(
                provider=ans.evaluator_provider,
                model=ans.evaluator_model,
                rubric_version=ans.rubric_version or "v1.0",
                latency_ms=ans.evaluation_latency_ms,
                contradiction_analysis=contra_data,
            )

            progress = SessionProgressSchema(
                completed_questions=session.completed_questions,
                total_questions=session.total_questions,
                session_overall_score=session.overall_score,
            )

            formatted_answers.append(
                EvaluationResponseSchema(
                    answer_id=ans.id,
                    question_id=ans.question_id,
                    session_id=ans.session_id,
                    attempt_number=getattr(ans, "attempt_number", 1) or 1,
                    candidate_answer=ans.candidate_answer,
                    status=ans.status,
                    overall_score=ans.overall_score,
                    scores=scores,
                    evaluation_confidence=ans.evaluation_confidence if ans.evaluation_confidence is not None else 1.0,
                    feedback=feedback,
                    evaluation_metadata=metadata,
                    session_progress=progress,
                    error_message=ans.error_message,
                    created_at=ans.created_at or datetime.now(timezone.utc),
                )
            )

        return SessionScorecardSchema(
            session_id=session.id,
            resume_id=session.resume_id,
            status=session.status,
            overall_score=session.overall_score,
            completed_questions=session.completed_questions,
            total_questions=session.total_questions,
            category_scores=session.category_scores or {},
            answers=formatted_answers,
            started_at=session.started_at,
            completed_at=session.completed_at,
        )


scorecard_aggregator = ScorecardAggregatorService()
