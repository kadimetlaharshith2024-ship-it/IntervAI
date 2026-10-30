import pytest
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError

from app.models.interview import InterviewAnswer, InterviewSession, InterviewQuestion
from app.schemas.evaluation import (
    AnswerSubmissionRequest,
    DimensionalScoresSchema,
    FeedbackBreakdownSchema,
    EvaluationResponseSchema,
    SessionScorecardSchema,
    EvaluationMetadataSchema
)


def test_new_interview_answer_defaults_to_pending():
    """Confirms that newly created interview answers default to PENDING status with null overall score."""
    qid = uuid4()
    answer = InterviewAnswer(
        question_id=qid,
        candidate_answer="In Spring Boot, dependency injection is managed by the IoC container via @Autowired or constructor injection."
    )
    assert answer.status == "PENDING"
    assert answer.overall_score is None
    assert answer.technical_accuracy_score is None
    assert answer.conceptual_understanding_score is None
    assert answer.completeness_score is None
    assert answer.depth_score is None
    assert answer.reasoning_score is None
    assert answer.evidence_alignment_score is None
    assert answer.evaluation_confidence == 1.0
    assert answer.error_message is None


def test_failed_evaluation_cannot_retain_fabricated_score():
    """Verifies that an evaluation failure sets status to FAILED, records the error, and does not fabricate a score."""
    qid = uuid4()
    answer_id = uuid4()
    failed_answer = InterviewAnswer(
        id=answer_id,
        question_id=qid,
        candidate_answer="Some technical answer text.",
        status="FAILED",
        overall_score=None,
        error_message="Provider timeout after 30s during structured evaluation."
    )
    assert failed_answer.status == "FAILED"
    assert failed_answer.overall_score is None
    assert failed_answer.error_message == "Provider timeout after 30s during structured evaluation."

    # Validate Pydantic response schema serialization
    response_schema = EvaluationResponseSchema(
        answer_id=failed_answer.id,
        question_id=failed_answer.question_id,
        status=failed_answer.status,
        overall_score=failed_answer.overall_score,
        scores=None,
        evaluation_confidence=0.0,
        feedback=None,
        error_message=failed_answer.error_message,
        created_at=datetime.now()
    )
    assert response_schema.status == "FAILED"
    assert response_schema.overall_score is None
    assert response_schema.scores is None
    assert response_schema.error_message is not None


def test_answer_submission_request_validation():
    """Tests validation of candidate answer input boundaries."""
    qid = uuid4()

    # Valid submission
    req = AnswerSubmissionRequest(
        question_id=qid,
        candidate_answer="This is a valid answer explaining microservices communication using REST and gRPC."
    )
    assert req.candidate_answer.startswith("This is a valid answer")

    # Rejection of whitespace-only answer
    with pytest.raises(ValidationError):
        AnswerSubmissionRequest(
            question_id=qid,
            candidate_answer="     \n\t  "
        )

    # Rejection of empty string
    with pytest.raises(ValidationError):
        AnswerSubmissionRequest(
            question_id=qid,
            candidate_answer=""
        )

    # Rejection of oversized answer (> 4000 chars)
    oversized = "a" * 4001
    with pytest.raises(ValidationError):
        AnswerSubmissionRequest(
            question_id=qid,
            candidate_answer=oversized
        )


def test_dimensional_scores_schema_boundaries():
    """Verifies that all dimensional scores adhere to the 0-100 scale."""
    valid_scores = DimensionalScoresSchema(
        technical_accuracy=85.0,
        conceptual_understanding=90.0,
        completeness=80.0,
        depth_and_specificity=75.0,
        practical_reasoning=88.0,
        evidence_alignment=100.0
    )
    assert valid_scores.technical_accuracy == 85.0

    # Out of bounds (< 0)
    with pytest.raises(ValidationError):
        DimensionalScoresSchema(
            technical_accuracy=-5.0,
            conceptual_understanding=90.0,
            completeness=80.0,
            depth_and_specificity=75.0,
            practical_reasoning=88.0,
            evidence_alignment=100.0
        )

    # Out of bounds (> 100)
    with pytest.raises(ValidationError):
        DimensionalScoresSchema(
            technical_accuracy=105.0,
            conceptual_understanding=90.0,
            completeness=80.0,
            depth_and_specificity=75.0,
            practical_reasoning=88.0,
            evidence_alignment=100.0
        )


def test_session_scorecard_serialization():
    """Tests session scorecard schema serialization and category aggregation."""
    sid = uuid4()
    rid = uuid4()
    scorecard = SessionScorecardSchema(
        session_id=sid,
        resume_id=rid,
        status="IN_PROGRESS",
        overall_score=86.5,
        completed_questions=2,
        total_questions=5,
        category_scores={"Architecture": 88.0, "Implementation": 85.0},
        answers=[]
    )
    assert scorecard.overall_score == 86.5
    assert scorecard.category_scores["Architecture"] == 88.0
    assert scorecard.completed_questions == 2
