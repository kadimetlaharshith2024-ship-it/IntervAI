import pytest
from unittest.mock import MagicMock
from app.schemas.rag import QuestionOutputSchema
from app.services.rag.validator import grounding_validator

def test_target_entity_java_focused_question_accepted():
    """
    Verifies that when target_entity is Java and the question specifically inquires
    about Java mechanics, it passes target entity validation.
    """
    ev_text = "Backend Microservices | Java, Spring Boot, PostgreSQL, Docker."
    resolved_ev = {
        "chk_java": {"text": ev_text, "source_type": "chunk"}
    }
    q = QuestionOutputSchema(
        question=(
            "In your backend services built with Java, how would you approach designing "
            "thread-safe concurrent data pipelines in Java to minimize garbage collection pauses?"
        ),
        category="Architecture",
        difficulty=3,
        target_project="Backend Microservices",
        target_entity="Java",
        evidence_ids=["chk_java"],
        evidence_text=ev_text,
        rationale="Evaluating Java concurrency and memory management architecture.",
        grounding_score=0.9,
        relevance_score=0.9,
        technical_quality_score=0.9,
        trace_id="trc_test_java"
    )
    is_valid, reason, details = grounding_validator.validate(
        question=q,
        allowed_projects=["Backend Microservices"],
        allowed_entities=["Java", "Spring Boot", "PostgreSQL", "Docker"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is True, f"Expected Java-focused question to be accepted, but got: {reason}"
    assert details.get("code") is None


def test_target_entity_java_with_unrelated_focus_rejected():
    """
    Verifies that if target_entity is Java, but the question only mentions Java in a preamble
    and purely inquires about REST APIs or another technology, it is rejected with target_entity_focus_mismatch.
    """
    ev_text = "Full Stack Platform | Java, REST APIs, PostgreSQL."
    resolved_ev = {
        "chk_fs": {"text": ev_text, "source_type": "chunk"}
    }
    q = QuestionOutputSchema(
        question=(
            "In a stack using Java and PostgreSQL, how would you approach designing "
            "REST APIs to handle distributed rate limiting and idempotency keys?"
        ),
        category="Architecture",
        difficulty=3,
        target_project="Full Stack Platform",
        target_entity="Java",  # Mismatch: question inquires about REST APIs
        evidence_ids=["chk_fs"],
        evidence_text=ev_text,
        rationale="Evaluating REST API design.",
        grounding_score=0.85,
        relevance_score=0.85,
        technical_quality_score=0.85,
        trace_id="trc_test_mismatch"
    )
    is_valid, reason, details = grounding_validator.validate(
        question=q,
        allowed_projects=["Full Stack Platform"],
        allowed_entities=["Java", "REST APIs", "PostgreSQL"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is False
    assert details.get("code") == "target_entity_focus_mismatch"
    assert "REST APIs" in reason or "actual_focus" in details


def test_target_entity_rest_apis_accepted_when_evidence_supports_it():
    """
    Verifies that when target_entity is genuinely REST APIs and supported by evidence,
    a REST API-focused question passes validation.
    """
    ev_text = "API Gateway Project | Developed REST APIs, OAuth2, and JWT authentication."
    resolved_ev = {
        "chk_gw": {"text": ev_text, "source_type": "chunk"}
    }
    q = QuestionOutputSchema(
        question=(
            "In the API Gateway Project, how would you approach designing REST APIs "
            "to enforce rate limiting, versioning, and standardized error envelopes?"
        ),
        category="Architecture",
        difficulty=3,
        target_project="API Gateway Project",
        target_entity="REST APIs",
        evidence_ids=["chk_gw"],
        evidence_text=ev_text,
        rationale="Evaluating REST API gateway architecture.",
        grounding_score=0.95,
        relevance_score=0.95,
        technical_quality_score=0.95,
        trace_id="trc_test_rest"
    )
    is_valid, reason, details = grounding_validator.validate(
        question=q,
        allowed_projects=["API Gateway Project"],
        allowed_entities=["REST APIs", "OAuth2", "JWT"],
        resolved_evidence_items=resolved_ev,
        existing_questions=[],
        available_claims={},
        available_relationships={}
    )
    assert is_valid is True, f"Expected REST APIs question to be valid, but got: {reason}"
