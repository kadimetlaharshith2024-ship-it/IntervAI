from typing import Dict, Any, Tuple, Optional, List
from app.services.evaluation.rubric import DIMENSION_KEYS, calculate_composite_score, calculate_evaluation_confidence

REQUIRED_FEEDBACK_KEYS = ["summary", "strengths", "weaknesses", "missing_concepts"]

def validate_raw_evaluation_output(
    raw_output: Dict[str, Any],
    question_category: str,
    answer_text: str,
    expected_concepts: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates structured LLM evaluation output for schema adherence, numeric bounds,
    required feedback fields, and computes deterministic composite scores.

    Returns:
        (is_valid, error_message, validated_payload)
    """
    if not isinstance(raw_output, dict):
        return False, f"Evaluator output must be a JSON dictionary, got {type(raw_output).__name__}.", None

    # 1. Extract and validate dimensional scores
    scores_raw = raw_output.get("scores") or raw_output.get("dimensional_scores")
    if not isinstance(scores_raw, dict):
        return False, "Missing 'scores' dictionary in evaluator output.", None

    missing_dims = [dim for dim in DIMENSION_KEYS if dim not in scores_raw]
    if missing_dims:
        return False, f"Missing required dimensional scores: {missing_dims}", None

    try:
        overall_score, normalized_scores = calculate_composite_score(question_category, scores_raw)
    except ValueError as e:
        return False, f"Invalid dimensional score value: {str(e)}", None

    # 2. Extract and validate feedback breakdown
    feedback_raw = raw_output.get("feedback")
    if not isinstance(feedback_raw, dict):
        # Fallback: check if feedback fields are at root level
        feedback_raw = {k: raw_output.get(k) for k in REQUIRED_FEEDBACK_KEYS if k in raw_output}
        if "summary" not in feedback_raw and "evaluation_feedback" in raw_output:
            feedback_raw["summary"] = raw_output["evaluation_feedback"]

    summary = str(feedback_raw.get("summary") or "").strip()
    if not summary:
        return False, "Evaluation summary is missing or empty.", None

    strengths = feedback_raw.get("strengths") or []
    weaknesses = feedback_raw.get("weaknesses") or []
    missing_concepts = feedback_raw.get("missing_concepts") or []
    key_takeaways = feedback_raw.get("key_takeaways")

    if not isinstance(strengths, list) or not isinstance(weaknesses, list) or not isinstance(missing_concepts, list):
        return False, "'strengths', 'weaknesses', and 'missing_concepts' must be lists of strings.", None

    # Clean list items
    clean_strengths = [str(s).strip() for s in strengths if str(s).strip()]
    clean_weaknesses = [str(w).strip() for w in weaknesses if str(w).strip()]
    clean_missing = [str(m).strip() for m in missing_concepts if str(m).strip()]

    # 3. Calculate evaluation confidence
    concept_cov = None
    if expected_concepts:
        # Check ratio of expected concepts mentioned or evaluated
        concept_cov = 1.0 - (len(clean_missing) / max(1, len(expected_concepts)))

    confidence = calculate_evaluation_confidence(
        answer_text=answer_text,
        question_category=question_category,
        dimensional_scores=normalized_scores,
        expected_concepts_coverage=concept_cov
    )

    validated_payload = {
        "overall_score": overall_score,
        "scores": normalized_scores,
        "evaluation_confidence": confidence,
        "feedback": {
            "summary": summary,
            "strengths": clean_strengths,
            "weaknesses": clean_weaknesses,
            "missing_concepts": clean_missing,
            "key_takeaways": str(key_takeaways).strip() if key_takeaways else None
        }
    }

    return True, None, validated_payload
