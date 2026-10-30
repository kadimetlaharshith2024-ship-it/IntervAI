import re
import math
from typing import Dict, Any, Tuple, Optional, List

RUBRIC_VERSION = "v1.0"

DIMENSION_KEYS: List[str] = [
    "technical_accuracy",
    "conceptual_understanding",
    "completeness",
    "depth_and_specificity",
    "practical_reasoning",
    "evidence_alignment",
]

# Category-adaptive dimension weights summing exactly to 1.00
CATEGORY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Architecture": {
        "technical_accuracy": 0.20,
        "conceptual_understanding": 0.20,
        "completeness": 0.15,
        "depth_and_specificity": 0.15,
        "practical_reasoning": 0.25,
        "evidence_alignment": 0.05,
    },
    "Implementation": {
        "technical_accuracy": 0.25,
        "conceptual_understanding": 0.15,
        "completeness": 0.15,
        "depth_and_specificity": 0.25,
        "practical_reasoning": 0.15,
        "evidence_alignment": 0.05,
    },
    "Security": {
        "technical_accuracy": 0.30,
        "conceptual_understanding": 0.15,
        "completeness": 0.15,
        "depth_and_specificity": 0.20,
        "practical_reasoning": 0.15,
        "evidence_alignment": 0.05,
    },
    "Performance": {
        "technical_accuracy": 0.20,
        "conceptual_understanding": 0.15,
        "completeness": 0.15,
        "depth_and_specificity": 0.25,
        "practical_reasoning": 0.20,
        "evidence_alignment": 0.05,
    },
    "Failure-Mode": {
        "technical_accuracy": 0.20,
        "conceptual_understanding": 0.15,
        "completeness": 0.15,
        "depth_and_specificity": 0.20,
        "practical_reasoning": 0.25,
        "evidence_alignment": 0.05,
    },
    "Fundamentals": {
        "technical_accuracy": 0.35,
        "conceptual_understanding": 0.35,
        "completeness": 0.15,
        "depth_and_specificity": 0.10,
        "practical_reasoning": 0.05,
        "evidence_alignment": 0.00,
    },
    "Claim-Verification": {
        "technical_accuracy": 0.15,
        "conceptual_understanding": 0.10,
        "completeness": 0.15,
        "depth_and_specificity": 0.20,
        "practical_reasoning": 0.15,
        "evidence_alignment": 0.25,
    },
    "Default": {
        "technical_accuracy": 0.20,
        "conceptual_understanding": 0.20,
        "completeness": 0.15,
        "depth_and_specificity": 0.20,
        "practical_reasoning": 0.20,
        "evidence_alignment": 0.05,
    }
}

REFUSAL_PATTERNS = [
    re.compile(r"(?i)^(?:i\s+don['’]?t\s+know(?:\s+(?:the\s+)?answer)?|no\s+idea|not\s+sure|skip|n\/?a|none|pass|don't\s+know|idk)[\.\?\!\s]*$"),
    re.compile(r"(?i)^i\s+(?:have\s+no|don['’]?t\s+have(?:\s+any)?)\s+experience\b"),
    re.compile(r"(?i)^i\s+am\s+not\s+(?:familiar|sure|aware)\b"),
    re.compile(r"^[\.\-\s]+$"),
]

def is_trivial_or_refusal_answer(answer_text: str) -> Tuple[bool, Optional[str]]:
    """
    Detects whether an answer is an explicit refusal, empty, or trivial filler.
    Does NOT penalize concise but technically substantive answers.
    """
    cleaned = (answer_text or "").strip()
    if not cleaned or len(cleaned) < 2:
        return True, "Answer is empty or single character."

    for pat in REFUSAL_PATTERNS:
        if pat.search(cleaned):
            # If the candidate provides substantive technical content after a contrastive conjunction
            # (e.g. "I am not familiar with Kafka, but in our project we used RabbitMQ with exchange routing..."),
            # do not treat as an outright refusal.
            if re.search(r"(?i)\b(?:but|however|although|instead|nevertheless)\b", cleaned):
                after_conj = re.split(r"(?i)\b(?:but|however|although|instead|nevertheless)\b", cleaned, maxsplit=1)[-1].strip()
                if len(after_conj.split()) >= 6:
                    return False, None
            return True, "Candidate explicitly stated lack of knowledge or requested to skip."

    return False, None

def normalize_score_value(val: Any) -> float:
    """Normalizes score to float in [0.0, 100.0]. Strictly rejects Booleans, NaN, and Inf."""
    if isinstance(val, bool):
        raise ValueError(f"Score value '{val}' cannot be a Boolean.")

    try:
        num = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Score value '{val}' is not a valid number.")

    if math.isnan(num) or math.isinf(num):
        raise ValueError(f"Score value '{val}' cannot be NaN or infinite.")

    if num < 0.0 or num > 100.0:
        raise ValueError(f"Score value {num} is outside the valid [0.0, 100.0] range.")

    return round(num, 2)

def calculate_composite_score(category: str, raw_dimensions: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the deterministic composite overall_score from raw dimension scores.

    Returns:
        (composite_score, normalized_scores_dict)
    """
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["Default"])
    normalized: Dict[str, float] = {}

    for dim in DIMENSION_KEYS:
        if dim not in raw_dimensions:
            raise ValueError(f"Missing required dimensional score '{dim}' for category '{category}'.")
        normalized[dim] = normalize_score_value(raw_dimensions[dim])

    composite = sum(normalized[dim] * weights[dim] for dim in DIMENSION_KEYS)
    overall_score = round(max(0.0, min(100.0, composite)), 2)
    return overall_score, normalized

def calculate_evaluation_confidence(
    answer_text: str,
    question_category: str,
    dimensional_scores: Dict[str, float],
    expected_concepts_coverage: Optional[float] = None,
    is_refusal: bool = False
) -> float:
    """
    Computes an evaluation confidence score (0.0 to 1.0) representing the reliability
    and certainty of the evaluation itself (NOT the candidate's skill level).

    Factors:
    - Explicit refusal / fast-path: 1.0 (high certainty of refusal).
    - Answer length adequacy: Extremely brief answers (< 10 words) have lower evaluation certainty.
    - Dimensional completeness and variance.
    - Expected concepts coverage if concepts were specified.
    """
    if is_refusal:
        return 1.0

    words = len((answer_text or "").split())
    if words == 0:
        return 0.0

    # Length adequacy factor (concise answers with 15+ words receive full length factor)
    if words >= 25:
        len_factor = 1.0
    elif words >= 15:
        len_factor = 0.90
    elif words >= 8:
        len_factor = 0.80
    else:
        len_factor = 0.65

    # Dimensional completeness factor (verifying all 6 dimensions are populated)
    scores = [dimensional_scores.get(k, 0.0) for k in DIMENSION_KEYS]
    dim_factor = 1.0 if len(scores) == 6 else 0.50

    # Concepts coverage factor
    if expected_concepts_coverage is not None:
        concept_factor = max(0.5, min(1.0, float(expected_concepts_coverage)))
    else:
        concept_factor = 1.0

    confidence = round(min(1.0, len_factor * 0.50 + dim_factor * 0.30 + concept_factor * 0.20), 4)
    return max(0.10, confidence)
