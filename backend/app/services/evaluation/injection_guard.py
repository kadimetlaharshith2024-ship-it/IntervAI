import re
from typing import Tuple, List

# Heuristic regex patterns for common prompt injection, instruction override, and scoring tampering attempts.
# NOTE: Heuristic detection provides a defense-in-depth layer but does NOT guarantee 100% protection against
# novel adversarial attacks. Robust containment is achieved through XML boundary isolation, structured
# output schema validation, and deterministic backend composite score aggregation.
PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules)\b"),
    re.compile(r"(?i)\b(?:system\s+prompt\s*(?:override|injection|jailbreak|leak|bypass)|override\s+system\s+prompt|ignore\s+system\s+prompt)\b"),
    re.compile(r"(?i)\b(?:developer\s+mode\s+enabled|jailbreak\s+mode|dan\s+mode|evaluator\s+override)\b"),
    re.compile(r"(?i)\b(?:give\s+(?:me\s+)?(?:a\s+)?(?:perfect|100|100%|maximum|full)\s*(?:score|marks|rating|points)?)\b"),
    re.compile(r"(?i)\b(?:you\s+must\s+(?:rate|score|give)\s+(?:this\s+answer\s+)?(?:100|10\/10|perfect))\b"),
    re.compile(r"(?i)\b(?:output\s+(?:only\s+)?(?:json\s+with\s+)?(?:score\s*:\s*100|100\/100))\b"),
    re.compile(r"(?i)\b(?:act\s+as\s+(?:a|an)\s+(?:unrestricted\s+ai|different\s+evaluator|jailbroken\s+model)\b)"),
    re.compile(r"(?i)\b(?:reveal|print|expose|display|show|leak)\b.*?\b(?:(?:system\s+)?(?:instruction|prompt)|api\s+key|hidden\s+prompt)\b"),
]

def scan_for_prompt_injection(answer_text: str) -> Tuple[bool, List[str]]:
    """
    Heuristically scans untrusted candidate answer text for prompt injection,
    jailbreak, or scoring override markers.

    Returns:
        (is_flagged, matched_reasons)
    """
    if not answer_text:
        return False, []

    matched = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(answer_text)
        if match:
            matched.append(match.group(0).strip())

    return bool(matched), matched

def format_untrusted_answer(answer_text: str) -> str:
    """
    Safely encapsulates candidate answer text within strict XML delimiters
    to prevent the LLM from executing candidate input as system instructions.
    Defangs internal XML delimiter tags to prevent prompt boundary escape.
    """
    cleaned = (answer_text or "").strip()
    safe_text = cleaned.replace("</untrusted_candidate_answer>", "&lt;/untrusted_candidate_answer&gt;")
    safe_text = safe_text.replace("<untrusted_candidate_answer>", "&lt;untrusted_candidate_answer&gt;")
    return f"<untrusted_candidate_answer>\n{safe_text}\n</untrusted_candidate_answer>"
