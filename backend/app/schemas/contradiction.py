from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ContradictionStatus(str, Enum):
    """
    Distinguishes the categorical outcome of deterministic evidence verification.
    """
    CONTRADICTION = "CONTRADICTION"                          # Confirmed/high-confidence factual contradiction
    SUPPORTED = "SUPPORTED"                                  # Supported or conceptually aligned with verified facts
    UNVERIFIED_ADDITION = "UNVERIFIED_ADDITION"              # Auxiliary detail/technology not mentioned on resume
    HYPOTHETICAL_DISCUSSION = "HYPOTHETICAL_DISCUSSION"      # Subjunctive, hypothetical, or architectural trade-off discussion
    NO_APPLICABLE_EVIDENCE = "NO_APPLICABLE_EVIDENCE"        # Theoretical/general question with no applicable resume anchor


class ContradictionType(str, Enum):
    """
    Specific deterministic rule that triggered the conflict.
    """
    TECHNOLOGY_SUBSTITUTION = "TECHNOLOGY_SUBSTITUTION"      # Direct competing technology substituted on anchored project
    EXPLICIT_NEGATION = "EXPLICIT_NEGATION"                  # Explicit denial of a verified technology or component
    ROLE_INCOMPATIBILITY = "ROLE_INCOMPATIBILITY"            # Seniority/role level incompatible with verified employer record


class ContradictionSeverity(str, Enum):
    """
    Confidence and impact severity of a detected conflict.
    """
    HIGH = "HIGH"        # Direct factual collision on anchored project or verified role
    MEDIUM = "MEDIUM"    # Significant architectural mismatch with ambiguous boundary
    LOW = "LOW"          # Minor contextual divergence


class ConflictDetail(BaseModel):
    """
    Detailed evidence grounding for an individual detected contradiction.
    """
    conflict_type: ContradictionType
    severity: ContradictionSeverity
    candidate_assertion: str = Field(..., description="The candidate's conflicting assertion snippet")
    verified_fact: str = Field(..., description="The verified ground-truth fact from the resume or knowledge graph")
    context_project: Optional[str] = Field(None, description="The specific project or employer context anchored")
    evidence_id: Optional[str] = Field(None, description="ID of the underlying Relationship or ResumeClaim record if available")
    evidence_text: Optional[str] = Field(None, description="Raw evidence snippet from the resume")
    explanation: str = Field(..., description="Explainable rationale for why this assertion contradicts verified evidence")


class ContradictionAnalysisResult(BaseModel):
    """
    Comprehensive result of deterministic contradiction analysis for a candidate answer.
    """
    status: ContradictionStatus
    has_contradiction: bool = False
    severity: ContradictionSeverity = ContradictionSeverity.LOW
    conflicts: List[ConflictDetail] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    unverified_additions: List[str] = Field(default_factory=list)
    hypothetical_claims: List[str] = Field(default_factory=list)
    explanation: str = Field(..., description="Summary explanation of the verification outcome")
