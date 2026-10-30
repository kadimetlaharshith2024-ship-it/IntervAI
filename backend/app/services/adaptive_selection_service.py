import logging
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.canonicalizer import canonicalize_entity
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim, ResumeChunk
from app.schemas.adaptive_selection import (
    AdaptiveStrategyType,
    NextQuestionSelectionResult,
    AdaptiveQuestionSelectionRequest,
    AdaptiveNextQuestionResponse,
)
from app.schemas.rag import QuestionGenerationRequest, QuestionGenerationResponse
from app.schemas.knowledge_state import (
    TopicCoverageStatus,
    ClaimVerificationStatus,
    TopicAssessmentStatus,
    GapNature,
    GapSeverity,
)
from app.services.knowledge_state_service import knowledge_state_service, JUNK_TOPIC_BLOCKLIST
from app.services.rag.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSelectionConfig:
    """Configurable parameters governing depth vs breadth balancing, score thresholds, and limits."""
    max_consecutive_followups: int = 2
    max_questions_per_topic: int = 3
    max_questions_per_project: int = 4
    default_max_session_questions: int = 10
    prefetch_enabled: bool = False

    # Score thresholds on a 0-100 scale (Phase 7.4 Score-Based Adaptation)
    high_performance_threshold: float = 75.0
    low_performance_threshold: float = 50.0
    gap_resolution_threshold: float = 75.0

    # Evaluation confidence thresholds (0.0 to 1.0)
    min_confidence_for_escalation: float = 0.65
    min_confidence_for_deescalation: float = 0.50

    # Difficulty bounds
    min_difficulty: int = 1
    max_difficulty: int = 5

    # Part B1 & B7: Quotas, rolling window, and tunable thresholds
    rolling_window_size: int = 3
    claim_verification_quota_percent: float = 0.20
    project_depth_quota_percent: float = 0.20
    duplicate_similarity_threshold: float = 0.82
    grounding_score_minimum: float = 0.65
    claim_similarity_minimum: float = 0.35
    max_candidate_retries: int = 4
    candidate_timeout_seconds: float = 25.0

    @classmethod
    def from_session_metadata(cls, metadata: Optional[Dict[str, Any]] = None) -> "AdaptiveSelectionConfig":
        if not metadata:
            return cls(
                high_performance_threshold=settings.ADAPTIVE_HIGH_SCORE_THRESHOLD,
                low_performance_threshold=settings.ADAPTIVE_LOW_SCORE_THRESHOLD,
                rolling_window_size=settings.ADAPTIVE_ROLLING_WINDOW_SIZE,
                claim_verification_quota_percent=settings.ADAPTIVE_CLAIM_QUOTA_PERCENT,
                project_depth_quota_percent=settings.ADAPTIVE_PROJECT_QUOTA_PERCENT,
                duplicate_similarity_threshold=settings.RAG_DUPLICATE_SIMILARITY_THRESHOLD,
                grounding_score_minimum=settings.RAG_GROUNDING_SCORE_MINIMUM,
                claim_similarity_minimum=settings.RAG_CLAIM_SIMILARITY_MINIMUM,
                prefetch_enabled=settings.PREFETCH_ENABLED,
                max_candidate_retries=settings.ADAPTIVE_MAX_CANDIDATE_RETRIES,
                candidate_timeout_seconds=settings.ADAPTIVE_CANDIDATE_TIMEOUT_SECONDS,
            )
        cfg = metadata.get("adaptive_config") or {}
        high_th = cfg.get("high_performance_threshold", cfg.get("high_score_threshold", settings.ADAPTIVE_HIGH_SCORE_THRESHOLD))
        if high_th is not None and 0.0 < high_th <= 10.0:
            high_th *= 10.0
        low_th = cfg.get("low_performance_threshold", cfg.get("low_score_threshold", settings.ADAPTIVE_LOW_SCORE_THRESHOLD))
        if low_th is not None and 0.0 < low_th <= 10.0:
            low_th *= 10.0
        prefetch_flag = cfg.get("prefetch_enabled", metadata.get("prefetch_enabled", settings.PREFETCH_ENABLED))
        return cls(
            max_consecutive_followups=cfg.get("max_consecutive_followups", 2),
            max_questions_per_topic=cfg.get("max_questions_per_topic", 3),
            max_questions_per_project=cfg.get("max_questions_per_project", 4),
            default_max_session_questions=cfg.get("max_questions", cfg.get("max_session_questions", 10)),
            high_performance_threshold=high_th,
            low_performance_threshold=low_th,
            gap_resolution_threshold=cfg.get("gap_resolution_threshold", 75.0),
            min_confidence_for_escalation=cfg.get("min_confidence_for_escalation", 0.65),
            min_confidence_for_deescalation=cfg.get("min_confidence_for_deescalation", 0.50),
            min_difficulty=cfg.get("min_difficulty", 1),
            max_difficulty=cfg.get("max_difficulty", 5),
            rolling_window_size=cfg.get("rolling_window_size", settings.ADAPTIVE_ROLLING_WINDOW_SIZE),
            claim_verification_quota_percent=cfg.get("claim_verification_quota_percent", settings.ADAPTIVE_CLAIM_QUOTA_PERCENT),
            project_depth_quota_percent=cfg.get("project_depth_quota_percent", settings.ADAPTIVE_PROJECT_QUOTA_PERCENT),
            duplicate_similarity_threshold=cfg.get("duplicate_similarity_threshold", settings.RAG_DUPLICATE_SIMILARITY_THRESHOLD),
            grounding_score_minimum=cfg.get("grounding_score_minimum", settings.RAG_GROUNDING_SCORE_MINIMUM),
            claim_similarity_minimum=cfg.get("claim_similarity_minimum", settings.RAG_CLAIM_SIMILARITY_MINIMUM),
            prefetch_enabled=bool(prefetch_flag),
            max_candidate_retries=cfg.get("max_candidate_retries", settings.ADAPTIVE_MAX_CANDIDATE_RETRIES),
            candidate_timeout_seconds=cfg.get("candidate_timeout_seconds", settings.ADAPTIVE_CANDIDATE_TIMEOUT_SECONDS),
        )


class AdaptiveSelectionEngine:
    """
    Adaptive Next-Question Selection Engine (Phases 7.3 & 7.4).
    A deterministic, explainable selector that balances depth and breadth using candidate
    knowledge state, demonstrated gaps, claim verification, relationship reasoning,
    score-based adaptive difficulty, confidence guards, and duplicate-prevention guardrails.
    """

    def determine_next_question_target(
        self,
        session_id: UUID,
        db: Session,
        config_override: Optional[AdaptiveSelectionConfig] = None,
        request_override: Optional[AdaptiveQuestionSelectionRequest] = None,
        excluded_topics: Optional[Set[str]] = None
    ) -> NextQuestionSelectionResult:
        """
        Determines the optimal next question target, intent, category, difficulty,
        and follow-up links deterministically without generating text or modifying DB.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found.")

        end_reason = (session.session_state or {}).get("end_reason")
        if session.status == "COMPLETED" and not end_reason:
            session.status = "IN_PROGRESS"
            session.completed_at = None
            db.flush()

        if session.status != "IN_PROGRESS":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot select next question: session is already {session.status}."
            )

        config = config_override or AdaptiveSelectionConfig.from_session_metadata(
            (session.session_state or {}).get("metadata")
        )
        if request_override and request_override.max_consecutive_followups is not None:
            config.max_consecutive_followups = request_override.max_consecutive_followups

        existing_q_count = (
            db.query(InterviewQuestion)
            .filter(InterviewQuestion.session_id == session_id)
            .count()
        )
        if existing_q_count >= config.default_max_session_questions:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot preview next question: session reached maximum question budget of {config.default_max_session_questions}."
            )

        def is_excluded(item: Optional[str]) -> bool:
            if not item or not excluded_topics:
                return False
            clean_item = item.strip().lower()
            return any(clean_item == ex.strip().lower() for ex in excluded_topics if ex)

        # 1. Retrieve knowledge state
        k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db)
        topics_map = k_state.get("topics", {})
        projects_map = k_state.get("projects", {})
        relationships_list = k_state.get("relationships", [])
        claims_list = k_state.get("claims", [])
        knowledge_gaps = k_state.get("knowledge_gaps", [])

        # TASK 3: Evidence gate - Preload chunks to verify project/experience backing
        from app.core.canonicalizer import CANONICAL_TAXONOMY, ALIAS_INDEX
        import re

        PROJECT_EXP_SECTIONS = {
            "projects", "project", "experience", "work_experience", 
            "employment", "professional_experience", "work", "work experience"
        }
        
        resume_chunks = (
            db.query(ResumeChunk)
            .filter(ResumeChunk.resume_id == session.resume_id)
            .all()
        )
        proj_exp_chunks = [
            c for c in resume_chunks
            if (c.section and c.section.strip().lower() in PROJECT_EXP_SECTIONS) or (c.project and c.project.strip())
        ]

        def get_topic_aliases(name: str) -> List[str]:
            can_name, _ = canonicalize_entity(name)
            canon = can_name or name
            aliases = {name.lower(), canon.lower()}
            tax_entry = CANONICAL_TAXONOMY.get(canon)
            if tax_entry and "aliases" in tax_entry:
                for a in tax_entry["aliases"]:
                    aliases.add(a.lower())
            return list(aliases)

        def has_project_evidence(name: str) -> Tuple[bool, Optional[str]]:
            # If name matches a project directly
            for p_name in projects_map:
                if p_name.lower() == name.lower():
                    return True, p_name

            aliases = get_topic_aliases(name)
            
            # 1. Check relationships linked to project context or project/experience section
            for r in relationships_list:
                src = (r.get("source_entity") or "").lower()
                tgt = (r.get("target_entity") or "").lower()
                ctx = r.get("context_entity")
                sec = (r.get("section") or "").lower()
                if any(a == src or a == tgt for a in aliases):
                    if ctx or sec in PROJECT_EXP_SECTIONS:
                        return True, ctx
            
            # 2. Check project/experience chunks
            for chk in proj_exp_chunks:
                chk_lower = (chk.chunk_text or "").lower()
                for a in aliases:
                    if re.search(r"\b" + re.escape(a) + r"\b", chk_lower):
                        return True, chk.project
            
            return False, None

        # 2. Retrieve chronological session questions and answers
        questions = (
            db.query(InterviewQuestion)
            .filter(InterviewQuestion.session_id == session_id)
            .order_by(InterviewQuestion.created_at.asc())
            .all()
        )
        answers = (
            db.query(InterviewAnswer)
            .filter(InterviewAnswer.session_id == session_id)
            .order_by(InterviewAnswer.created_at.asc())
            .all()
        )

        # Filter out unconsumed prefetched question from history analysis
        prefetched = (session.session_state or {}).get("prefetched_question")
        unconsumed_id = prefetched.get("question_id") if (prefetched and not prefetched.get("consumed")) else None
        if unconsumed_id:
            questions = [q for q in questions if str(q.id) != str(unconsumed_id)]

        # 3. Analyze question history, counts, and consecutive follow-up depth
        questions_per_topic: Dict[str, int] = {}
        questions_per_project: Dict[str, int] = {}
        followups_per_topic: Dict[str, int] = {}
        q_map: Dict[UUID, InterviewQuestion] = {q.id: q for q in questions}

        for q in questions:
            t = q.target_entity or q.category
            can_t, _ = canonicalize_entity(t) if t else (t, "")
            key_t = can_t or t
            if key_t:
                questions_per_topic[key_t] = questions_per_topic.get(key_t, 0) + 1
                if q.parent_question_id is not None or getattr(q, "is_follow_up", False):
                    followups_per_topic[key_t] = followups_per_topic.get(key_t, 0) + 1
            if q.target_project:
                p_key = q.target_project.strip()
                questions_per_project[p_key] = questions_per_project.get(p_key, 0) + 1

        consecutive_followup_count = 0
        last_question: Optional[InterviewQuestion] = questions[-1] if questions else None

        if last_question:
            curr = last_question
            while curr and curr.parent_question_id and curr.parent_question_id in q_map:
                consecutive_followup_count += 1
                curr = q_map[curr.parent_question_id]

        allow_followup = consecutive_followup_count < config.max_consecutive_followups
        # TASK 4: After a follow-up, force one breadth (uncovered topic) question before returning to gaps
        last_was_followup = bool(
            last_question and (
                last_question.parent_question_id is not None
                or consecutive_followup_count >= 1
                or getattr(last_question, "is_follow_up", False)
            )
        )

        # 4. Analyze latest evaluation performance and compute score-based adaptive difficulty (Phase 7.4)
        curr_diff = session.current_difficulty if (config.min_difficulty <= session.current_difficulty <= config.max_difficulty) else 3
        latest_eval_score: Optional[float] = None
        latest_eval_conf: Optional[float] = None
        is_refusal: bool = False
        latest_eval_answer: Optional[InterviewAnswer] = None

        window_size = config.rolling_window_size
        window_avg_score: Optional[float] = None

        if answers:
            # Look at chronologically latest evaluated answer
            eval_answers = [a for a in answers if a.status == "EVALUATED" and a.overall_score is not None]
            if eval_answers:
                latest_eval_answer = eval_answers[-1]
                raw_score = latest_eval_answer.overall_score
                # Normalize 0-10 score scale if passed explicitly
                if raw_score is not None and 0.0 < raw_score <= 10.0 and config.high_performance_threshold > 10.0:
                    latest_eval_score = raw_score * 10.0
                else:
                    latest_eval_score = raw_score
                latest_eval_conf = (
                    latest_eval_answer.evaluation_confidence
                    if latest_eval_answer.evaluation_confidence is not None
                    else 1.0
                )
                is_refusal = (
                    getattr(latest_eval_answer, "evaluator_model", "") == "fast_path_refusal"
                    or (latest_eval_score == 0.0 and "refusal" in (latest_eval_answer.evaluation_feedback or "").lower())
                )

                # B2: Rolling window of last N answers (raw scores)
                recent_evals = eval_answers[-window_size:]
                recent_scores = []
                for a in recent_evals:
                    s = a.overall_score
                    if s is not None and 0.0 < s <= 10.0 and config.high_performance_threshold > 10.0:
                        s *= 10.0
                    recent_scores.append(s)
                if recent_scores:
                    window_avg_score = sum(recent_scores) / len(recent_scores)

        target_difficulty = curr_diff
        performance_signal = "MODERATE"
        adaptation_rationale = ""

        if latest_eval_score is not None:
            if is_refusal:
                performance_signal = "REFUSAL"
                if curr_diff > config.min_difficulty:
                    target_difficulty = max(config.min_difficulty, curr_diff - 1)
                    adaptation_rationale = f"Candidate declined/refused previous question. De-escalating difficulty from {curr_diff} to {target_difficulty}."
                else:
                    target_difficulty = config.min_difficulty
                    adaptation_rationale = f"Candidate declined/refused previous question. Maintaining minimum difficulty {config.min_difficulty}."
            elif latest_eval_score > config.high_performance_threshold:
                # Tier C: Score > 75.0 (Score > 7.5 on 0-10 scale)
                # B2: Escalate only if latest score > high threshold AND window average > high threshold
                is_window_high = (window_avg_score is not None and window_avg_score > config.high_performance_threshold)
                if not is_window_high:
                    target_difficulty = curr_diff
                    performance_signal = "STRONG_WINDOW_NOT_MET"
                    adaptation_rationale = (
                        f"Candidate achieved high score ({round(latest_eval_score, 1)}/100), but rolling window "
                        f"average ({round(window_avg_score or 0.0, 1)}/100) is not above threshold ({config.high_performance_threshold}). "
                        f"Maintaining difficulty {curr_diff}."
                    )
                elif latest_eval_conf is not None and latest_eval_conf >= config.min_confidence_for_escalation:
                    target_difficulty = min(config.max_difficulty, curr_diff + 1)
                    performance_signal = "STRONG"
                    adaptation_rationale = (
                        f"Candidate demonstrated high mastery (latest {round(latest_eval_score, 1)}/100, window avg {round(window_avg_score, 1)}/100) "
                        f"with high evaluation confidence ({round(latest_eval_conf, 2)} >= {config.min_confidence_for_escalation}). "
                        f"Escalating difficulty from {curr_diff} to {target_difficulty}."
                    )
                else:
                    target_difficulty = curr_diff
                    performance_signal = "STRONG_LOW_CONFIDENCE"
                    adaptation_rationale = (
                        f"Candidate achieved high score ({round(latest_eval_score, 1)}/100), but evaluation confidence "
                        f"({round(latest_eval_conf or 0.0, 2)}) is below threshold ({config.min_confidence_for_escalation}). "
                        f"Maintaining difficulty {curr_diff}."
                    )
            elif latest_eval_score < config.low_performance_threshold:
                # Tier A: Score < 50.0 (Score < 5.0 on 0-10 scale)
                # B2: De-escalate symmetrically only if latest score < low threshold AND window average < low threshold
                is_window_low = (window_avg_score is not None and window_avg_score < config.low_performance_threshold)
                if not is_window_low:
                    target_difficulty = curr_diff
                    performance_signal = "WEAK_WINDOW_NOT_MET"
                    adaptation_rationale = (
                        f"Candidate had low latest score ({round(latest_eval_score, 1)}/100), but rolling window "
                        f"average ({round(window_avg_score or 0.0, 1)}/100) is not below threshold ({config.low_performance_threshold}). "
                        f"Maintaining difficulty {curr_diff}."
                    )
                elif latest_eval_conf is not None and latest_eval_conf >= config.min_confidence_for_deescalation:
                    target_difficulty = max(config.min_difficulty, curr_diff - 1)
                    performance_signal = "WEAK"
                    adaptation_rationale = (
                        f"Candidate demonstrated knowledge weakness (latest {round(latest_eval_score, 1)}/100, window avg {round(window_avg_score, 1)}/100) "
                        f"with confidence ({round(latest_eval_conf, 2)} >= {config.min_confidence_for_deescalation}). "
                        f"De-escalating difficulty from {curr_diff} to {target_difficulty}."
                    )
                else:
                    target_difficulty = curr_diff
                    performance_signal = "WEAK_LOW_CONFIDENCE"
                    adaptation_rationale = (
                        f"Candidate scored ({round(latest_eval_score, 1)}/100), but evaluation confidence "
                        f"({round(latest_eval_conf or 0.0, 2)}) is below de-escalation threshold. "
                        f"Maintaining difficulty {curr_diff}."
                    )
            else:
                # Tier B: 50.0 <= Score <= 75.0 (Score 5.0 through 7.5 on 0-10 scale)
                target_difficulty = curr_diff
                performance_signal = "MODERATE"
                adaptation_rationale = (
                    f"Candidate demonstrated solid baseline competence ({round(latest_eval_score, 1)}/100). "
                    f"Maintaining difficulty level {curr_diff} to evaluate practical implementation trade-offs."
                )
        else:
            performance_signal = "NO_EVALUATION"
            target_difficulty = curr_diff
            adaptation_rationale = f"No evaluated answers available yet. Retaining difficulty {curr_diff}."

        # 5. Helper sets of already asked scopes to avoid duplicates
        asked_topic_categories: Set[Tuple[str, str]] = set()
        for q in questions:
            ent = q.target_entity or q.category or ""
            can_ent, _ = canonicalize_entity(ent)
            asked_topic_categories.add((can_ent or ent, q.category))

        # Check force_strategy override if requested
        force_strat = request_override.force_strategy if request_override else None

        # =========================================================================
        # SELECTION DECISION RULES
        # =========================================================================

        # 0. EXPLICIT FORCE STRATEGY DISPATCH (When caller explicitly requests a strategy)
        if force_strat == AdaptiveStrategyType.CLAIM_VERIFICATION and claims_list:
            untested_claims = [
                c for c in claims_list
                if (c.get("status") in [ClaimVerificationStatus.UNTESTED.value, "UNTESTED"])
                and not is_excluded(c.get("target_entity"))
                and not is_excluded(c.get("project"))
            ] or [
                c for c in claims_list
                if not is_excluded(c.get("target_entity"))
                and not is_excluded(c.get("project"))
            ] or claims_list
            untested_claims.sort(key=lambda x: x.get("claim_id") or "")
            selected_claim = untested_claims[0]
            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.CLAIM_VERIFICATION,
                target_topic=selected_claim.get("target_entity") or selected_claim.get("project") or "Resume Claim",
                target_entity=selected_claim.get("target_entity"),
                target_project=selected_claim.get("project"),
                target_claim_id=selected_claim.get("claim_id"),
                target_relationship=None,
                category="Claim-Verification",
                difficulty=target_difficulty,
                rationale=f"Verifying candidate resume claim on '{selected_claim.get('project') or 'project'}': \"{selected_claim.get('claim_text', '')[:100]}\".",
                is_follow_up=False,
                parent_question_id=None,
                consecutive_followup_count=0,
            )

        if force_strat == AdaptiveStrategyType.RELATIONSHIP_VERIFICATION and relationships_list:
            eligible_rels = [
                r for r in relationships_list
                if not is_excluded(r.get("source_entity"))
                and not is_excluded(r.get("target_entity"))
                and not is_excluded(r.get("context_entity"))
            ] or relationships_list
            top_rel = eligible_rels[0]
            rel_src = top_rel.get("source_entity")
            rel_tgt = top_rel.get("target_entity")
            rel_type = top_rel.get("relation_type") or "relates_to"
            rel_str = f"{rel_src} {rel_type} {rel_tgt}"
            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.RELATIONSHIP_VERIFICATION,
                target_topic=rel_src,
                target_entity=rel_tgt,
                target_project=top_rel.get("context_entity"),
                target_claim_id=None,
                target_relationship=rel_str,
                category=request_override.preferred_category or "Architecture",
                difficulty=target_difficulty,
                rationale=f"Testing candidate's architectural reasoning across extracted relationship: '{rel_str}'.",
                is_follow_up=False,
                parent_question_id=None,
                consecutive_followup_count=0,
            )

        if force_strat == AdaptiveStrategyType.CONTRADICTION_CLARIFICATION:
            proj_context = last_question.target_project if last_question else None
            entity_anchor = last_question.target_entity if last_question else None
            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.CONTRADICTION_CLARIFICATION,
                target_topic=entity_anchor or proj_context or "Contradiction Clarification",
                target_entity=entity_anchor,
                target_project=proj_context,
                target_claim_id=None,
                target_relationship=None,
                category=request_override.preferred_category or "Clarification",
                difficulty=target_difficulty,
                rationale="Forced neutral clarification to investigate conflicting evidence.",
                is_follow_up=True,
                parent_question_id=last_question.id if last_question else None,
                consecutive_followup_count=consecutive_followup_count + 1,
                follow_up_reason="Clarifying discrepancy between candidate statement and verified evidence.",
            )

        if force_strat == AdaptiveStrategyType.PARTIAL_CLARIFICATION and last_question:
            t_ent = last_question.target_entity or last_question.category or "General"
            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.PARTIAL_CLARIFICATION,
                target_topic=t_ent,
                target_entity=last_question.target_entity,
                target_project=last_question.target_project,
                target_claim_id=None,
                target_relationship=None,
                category=request_override.preferred_category or "Implementation",
                difficulty=target_difficulty,
                rationale=f"Forced targeted clarification on omitted mechanisms for '{t_ent}'.",
                is_follow_up=True,
                parent_question_id=last_question.id,
                consecutive_followup_count=consecutive_followup_count + 1,
                follow_up_reason=f"Targeted clarification on partial answer for '{t_ent}'.",
            )

        # RULE 1: EVIDENCE-BACKED CONTRADICTION CLARIFICATION (Phase 7.5 - Priority 2)
        # When candidate's latest evaluated answer produced confirmed factual contradiction
        # backed by verified evidence snippet or fact, ask a neutral, non-adversarial clarification.
        # Guard: Only genuine contradictions (status == CONTRADICTION), NOT hypothetical discussions or additions.
        # Guard: Must be supported by verified evidence.
        # Guard: Respects follow-up budget (allow_followup).
        if (
            allow_followup
            and latest_eval_answer
            and force_strat not in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION, AdaptiveStrategyType.RELATIONSHIP_VERIFICATION]
        ):
            contra_data = getattr(latest_eval_answer, "contradiction_analysis", None)
            if contra_data is None and isinstance(latest_eval_answer.dimension_scores, dict):
                contra_data = (
                    latest_eval_answer.dimension_scores.get("_contradiction_analysis")
                    or latest_eval_answer.dimension_scores.get("contradiction_analysis")
                )

            if contra_data and contra_data.get("has_contradiction"):
                contra_status = contra_data.get("status")
                # Strictly distinguish confirmed contradiction from unverified additions or hypotheticals
                if contra_status in ["CONTRADICTION", "ContradictionStatus.CONTRADICTION"]:
                    conflicts = contra_data.get("conflicts", [])
                    # Verify presence of supporting evidence
                    verified_conflicts = [
                        c for c in conflicts
                        if isinstance(c, dict) and (c.get("verified_fact") or c.get("evidence_text") or c.get("explanation"))
                    ]
                    if verified_conflicts:
                        top_conflict = verified_conflicts[0]
                        contra_proj = top_conflict.get("context_project") or (last_question.target_project if last_question else None)
                        contra_ent = last_question.target_entity if last_question else None

                        if not is_excluded(contra_ent) and not is_excluded(contra_proj):
                            cand_assert = (top_conflict.get("candidate_assertion") or "")[:80]
                            verified_fact = (top_conflict.get("verified_fact") or top_conflict.get("evidence_text") or "")[:80]

                            return NextQuestionSelectionResult(
                                strategy=AdaptiveStrategyType.CONTRADICTION_CLARIFICATION,
                                target_topic=contra_ent or contra_proj or "Contradiction Clarification",
                                target_entity=contra_ent,
                                target_project=contra_proj,
                                target_claim_id=top_conflict.get("evidence_id"),
                                target_relationship=None,
                                category=request_override.preferred_category if (request_override and request_override.preferred_category) else "Clarification",
                                difficulty=target_difficulty,
                                rationale=(
                                    f"Candidate answer on '{contra_proj or 'experience'}' contradicted verified resume evidence. "
                                    f"Candidate stated: \"{cand_assert}\", while verified fact is: \"{verified_fact}\". "
                                    f"Asking neutral, evidence-grounded clarification to explore discrepancy without speculation."
                                ),
                                is_follow_up=True,
                                parent_question_id=last_question.id if last_question else None,
                                consecutive_followup_count=consecutive_followup_count + 1,
                                follow_up_reason=(
                                    f"Clarifying discrepancy between candidate assertion ('{cand_assert}') and verified evidence ('{verified_fact}')."
                                ),
                            )

        # RULE 2: PARTIAL-ANSWER CLARIFICATION (Phase 7.5 - Priority 2)
        # When candidate gave a partially correct answer (50 <= score <= 75) that omitted specific key concepts.
        # Asks targeted clarification on the missing component without repeating the parent question.
        # TASK 4: After a follow-up, force one breadth question before returning to gaps or clarifications.
        if (
            ((allow_followup and not last_was_followup) or force_strat == AdaptiveStrategyType.PARTIAL_CLARIFICATION)
            and last_question
            and latest_eval_answer
            and not is_refusal
            and not is_excluded(last_question.target_entity)
            and not is_excluded(last_question.target_project)
            and force_strat not in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION, AdaptiveStrategyType.RELATIONSHIP_VERIFICATION]
        ):
            missing_concepts_list = [
                c.strip() for c in (latest_eval_answer.missing_concepts or [])
                if isinstance(c, str) and c.strip()
            ]
            is_partial_score = (
                latest_eval_score is not None
                and config.low_performance_threshold <= latest_eval_score <= config.high_performance_threshold
            )
            if missing_concepts_list and (is_partial_score or force_strat == AdaptiveStrategyType.PARTIAL_CLARIFICATION):
                omitted_concept = missing_concepts_list[0]
                can_topic_str, _ = canonicalize_entity(last_question.target_entity or last_question.category or "General")
                target_topic_name = can_topic_str or last_question.target_entity or last_question.category or "General"

                topic_q_count = questions_per_topic.get(target_topic_name, 0)
                topic_fu_count = followups_per_topic.get(target_topic_name, 0)
                # TASK 4: Max 2 follow-ups per topic per session
                if (topic_q_count < config.max_questions_per_topic and topic_fu_count < 2 or force_strat == AdaptiveStrategyType.PARTIAL_CLARIFICATION) and not is_excluded(target_topic_name):
                    next_cat = "Implementation" if last_question.category != "Implementation" else "Clarification"
                    if request_override and request_override.preferred_category:
                        next_cat = request_override.preferred_category

                    return NextQuestionSelectionResult(
                        strategy=AdaptiveStrategyType.PARTIAL_CLARIFICATION,
                        target_topic=target_topic_name,
                        target_entity=last_question.target_entity,
                        target_project=last_question.target_project,
                        target_claim_id=None,
                        target_relationship=None,
                        category=next_cat,
                        difficulty=target_difficulty,
                        rationale=(
                            f"Candidate provided a partially correct answer ({round(latest_eval_score or 0.0, 1)}/100) on '{target_topic_name}', "
                            f"omitting expected mechanism: '{omitted_concept}'. {adaptation_rationale} "
                            f"Asking targeted clarification focusing specifically on the missing component."
                        ),
                        is_follow_up=True,
                        parent_question_id=last_question.id,
                        consecutive_followup_count=consecutive_followup_count + 1,
                        follow_up_reason=(
                            f"Clarifying omitted architectural mechanism '{omitted_concept}' from previous partial answer on '{target_topic_name}'."
                        ),
                    )

        # RULE 3: KNOWLEDGE GAP FOLLOW-UP (B5: Ranked by severity first, then recency)
        # Follow-up rules keep higher priority (B1).
        # TASK 4: Never trigger gap follow-up on a topic whose mastery score is >= 85.
        # TASK 4: Max 2 follow-ups per topic per session; after a follow-up, force one breadth (uncovered topic) question before returning to gaps.
        if (
            ((allow_followup and not last_was_followup) or force_strat == AdaptiveStrategyType.GAP_FOLLOW_UP)
            and force_strat not in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION, AdaptiveStrategyType.RELATIONSHIP_VERIFICATION]
        ):
            active_gaps = [
                g for g in knowledge_gaps
                if not g.get("is_addressed")
                and g.get("gap_nature") != GapNature.INSUFFICIENT_EVIDENCE.value
                and not is_excluded(g.get("topic"))
                and topics_map.get(canonicalize_entity(g.get("topic") or "")[0] or g.get("topic") or "", {}).get("mastery_score", 0.0) < 85.0
                and followups_per_topic.get(canonicalize_entity(g.get("topic") or "")[0] or g.get("topic") or "", 0) < 2
            ]
            if active_gaps:
                severity_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                active_gaps.sort(
                    key=lambda x: (severity_map.get(str(x.get("severity", "")).upper(), 0), x.get("created_at") or ""),
                    reverse=True
                )
                top_gap = active_gaps[0]
                gap_topic = top_gap.get("topic") or "General"
                can_gap_topic, _ = canonicalize_entity(gap_topic)
                gap_topic_clean = can_gap_topic or gap_topic

                topic_q_count = questions_per_topic.get(gap_topic_clean, 0)
                topic_fu_count = followups_per_topic.get(gap_topic_clean, 0)
                if (topic_q_count < config.max_questions_per_topic and topic_fu_count < 2) or force_strat == AdaptiveStrategyType.GAP_FOLLOW_UP:
                    parent_q_id = UUID(top_gap["question_id"]) if top_gap.get("question_id") else None
                    parent_q = q_map.get(parent_q_id) if parent_q_id else None

                    next_cat = "Implementation"
                    if parent_q and parent_q.category in ["Architecture", "Implementation", "Failure-Mode"]:
                        next_cat = "Fundamentals"
                    if request_override and request_override.preferred_category:
                        next_cat = request_override.preferred_category

                    return NextQuestionSelectionResult(
                        strategy=AdaptiveStrategyType.GAP_FOLLOW_UP,
                        target_topic=gap_topic_clean,
                        target_entity=gap_topic_clean,
                        target_project=parent_q.target_project if parent_q else None,
                        target_claim_id=None,
                        target_relationship=None,
                        category=next_cat,
                        difficulty=target_difficulty,
                        rationale=(
                            f"Candidate demonstrated a knowledge gap in '{gap_topic_clean}' "
                            f"({top_gap.get('description', '')}). {adaptation_rationale} "
                            f"Probing foundational mechanics to evaluate core understanding."
                        ),
                        is_follow_up=True,
                        parent_question_id=parent_q_id,
                        consecutive_followup_count=consecutive_followup_count + 1,
                        follow_up_reason=(
                            f"Probing foundational mechanics on '{gap_topic_clean}' to address demonstrated knowledge gap."
                        ),
                    )

        # Compute quotas and budget tightness ahead of Rule 4 so optional difficulty follow-up
        # does not starve mandatory claim or project-depth quotas when budget is tight (Part B1)
        budget = config.default_max_session_questions
        remaining_budget = max(0, budget - len(questions))
        claim_quota = max(1, int(round(budget * config.claim_verification_quota_percent))) if claims_list else 0
        project_quota = max(1, int(round(budget * config.project_depth_quota_percent))) if projects_map else 0

        claims_asked_count = sum(
            1 for q in questions
            if q.target_claim_id
            or q.category == "Claim-Verification"
            or (q.targeted_claim and q.targeted_claim not in ["Evidence-Grounded Question", ""] and q.targeted_claim.strip())
        )
        project_depth_asked_count = sum(
            1 for q in questions
            if q.target_project and (not q.target_entity or q.category == "Project-Depth" or "Project Depth" in (q.rationale or ""))
        )

        asked_claim_ids = {q.target_claim_id for q in questions if getattr(q, "target_claim_id", None)}
        has_available_claims = any(
            (c.get("status") in [ClaimVerificationStatus.UNTESTED.value, "UNTESTED"]
             or (c.get("claim_id") and c.get("claim_id") not in asked_claim_ids))
            and not is_excluded(c.get("target_entity"))
            and not is_excluded(c.get("project"))
            for c in claims_list
        )
        has_available_projects = any(
            not is_excluded(p_name)
            and p_data.get("remaining_areas_to_assess")
            and questions_per_project.get(p_name, 0) < config.max_questions_per_project
            for p_name, p_data in projects_map.items()
        )

        claim_deficit = max(0, claim_quota - claims_asked_count) if has_available_claims else 0
        project_deficit = max(0, project_quota - project_depth_asked_count) if has_available_projects else 0
        total_quota_deficit = claim_deficit + project_deficit
        is_budget_tight = (remaining_budget <= total_quota_deficit)

        # RULE 4: DIFFICULTY ESCALATION ON STRONG TOPICS (Tier C: Seniority Probing - Priority 4)
        # Triggered when candidate scored high with reliable evaluation confidence.
        # Follows up with advanced failure-mode/performance questions if within single follow-up budget.
        # If topic was already followed up on or breadth is needed, allows selector to proceed to Rule 5.
        if (
            allow_followup
            and not last_was_followup
            and not (is_budget_tight and total_quota_deficit > 0)
            and performance_signal == "STRONG"
            and last_question
            and last_question.target_entity
            and not is_excluded(last_question.target_entity)
            and not is_excluded(last_question.target_project)
            and force_strat not in [AdaptiveStrategyType.UNCOVERED_TOPIC, AdaptiveStrategyType.CLAIM_VERIFICATION, AdaptiveStrategyType.RELATIONSHIP_VERIFICATION]
        ):
            can_last_t, _ = canonicalize_entity(last_question.target_entity)
            last_t = can_last_t or last_question.target_entity
            topic_data = topics_map.get(last_t, {})

            # Only escalate on same concept if topic hasn't already had 2 follow-ups or if explicitly forced
            if ((consecutive_followup_count == 0 and questions_per_topic.get(last_t, 0) <= 1 and followups_per_topic.get(last_t, 0) < 2) or force_strat == AdaptiveStrategyType.DIFFICULTY_ESCALATION) and not is_excluded(last_t):
                escalated_cat = "Failure-Mode"
                if last_question.category == "Failure-Mode":
                    escalated_cat = "Performance"
                elif last_question.category == "Performance":
                    escalated_cat = "Security"
                if request_override and request_override.preferred_category:
                    escalated_cat = request_override.preferred_category

                return NextQuestionSelectionResult(
                    strategy=AdaptiveStrategyType.DIFFICULTY_ESCALATION,
                    target_topic=last_t,
                    target_entity=last_t,
                    target_project=last_question.target_project,
                    target_claim_id=None,
                    target_relationship=None,
                    category=escalated_cat,
                    difficulty=target_difficulty,
                    rationale=(
                        f"Candidate achieved high score ({round(latest_eval_score or 0.0, 1)}/100) on '{last_t}'. "
                        f"{adaptation_rationale} Probing architectural resilience and failure recovery."
                    ),
                    is_follow_up=True,
                    parent_question_id=last_question.id,
                    consecutive_followup_count=consecutive_followup_count + 1,
                    follow_up_reason=(
                        f"Deepening strong answer ({round(latest_eval_score or 0.0, 1)}/100) on '{last_t}' with advanced failure modes, architecture, and scalability trade-offs."
                    ),
                )

        # =========================================================================
        # BREADTH & NON-FOLLOW-UP STRATEGIES (WITH QUOTA RESERVATION - Part B1)
        # =========================================================================
        budget = config.default_max_session_questions
        remaining_budget = max(0, budget - len(questions))

        # 1. Candidate item selectors
        def _get_uncovered_topic_result() -> Optional[NextQuestionSelectionResult]:
            uncovered_topics: List[Tuple[str, Dict[str, Any]]] = []
            for t_name, t_val in topics_map.items():
                if not t_val.get("is_assessable", False):
                    continue
                if t_name.lower() in JUNK_TOPIC_BLOCKLIST:
                    continue
                asked_count = t_val.get("questions_asked_count", 0)
                eval_count = t_val.get("answers_evaluated_count", 0)
                cov_status = t_val.get("coverage_status")
                if (
                    eval_count == 0
                    and asked_count == 0
                    and cov_status in [TopicCoverageStatus.NOT_CONSIDERED.value, TopicCoverageStatus.SELECTED.value]
                ):
                    can_name, _ = canonicalize_entity(t_name)
                    norm_name = can_name or t_name
                    if questions_per_topic.get(norm_name, 0) == 0:
                        uncovered_topics.append((t_name, t_val))

            if uncovered_topics and excluded_topics:
                uncovered_topics = [
                    item for item in uncovered_topics
                    if item[0] not in excluded_topics and item[0].lower() not in {x.lower() for x in excluded_topics}
                ]

            if not uncovered_topics:
                return None

            def topic_sort_key(item: Tuple[str, Dict[str, Any]]):
                name, data = item
                ttype = (data.get("topic_type") or "").lower()
                if any(x in ttype for x in ["framework", "database", "language"]):
                    priority = 1
                elif any(x in ttype for x in ["cloud", "devops", "tool"]):
                    priority = 2
                else:
                    priority = 3

                has_proj, proj = has_project_evidence(name)
                # Skills-list-only topics (no project/experience evidence) sort LAST!
                evidence_priority = 0 if has_proj else 1

                at_capacity = 1 if (proj and questions_per_project.get(proj, 0) >= config.max_questions_per_project) else 0
                return (evidence_priority, at_capacity, priority, name.lower())

            uncovered_topics.sort(key=topic_sort_key)
            chosen_topic_name, chosen_topic_data = uncovered_topics[0]
            chosen_type = (chosen_topic_data.get("topic_type") or "Skill").title()

            has_proj, associated_proj = has_project_evidence(chosen_topic_name)
            if not has_proj:
                # Skills-list-only topic: must be conceptual and must NOT reference any project
                associated_proj = None
                chosen_cat = "Fundamentals"
                rationale_text = (
                    f"Topic '{chosen_topic_name}' ({chosen_type}) appears only in candidate's skills list with no project evidence. "
                    f"Selecting as a conceptual question for interview breadth coverage."
                )
            else:
                chosen_cat = "Architecture"
                if performance_signal == "STRONG":
                    chosen_cat = "Architecture" if target_difficulty >= 4 else "Implementation"
                elif performance_signal in ["WEAK", "REFUSAL"]:
                    chosen_cat = "Fundamentals" if target_difficulty <= 2 else "Implementation"
                elif performance_signal == "MODERATE":
                    chosen_cat = "Implementation" if target_difficulty <= 3 else "Trade-Offs"
                elif "language" in chosen_type.lower():
                    chosen_cat = "Fundamentals" if target_difficulty <= 2 else "Implementation"
                elif "database" in chosen_type.lower():
                    chosen_cat = "Architecture" if target_difficulty >= 3 else "Fundamentals"
                rationale_text = (
                    f"Topic '{chosen_topic_name}' ({chosen_type}) extracted from candidate's resume has not yet been assessed. "
                    f"{adaptation_rationale} Selecting for interview breadth coverage."
                )

            if request_override and request_override.preferred_category:
                chosen_cat = request_override.preferred_category

            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.UNCOVERED_TOPIC,
                target_topic=chosen_topic_name,
                target_entity=chosen_topic_name,
                target_project=associated_proj,
                target_claim_id=None,
                target_relationship=None,
                category=chosen_cat,
                difficulty=target_difficulty,
                rationale=rationale_text,
                is_follow_up=False,
                parent_question_id=None,
                consecutive_followup_count=0,
            )

        def _get_claim_result() -> Optional[NextQuestionSelectionResult]:
            if not claims_list:
                return None
            asked_claim_ids = {
                q.target_claim_id for q in questions
                if getattr(q, "target_claim_id", None)
            }
            untested_claims = [
                c for c in claims_list
                if (c.get("status") in [ClaimVerificationStatus.UNTESTED.value, "UNTESTED"]
                    or (c.get("claim_id") and c.get("claim_id") not in asked_claim_ids))
                and not is_excluded(c.get("target_entity"))
                and not is_excluded(c.get("project"))
                and (not c.get("project") or questions_per_project.get(c.get("project"), 0) < config.max_questions_per_project)
            ]
            if not untested_claims:
                untested_claims = [
                    c for c in claims_list
                    if (c.get("status") in [ClaimVerificationStatus.UNTESTED.value, "UNTESTED"]
                        or (c.get("claim_id") and c.get("claim_id") not in asked_claim_ids))
                    and not is_excluded(c.get("target_entity"))
                    and not is_excluded(c.get("project"))
                ]
            if not untested_claims:
                return None

            def claim_sort_key(c_item):
                p = c_item.get("project") or ""
                is_current_proj = 1 if (last_question and last_question.target_project == p) else 0
                return (-is_current_proj, c_item.get("claim_id") or "")

            untested_claims.sort(key=claim_sort_key)
            selected_claim = untested_claims[0]
            claim_proj = selected_claim.get("project")
            claim_ent = selected_claim.get("target_entity")

            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.CLAIM_VERIFICATION,
                target_topic=claim_ent or claim_proj or "Resume Claim",
                target_entity=claim_ent,
                target_project=claim_proj,
                target_claim_id=selected_claim.get("claim_id"),
                target_relationship=None,
                category="Claim-Verification",
                difficulty=target_difficulty,
                rationale=(
                    f"Candidate has an unverified resume claim on '{claim_proj or 'project'}': "
                    f"\"{selected_claim.get('claim_text', '')[:100]}\". Probing hands-on execution depth."
                ),
                is_follow_up=False,
                parent_question_id=None,
                consecutive_followup_count=0,
            )

        def _get_relationship_result() -> Optional[NextQuestionSelectionResult]:
            untested_rels = [
                r for r in relationships_list
                if not r.get("is_tested")
                and r.get("status") == "UNTESTED"
                and (r.get("source_entity") or "").lower() not in JUNK_TOPIC_BLOCKLIST
                and (r.get("target_entity") or "").lower() not in JUNK_TOPIC_BLOCKLIST
                and (r.get("section") or "").lower() not in ("education", "achievements", "languages", "leadership", "header", "contact")
                and not is_excluded(r.get("source_entity"))
                and not is_excluded(r.get("target_entity"))
                and not is_excluded(r.get("context_entity"))
            ]
            if not untested_rels:
                return None
            untested_rels.sort(key=lambda r: (r.get("source_entity") or "", r.get("target_entity") or ""))
            top_rel = untested_rels[0]
            rel_src = top_rel.get("source_entity")
            rel_tgt = top_rel.get("target_entity")
            rel_type = top_rel.get("relation_type") or "relates_to"
            rel_ctx = top_rel.get("context_entity")
            rel_str = f"{rel_src} {rel_type} {rel_tgt}"

            return NextQuestionSelectionResult(
                strategy=AdaptiveStrategyType.RELATIONSHIP_VERIFICATION,
                target_topic=rel_src,
                target_entity=rel_tgt,
                target_project=rel_ctx,
                target_claim_id=None,
                target_relationship=rel_str,
                category="Architecture",
                difficulty=target_difficulty,
                rationale=f"Testing candidate's architectural reasoning across extracted relationship: '{rel_str}'.",
                is_follow_up=False,
                parent_question_id=None,
                consecutive_followup_count=0,
            )

        def _get_project_depth_result() -> Optional[NextQuestionSelectionResult]:
            for proj_name, proj_data in sorted(projects_map.items()):
                if is_excluded(proj_name):
                    continue
                rem_areas = proj_data.get("remaining_areas_to_assess", [])
                q_cnt = questions_per_project.get(proj_name, 0)
                if rem_areas and q_cnt < config.max_questions_per_project:
                    next_area = rem_areas[0]
                    return NextQuestionSelectionResult(
                        strategy=AdaptiveStrategyType.PROJECT_DEPTH,
                        target_topic=proj_name,
                        target_entity=None,
                        target_project=proj_name,
                        target_claim_id=None,
                        target_relationship=None,
                        category=next_area,
                        difficulty=target_difficulty,
                        rationale=(
                            f"Project '{proj_name}' has remaining unassessed dimension '{next_area}'. Probing candidate implementation choices."
                        ),
                        is_follow_up=False,
                        parent_question_id=None,
                        consecutive_followup_count=0,
                    )
            return None

        # 2. Compute quotas and check whether remaining budget is tight (Part B1)
        claim_quota = max(1, int(round(budget * config.claim_verification_quota_percent))) if claims_list else 0
        project_quota = max(1, int(round(budget * config.project_depth_quota_percent))) if projects_map else 0

        claims_asked_count = sum(
            1 for q in questions
            if q.target_claim_id
            or q.category == "Claim-Verification"
            or (q.targeted_claim and q.targeted_claim not in ["Evidence-Grounded Question", ""] and q.targeted_claim.strip())
        )
        project_depth_asked_count = sum(
            1 for q in questions
            if q.target_project and (not q.target_entity or q.category == "Project-Depth" or "Project Depth" in (q.rationale or ""))
        )

        has_available_claims = any(
            (c.get("status") in [ClaimVerificationStatus.UNTESTED.value, "UNTESTED"]
             or (c.get("claim_id") and c.get("claim_id") not in asked_claim_ids))
            and not is_excluded(c.get("target_entity"))
            and not is_excluded(c.get("project"))
            for c in claims_list
        )
        has_available_projects = any(
            not is_excluded(p_name)
            and p_data.get("remaining_areas_to_assess")
            and questions_per_project.get(p_name, 0) < config.max_questions_per_project
            for p_name, p_data in projects_map.items()
        )

        claim_deficit = max(0, claim_quota - claims_asked_count) if has_available_claims else 0
        project_deficit = max(0, project_quota - project_depth_asked_count) if has_available_projects else 0
        total_quota_deficit = claim_deficit + project_deficit

        # If quota is unmet and remaining budget is tight,
        # prefer claim verification or project depth over uncovered topics (B1).
        is_budget_tight = (remaining_budget <= total_quota_deficit)
        if is_budget_tight:
            if claim_deficit > 0:
                claim_res = _get_claim_result()
                if claim_res:
                    return claim_res
            if project_deficit > 0:
                proj_res = _get_project_depth_result()
                if proj_res:
                    return proj_res

        # 3. Standard non-follow-up priority order
        uncovered_res = _get_uncovered_topic_result()
        if uncovered_res:
            return uncovered_res

        claim_res = _get_claim_result()
        if claim_res:
            return claim_res

        rel_res = _get_relationship_result()
        if rel_res:
            return rel_res

        proj_res = _get_project_depth_result()
        if proj_res:
            return proj_res

        # RULE 8: FALLBACK COVERAGE EXPANSION (All Topics Initially Visited)
        # If all topics have been visited and no active gaps exist, rotate to deepest topic with unasked category
        sorted_topics = sorted(
            topics_map.items(),
            key=lambda x: x[1].get("confidence", 0.0)
        )
        eligible_fallback_topics = [
            item for item in sorted_topics if not is_excluded(item[0])
        ]
        fallback_topic = eligible_fallback_topics[0][0] if eligible_fallback_topics else (sorted_topics[0][0] if sorted_topics else "Architecture")

        categories_pool = ["Failure-Mode", "Security", "Performance", "Architecture", "Implementation", "Trade-Offs"]
        used_cats = {cat for (top, cat) in asked_topic_categories if top == fallback_topic}
        avail_cats = [c for c in categories_pool if c not in used_cats and not is_excluded(c)]
        if not avail_cats:
            avail_cats = [c for c in categories_pool if not is_excluded(c)]
        fallback_cat = avail_cats[0] if avail_cats else "Trade-Offs"

        if request_override and request_override.preferred_category:
            fallback_cat = request_override.preferred_category

        return NextQuestionSelectionResult(
            strategy=AdaptiveStrategyType.COVERAGE_EXPANSION,
            target_topic=fallback_topic,
            target_entity=fallback_topic,
            target_project=None,
            target_claim_id=None,
            target_relationship=None,
            category=fallback_cat,
            difficulty=target_difficulty,
            rationale=(
                f"All extracted resume topics have received baseline assessment. "
                f"Re-sampling topic '{fallback_topic}' with '{fallback_cat}' to evaluate advanced engineering depth."
            ),
            is_follow_up=False,
            parent_question_id=None,
            consecutive_followup_count=0,
        )

    BREADTH_STRATEGIES = {
        AdaptiveStrategyType.UNCOVERED_TOPIC,
        AdaptiveStrategyType.CLAIM_VERIFICATION,
        AdaptiveStrategyType.RELATIONSHIP_VERIFICATION,
        AdaptiveStrategyType.PROJECT_DEPTH,
    }

    def _get_state_version(self, session: InterviewSession, db: Session) -> str:
        q_cnt = db.query(InterviewQuestion).filter(InterviewQuestion.session_id == session.id).count()
        prefetched = (session.session_state or {}).get("prefetched_question")
        if prefetched and not prefetched.get("consumed"):
            if session.total_questions is not None and q_cnt > session.total_questions:
                q_cnt = session.total_questions
        ans_cnt = db.query(InterviewAnswer).filter(InterviewAnswer.session_id == session.id).count()
        eval_cnt = db.query(InterviewAnswer).filter(InterviewAnswer.session_id == session.id, InterviewAnswer.status == "EVALUATED").count()
        return f"q:{q_cnt}_ans:{ans_cnt}_eval:{eval_cnt}_diff:{session.current_difficulty}"

    def _is_prefetch_enabled(self, session: InterviewSession, config: AdaptiveSelectionConfig) -> bool:
        if config.prefetch_enabled:
            return True
        sess_meta = (session.session_state or {}).get("metadata") or {}
        if sess_meta.get("prefetch_enabled") is not None:
            return bool(sess_meta.get("prefetch_enabled"))
        if getattr(session, "session_metadata", None):
            sm = session.session_metadata or {}
            if "prefetch_enabled" in sm:
                return bool(sm["prefetch_enabled"])
            if "adaptive_config" in sm and "prefetch_enabled" in sm["adaptive_config"]:
                return bool(sm["adaptive_config"]["prefetch_enabled"])
        return bool(settings.PREFETCH_ENABLED)

    def _maybe_schedule_prefetch(
        self,
        session: InterviewSession,
        background_tasks: Optional[BackgroundTasks],
        db: Session,
        config: AdaptiveSelectionConfig
    ) -> None:
        if not self._is_prefetch_enabled(session, config):
            return
        if session.status != "IN_PROGRESS":
            return
        if session.total_questions >= config.default_max_session_questions:
            return

        expected_version = self._get_state_version(session, db)
        next_sel = self.determine_next_question_target(session.id, db, config_override=config)
        if next_sel.strategy not in self.BREADTH_STRATEGIES:
            return

        if background_tasks is not None:
            background_tasks.add_task(background_prefetch_next_question, session.id, expected_version)
        else:
            try:
                self.execute_prefetch(session.id, expected_version, db)
            except Exception as e:
                logger.warning(f"Direct prefetch failed for session {session.id}: {e}")

    def execute_prefetch(self, session_id: UUID, expected_state_version: str, db: Session) -> None:
        query = db.query(InterviewSession).filter(InterviewSession.id == session_id)
        if db.bind and getattr(db.bind.dialect, "name", "") == "postgresql":
            query = query.with_for_update()
        session = query.first()
        if not session or session.status != "IN_PROGRESS":
            return

        current_version = self._get_state_version(session, db)
        if current_version != expected_state_version:
            logger.info("Prefetch aborted: session state mutated before generation.")
            return

        config = AdaptiveSelectionConfig.from_session_metadata(
            (session.session_state or {}).get("metadata")
        )
        next_sel = self.determine_next_question_target(session.id, db, config_override=config)
        if next_sel.strategy not in self.BREADTH_STRATEGIES:
            logger.info("Prefetch aborted: next strategy is not a breadth strategy.")
            return

        rag_req = QuestionGenerationRequest(
            category=next_sel.category,
            difficulty=next_sel.difficulty,
            target_project=next_sel.target_project,
            target_entity=next_sel.target_entity,
            session_id=session.id,
            parent_question_id=next_sel.parent_question_id,
            target_claim_id=next_sel.target_claim_id,
            target_relationship=next_sel.target_relationship,
            follow_up_reason=next_sel.follow_up_reason
        )

        try:
            rag_resp = rag_pipeline.generate_grounded_question(
                resume_id=session.resume_id,
                req=rag_req,
                db=db,
                persist=False
            )
        except TypeError:
            rag_resp = rag_pipeline.generate_grounded_question(
                resume_id=session.resume_id,
                req=rag_req,
                db=db
            )

        if rag_resp.validation_status == "ACCEPTED" and rag_resp.question:
            # Check state version while excluding the newly generated prefetch question
            q_cnt_after = db.query(InterviewQuestion).filter(
                InterviewQuestion.session_id == session.id,
                InterviewQuestion.id != rag_resp.question_id
            ).count()
            ans_cnt_after = db.query(InterviewAnswer).filter(InterviewAnswer.session_id == session.id).count()
            eval_cnt_after = db.query(InterviewAnswer).filter(InterviewAnswer.session_id == session.id, InterviewAnswer.status == "EVALUATED").count()
            current_version_after = f"q:{q_cnt_after}_ans:{ans_cnt_after}_eval:{eval_cnt_after}_diff:{session.current_difficulty}"

            if current_version_after != expected_state_version:
                logger.info("Prefetch discarded: state mutated during question generation.")
                if rag_resp.question_id:
                    try:
                        db.query(InterviewQuestion).filter(
                            InterviewQuestion.id == rag_resp.question_id
                        ).delete()
                        db.commit()
                    except Exception as clean_err:
                        logger.debug(f"Failed to clean up discarded prefetch question {rag_resp.question_id}: {clean_err}")
                return

            sess_state = dict(session.session_state or {})
            sess_state["prefetched_question"] = {
                "question_id": str(rag_resp.question_id),
                "question": rag_resp.question.model_dump(),
                "selection": next_sel.model_dump(),
                "knowledge_state_version": expected_state_version,
                "difficulty": next_sel.difficulty,
                "strategy": next_sel.strategy.value,
                "created_at": datetime.utcnow().isoformat(),
                "consumed": False
            }
            prefetch_meta = sess_state.setdefault("_prefetch_metrics", {
                "prefetches_generated": 0,
                "prefetches_consumed": 0,
                "prefetches_discarded": 0,
            })
            prefetch_meta["prefetches_generated"] += 1

            session.session_state = sess_state
            flag_modified(session, "session_state")
            db.commit()
            logger.info(f"Successfully prefetched question for session {session_id} (target='{next_sel.target_topic}').")

    def select_and_generate_next_question(
        self,
        session_id: UUID,
        db: Session,
        config_override: Optional[AdaptiveSelectionConfig] = None,
        request_override: Optional[AdaptiveQuestionSelectionRequest] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> AdaptiveNextQuestionResponse:
        """
        Full lifecycle: Determines next question target using adaptive rules,
        invokes grounded RAG question generation, links question to session and parent,
        updates difficulty and topic coverage state in PostgreSQL, and returns response.
        """
        query = db.query(InterviewSession).filter(InterviewSession.id == session_id)
        if db.bind and getattr(db.bind.dialect, "name", "") == "postgresql":
            query = query.with_for_update()
        session = query.first()
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found.")

        # If session was marked COMPLETED simply because previous questions were evaluated,
        # but the interview has not been explicitly ended and question budget remains, reopen to IN_PROGRESS
        end_reason = (session.session_state or {}).get("end_reason")
        if session.status == "COMPLETED" and not end_reason:
            session.status = "IN_PROGRESS"
            session.completed_at = None
            db.flush()

        if session.status != "IN_PROGRESS":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot generate next question: session is already {session.status}."
            )

        # Idempotency check: if client sent request_idempotency_key, return cached result on retry
        req_key = getattr(request_override, "request_idempotency_key", None) if request_override else None
        if req_key:
            state_dict = session.session_state or {}
            idem_cache = state_dict.get("_idempotency_cache", {})
            if req_key in idem_cache:
                logger.info(f"Returning cached response for idempotency key '{req_key}' on session {session_id}")
                return AdaptiveNextQuestionResponse.model_validate(idem_cache[req_key])

        config = config_override or AdaptiveSelectionConfig.from_session_metadata(
            (session.session_state or {}).get("metadata")
        )

        # Check total question budget (excluding unconsumed prefetch)
        existing_q_count = (
            db.query(InterviewQuestion)
            .filter(InterviewQuestion.session_id == session_id)
            .count()
        )
        prefetched_peek = (session.session_state or {}).get("prefetched_question")
        if prefetched_peek and not prefetched_peek.get("consumed"):
            pf_id = prefetched_peek.get("question_id")
            if pf_id:
                try:
                    if db.query(InterviewQuestion).filter(InterviewQuestion.id == UUID(str(pf_id))).first() is not None:
                        existing_q_count = max(0, existing_q_count - 1)
                except Exception:
                    pass

        if existing_q_count >= config.default_max_session_questions:
            return AdaptiveNextQuestionResponse(
                session_id=session.id,
                resume_id=session.resume_id,
                status="BUDGET_EXHAUSTED",
                selection=None,
                message=f"Session reached maximum question budget of {config.default_max_session_questions}."
            )

        t_start = time.perf_counter()

        # Check for cached prefetched question (Part C: A7)
        prefetched = (session.session_state or {}).get("prefetched_question")
        if prefetched and not prefetched.get("consumed"):
            current_version = self._get_state_version(session, db)
            cached_version = prefetched.get("knowledge_state_version")
            cached_diff = prefetched.get("difficulty")
            cached_strat = prefetched.get("strategy")

            is_valid_cache = False
            current_selection = None
            if (
                self._is_prefetch_enabled(session, config)
                and (config_override is None or config_override.prefetch_enabled)
                and current_version == cached_version
                and session.current_difficulty == cached_diff
                and session.status == "IN_PROGRESS"
            ):
                current_selection = self.determine_next_question_target(
                    session_id=session.id,
                    db=db,
                    config_override=config,
                    request_override=request_override
                )
                breadth_strat_values = {s.value for s in self.BREADTH_STRATEGIES}
                if (
                    current_selection.strategy.value in breadth_strat_values
                    and current_selection.strategy.value == cached_strat
                    and current_selection.target_topic == prefetched.get("selection", {}).get("target_topic")
                ):
                    is_valid_cache = True

            if is_valid_cache and current_selection is not None:
                # Consume prefetched question
                sess_state = dict(session.session_state or {})
                consumed_data = sess_state.pop("prefetched_question")
                prefetch_meta = sess_state.setdefault("_prefetch_metrics", {
                    "prefetches_generated": 0,
                    "prefetches_consumed": 0,
                    "prefetches_discarded": 0,
                })
                prefetch_meta["prefetches_consumed"] += 1

                k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db)
                if current_selection.target_topic and current_selection.target_topic in k_state.get("topics", {}):
                    topic_item = k_state["topics"][current_selection.target_topic]
                    topic_item["coverage_status"] = TopicCoverageStatus.QUESTION_ASKED.value
                    topic_item["questions_asked_count"] = topic_item.get("questions_asked_count", 0) + 1

                if current_selection.target_project and current_selection.target_project in k_state.get("projects", {}):
                    proj_item = k_state["projects"][current_selection.target_project]
                    proj_item["coverage_status"] = TopicCoverageStatus.QUESTION_ASKED.value

                sess_state["knowledge_state"] = k_state
                session.session_state = sess_state
                flag_modified(session, "session_state")
                session.total_questions = existing_q_count + 1

                # If question was prefetched with persist=False, insert it now into DB
                q_id = UUID(str(consumed_data["question_id"]))
                cached_q = consumed_data["question"]
                existing_q_row = db.query(InterviewQuestion).filter(InterviewQuestion.id == q_id).first()
                if not existing_q_row:
                    new_q = InterviewQuestion(
                        id=q_id,
                        resume_id=session.resume_id,
                        session_id=session.id,
                        parent_question_id=current_selection.parent_question_id,
                        question_text=cached_q.get("question"),
                        category=cached_q.get("category", current_selection.category),
                        difficulty=cached_q.get("difficulty", current_selection.difficulty),
                        targeted_claim=cached_q.get("target_claim_id") or "Evidence-Grounded Question",
                        target_project=cached_q.get("target_project", current_selection.target_project),
                        target_entity=cached_q.get("target_entity", current_selection.target_entity),
                        target_claim_id=cached_q.get("target_claim_id"),
                        target_relationship=cached_q.get("target_relationship"),
                        evidence_ids=cached_q.get("evidence_ids", []) or [],
                        evidence_text=cached_q.get("evidence_text"),
                        rationale=cached_q.get("rationale") or current_selection.rationale,
                        grounding_score=cached_q.get("grounding_score", 1.0),
                        relevance_score=cached_q.get("relevance_score", 1.0),
                        technical_quality_score=cached_q.get("technical_quality_score", 1.0),
                        retrieval_trace_id=cached_q.get("trace_id"),
                        assessed_difficulty=cached_q.get("assessed_difficulty") or cached_q.get("difficulty", current_selection.difficulty),
                        rubric_version=cached_q.get("rubric_version", "v2.0-canonical-5level")
                    )
                    db.add(new_q)

                try:
                    db.commit()
                    db.refresh(session)
                except Exception as c_err:
                    db.rollback()
                    logger.error(f"Persistence error while consuming prefetch for session {session_id}: {c_err}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Database persistence failure: {str(c_err)}")

                t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
                resp = AdaptiveNextQuestionResponse(
                    session_id=session.id,
                    resume_id=session.resume_id,
                    status="GENERATED",
                    selection=current_selection,
                    question_id=q_id,
                    question_text=cached_q.get("question"),
                    category=cached_q.get("category", current_selection.category),
                    difficulty=cached_q.get("difficulty", current_selection.difficulty),
                    target_project=cached_q.get("target_project", current_selection.target_project),
                    target_entity=cached_q.get("target_entity", current_selection.target_entity),
                    evidence_text=cached_q.get("evidence_text"),
                    grounding_score=cached_q.get("grounding_score", 1.0),
                    assessed_difficulty=cached_q.get("assessed_difficulty") or cached_q.get("difficulty", current_selection.difficulty),
                    difficulty_rationale=cached_q.get("difficulty_rationale"),
                    evidence_ids=cached_q.get("evidence_ids", []) or [],
                    rubric_version=cached_q.get("rubric_version", "v2.0-canonical-5level"),
                    grounding_status=cached_q.get("grounding_status", "VERIFIED"),
                    message="Adaptive next question served from valid prefetch cache.",
                    latencies_ms={"total_ms": t_elapsed_ms, "rag_ms": 0.0, "prefetch_hit": 1.0}
                )
                if req_key:
                    idem_cache = session.session_state.setdefault("_idempotency_cache", {})
                    idem_cache[req_key] = resp.model_dump(mode="json")
                    flag_modified(session, "session_state")
                    try:
                        db.commit()
                    except Exception as c_err:
                        db.rollback()
                        logger.error(f"Persistence error while saving idempotency cache for session {session_id}: {c_err}", exc_info=True)
                        raise HTTPException(status_code=500, detail=f"Database persistence failure: {str(c_err)}")

                # Schedule next prefetch in background if enabled
                try:
                    self._maybe_schedule_prefetch(session, background_tasks, db, config)
                except Exception as pf_err:
                    logger.debug(f"Failed to schedule prefetch for session {session.id}: {pf_err}")
                return resp
            else:
                logger.info(f"Discarding stale or mismatched prefetched question for session {session_id}.")
                sess_state = dict(session.session_state or {})
                discarded_data = sess_state.pop("prefetched_question", None)
                prefetch_meta = sess_state.setdefault("_prefetch_metrics", {
                    "prefetches_generated": 0,
                    "prefetches_consumed": 0,
                    "prefetches_discarded": 0,
                })
                prefetch_meta["prefetches_discarded"] += 1
                session.session_state = sess_state
                flag_modified(session, "session_state")
                # Clean up unserved discarded question from DB
                if discarded_data and discarded_data.get("question_id"):
                    try:
                        db.query(InterviewQuestion).filter(
                            InterviewQuestion.id == UUID(str(discarded_data["question_id"]))
                        ).delete()
                    except Exception as del_err:
                        logger.debug(f"Could not delete discarded prefetch question from DB: {del_err}")
                try:
                    db.commit()
                except Exception as c_err:
                    db.rollback()
                    logger.error(f"Persistence error while discarding stale prefetch for session {session_id}: {c_err}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Database persistence failure during prefetch discard: {str(c_err)}")

        # 1 & 2. Determine next question selection target and execute RAG generation
        # Bounded by configurable retry limit and candidate latency budget
        excluded_targets: Set[str] = set()
        excluded_question_texts: List[str] = []
        TOTAL_DEADLINE_S = getattr(settings, "TOTAL_QUESTION_DEADLINE_SECONDS", 12.0)
        TIMEOUT_S = config.candidate_timeout_seconds if (config.candidate_timeout_seconds and config.candidate_timeout_seconds > 0) else TOTAL_DEADLINE_S
        MIN_LLM_BUDGET_S = min(getattr(settings, "MIN_LLM_CALL_BUDGET", 2.5), max(0.0001, TIMEOUT_S * 0.5))
        MAX_CANDIDATE_ATTEMPTS = max(1, config.max_candidate_retries)
        rag_resp = None
        selection = None
        t_rag_total = 0.0
        rejection_reasons: List[str] = []

        for candidate_idx in range(MAX_CANDIDATE_ATTEMPTS):
            elapsed_candidate = time.perf_counter() - t_start
            remaining_candidate = TIMEOUT_S - elapsed_candidate
            # Check elapsed time against total deadline budget and min LLM budget
            if remaining_candidate < MIN_LLM_BUDGET_S:
                logger.warning(
                    f"Candidate generation remaining budget {remaining_candidate:.2f}s < minimum LLM budget {MIN_LLM_BUDGET_S}s "
                    f"(attempt {candidate_idx + 1}/{MAX_CANDIDATE_ATTEMPTS}) for session {session_id}; terminating retry loop."
                )
                break
            selection = self.determine_next_question_target(
                session_id=session_id,
                db=db,
                config_override=config,
                request_override=request_override,
                excluded_topics=excluded_targets
            )

            # TASK 3 Evidence gate before any LLM call:
            # If candidate target has target_project, verify >= 1 chunk/relationship/claim from projects/experience exists
            if selection.target_project:
                PROJECT_EXP_SECTIONS = {
                    "projects", "project", "experience", "work_experience", 
                    "employment", "professional_experience", "work", "work experience"
                }
                p_norm = selection.target_project.strip().lower()
                all_chunks = (
                    db.query(ResumeChunk)
                    .filter(ResumeChunk.resume_id == session.resume_id)
                    .all()
                )
                all_rels = (
                    db.query(Relationship)
                    .filter(Relationship.resume_id == session.resume_id)
                    .all()
                )
                all_claims = (
                    db.query(ResumeClaim)
                    .filter(ResumeClaim.resume_id == session.resume_id)
                    .all()
                )
                matching_proj_chunks = [
                    c for c in all_chunks
                    if ((c.section and c.section.strip().lower() in PROJECT_EXP_SECTIONS) or (c.project and c.project.strip()))
                    and ((c.project and p_norm in c.project.lower()) or (p_norm in (c.chunk_text or "").lower()))
                ]
                matching_proj_rels = [
                    r for r in all_rels
                    if (r.context_entity and p_norm in r.context_entity.lower())
                    or ((r.section and r.section.strip().lower() in PROJECT_EXP_SECTIONS) and (p_norm in (r.evidence_text or "").lower()))
                ]
                matching_proj_claims = [
                    clm for clm in all_claims
                    if (clm.project and p_norm in clm.project.lower())
                    or ((clm.section and clm.section.strip().lower() in PROJECT_EXP_SECTIONS) and (p_norm in (clm.claim_text or "").lower()))
                ]
                if not (matching_proj_chunks or matching_proj_rels or matching_proj_claims):
                    logger.warning(
                        f"Skipping candidate target '{selection.target_topic}' on project '{selection.target_project}': "
                        f"zero project/experience evidence found. Skipping without LLM call."
                    )
                    if selection.target_topic:
                        excluded_targets.add(selection.target_topic)
                    if selection.target_project:
                        excluded_targets.add(selection.target_project)
                    continue

                if selection.target_entity:
                    from app.core.canonicalizer import CANONICAL_TAXONOMY
                    can_ent, _ = canonicalize_entity(selection.target_entity)
                    e_canon = can_ent or selection.target_entity
                    e_aliases = {selection.target_entity.lower(), e_canon.lower()}
                    tax = CANONICAL_TAXONOMY.get(e_canon)
                    if tax and "aliases" in tax:
                        for a in tax["aliases"]:
                            e_aliases.add(a.lower())

                    has_entity_in_proj = (
                        any(
                            any(re.search(r"\b" + re.escape(a) + r"\b", (c.chunk_text or "").lower()) for a in e_aliases)
                            for c in matching_proj_chunks
                        ) or any(
                            any(a in (r.source_entity or "").lower() or a in (r.target_entity or "").lower() or a in (r.evidence_text or "").lower() for a in e_aliases)
                            for r in matching_proj_rels
                        ) or any(
                            any(a in (clm.target_entity or "").lower() or a in (clm.claim_text or "").lower() for a in e_aliases)
                            for clm in matching_proj_claims
                        )
                    )
                    if not has_entity_in_proj:
                        logger.warning(
                            f"Skipping candidate target entity '{selection.target_entity}' on project '{selection.target_project}': "
                            f"entity is absent from project chunks/relationships/claims. Skipping without LLM call."
                        )
                        if selection.target_topic:
                            excluded_targets.add(selection.target_topic)
                        if selection.target_entity:
                            excluded_targets.add(selection.target_entity)
                        continue

            rag_req = QuestionGenerationRequest(
                category=selection.category,
                difficulty=selection.difficulty,
                target_project=selection.target_project,
                target_entity=selection.target_entity,
                session_id=session.id,
                parent_question_id=selection.parent_question_id,
                target_claim_id=selection.target_claim_id,
                target_relationship=selection.target_relationship,
                follow_up_reason=selection.follow_up_reason,
                excluded_question_texts=list(excluded_question_texts)
            )

            t_rag_s = time.perf_counter()
            try:
                rag_resp = rag_pipeline.generate_grounded_question(
                    resume_id=session.resume_id,
                    req=rag_req,
                    db=db,
                    deadline_seconds=TOTAL_DEADLINE_S,
                    t_global_start=t_start
                )
            except TypeError as te:
                if "deadline_seconds" in str(te) or "unexpected keyword argument" in str(te):
                    rag_resp = rag_pipeline.generate_grounded_question(
                        resume_id=session.resume_id,
                        req=rag_req,
                        db=db
                    )
                else:
                    raise
            t_rag_total += (time.perf_counter() - t_rag_s) * 1000

            if rag_resp.validation_status == "ACCEPTED" and rag_resp.question:
                break

            # Discard rejected candidate and collect concepts/texts to exclude from subsequent attempts
            rejection_reasons.append(rag_resp.message or rag_resp.error_code or "rejected")
            if selection.target_topic:
                excluded_targets.add(selection.target_topic)
            if selection.target_entity:
                excluded_targets.add(selection.target_entity)

            # Accumulate rejected candidate texts so subsequent candidates and targets avoid them
            if getattr(rag_resp, "rejected_candidates", None):
                for rej_text in rag_resp.rejected_candidates:
                    if rej_text and rej_text not in excluded_question_texts:
                        excluded_question_texts.append(rej_text)

            logger.info(
                f"Candidate attempt {candidate_idx + 1}/{MAX_CANDIDATE_ATTEMPTS} for target '{selection.target_topic}' "
                f"was rejected during RAG generation ({rag_resp.message}, code={rag_resp.error_code}). "
                f"Discarded candidate; rotating to next technical dimension..."
            )

        t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # 3. Handle RAG validation failures gracefully without corrupting session
        if not rag_resp or rag_resp.validation_status != "ACCEPTED" or not rag_resp.question:
            elapsed_s = time.perf_counter() - t_start
            is_timeout = (elapsed_s >= (TIMEOUT_S - MIN_LLM_BUDGET_S)) or (rag_resp and getattr(rag_resp, "error_code", "") in ("timeout", "provider_error"))
            err_code = (
                "timeout" if is_timeout else (
                    rag_resp.error_code if rag_resp and rag_resp.error_code else "insufficient_evidence"
                )
            )
            if is_timeout:
                failure_msg = f"Question generation timed out after {round(elapsed_s, 1)}s. Question is still preparing, please retry."
                resp_status = "EVIDENCE_UNAVAILABLE"
            elif err_code == "redundancy_rejection":
                failure_msg = "All candidate questions across attempted technical dimensions were rejected as semantic duplicates. Session state remains active. Please retry to explore alternative technical dimensions."
                resp_status = "EVIDENCE_UNAVAILABLE"
            else:
                failure_msg = f"Could not generate grounded question for '{selection.target_topic if selection else 'target'}': {rag_resp.message if rag_resp else 'No candidate evidence'}"
                resp_status = "EVIDENCE_UNAVAILABLE"

            logger.warning(
                f"RAG question generation was unable to generate question for session {session_id} (error_code={err_code}): {failure_msg}"
            )
            return AdaptiveNextQuestionResponse(
                session_id=session.id,
                resume_id=session.resume_id,
                status=resp_status,
                error_code=err_code,
                rubric_version="v2.0-canonical-5level",
                selection=selection,
                message=failure_msg
            )

        # 4. Successfully generated: update knowledge state topic coverage and session difficulty
        try:
            k_state = knowledge_state_service.get_or_initialize_knowledge_state(session, db)
            session.current_difficulty = selection.difficulty

            # Update topic coverage status in session_state JSON
            if selection.target_topic and selection.target_topic in k_state.get("topics", {}):
                topic_item = k_state["topics"][selection.target_topic]
                topic_item["coverage_status"] = TopicCoverageStatus.QUESTION_ASKED.value
                topic_item["questions_asked_count"] = topic_item.get("questions_asked_count", 0) + 1

            if selection.target_project and selection.target_project in k_state.get("projects", {}):
                proj_item = k_state["projects"][selection.target_project]
                proj_item["coverage_status"] = TopicCoverageStatus.QUESTION_ASKED.value

            sess_state = dict(session.session_state or {})
            sess_state["knowledge_state"] = k_state

            generated_q = rag_resp.question
            resp = AdaptiveNextQuestionResponse(
                session_id=session.id,
                resume_id=session.resume_id,
                status="GENERATED",
                selection=selection,
                question_id=rag_resp.question_id,
                question_text=generated_q.question if generated_q else None,
                category=generated_q.category if generated_q else selection.category,
                difficulty=generated_q.difficulty if generated_q else selection.difficulty,
                target_project=generated_q.target_project if generated_q else selection.target_project,
                target_entity=generated_q.target_entity if generated_q else selection.target_entity,
                evidence_text=generated_q.evidence_text if generated_q else None,
                grounding_score=generated_q.grounding_score if generated_q else 1.0,
                assessed_difficulty=getattr(generated_q, "assessed_difficulty", None) or (rag_resp.assessed_difficulty if rag_resp else None) or (generated_q.difficulty if generated_q else selection.difficulty),
                difficulty_rationale=getattr(generated_q, "difficulty_rationale", None),
                evidence_ids=getattr(generated_q, "evidence_ids", []) or [],
                rubric_version=getattr(generated_q, "rubric_version", "v2.0-canonical-5level"),
                grounding_status=getattr(generated_q, "grounding_status", "VERIFIED"),
                message="Adaptive next question selected and generated successfully.",
                latencies_ms={"total_ms": t_elapsed_ms, "rag_ms": round(t_rag_total, 2)}
            )

            # Store in idempotency cache if request key was provided
            if req_key:
                idem_cache = sess_state.setdefault("_idempotency_cache", {})
                idem_cache[req_key] = resp.model_dump(mode="json")

            session.session_state = sess_state
            flag_modified(session, "session_state")
            session.total_questions = existing_q_count + 1

            db.commit()
            db.refresh(session)
            logger.info(
                f"[TIMING] select_and_generate_next_question: session={session.id} "
                f"total_elapsed_ms={t_elapsed_ms} rag_ms={round(t_rag_total, 2)} "
                f"target='{selection.target_topic}' difficulty={selection.difficulty}"
            )

            # Schedule next prefetch in background if enabled
            try:
                self._maybe_schedule_prefetch(session, background_tasks, db, config)
            except Exception as pf_err:
                logger.debug(f"Failed to schedule prefetch for session {session.id}: {pf_err}")
            return resp
        except Exception as e:
            db.rollback()
            logger.error(f"Persistence error while updating session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database persistence failure during session update: {str(e)}")


def background_prefetch_next_question(session_id: UUID, expected_state_version: str) -> None:
    """FastAPI BackgroundTask worker for prefetching the next breadth question."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        adaptive_selection_engine.execute_prefetch(session_id, expected_state_version, db)
    except Exception as e:
        logger.warning(f"Background prefetch failed for session {session_id}: {e}")
    finally:
        db.close()


adaptive_selection_engine = AdaptiveSelectionEngine()
adaptive_selection_service = adaptive_selection_engine
