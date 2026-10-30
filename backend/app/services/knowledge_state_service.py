import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.canonicalizer import canonicalize_entity
from app.models.resume import Resume
from app.models.skill import ResumeSkill, Skill
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import InterviewSession, InterviewQuestion, InterviewAnswer
from app.schemas.knowledge_state import (
    TopicMasteryState,
    ProjectUnderstandingState,
    RelationshipUnderstandingState,
    ResumeClaimVerificationState,
    DetailedKnowledgeGap,
    StrongWeakSummary,
    TopicCoverageSummary,
    CandidateKnowledgeStateResponse,
    TopicAssessmentStatus,
    TopicCoverageStatus,
    ClaimVerificationStatus,
    GapSeverity,
    GapNature,
)
from app.schemas.adaptive_selection import AdaptiveStrategyType

logger = logging.getLogger(__name__)

def calculate_difficulty_aware_mastery(
    historical_evaluations: List[Dict[str, Any]],
    latest_score: float,
    latest_difficulty: int
) -> float:
    """
    Calculates difficulty-aware, recency-weighted topic mastery score in [0.0, 100.0].

    FORMULA SPECIFICATION (Part B2):
    1. Raw scores (0-100) are stored directly without pre-multiplying.
    2. Difficulty levels 1 to 5 map to linear weights:
       weight(d) = 1.0 + 0.25 * (d - 3)
       - Level 1: weight = 0.50
       - Level 2: weight = 0.75
       - Level 3: weight = 1.00 (Neutral baseline)
       - Level 4: weight = 1.25
       - Level 5: weight = 1.50

    3. Bounded difficulty adjustment per question:
       To reflect demonstrated question depth without overflowing [0, 100] or requiring clamping:
       adj(s, d) = ((d - 3) / 2.0) * 5.0 * ((100.0 - s) / 100.0 if d >= 3 else (s / 100.0))
       v(s, d) = s + adj(s, d)

       Mathematical Properties:
       - Strictly bounded in [0, 100]: at s=100, adj=0 -> v=100. At s=0, adj=0 -> v=0.
       - Max adjustment is capped at +/- 5.0 points total across any difficulty.
       - Monotonic: for any fixed difficulty d, d(v)/ds > 0 strictly.
       - A score of 60 at difficulty 5 (v = 62.0) never exceeds a score of 80 at difficulty 1 (v = 76.0).
       - Identical scores at different difficulties differ by at most the documented 5.0-point bound.
       - No input combination requires artificial clamping.

    4. Recency-weighted aggregation across multiple questions:
       - Single evaluation (n = 1):
         mastery = v(s_1, d_1)
       - Multiple evaluations (n > 1):
         Prior questions are aggregated via weighted average:
           prior_weighted_avg = sum(w_i * v_i for i in 1..n-1) / sum(w_i for i in 1..n-1)
           prior_avg_weight = sum(w_i for i in 1..n-1) / (n - 1)
         Combined with 60% latest / 40% prior weighting applied to the weighted values:
           w_latest = 0.60 * weight(d_n)
           w_prior  = 0.40 * prior_avg_weight
           mastery = (w_latest * v_latest + w_prior * prior_weighted_avg) / (w_latest + w_prior)

    WORKED EXAMPLE:
    Candidate answers two questions on PostgreSQL:
    - Attempt 1: Difficulty 3 (weight = 1.0), raw score = 70.0.
      adj = 0.0, v_1 = 70.0.
      Mastery = 70.0.
    - Attempt 2: Difficulty 5 (weight = 1.5), raw score = 80.0.
      adj = (2/2) * 5.0 * (20/100) = +1.0 -> v_2 = 81.0.
      w_latest = 0.60 * 1.5 = 0.90
      w_prior  = 0.40 * 1.0 = 0.40
      mastery = (0.90 * 81.0 + 0.40 * 70.0) / (0.90 + 0.40)
              = (72.9 + 28.0) / 1.30 = 100.9 / 1.30 = 77.62 -> 77.6.
    """
    def _diff_weight(d: int) -> float:
        d_clamped = max(1, min(5, d))
        return 1.0 + 0.25 * (d_clamped - 3)

    def _adjusted_val(s: float, d: int) -> float:
        d_clamped = max(1, min(5, d))
        if d_clamped >= 3:
            adj = ((d_clamped - 3) / 2.0) * 5.0 * ((100.0 - s) / 100.0)
        else:
            adj = ((d_clamped - 3) / 2.0) * 5.0 * (s / 100.0)
        return s + adj

    v_latest = _adjusted_val(latest_score, latest_difficulty)

    if not historical_evaluations:
        return round(v_latest, 1)

    prior_vals = [
        _adjusted_val(float(h.get("score", latest_score)), int(h.get("difficulty", 3)))
        for h in historical_evaluations
        if isinstance(h, dict) and "score" in h and h["score"] is not None
    ]

    if not prior_vals:
        return round(v_latest, 1)

    prior_mastery = prior_vals[0]
    for p in prior_vals[1:]:
        prior_mastery = 0.60 * p + 0.40 * prior_mastery

    mastery = 0.60 * v_latest + 0.40 * prior_mastery
    return round(mastery, 1)


# Generic blocklist for non-assessable junk topics (Task 2)
JUNK_TOPIC_BLOCKLIST: Set[str] = {
    # Generic entity labels
    "candidate", "applicant", "student", "developer", "engineer", "lead", "fresher", "person",
    # Academic & educational terms
    "institute", "college", "university", "school", "campus", "academy", "department",
    "bachelor", "master", "phd", "degree", "b.tech", "m.tech", "btech", "mtech", "b.e", "m.e", "bca", "mca",
    "curriculum", "cgpa", "gpa", "percentage", "marks", "grade", "education",
    # Competitive ranks & tests
    "rank", "ranking", "ranks", "eapcet", "jee", "gate", "cet", "cat", "gre", "toefl", "ielts",
    "1st place", "2nd place", "3rd place", "first place", "second place", "third place",
    "hackathon", "competition", "contest", "winner", "participant", "runner up",
    "award", "scholarship", "medal", "honors", "honor", "certificate", "certification", "achievements",
    # Spoken natural languages (languages spoken vs programming languages)
    "english", "hindi", "telugu", "tamil", "kannada", "malayalam", "marathi", "bengali", "gujarati",
    "spanish", "french", "german", "mandarin", "japanese", "native speaker", "languages",
    # Generic metrics and performance words
    "accuracy", "latency", "performance", "speed", "score", "f1-score", "metric", "efficiency",
    # Contact / header artifacts
    "header", "contact", "phone", "email", "mobile", "address", "linkedin", "github", "gitlab", "portfolio",
    "resume", "curriculum vitae", "cv", "unknown", "general", "redacted"
}


def is_assessable_topic(
    name: str,
    topic_type: Optional[str] = None,
    section: Optional[str] = None,
    is_project: bool = False
) -> bool:
    """
    Validates whether a topic is technically assessable.
    Only creates assessable topics from:
    (a) canonical skill taxonomy, and
    (b) project/experience names.
    Excludes education, ranks, spoken languages, achievements, and blocklisted words.
    """
    if not name or not name.strip():
        return False

    name_clean = name.strip()
    name_lower = name_clean.lower()

    # 1. Direct blocklist check
    if name_lower in JUNK_TOPIC_BLOCKLIST:
        return False

    # 2. Substring/token check against generic words
    for junk_pattern in [
        r"\b(?:institute|college|university|school|academy)\b",
        r"\b(?:bachelor|master|b\.tech|m\.tech|degree|cgpa|gpa)\b",
        r"\b(?:rank\s+of|rank\s+\d+|eapcet|jee|gate)\b",
        r"\b(?:1st|2nd|3rd|first|second|third)\s+place\b",
        r"\b(?:hackathon|competition|winner|runner\s+up)\b",
    ]:
        if re.search(junk_pattern, name_lower):
            return False

    # 3. Non-assessable section check
    if section and section.lower() in ("education", "achievements", "languages", "leadership", "header", "contact"):
        return False

    # 4. Check if it's a project/experience name
    if is_project:
        return True

    # 5. Check if it's recognized in canonical skill taxonomy
    canon, cat = canonicalize_entity(name_clean)
    if canon and canon.lower() not in JUNK_TOPIC_BLOCKLIST and cat != "General":
        return True

    # Also check if declared topic_type is a valid technical category
    if topic_type and topic_type.lower() in [
        "programming language", "framework", "database", "devops", "cloud", "ai/ml", "tool", "skill", "architecture", "software"
    ]:
        return True

    return False


class CandidateKnowledgeStateEngine:
    """
    Candidate Knowledge State Engine (Phase 7.2).
    Maintains a deterministic, persistent, and auditable representation of candidate
    skill/technology mastery, project understanding, relationship reasoning,
    resume-claim verification, knowledge gaps, and topic coverage.
    """

    def initialize_knowledge_state(
        self,
        session: InterviewSession,
        db: Session
    ) -> Dict[str, Any]:
        """
        Initializes an empty knowledge state baseline grounded in the candidate's parsed resume.
        Pre-registers all extracted skills, projects, relationships, and claims as UNASSESSED / UNTESTED.
        """
        resume_id = session.resume_id
        now_utc = datetime.now(timezone.utc)

        # 1. Discover all resume skills
        topics_dict: Dict[str, Dict[str, Any]] = {}
        resume_skills = (
            db.query(ResumeSkill, Skill)
            .join(Skill, ResumeSkill.skill_id == Skill.id)
            .filter(ResumeSkill.resume_id == resume_id)
            .all()
        )
        for rs, s in resume_skills:
            canonical_name, topic_type = canonicalize_entity(s.name)
            if not canonical_name:
                canonical_name = s.name
            t_type = topic_type if topic_type != "General" else (s.category or "Skill")
            is_ass = is_assessable_topic(canonical_name, topic_type=t_type)

            topics_dict[canonical_name] = {
                "topic_id": canonical_name,
                "topic_name": canonical_name,
                "topic_type": t_type,
                "is_assessable": is_ass,
                "assessment_status": TopicAssessmentStatus.UNASSESSED.value,
                "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                "questions_asked_count": 0,
                "answers_evaluated_count": 0,
                "latest_score": None,
                "historical_scores": [],
                "mastery_score": 0.0,
                "confidence": 0.0,
                "last_assessed_at": None,
            }

        # 2. Discover all extracted relationships
        rels = db.query(Relationship).filter(Relationship.resume_id == resume_id).all()
        relationships_list: List[Dict[str, Any]] = []
        discovered_projects: Set[str] = set()

        # First discover all genuine project names
        for r in rels:
            if r.context_entity and r.context_entity.strip():
                clean_ctx = r.context_entity.strip()
                if (r.section == "Projects" or "project" in (r.section or "").lower() or "built" in (r.relation_type or "").lower()) and clean_ctx.lower() not in JUNK_TOPIC_BLOCKLIST:
                    discovered_projects.add(clean_ctx)

        for r in rels:
            sec_lower = (r.section or "").lower()
            is_non_tech_section = sec_lower in ("education", "achievements", "languages", "leadership", "header", "contact")

            # Also register entities from relationships into topics with strict assessability gating
            for entity in [r.source_entity, r.target_entity]:
                if not entity or not entity.strip():
                    continue
                can_ent, ent_type = canonicalize_entity(entity)
                ent_name = can_ent or entity.strip()
                ent_lower = ent_name.lower()

                # Block generic and junk words completely
                if ent_lower in JUNK_TOPIC_BLOCKLIST or any(re.search(p, ent_lower) for p in [
                    r"\b(?:institute|college|university|school|academy)\b",
                    r"\b(?:bachelor|master|b\.tech|m\.tech|degree|cgpa|gpa)\b",
                    r"\b(?:rank\s+of|rank\s+\d+|eapcet|jee|gate)\b",
                    r"\b(?:1st|2nd|3rd|first|second|third)\s+place\b",
                    r"\b(?:hackathon|competition|winner|runner\s+up)\b",
                ]):
                    continue

                is_proj = (ent_name in discovered_projects) or (r.section == "Projects" and ent_name == r.context_entity)
                is_ass = is_assessable_topic(ent_name, topic_type=ent_type, section=r.section, is_project=is_proj)
                if is_non_tech_section:
                    is_ass = False

                if ent_name not in topics_dict:
                    topics_dict[ent_name] = {
                        "topic_id": ent_name,
                        "topic_name": ent_name,
                        "topic_type": ent_type,
                        "is_assessable": is_ass,
                        "assessment_status": TopicAssessmentStatus.UNASSESSED.value,
                        "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                        "questions_asked_count": 0,
                        "answers_evaluated_count": 0,
                        "latest_score": None,
                        "historical_scores": [],
                        "mastery_score": 0.0,
                        "confidence": 0.0,
                        "last_assessed_at": None,
                    }
                elif not topics_dict[ent_name].get("is_assessable"):
                    topics_dict[ent_name]["is_assessable"] = is_ass

            rel_key = f"{r.source_entity}->{r.relation_type}->{r.target_entity}"
            relationships_list.append({
                "relationship_id": str(r.id) if r.id else rel_key,
                "source_entity": r.source_entity,
                "relation_type": r.relation_type,
                "target_entity": r.target_entity,
                "context_entity": r.context_entity,
                "is_tested": False,
                "status": "UNTESTED",
                "demonstrated_understanding": "UNTESTED",
                "score": None,
                "evidence": r.evidence_text,
                "supporting_question_id": None,
                "notes": None,
            })

        # 3. Discover all resume claims
        claims = db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).all()
        claims_list: List[Dict[str, Any]] = []
        for c in claims:
            if c.project and c.project.strip():
                clean_p = c.project.strip()
                if clean_p.lower() not in JUNK_TOPIC_BLOCKLIST:
                    discovered_projects.add(clean_p)

            claims_list.append({
                "claim_id": c.claim_id or str(c.id),
                "claim_text": c.claim_text,
                "project": c.project,
                "target_entity": c.target_entity,
                "relation_type": c.relation_type,
                "status": ClaimVerificationStatus.UNTESTED.value,
                "evidence_text": c.evidence_text,
                "verification_source_question_id": None,
                "confidence": 0.0,
                "last_updated_at": None,
                "notes": None,
            })

        # 4. Initialize projects dictionary and ensure projects are in topics_dict as assessable
        projects_dict: Dict[str, Dict[str, Any]] = {}
        for proj in sorted(list(discovered_projects)):
            if proj.lower() in JUNK_TOPIC_BLOCKLIST:
                continue
            proj_id = re.sub(r"[^\w\s-]", "", proj).strip().replace(" ", "_")
            projects_dict[proj] = {
                "project_id": proj_id or proj,
                "project_name": proj,
                "is_assessed": False,
                "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                "questions_asked_count": 0,
                "answers_evaluated_count": 0,
                "average_score": None,
                "latest_score": None,
                "architecture_understanding": "UNASSESSED",
                "implementation_understanding": "UNASSESSED",
                "known_knowledge_gaps": [],
                "remaining_areas_to_assess": ["Architecture", "Implementation", "Tech Stack Trade-offs"],
                "tested_question_ids": [],
            }

            if proj not in topics_dict:
                topics_dict[proj] = {
                    "topic_id": proj,
                    "topic_name": proj,
                    "topic_type": "Project",
                    "is_assessable": True,
                    "assessment_status": TopicAssessmentStatus.UNASSESSED.value,
                    "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                    "questions_asked_count": 0,
                    "answers_evaluated_count": 0,
                    "latest_score": None,
                    "historical_scores": [],
                    "mastery_score": 0.0,
                    "confidence": 0.0,
                    "last_assessed_at": None,
                }
            else:
                topics_dict[proj]["is_assessable"] = True

        # 5. Build summary and coverage computed ONLY over assessable topics
        assessable_topics = {k: v for k, v in topics_dict.items() if v.get("is_assessable", False)}
        unassessed = sorted(list(assessable_topics.keys()))
        total_assessable = len(assessable_topics)

        summary = {
            "demonstrated_strengths": [],
            "demonstrated_weaknesses": [],
            "partially_understood": [],
            "unassessed_topics": unassessed,
            "topics_requiring_followup": [],
        }

        coverage = {
            "total_resume_topics": total_assessable,
            "total_assessable_topics": total_assessable,
            "not_considered_count": total_assessable,
            "question_asked_count": 0,
            "partially_assessed_count": 0,
            "adequately_assessed_count": 0,
            "requires_follow_up_count": 0,
            "coverage_percentage": 0.0,
        }

        knowledge_state = {
            "session_id": str(session.id),
            "resume_id": str(resume_id),
            "topics": topics_dict,
            "projects": projects_dict,
            "relationships": relationships_list,
            "claims": claims_list,
            "knowledge_gaps": [],
            "summary": summary,
            "coverage": coverage,
            "last_updated_at": now_utc.isoformat(),
            "_processed_answer_ids": [],
            "_question_latest_attempts": {},
        }

        return knowledge_state

    def get_or_initialize_knowledge_state(
        self,
        session: InterviewSession,
        db: Session
    ) -> Dict[str, Any]:
        """
        Retrieves existing knowledge state from session_state or initializes it if absent or corrupted.
        """
        try:
            db.refresh(session)
        except Exception:
            pass
        state = dict(session.session_state or {})
        k_state = state.get("knowledge_state")
        is_corrupt = False
        if not k_state or not isinstance(k_state, dict):
            is_corrupt = True
        else:
            required_dict_keys = ["topics", "projects", "summary", "coverage"]
            required_list_keys = ["relationships", "claims", "knowledge_gaps"]
            for k in required_dict_keys:
                if not isinstance(k_state.get(k), dict):
                    is_corrupt = True
                    break
            if not is_corrupt:
                for k in required_list_keys:
                    if not isinstance(k_state.get(k), list):
                        is_corrupt = True
                        break

        if is_corrupt:
            logger.warning(f"Corrupt or missing knowledge state detected for session {session.id}. Re-initializing cleanly.")
            k_state = self.initialize_knowledge_state(session, db)
            state["knowledge_state"] = k_state
            session.session_state = state
            flag_modified(session, "session_state")
            try:
                db.commit()
                db.refresh(session)
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to commit re-initialized knowledge state for session {session.id}: {e}")
        return k_state

    def update_knowledge_state_with_answer(
        self,
        session: InterviewSession,
        question: InterviewQuestion,
        answer: InterviewAnswer,
        db: Session
    ) -> Dict[str, Any]:
        return self.update_from_evaluation(session.id, question.id, answer, db)

    def update_from_evaluation(
        self,
        session_id: UUID,
        question_id: UUID,
        answer: InterviewAnswer,
        db: Session
    ) -> Dict[str, Any]:
        """
        Updates the Candidate Knowledge State in response to an evaluated answer.
        Guarantees idempotency and protects previously recorded knowledge.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            logger.warning(f"Session {session_id} not found when updating knowledge state.")
            return {}

        question = db.query(InterviewQuestion).filter(InterviewQuestion.id == question_id).first()
        if not question:
            logger.warning(f"Question {question_id} not found when updating knowledge state.")
            return {}

        # Validate resume & session integrity
        if session.resume_id != question.resume_id:
            logger.warning(f"Resume mismatch for question {question_id} and session {session_id}.")
            return {}

        # Safeguard: Do not process answers that are not evaluated or lack score
        if answer.status != "EVALUATED" or answer.overall_score is None:
            logger.info(f"Answer {answer.id} is not evaluated or lacks overall_score (status={answer.status}). Skipping knowledge state update.")
            return self.get_or_initialize_knowledge_state(session, db)

        k_state = self.get_or_initialize_knowledge_state(session, db)
        now_utc = datetime.now(timezone.utc)

        # Idempotency check: track processed answer attempt
        processed_ids: List[str] = k_state.get("_processed_answer_ids", [])
        ans_id_str = str(answer.id)
        if ans_id_str in processed_ids:
            logger.info(f"Answer {ans_id_str} has already been processed for knowledge state. Skipping duplicate processing.")
            return k_state

        topics_map: Dict[str, Dict[str, Any]] = k_state.setdefault("topics", {})
        projects_map: Dict[str, Dict[str, Any]] = k_state.setdefault("projects", {})
        relationships_list: List[Dict[str, Any]] = k_state.setdefault("relationships", [])
        claims_list: List[Dict[str, Any]] = k_state.setdefault("claims", [])
        gaps_list: List[Dict[str, Any]] = k_state.setdefault("knowledge_gaps", [])
        q_latest_map: Dict[str, Any] = k_state.setdefault("_question_latest_attempts", {})

        raw_score = answer.overall_score if answer.overall_score is not None else 0.0
        # Support explicit 0-10 scale mapping to 0-100 rubric scale
        score = raw_score * 10.0 if (0.0 < raw_score <= 10.0) else raw_score
        eval_conf = answer.evaluation_confidence if answer.evaluation_confidence is not None else 1.0
        is_refusal = (
            getattr(answer, "evaluator_model", "") == "fast_path_refusal"
            or (answer.overall_score == 0.0 and "refusal" in (answer.evaluation_feedback or "").lower())
        )

        # 1. Identify Target Topics for this Question
        associated_topics: Set[str] = set()
        if question.target_entity:
            can_t, _ = canonicalize_entity(question.target_entity)
            if can_t:
                associated_topics.add(can_t)

        if question.expected_concepts:
            for c in question.expected_concepts:
                if isinstance(c, str) and c.strip():
                    can_c, _ = canonicalize_entity(c.strip())
                    if can_c:
                        associated_topics.add(can_c)

        if not associated_topics:
            # Fallback to category if no specific entity was targeted
            cat_name = question.category or "General"
            associated_topics.add(cat_name)

        q_id_str = str(question.id)
        is_first_eval_for_question = q_id_str not in q_latest_map

        # Record question latest attempt
        q_latest_map[q_id_str] = {
            "latest_answer_id": ans_id_str,
            "attempt_number": answer.attempt_number,
            "score": score,
            "status": answer.status,
            "updated_at": now_utc.isoformat(),
        }

        # 2. Update Topic Mastery for Associated Topics
        for topic_name in associated_topics:
            if topic_name not in topics_map:
                can_name, t_type = canonicalize_entity(topic_name)
                topics_map[topic_name] = {
                    "topic_id": topic_name,
                    "topic_name": topic_name,
                    "topic_type": t_type if t_type != "General" else "Skill",
                    "assessment_status": TopicAssessmentStatus.UNASSESSED.value,
                    "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                    "questions_asked_count": 0,
                    "answers_evaluated_count": 0,
                    "latest_score": None,
                    "historical_scores": [],
                    "mastery_score": 0.0,
                    "confidence": 0.0,
                    "last_assessed_at": None,
                }

            t_data = topics_map[topic_name]

            if is_first_eval_for_question:
                t_data["questions_asked_count"] = t_data.get("questions_asked_count", 0) + 1
            t_data["answers_evaluated_count"] = t_data.get("answers_evaluated_count", 0) + 1
            t_data["latest_score"] = round(score, 1)

            historical_scores = t_data.get("historical_scores", [])
            historical_scores.append(round(score, 1))
            t_data["historical_scores"] = historical_scores

            # Maintain historical evaluations with difficulty for B2 mastery calculation
            hist_evals = t_data.setdefault("historical_evaluations", [])
            prior_evals = list(hist_evals)
            q_diff = getattr(question, "difficulty", 3) or 3
            hist_evals.append({"score": round(score, 1), "difficulty": q_diff})

            t_data["mastery_score"] = calculate_difficulty_aware_mastery(
                prior_evals, round(score, 1), q_diff
            )

            m_count = t_data["answers_evaluated_count"]

            # Calculate deterministic confidence
            if m_count == 1:
                conf = min(0.50, round(0.50 * eval_conf, 2))
            elif m_count == 2:
                conf = min(0.80, round(0.80 * eval_conf, 2))
            else:
                conf = min(0.95, round((0.80 + 0.05 * (m_count - 2)) * eval_conf, 2))
            t_data["confidence"] = conf

            # Assessment status progression
            if m_count == 1:
                t_data["assessment_status"] = TopicAssessmentStatus.PARTIALLY_ASSESSED.value
            elif m_count >= 2 and conf >= 0.65:
                t_data["assessment_status"] = TopicAssessmentStatus.ADEQUATELY_ASSESSED.value
            else:
                t_data["assessment_status"] = TopicAssessmentStatus.ASSESSED.value

            # Coverage status progression
            if score < 60.0 or is_refusal:
                t_data["coverage_status"] = TopicCoverageStatus.REQUIRES_FOLLOW_UP.value
            elif m_count >= 2 and score >= 60.0:
                t_data["coverage_status"] = TopicCoverageStatus.ADEQUATELY_ASSESSED.value
            else:
                t_data["coverage_status"] = TopicCoverageStatus.PARTIALLY_ASSESSED.value

            t_data["last_assessed_at"] = now_utc.isoformat()

        # 3. Update Project Understanding
        if question.target_project:
            proj_name = question.target_project.strip()
            if proj_name not in projects_map:
                projects_map[proj_name] = {
                    "project_id": re.sub(r"[^\w\s-]", "", proj_name).strip().replace(" ", "_"),
                    "project_name": proj_name,
                    "is_assessed": False,
                    "coverage_status": TopicCoverageStatus.NOT_CONSIDERED.value,
                    "questions_asked_count": 0,
                    "answers_evaluated_count": 0,
                    "average_score": None,
                    "latest_score": None,
                    "architecture_understanding": "UNASSESSED",
                    "implementation_understanding": "UNASSESSED",
                    "known_knowledge_gaps": [],
                    "remaining_areas_to_assess": ["Architecture", "Implementation"],
                    "tested_question_ids": [],
                }

            p_data = projects_map[proj_name]
            p_data["is_assessed"] = True
            p_data["latest_score"] = round(score, 1)

            tested_q_ids = [str(x) for x in p_data.get("tested_question_ids", [])]
            if q_id_str not in tested_q_ids:
                tested_q_ids.append(q_id_str)
                p_data["questions_asked_count"] = p_data.get("questions_asked_count", 0) + 1
            p_data["tested_question_ids"] = tested_q_ids

            p_data["answers_evaluated_count"] = p_data.get("answers_evaluated_count", 0) + 1

            prev_avg = p_data.get("average_score")
            if prev_avg is None:
                p_data["average_score"] = round(score, 1)
            else:
                # Rolling average of answers on this project
                cnt = p_data["answers_evaluated_count"]
                new_avg = ((prev_avg * (cnt - 1)) + score) / cnt
                p_data["average_score"] = round(new_avg, 1)

            # Update architecture/implementation understanding based on category
            if question.category == "Architecture":
                if score >= 75.0:
                    p_data["architecture_understanding"] = "STRONG"
                elif score >= 55.0:
                    p_data["architecture_understanding"] = "ADEQUATE"
                else:
                    p_data["architecture_understanding"] = "WEAK"
                rem = [a for a in p_data.get("remaining_areas_to_assess", []) if a != "Architecture"]
                p_data["remaining_areas_to_assess"] = rem

            elif question.category in ["Implementation", "Fundamentals"]:
                if score >= 75.0:
                    p_data["implementation_understanding"] = "STRONG"
                elif score >= 55.0:
                    p_data["implementation_understanding"] = "ADEQUATE"
                else:
                    p_data["implementation_understanding"] = "WEAK"
                rem = [a for a in p_data.get("remaining_areas_to_assess", []) if a != "Implementation"]
                p_data["remaining_areas_to_assess"] = rem

            if score < 60.0:
                p_data["coverage_status"] = TopicCoverageStatus.REQUIRES_FOLLOW_UP.value
            elif p_data["answers_evaluated_count"] >= 2:
                p_data["coverage_status"] = TopicCoverageStatus.ADEQUATELY_ASSESSED.value
            else:
                p_data["coverage_status"] = TopicCoverageStatus.PARTIALLY_ASSESSED.value

        # 4. Update Relationship Understanding
        contra_data = getattr(answer, "contradiction_analysis", None)
        if contra_data is None and isinstance(answer.dimension_scores, dict):
            contra_data = answer.dimension_scores.get("_contradiction_analysis") or answer.dimension_scores.get("contradiction_analysis")

        has_contradiction = bool(contra_data and contra_data.get("has_contradiction"))

        for r_item in relationships_list:
            matches_rel = False
            # Direct match by target_relationship string
            if question.target_relationship and (
                question.target_relationship.lower() in r_item["relationship_id"].lower()
                or r_item["source_entity"].lower() in question.target_relationship.lower()
            ):
                matches_rel = True
            # Match by entity + project context
            elif (
                question.target_entity
                and (
                    question.target_entity.lower() in r_item["source_entity"].lower()
                    or question.target_entity.lower() in r_item["target_entity"].lower()
                )
                and question.target_project
                and r_item.get("context_entity")
                and question.target_project.lower() in r_item["context_entity"].lower()
            ):
                matches_rel = True

            if matches_rel:
                r_item["is_tested"] = True
                r_item["score"] = round(score, 1)
                r_item["supporting_question_id"] = q_id_str

                if has_contradiction:
                    r_item["status"] = "MISUNDERSTOOD"
                    r_item["demonstrated_understanding"] = "UNCLEAR"
                    r_item["notes"] = "Candidate assertion contradicted verified resume evidence."
                elif score >= 75.0:
                    r_item["status"] = "DEMONSTRATED"
                    r_item["demonstrated_understanding"] = "DEMONSTRATED"
                    r_item["notes"] = "Candidate provided sound reasoning aligning with relationship."
                elif score >= 50.0:
                    r_item["status"] = "PARTIAL"
                    r_item["demonstrated_understanding"] = "WEAK_EXPLANATION"
                    r_item["notes"] = "Candidate showed partial understanding with missing depth."
                else:
                    r_item["status"] = "MISUNDERSTOOD"
                    r_item["demonstrated_understanding"] = "WEAK_EXPLANATION"
                    r_item["notes"] = "Candidate gave an inadequate explanation of relationship mechanics."

        # 5. Update Resume Claim Verification
        for c_item in claims_list:
            matches_claim = False
            if question.target_claim_id and question.target_claim_id == c_item.get("claim_id"):
                matches_claim = True
            elif question.targeted_claim and question.targeted_claim not in ["Evidence-Grounded Question", ""] and (
                question.targeted_claim.lower() in c_item["claim_text"].lower()
                or c_item["claim_text"].lower() in question.targeted_claim.lower()
            ):
                matches_claim = True
            elif (
                (question.category == "Claim-Verification" or getattr(question, "strategy", None) in [AdaptiveStrategyType.CLAIM_VERIFICATION, "CLAIM_VERIFICATION"])
                and question.target_project
                and c_item.get("project")
                and question.target_project.lower() in c_item["project"].lower()
                and question.target_entity
                and c_item.get("target_entity")
                and question.target_entity.lower() in c_item["target_entity"].lower()
            ):
                matches_claim = True

            if matches_claim:
                c_item["verification_source_question_id"] = q_id_str
                c_item["last_updated_at"] = now_utc.isoformat()

                if has_contradiction:
                    # Verified contradiction detected
                    c_item["status"] = ClaimVerificationStatus.CONTRADICTED.value
                    c_item["confidence"] = 0.95
                    c_item["notes"] = "Deterministic contradiction analysis identified factual conflict."
                elif is_refusal:
                    # Candidate refused or skipped question: NOT contradiction
                    c_item["status"] = ClaimVerificationStatus.INSUFFICIENT_EVIDENCE.value
                    c_item["confidence"] = 0.40
                    c_item["notes"] = "Candidate did not provide substantive explanation for claim verification."
                elif score >= 70.0 and (answer.evidence_alignment_score is None or answer.evidence_alignment_score >= 65.0):
                    c_item["status"] = ClaimVerificationStatus.SUPPORTED.value
                    c_item["confidence"] = eval_conf
                    c_item["notes"] = "Candidate demonstrated deep grounding supporting the claim."
                elif score >= 50.0:
                    c_item["status"] = ClaimVerificationStatus.PARTIALLY_SUPPORTED.value
                    c_item["confidence"] = eval_conf
                    c_item["notes"] = "Candidate provided partial details supporting the claim."
                else:
                    # Low technical explanation score does NOT mean the resume fact is false
                    c_item["status"] = ClaimVerificationStatus.INSUFFICIENT_EVIDENCE.value
                    c_item["confidence"] = eval_conf
                    c_item["notes"] = "Candidate struggled with technical explanation; claim remains unverified."

                # Sync claim status back to ResumeClaim model row if found
                try:
                    c_row = db.query(ResumeClaim).filter(
                        ResumeClaim.resume_id == session.resume_id,
                        ResumeClaim.claim_id == c_item["claim_id"]
                    ).first()
                    if c_row:
                        c_row.status = c_item["status"]
                except Exception as e:
                    logger.debug(f"Failed to sync status to ResumeClaim row: {e}")

        # 6. Knowledge Gap Resolution & Creation
        primary_topic = list(associated_topics)[0] if associated_topics else (question.target_project or question.category)

        # Check for gap resolution: if score >= 75, only close gaps tied to the same question_id,
        # or to the same topic AND whose missing concepts are covered by the new answer (B5).
        if score >= 75.0:
            new_missing_lower = {str(m).strip().lower() for m in (answer.missing_concepts or []) if str(m).strip()}
            matched_concepts_lower = {str(m).strip().lower() for m in (getattr(answer, "matched_concepts", []) or []) if str(m).strip()}

            for g in gaps_list:
                if g.get("is_addressed"):
                    continue

                is_same_q = (g.get("question_id") == q_id_str)
                is_same_topic = (g.get("topic") in associated_topics or g.get("topic") == primary_topic)

                should_resolve = False
                if is_same_q:
                    should_resolve = True
                elif is_same_topic:
                    g_missing = g.get("missing_concepts", [])
                    if not g_missing:
                        should_resolve = len(new_missing_lower) == 0
                    else:
                        g_missing_lower = [str(m).strip().lower() for m in g_missing if str(m).strip()]
                        still_missing = any(m in new_missing_lower for m in g_missing_lower)
                        covered_in_matched = all(m in matched_concepts_lower for m in g_missing_lower) if matched_concepts_lower else False
                        if not still_missing or covered_in_matched:
                            should_resolve = True

                if should_resolve:
                    g["is_addressed"] = True
                    g["addressed_by_answer_id"] = ans_id_str
                    g["resolution_notes"] = f"Addressed in attempt {answer.attempt_number} with score {round(score, 1)}."
                    g["updated_at"] = now_utc.isoformat()

        # TASK 4 Hardened gap registration logic:
        # Register a gap ONLY if:
        # - overall score < 75, OR
        # - (score < 85 AND missing_concepts has >=1 substantive item, ignoring "None", "None significant", "N/A", etc.)
        # Never register a gap on a topic whose mastery score is >= 85.
        missing = list(answer.missing_concepts or [])
        weaknesses = list(answer.weaknesses or [])

        NON_SUBSTANTIVE_CONCEPTS = {
            "none", "none significant", "n/a", "na", "no major omissions",
            "nothing substantive", "none noted", "nil", "nothing", "no missing concepts",
            "not applicable", "no critical omissions", "no significant omissions",
            "no omissions", "no major gaps", "none observed"
        }

        substantive_missing = [
            c for c in missing
            if isinstance(c, str)
            and c.strip().lower() not in NON_SUBSTANTIVE_CONCEPTS
            and not any(c.strip().lower().startswith(p) for p in ["none ", "none.", "no major", "no significant", "no critical", "nothing "])
        ]

        topic_mastery = topics_map.get(primary_topic, {}).get("mastery_score", score)

        # Resolve any existing gaps if candidate has now reached mastery >= 85
        if topic_mastery >= 85.0:
            for g in gaps_list:
                if g.get("topic") == primary_topic and not g.get("is_addressed"):
                    g["is_addressed"] = True
                    g["addressed_by_answer_id"] = ans_id_str
                    g["resolution_notes"] = f"Resolved: Candidate demonstrated mastery ({round(topic_mastery, 1)} >= 85) on {primary_topic}."
                    g["updated_at"] = now_utc.isoformat()

        should_create_gap = (
            topic_mastery < 85.0
            and (
                score < 75.0
                or (score < 85.0 and len(substantive_missing) >= 1)
                or has_contradiction
            )
        )

        if should_create_gap:
            if score < 50.0 or has_contradiction:
                severity = GapSeverity.HIGH.value
                nature = GapNature.DEMONSTRATED_WEAKNESS.value
            elif score < 75.0:
                severity = GapSeverity.MEDIUM.value
                nature = GapNature.INCOMPLETE_UNDERSTANDING.value
            else:
                severity = GapSeverity.LOW.value
                nature = GapNature.INCOMPLETE_UNDERSTANDING.value

            if is_refusal:
                nature = GapNature.INSUFFICIENT_EVIDENCE.value

            desc = "; ".join(substantive_missing or weaknesses) if (substantive_missing or weaknesses) else f"Low performance on {primary_topic}"
            if has_contradiction:
                desc = f"Factual contradiction on {primary_topic}: {desc}"

            gaps_list.append({
                "gap_id": str(uuid4()),
                "target_type": "project" if question.target_project else "skill",
                "topic": primary_topic,
                "description": desc,
                "missing_concepts": substantive_missing,
                "question_id": q_id_str,
                "answer_id": ans_id_str,
                "evidence_or_feedback": answer.evaluation_feedback or "",
                "severity": severity,
                "gap_nature": nature,
                "confidence": eval_conf,
                "is_addressed": False,
                "addressed_by_answer_id": None,
                "resolution_notes": None,
                "created_at": now_utc.isoformat(),
                "updated_at": now_utc.isoformat(),
            })

            if question.target_project and question.target_project in projects_map:
                p_gaps = projects_map[question.target_project].setdefault("known_knowledge_gaps", [])
                if desc not in p_gaps:
                    p_gaps.append(desc)

        # 7. Recompute Strong/Weak Summary
        strengths: List[str] = []
        weaknesses_list: List[str] = []
        partial_list: List[str] = []
        unassessed_list: List[str] = []
        followup_list: List[str] = []

        for t_name, t_val in topics_map.items():
            if not t_val.get("is_assessable", False):
                continue
            cnt = t_val.get("answers_evaluated_count", 0)
            m_score = t_val.get("mastery_score", 0.0)
            conf = t_val.get("confidence", 0.0)
            cov_status = t_val.get("coverage_status")

            if cnt == 0:
                unassessed_list.append(t_name)
            else:
                if m_score >= 75.0 and conf >= 0.40:
                    strengths.append(t_name)
                elif m_score < 50.0 and conf >= 0.40:
                    weaknesses_list.append(t_name)
                elif 50.0 <= m_score < 75.0:
                    partial_list.append(t_name)

            if cov_status == TopicCoverageStatus.REQUIRES_FOLLOW_UP.value:
                followup_list.append(t_name)

        # Also add topics with active unaddressed gaps to followup
        for g in gaps_list:
            if not g.get("is_addressed") and g.get("topic") in topics_map:
                if topics_map[g.get("topic")].get("is_assessable", False) and g.get("topic") not in followup_list:
                    followup_list.append(g.get("topic"))

        k_state["summary"] = {
            "demonstrated_strengths": sorted(strengths),
            "demonstrated_weaknesses": sorted(weaknesses_list),
            "partially_understood": sorted(partial_list),
            "unassessed_topics": sorted(unassessed_list),
            "topics_requiring_followup": sorted(followup_list),
        }

        # 8. Recompute Topic Coverage Summary ONLY over assessable topics (Task 2)
        assessable_topics = {k: v for k, v in topics_map.items() if v.get("is_assessable", False)}
        total_assessable = len(assessable_topics)
        not_considered = sum(1 for t in assessable_topics.values() if t.get("coverage_status") == TopicCoverageStatus.NOT_CONSIDERED.value)
        q_asked = sum(1 for t in assessable_topics.values() if t.get("coverage_status") == TopicCoverageStatus.QUESTION_ASKED.value)
        part_assessed = sum(1 for t in assessable_topics.values() if t.get("coverage_status") == TopicCoverageStatus.PARTIALLY_ASSESSED.value)
        adeq_assessed = sum(1 for t in assessable_topics.values() if t.get("coverage_status") == TopicCoverageStatus.ADEQUATELY_ASSESSED.value)
        req_followup = sum(1 for t in assessable_topics.values() if t.get("coverage_status") == TopicCoverageStatus.REQUIRES_FOLLOW_UP.value)

        assessed_cnt = part_assessed + adeq_assessed + req_followup
        cov_pct = round((assessed_cnt / total_assessable) * 100, 1) if total_assessable > 0 else 0.0

        k_state["coverage"] = {
            "total_resume_topics": total_assessable,
            "total_assessable_topics": total_assessable,
            "not_considered_count": not_considered,
            "question_asked_count": q_asked,
            "partially_assessed_count": part_assessed,
            "adequately_assessed_count": adeq_assessed,
            "requires_follow_up_count": req_followup,
            "coverage_percentage": cov_pct,
        }

        # 9. Mark answer attempt processed and persist state
        processed_ids.append(ans_id_str)
        k_state["_processed_answer_ids"] = processed_ids
        k_state["last_updated_at"] = now_utc.isoformat()

        # Synchronize back to session.session_state
        sess_state = dict(session.session_state or {})
        sess_state["knowledge_state"] = k_state

        # Keep Phase 7.1 top-level keys in sync for backward compatibility
        tested_skills_top = set(sess_state.get("tested_skills", []))
        tested_tech_top = set(sess_state.get("tested_technologies", []))
        for t_name in associated_topics:
            tested_skills_top.add(t_name)
            tested_tech_top.add(t_name)
        sess_state["tested_skills"] = sorted(list(tested_skills_top))
        sess_state["tested_technologies"] = sorted(list(tested_tech_top))

        if question.target_project:
            tested_proj_top = set(sess_state.get("tested_projects", []))
            tested_proj_top.add(question.target_project.strip())
            sess_state["tested_projects"] = sorted(list(tested_proj_top))

        if question.target_relationship:
            tested_rel_top = set(sess_state.get("tested_relationships", []))
            tested_rel_top.add(question.target_relationship.strip())
            sess_state["tested_relationships"] = sorted(list(tested_rel_top))

        if question.targeted_claim:
            tested_claim_top = set(sess_state.get("tested_claims", []))
            tested_claim_top.add(question.targeted_claim.strip())
            sess_state["tested_claims"] = sorted(list(tested_claim_top))

        session.session_state = sess_state
        try:
            flag_modified(session, "session_state")
        except Exception:
            pass

        try:
            db.commit()
            db.refresh(session)
        except Exception:
            pass
        logger.info(f"Updated candidate knowledge state for session {session_id} on question {question_id} (topics={list(associated_topics)})")
        return k_state

    def reconcile_knowledge_state(
        self,
        session_id: UUID,
        db: Session
    ) -> Dict[str, Any]:
        """
        Reconstructs the full Candidate Knowledge State from scratch by cleanly replaying
        all evaluated answers and attempts in chronological order.
        Provides robust recovery if any transient failure or concurrency race occurred.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            logger.warning(f"Session {session_id} not found during knowledge state reconciliation.")
            return {}

        # Reset state to clean baseline from resume
        fresh_state = self.initialize_knowledge_state(session, db)
        sess_state = dict(session.session_state or {})
        sess_state["knowledge_state"] = fresh_state
        session.session_state = sess_state
        flag_modified(session, "session_state")
        db.commit()
        db.refresh(session)

        # Retrieve all answers for this session ordered chronologically
        answers = (
            db.query(InterviewAnswer)
            .filter(
                InterviewAnswer.session_id == session_id,
                InterviewAnswer.status == "EVALUATED"
            )
            .order_by(
                InterviewAnswer.attempt_number.asc(),
                InterviewAnswer.created_at.asc(),
                InterviewAnswer.id.asc()
            )
            .all()
        )

        for ans in answers:
            self.update_from_evaluation(
                session_id=session_id,
                question_id=ans.question_id,
                answer=ans,
                db=db
            )

        db.refresh(session)
        final_state = dict(session.session_state or {}).get("knowledge_state", {})
        logger.info(f"Reconciled knowledge state for session {session_id} across {len(answers)} evaluated attempts.")
        return final_state

    def format_knowledge_state_response(
        self,
        session_id: UUID,
        db: Session
    ) -> CandidateKnowledgeStateResponse:
        """
        Loads knowledge state and formats it into the validated Pydantic CandidateKnowledgeStateResponse.
        """
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Interview session not found.")

        k_state = self.get_or_initialize_knowledge_state(session, db)

        # Parse topics
        topics: Dict[str, TopicMasteryState] = {}
        for t_k, t_v in k_state.get("topics", {}).items():
            topics[t_k] = TopicMasteryState.model_validate(t_v)

        # Parse projects
        projects: Dict[str, ProjectUnderstandingState] = {}
        for p_k, p_v in k_state.get("projects", {}).items():
            projects[p_k] = ProjectUnderstandingState.model_validate(p_v)

        # Parse relationships
        relationships: List[RelationshipUnderstandingState] = [
            RelationshipUnderstandingState.model_validate(r)
            for r in k_state.get("relationships", [])
        ]

        # Parse claims
        claims: List[ResumeClaimVerificationState] = [
            ResumeClaimVerificationState.model_validate(c)
            for c in k_state.get("claims", [])
        ]

        # Parse knowledge gaps
        gaps: List[DetailedKnowledgeGap] = [
            DetailedKnowledgeGap.model_validate(g)
            for g in k_state.get("knowledge_gaps", [])
        ]

        # Parse summary & coverage
        summary = StrongWeakSummary.model_validate(k_state.get("summary", {}))
        coverage = TopicCoverageSummary.model_validate(k_state.get("coverage", {}))

        last_updated = None
        if k_state.get("last_updated_at"):
            try:
                last_updated = datetime.fromisoformat(k_state["last_updated_at"])
            except Exception:
                last_updated = None

        return CandidateKnowledgeStateResponse(
            session_id=session.id,
            resume_id=session.resume_id,
            topics=topics,
            projects=projects,
            relationships=relationships,
            claims=claims,
            knowledge_gaps=gaps,
            summary=summary,
            coverage=coverage,
            last_updated_at=last_updated,
        )


knowledge_state_service = CandidateKnowledgeStateEngine()
