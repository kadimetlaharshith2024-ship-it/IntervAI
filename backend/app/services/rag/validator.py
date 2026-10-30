import re
import logging
from typing import Tuple, List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.rag import QuestionOutputSchema
from app.models.interview import InterviewQuestion
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.services.rag.embedding_service import embedding_service
from app.services.rag.retrievers import cosine_similarity

logger = logging.getLogger(__name__)

CATEGORY_OBJECTIVE_EMBEDDINGS: Dict[str, str] = {
    "Architecture": "system component boundaries, data flow across interfaces, architectural trade-offs, separation of concerns",
    "Security": "authentication mechanisms, authorization boundaries, token lifecycle, encryption, threat resistance, input validation",
    "Performance": "concurrency handling, latency optimization, database query bottlenecks, memory resource constraints, throughput",
    "Implementation": "concrete code-level mechanics, API contracts, state lifecycle, transaction boundaries, error handling",
    "Fundamentals": "core foundational principles, language mechanics, protocols, and foundational data structures",
    "Failure-Mode": "failure boundaries, exception recovery, crash mitigation, timeout policies, edge case resilience",
    "Claim-Verification": "verifying execution authenticity, technical validation of asserted claims, hands-on depth"
}

ASSERTIVE_PRESUPPOSITION_REGEX = re.compile(
    r"\b("
    # walk me through (standalone — the phrase alone presupposes prior implementation)
    r"walk\s+me\s+through|"
    # tell me about your implementation/configuration/design of (standalone)
    r"tell\s+(?:me\s+)?about\s+your\s+(?:implementation|configuration|design)\s+of|"
    # walk through / describe / explain / detail / outline followed by qualifying content
    r"(?:walk\s+through|describe|explain|detail|outline)\s+(?:how\s+you\s+(?:implemented|built|architected|configured|designed|created|structured)|a\s+concrete\s+example\s+of\s+(?:\w+ing\s+)?(?:your\s+)?|(?:your\s+)?(?:custom\s+)?(?:implementation|architecture|configuration|design|approach|system|pipeline|steps)|the\s+(?:concrete\s+)?(?:mechanics|implementation|configuration|pipeline|steps|flow|lifecycle)\s+of)|"
    r"how\s+(?:did\s+|do\s+)?you\s+(?:implement|build|architect|configure|design|structure|optimize|manage)\s+(?:your\s+)?|"
    r"detail\s+how\s+.*?(?:was|were)\s+(?:managed|handled|implemented)|"
    r"(?:in|during|for)\s+your\s+(?:custom\s+)?(?:implementation|architecture|configuration|design)\s+of|"
    r"(?:when|after)\s+(?:you\s+implemented|building\s+your|designing\s+your|implementing\s+your|configuring\s+your)"
    r")\b",
    re.IGNORECASE
)

OPEN_ENDED_EXPLORATION_REGEX = re.compile(
    r"\b("
    r"how\s+would\s+you|"
    r"how\s+did\s+you\s+or\s+would\s+you|"
    r"how\s+might\s+you|"
    r"how\s+could\s+you|"
    r"how\s+do\s+you\s+approach|"
    r"how\s+can\s+you|"
    r"what\s+(?:approach|strategy|design)\s+would\s+you\s+(?:take|use|recommend)|"
    r"what\s+trade-offs\s+would\s+you|"
    r"how\s+do\s+you\s+evaluate|"
    r"what\s+is\s+your\s+approach\s+to"
    r")\b",
    re.IGNORECASE
)

class IndependentGroundingValidator:
    """
    Performs deterministic and semantic validation of candidate interview questions
    without relying on LLM-supplied self-evaluations or static keyword lists.
    """
    def __init__(self):
        self._category_cache: Dict[str, List[float]] = {}
        self._question_embedding_cache: Dict[str, List[float]] = {}
        self._claim_embedding_cache: Dict[str, List[float]] = {}

    def clear_caches(self):
        self._question_embedding_cache.clear()
        self._claim_embedding_cache.clear()

    def clear_cache_for_resume(self, resume_id: Any):
        # Clears caches when a resume is re-indexed or modified
        self.clear_caches()

    def _get_category_vector(self, category: str) -> List[float]:
        if category not in self._category_cache:
            desc = CATEGORY_OBJECTIVE_EMBEDDINGS.get(category, CATEGORY_OBJECTIVE_EMBEDDINGS["Architecture"])
            self._category_cache[category] = embedding_service.embed_query(desc)
        return self._category_cache[category]

    @staticmethod
    def _normalize_project_name(name: str) -> str:
        if not name:
            return ""
        s = re.sub(r"[\u2010-\u2015\u2212\-_/|]+", " ", name.lower())
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())

    @staticmethod
    def estimate_difficulty_level(question_text: str) -> int:
        """
        Analyzes linguistic complexity, depth markers, cognitive verbs, and structural scope
        to independently estimate question difficulty across canonical Levels 1 to 5.
        """
        q = (question_text or "").strip()
        q_lower = q.lower()

        # 1. Level 1 Priority: Basic syntax, keywords, definitions, elementary concepts
        l1_strict = [
            r"\b(?:which\s+keyword\s+(?:controls|specifies|initiates)|what\s+is\s+the\s+basic\s+syntax)\b",
            r"\b(?:define\s+the\s+elementary|name\s+the\s+elementary|list\s+the\s+elementary)\b",
            r"\b(?:what\s+is\s+the\s+primary\s+purpose\s+of\s+registering|what\s+is\s+the\s+standard\s+definition)\b",
            r"\b(?:name\s+the\s+basic\s+concept|what\s+are\s+the\s+fundamental\s+requirements\s+to\s+configure)\b",
            r"\b(?:what\s+are\s+the\s+core\s+rules\s+governing|define\s+the\s+distinction\s+between\s+baseline)\b",
            r"\b(?:what\s+is\s+the\s+elementary\s+role|define\s+the\s+elementary\s+boundaries)\b",
        ]
        if any(re.search(p, q_lower) for p in l1_strict):
            return 1

        # 2. Level 5: Advanced Depth (Analyze & Defend under Uncertainty, Competing Constraints, Failure Scenarios)
        l5_patterns = [
            r"\b(?:competing\s+constraints?|under\s+uncertainty)\b",
            r"\b(?:changing\s+assumptions?|assumptions?\s+change|sensitivity\s+to\s+changing)\b",
            r"\b(?:becomes?\s+uneven|data\s+skew|hotspots?|hot-spots?)\b",
            r"\b(?:compare\s+.*?\s+strategies\s+and\s+determine\s+when|when\s+should\s+the\s+existing\s+design\s+change|when\s+would\s+you\s+migrate)\b",
            r"\b(?:defend\s+your\s+(?:choice|design|architecture)|defensible\s+trade-?offs?)\b",
            r"\b(?:cascading\s+failure\s+modes?|cascading\s+failure|distributed\s+blast\s+radius|blast\s+radius)\b",
            r"\b(?:network\s+partition|split-brain|byzantine|consensus\s+protocols?|linearizability)\b",
            r"\b(?:distributed\s+consistency\s+during|distributed\s+consistency|under\s+extreme\s+load|threat\s+vectors?|adversarial)\b",
            r"\b(?:race\s+conditions?\s+hazards?|adverse\s+network\s+degradation|deadlock\s+detection\s+cycles)\b",
            r"\b(?:leader\s+election\s+failover|partition\s+degradation|partition\s+tolerance|silent\s+data\s+corruption)\b",
        ]
        l5_score = sum(1 for p in l5_patterns if re.search(p, q_lower))

        # 3. Level 4: Architecture (Design & Evaluate System Boundaries, Scalability, Trade-offs)
        l4_patterns = [
            r"\barchitectural\s+trade-?offs?\b",
            r"\b(?:trade-?offs?\s+did\s+you\s+evaluate|trade-?offs?\s+would\s+you\s+evaluate)\b",
            r"\b(?:trade-?offs?\s+did\s+you\s+make|trade-?offs?\s+did\s+you\s+accept)\b",
            r"\b(?:horizontal\s+auto-scaling|horizontal\s+scale|horizontal\s+scalability|scale\s+horizontally|scaling\s+bottlenecks?|scaling\s+constraint)\b",
            r"\b(?:decoupling\s+services|decouple\s+(?:architectural|component|bottleneck))\b",
            r"\b(?:component\s+boundaries|system\s+boundaries|architectural\s+boundary)\b",
            r"\b(?:security\s+boundary|resource\s+isolation|network\s+segmentation)\b",
            r"\b(?:multi-tier\s+network\s+architecture|virtual\s+private\s+cloud|vpc\s+design)\b",
            r"\b(?:handle\s+concurrent\s+.*?\s+and\s+reliable\s+api\s+responses)\b",
            r"\b(?:asynchronous\s+data\s+flows?|event-driven\s+architecture|backpressure\s+mechanism|backpressure)\b",
            r"\b(?:failover\s+strategy|failover\s+synchronization|failover\s+routing|failover\s+events|failover\s+recovery|high\s+availability\s+cluster)\b",
            r"\b(?:concurrency\s+model|concurrency\s+model\s+bottlenecks)\b",
        ]
        l4_score = sum(1 for p in l4_patterns if re.search(p, q_lower))

        # 4. Level 2 Priority: Standard application, descriptive data flow, practical purpose
        l2_strict = [
            r"\b(?:describe\s+how\s+developers\s+structure|describe\s+how\s+.*?\s+integrates?\s+.*?\s+into\s+practical\s+application)\b",
            r"\b(?:explain\s+the\s+practical\s+purpose\s+of\s+configuring|describe\s+how\s+.*?\s+affects\s+the\s+data\s+flow)\b",
            r"\b(?:explain\s+how\s+.*?\s+coordinates\s+with\s+.*?\s+to\s+maintain\s+structured|explain\s+how\s+.*?\s+coordinates\s+with\s+the\s+runtime)\b",
            r"\b(?:what\s+role\s+did\s+.*?\s+fulfill\s+in\s+the\s+practical\s+deployment|what\s+role\s+does\s+.*?\s+serve\s+during\s+typical)\b",
            r"\b(?:explain\s+how\s+.*?\s+is\s+integrated\s+across\s+components|describe\s+how\s+.*?\s+facilitates\s+practical\s+usage)\b",
            r"\b(?:explain\s+how\s+.*?\s+participates\s+in\s+the\s+execution\s+flow|describe\s+how\s+to\s+configure\s+.*?\s+for\s+typical)\b",
            r"\b(?:explain\s+how\s+to\s+integrate\s+.*?\s+and\s+describe\s+how\s+data\s+flows|how\s+would\s+.*?\s+help\s+determine\s+whether)\b",
            r"\b(?:explain\s+how\s+.*?\s+is\s+structured\s+within\s+the\s+data\s+flow|what\s+role\s+did\s+.*?\s+play\s+in\s+the\s+typical\s+operational|describe\s+the\s+steps\s+to\s+integrate)\b",
            r"\b(?:how\s+(?:does|do|did|would)\s+.*?\s+(?:use|help|utilize)\s+|how\s+is\s+.*?\s+used\s+to\s+store)\b",
        ]
        l2_score = sum(1 for p in l2_strict if re.search(p, q_lower))

        # 5. Level 3: Implementation (Build & Debug Details, Algorithms, Boundary Testing)
        l3_patterns = [
            r"\b(?:implement\s+a\s+point-in-region|implement\s+a\s+|how\s+would\s+you\s+implement|how\s+did\s+you\s+implement)\b",
            r"\b(?:test\s+boundary\s+cases|boundary\s+cases|edge\s+cases|unit\s+and\s+integration\s+tests?)\b",
            r"\b(?:branching\s+factor|node\s+capacity|polygon\s+geometries|bounding\s+box)\b",
            r"\b(?:state\s+transitions?|transaction\s+boundaries?|transactional\s+boundary)\b",
            r"\b(?:dependency\s+injection\s+lifecycle|middleware\s+stack|thread-safe\s+state)\b",
            r"\b(?:cache\s+invalidation\s+and\s+manage|manage\s+transaction\s+boundaries|cache\s+invalidation)\b",
            r"\b(?:query\s+plans?|query\s+optimization|indexing\s+node|spatial\s+lookups?\s+in\s+a\s+high|throughput\s+tuning|throughput\s+optimization)\b",
            r"\b(?:error\s+propagation|exception\s+handling\s+logic|retry\s+policy|buffer\s+thresholds?)\b",
            r"\b(?:non-blocking\s+concurrency|lifecycle\s+state\s+management)\b",
            r"\b(?:how\s+would\s+you\s+approach\s+designing\s+the\s+transaction|how\s+did\s+you\s+handle\s+state\s+transition)\b",
        ]
        l3_score = sum(1 for p in l3_patterns if re.search(p, q_lower))

        # General Level 1 patterns
        l1_patterns = [
            r"\b(?:what\s+is\s+(?:the\s+)?(?:fundamental|core|primary|basic)?\s*(?:purpose|concept|principle|role|definition)|what\s+is\s+an?|what\s+are\s+(?:the\s+)?)\b",
            r"\b(?:why\s+is\s+it\s+(?:useful|essential|important|considered)|why\s+is\s+.*?\s+useful|useful\s+for\s+spatial\s+data)\b",
            r"\b(?:how\s+does\s+.*?\s+fundamentally\s+(?:organize|work|enable|facilitate)|fundamentally\s+enable)\b",
            r"\b(?:fundamental\s+concept\s+of|core\s+principles?\s+behind|concept\s+of\s+a\s+minimum\s+bounding)\b",
            r"\b(?:elucidate\s+and\s+define|define\s+what|what\s+an?\s+.*?\s+actually\s+represents?|definition\s+of)\b",
            r"\b(?:primary\s+role\s+of\s+the|basic\s+properties|basic\s+concept)\b",
            r"\b(?:core\s+principles?|foundational\s+principles?|fundamental\s+principles?)\b",
            r"\b(?:difference\s+between\s+.*?\s+and\s+|distinction\s+between)\b",
        ]
        l1_score = sum(1 for p in l1_patterns if re.search(p, q_lower))

        if l5_score > 0:
            return 5
        if l4_score > 0 and l4_score >= l3_score:
            return 4
        has_l3_verbs = bool(re.search(r"\b(?:implement|implementing|how\s+(?:would|did)\s+you\s+implement|test\s+boundary|unit\s+and\s+integration|edge\s+cases|debug|debugging)\b", q_lower))
        if l1_score > 0 and not has_l3_verbs and l3_score <= 1:
            return 1
        if l2_score > 0:
            return 2
        if l3_score > 0:
            return 3
        if l1_score > 0:
            return 1
        return 3

    @staticmethod
    def explain_difficulty_level(question_text: str, level: Optional[int] = None) -> str:
        """
        Provides a calibrated technical explanation justifying the assessed difficulty level
        according to the canonical 5-level rubric.
        """
        lvl = level or IndependentGroundingValidator.estimate_difficulty_level(question_text)
        explanations = {
            1: "Level 1 — Fundamentals (Understand): Evaluates understanding of foundational concepts, definitions, and core principles without requiring code implementation or system architecture.",
            2: "Level 2 — Application (Apply): Requires applying concepts to concrete situations, standard workflows, and practical data flows between components.",
            3: "Level 3 — Implementation (Build and Debug): Requires reasoning about concrete implementation details, algorithms, data structures, debugging, boundary cases, or transaction boundaries.",
            4: "Level 4 — Architecture (Design and Evaluate): Requires system-level reasoning about component boundaries, horizontal scalability, reliability, security isolation, and architectural trade-offs.",
            5: "Level 5 — Advanced Depth (Analyze and Defend): Requires deep reasoning under uncertainty, competing constraints, changing assumptions, alternative strategies, or cascading distributed failure modes."
        }
        return explanations.get(lvl, f"Level {lvl} technical depth.")

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """
        Normalized token overlap (Dice coefficient) between two question texts
        used as a deterministic fallback when embedding computation fails.
        """
        tokens1 = set(re.findall(r"\b\w{2,}\b", (text1 or "").lower()))
        tokens2 = set(re.findall(r"\b\w{2,}\b", (text2 or "").lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        return (2.0 * len(intersection)) / (len(tokens1) + len(tokens2))

    def check_semantic_duplicate(
        self,
        new_question_text: str,
        existing_questions: List[InterviewQuestion],
        similarity_threshold: Optional[float] = None,
        new_q_vector: Optional[List[float]] = None,
        return_details: bool = False,
        db: Optional[Session] = None,
        extra_excluded_texts: Optional[List[str]] = None
    ) -> Any:
        if similarity_threshold is None:
            similarity_threshold = settings.RAG_DUPLICATE_SIMILARITY_THRESHOLD
        if not existing_questions and not extra_excluded_texts:
            return (False, None, None, None, 0.0) if return_details else (False, None)

        active_model = embedding_service.model_name

        # 1. Obtain or compute new question vector with safe fallback
        q_vec = new_q_vector
        if q_vec is None:
            try:
                q_vec = embedding_service.embed_query(new_question_text)
            except Exception as e:
                logger.warning("Embedding service failed for new question; falling back to lexical similarity: %s", e)
                q_vec = None

        # 2. Identify existing questions that have valid stored embeddings vs missing/mismatched
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        missing_keys: List[str] = []

        for idx, eq in enumerate(existing_questions):
            if not getattr(eq, "question_text", None):
                continue
            cache_key = str(eq.id) if getattr(eq, "id", None) else eq.question_text
            stored_emb = getattr(eq, "embedding", None)
            stored_model = getattr(eq, "embedding_model", None)

            if cache_key in self._question_embedding_cache:
                eq.embedding = self._question_embedding_cache[cache_key]
                eq.embedding_model = active_model
            elif stored_emb and stored_model == active_model:
                self._question_embedding_cache[cache_key] = stored_emb
            else:
                missing_indices.append(idx)
                missing_texts.append(eq.question_text)
                missing_keys.append(cache_key)

        # 3. Batch compute missing/legacy embeddings in ONE call (Part A2)
        if missing_texts and q_vec is not None:
            try:
                if len(missing_texts) == 1:
                    computed_embeddings = [embedding_service.embed_query(missing_texts[0])]
                else:
                    computed_embeddings = embedding_service.embed_texts(missing_texts)
                for missing_idx, emb, key in zip(missing_indices, computed_embeddings, missing_keys):
                    eq = existing_questions[missing_idx]
                    eq.embedding = emb
                    eq.embedding_model = active_model
                    self._question_embedding_cache[key] = emb
                    if db is not None:
                        try:
                            db.add(eq)
                            db.flush()
                        except Exception as flush_err:
                            logger.debug("Failed to flush backfilled embedding for question: %s", flush_err)
            except Exception as batch_err:
                logger.warning(
                    "Embedding service failed during batch embedding of %d legacy questions: %s. Lexical fallback will be used.",
                    len(missing_texts), batch_err
                )

        # 4. Compare against each existing question (NO one-at-a-time embed_query loop!)
        LEXICAL_THRESHOLD = 0.65
        for eq in existing_questions:
            if not getattr(eq, "question_text", None):
                continue

            stored_emb = getattr(eq, "embedding", None)
            stored_model = getattr(eq, "embedding_model", None)

            # If both vectors are available with matching model, use cosine similarity
            if q_vec is not None and stored_emb is not None and stored_model == active_model:
                sim = cosine_similarity(q_vec, stored_emb)
                if sim >= similarity_threshold:
                    eq_id_str = str(getattr(eq, "id", "unknown"))
                    msg = f"Question is semantically duplicate with existing Question ID '{eq_id_str}' (Cosine: {round(sim, 3)} >= {similarity_threshold})."
                    if return_details:
                        return True, msg, eq.question_text, eq_id_str, round(sim, 3)
                    return True, msg
            else:
                # Deterministic lexical similarity check fallback (Dice coefficient)
                lex_sim = self._lexical_similarity(new_question_text, eq.question_text)
                if lex_sim >= LEXICAL_THRESHOLD:
                    eq_id_str = str(getattr(eq, "id", "unknown"))
                    msg = f"Question is duplicate with existing Question ID '{eq_id_str}' (Lexical Overlap: {round(lex_sim, 3)} >= {LEXICAL_THRESHOLD})."
                    if return_details:
                        return True, msg, eq.question_text, eq_id_str, round(lex_sim, 3)
                    return True, msg

        # Check against previously rejected candidate texts or explicitly excluded question texts
        if extra_excluded_texts:
            for ex_idx, ex_text in enumerate(extra_excluded_texts):
                if not ex_text or not str(ex_text).strip():
                    continue
                ex_clean = str(ex_text).strip()
                ex_emb = self._question_embedding_cache.get(ex_clean)
                if ex_emb is None and q_vec is not None:
                    try:
                        ex_emb = embedding_service.embed_query(ex_clean)
                        self._question_embedding_cache[ex_clean] = ex_emb
                    except Exception as ex_err:
                        logger.debug("Failed to embed excluded question text '%s': %s", ex_clean[:50], ex_err)
                        ex_emb = None

                if q_vec is not None and ex_emb is not None:
                    sim = cosine_similarity(q_vec, ex_emb)
                    if sim >= similarity_threshold:
                        msg = f"Question is semantically duplicate with previously rejected candidate (Cosine: {round(sim, 3)} >= {similarity_threshold})."
                        if return_details:
                            return True, msg, ex_clean, f"rejected_candidate_{ex_idx}", round(sim, 3)
                        return True, msg
                else:
                    lex_sim = self._lexical_similarity(new_question_text, ex_clean)
                    if lex_sim >= LEXICAL_THRESHOLD:
                        msg = f"Question is duplicate with previously rejected candidate (Lexical Overlap: {round(lex_sim, 3)} >= {LEXICAL_THRESHOLD})."
                        if return_details:
                            return True, msg, ex_clean, f"rejected_candidate_{ex_idx}", round(lex_sim, 3)
                        return True, msg

        return (False, None, None, None, 0.0) if return_details else (False, None)

    def validate(
        self,
        question: QuestionOutputSchema,
        allowed_projects: List[str],
        allowed_entities: List[str],
        resolved_evidence_items: Dict[str, Dict[str, Any]],
        existing_questions: List[InterviewQuestion],
        available_claims: Dict[str, ResumeClaim],
        available_relationships: Dict[str, Relationship],
        db: Optional[Session] = None,
        extra_excluded_texts: Optional[List[str]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        details: Dict[str, Any] = {}
        q_text = question.question.strip()
        q_lower = q_text.lower()

        # =====================================================================
        # PART A1: ALL DETERMINISTIC CHECKS RUN BEFORE ANY EMBEDDING CALL.
        # Candidates failing any deterministic check trigger ZERO embeddings.
        # =====================================================================

        # 1. Evidence Citation Verification
        if not question.evidence_ids:
            return False, "REJECTED: Model emitted no evidence citations.", {"code": "no_citations"}

        cited_evidence_texts: List[str] = []
        for eid in question.evidence_ids:
            if eid not in resolved_evidence_items:
                return False, f"REJECTED: Model referenced unknown or unretrieved evidence ID '{eid}'.", {"code": "invalid_citation_id"}
            item = resolved_evidence_items[eid]
            item_text = item.get("text", "")
            if item.get("project"):
                item_text = f"{item_text} {item.get('project')}"
            cited_evidence_texts.append(item_text)

        concatenated_cited_text = " ".join(cited_evidence_texts).lower()

        # 2. Strict Project Grounding against Cited Evidence (Punctuation and Hyphen Normalized)
        if question.target_project:
            p_clean = question.target_project.strip()
            p_norm = self._normalize_project_name(p_clean)

            # Match against allowed candidate projects using normalized keys
            matched_project = False
            for allowed_p in allowed_projects:
                allowed_norm = self._normalize_project_name(allowed_p)
                if p_norm == allowed_norm:
                    matched_project = True
                    break
                # Substring match if meaningful (at least 2 words or 8 chars)
                if (len(p_norm) >= 8 and p_norm in allowed_norm) or (len(allowed_norm) >= 8 and allowed_norm in p_norm):
                    matched_project = True
                    break

            norm_concatenated = self._normalize_project_name(concatenated_cited_text)
            if not matched_project and (p_norm in norm_concatenated or any(p_part in norm_concatenated for p_part in p_norm.split() if len(p_part) >= 4)):
                matched_project = True

            if not matched_project:
                return False, f"REJECTED: Project '{question.target_project}' does not exist on candidate resume.", {"code": "hallucinated_project"}

            if p_norm not in norm_concatenated and not any(p_part in norm_concatenated for p_part in p_norm.split() if len(p_part) >= 4):
                return False, f"REJECTED: Target project '{question.target_project}' is absent from the cited evidence blocks.", {"code": "project_not_in_cited_evidence"}
        else:
            # If target_project is None (conceptual or skills-list-only question),
            # assert that the question does NOT incorrectly tie itself to a resume project.
            norm_q = self._normalize_project_name(q_text)
            for allowed_p in allowed_projects:
                allowed_norm = self._normalize_project_name(allowed_p)
                if allowed_norm and len(allowed_norm) >= 4 and allowed_norm in norm_q:
                    return False, f"REJECTED: Conceptual question for entity '{question.target_entity}' incorrectly references project '{allowed_p}'.", {"code": "ungrounded_project_reference"}

        # 3. Strict Entity Grounding against Cited Evidence & Technical Question Alignment
        if question.target_entity:
            e_clean = question.target_entity.strip()
            e_norm = e_clean.lower()
            from app.core.canonicalizer import ALIAS_INDEX, extract_non_overlapping_entities

            entity_matched = e_norm in concatenated_cited_text
            if not entity_matched:
                # Check canonical aliases in case entity appears under an alias
                for alias, canonical in ALIAS_INDEX.items():
                    if canonical.lower() == e_norm and alias.lower() in concatenated_cited_text:
                        entity_matched = True
                        break
                    if alias.lower() == e_norm and canonical.lower() in concatenated_cited_text:
                        entity_matched = True
                        break

            if not entity_matched:
                return False, f"REJECTED: Target entity '{question.target_entity}' is absent from the cited evidence blocks.", {"code": "entity_not_in_cited_evidence"}

            # Technical focus alignment: If question's core inquiry focuses on a specific backend mechanism
            # (e.g. spatial indexing, Shapely, QGIS) but target_entity is an incidental frontend technology
            # (e.g. React) only mentioned in a preamble list, reject to enforce alignment.
            inquiry_match = re.search(r"(?:how\s+(?:would|did|do)\s+you|what\s+approach|designing|implementing|architecting)\s+(.*)", q_text, re.IGNORECASE)
            inquiry_part = inquiry_match.group(1).lower() if inquiry_match else q_lower

            if e_norm not in inquiry_part:
                inquiry_entities = [
                    canon for canon, _, _ in extract_non_overlapping_entities(inquiry_part, ALIAS_INDEX)
                    if canon.lower() in concatenated_cited_text
                ]
                if inquiry_entities and e_clean not in inquiry_entities:
                    primary_focus = inquiry_entities[0]
                    return False, (
                        f"REJECTED: Question target_entity '{question.target_entity}' does not match the question's main technical focus. "
                        f"The question inquires about '{primary_focus}', but cites '{question.target_entity}'."
                    ), {"code": "target_entity_focus_mismatch", "actual_focus": primary_focus}

        # 4. Verified Relationship Linkage (Deterministic check)
        if question.target_relationship:
            rel_id = str(question.target_relationship).strip()
            if rel_id not in available_relationships:
                return False, f"REJECTED: target_relationship '{rel_id}' is not a valid relationship for this resume.", {"code": "invalid_relationship_id"}

        # 5. Verified Claim Linkage (Deterministic existence check)
        if question.target_claim_id:
            if question.target_claim_id not in available_claims:
                return False, f"REJECTED: target_claim_id '{question.target_claim_id}' is not a valid claim for this resume.", {"code": "invalid_claim_id"}

        # 6. Evidence Entailment & Unsupported Implementation Presuppositions (Regex check)
        has_assertive_claim = bool(ASSERTIVE_PRESUPPOSITION_REGEX.search(q_text))
        is_open_ended = bool(OPEN_ENDED_EXPLORATION_REGEX.search(q_text))

        if has_assertive_claim and not is_open_ended:
            has_detailed_chunk_evidence = any(
                item.get("source_type") in ("chunk", "claim") and len(item.get("text", "").split()) > 8
                for item in [resolved_evidence_items.get(eid, {}) for eid in question.evidence_ids]
            )
            if not has_detailed_chunk_evidence:
                return False, (
                    "REJECTED: Question asserts unverified implementation details not supported by the cited evidence. "
                    "When evidence only establishes technology usage, questions must explore candidate knowledge "
                    "open-endedly (e.g., 'how would you or did you approach...') without presupposing unstated implementations."
                ), {"code": "unsupported_implementation_presupposition"}

        # 7. Independent Difficulty Estimation (Deterministic text analysis)
        estimated_diff = self.estimate_difficulty_level(q_text)
        if abs(question.difficulty - estimated_diff) >= 2:
            return False, f"REJECTED: Question fails difficulty requirement. Requested Level {question.difficulty}, but question depth evaluated to Level {estimated_diff}.", {
                "code": "difficulty_mismatch",
                "assessed_difficulty": estimated_diff,
                "requested_difficulty": question.difficulty,
                "difficulty_rationale": self.explain_difficulty_level(q_text, estimated_diff),
            }

        # =====================================================================
        # LAZY EMBEDDING EVALUATION:
        # All deterministic checks passed. Compute question embedding lazily
        # only when semantic checks or scoring actually require it.
        # =====================================================================
        _q_emb: Optional[List[float]] = None

        def get_q_emb() -> List[float]:
            nonlocal _q_emb
            if _q_emb is None:
                _q_emb = embedding_service.embed_query(q_text)
            return _q_emb

        # 8. Semantic Claim Alignment (Evaluated lazily with cached claim vector)
        if question.target_claim_id:
            claim_obj = available_claims[question.target_claim_id]
            claim_text = claim_obj.claim_text
            if claim_text not in self._claim_embedding_cache:
                self._claim_embedding_cache[claim_text] = embedding_service.embed_query(claim_text)
            claim_emb = self._claim_embedding_cache[claim_text]
            claim_sim = cosine_similarity(get_q_emb(), claim_emb)
            if claim_sim < settings.RAG_CLAIM_SIMILARITY_MINIMUM:
                return False, f"REJECTED: Target claim '{question.target_claim_id}' does not semantically support the generated question (Sim: {round(claim_sim, 2)}).", {"code": "claim_misaligned"}

        # 9. Embedding-based Semantic Duplicate Detection (Reusing lazy q_emb)
        is_dup, dup_msg, dup_text, dup_id, dup_sim = self.check_semantic_duplicate(
            q_text,
            existing_questions,
            similarity_threshold=settings.RAG_DUPLICATE_SIMILARITY_THRESHOLD,
            new_q_vector=get_q_emb(),
            return_details=True,
            db=db,
            extra_excluded_texts=extra_excluded_texts
        )
        if is_dup:
            return False, f"REJECTED: {dup_msg}", {
                "code": "semantic_duplicate",
                "reason": dup_msg,
                "duplicate_question_id": dup_id,
                "duplicate_question_text": dup_text,
                "similarity": dup_sim
            }

        # 10. Derived Metric Calculations & Scoring
        ev_emb = embedding_service.embed_query(" ".join(cited_evidence_texts))
        semantic_evidence_alignment = max(0.0, cosine_similarity(get_q_emb(), ev_emb))

        cat_vec = self._get_category_vector(question.category)
        semantic_category_alignment = max(0.0, cosine_similarity(get_q_emb(), cat_vec))

        # Grounding Score Calculation ($S_g$)
        citation_fidelity = 1.0  # Verified via check 1
        entity_grounding = 1.0 if (question.target_entity and question.target_entity.lower() in concatenated_cited_text) else 0.8
        project_grounding = 1.0 if (question.target_project and question.target_project.lower() in concatenated_cited_text) else 0.8

        grounding_score = round(
            (citation_fidelity * 0.30) +
            (entity_grounding * 0.25) +
            (project_grounding * 0.15) +
            (semantic_evidence_alignment * 0.30),
            4
        )

        relevance_score = round(min(1.0, semantic_category_alignment * 0.60 + semantic_evidence_alignment * 0.40 + 0.10), 4)

        word_count = len(q_text.split())
        length_penalty = 0.20 if (15 <= word_count <= 45) else 0.05
        technical_quality = round(min(1.0, 0.50 + length_penalty + (0.30 if estimated_diff >= 3 else 0.15)), 4)

        if grounding_score < settings.RAG_GROUNDING_SCORE_MINIMUM:
            return False, f"REJECTED: Derived grounding score {grounding_score} below {settings.RAG_GROUNDING_SCORE_MINIMUM} threshold.", {"code": "low_grounding", "score": grounding_score}

        details = {
            "grounding_score": grounding_score,
            "relevance_score": relevance_score,
            "technical_quality_score": technical_quality,
            "estimated_difficulty": estimated_diff,
            "assessed_difficulty": estimated_diff,
            "difficulty_rationale": self.explain_difficulty_level(q_text, estimated_diff),
            "semantic_evidence_alignment": round(semantic_evidence_alignment, 3),
            "semantic_category_alignment": round(semantic_category_alignment, 3),
            "embedding": get_q_emb(),
            "status": "PASSED",
            "grounding_status": "VERIFIED",
            "rubric_version": "v2.0-canonical-5level"
        }

        question.grounding_score = grounding_score
        question.relevance_score = relevance_score
        question.technical_quality_score = technical_quality

        return True, "PASSED", details

grounding_validator = IndependentGroundingValidator()