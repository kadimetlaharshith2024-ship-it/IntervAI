import re
from typing import List, Dict, Optional, Tuple, Set, Iterable, Any
from sqlalchemy.orm import Session

from app.core.canonicalizer import (
    CANONICAL_TAXONOMY,
    ALIAS_INDEX,
    canonicalize_entity,
    strip_formatting_artifacts,
)
from app.models.relationship import Relationship
from app.models.rag import ResumeClaim
from app.models.interview import InterviewQuestion
from app.schemas.contradiction import (
    ContradictionStatus,
    ContradictionType,
    ContradictionSeverity,
    ConflictDetail,
    ContradictionAnalysisResult,
)
from app.services.evaluation.rubric import calculate_composite_score

# ------------------------------------------------------------------------------
# 1. Functional Substitution Groups (Mutually Exclusive Primary Roles)
# ------------------------------------------------------------------------------
FUNCTIONAL_SUBSTITUTION_GROUPS: Dict[str, Set[str]] = {
    "RELATIONAL_DATABASE": {
        "postgresql", "mysql", "oracle", "oracle dbms", "oracle database",
        "microsoft sql server", "mssql", "mariadb", "sqlite"
    },
    "CACHE_IN_MEMORY": {
        "redis", "memcached", "hazelcast", "aerospike"
    },
    "DOCUMENT_NOSQL": {
        "mongodb", "couchbase", "dynamodb", "firestore", "couchdb"
    },
    "CLOUD_DEPLOYMENT": {
        "aws", "google cloud platform", "gcp", "microsoft azure", "azure",
        "vercel", "netlify", "render", "heroku", "firebase"
    },
    "PYTHON_WEB_FRAMEWORK": {
        "fastapi", "django", "flask"
    },
    "JAVA_WEB_FRAMEWORK": {
        "spring boot", "quarkus", "micronaut"
    },
    "FRONTEND_FRAMEWORK": {
        "react", "angular", "vue", "svelte"
    }
}

# ------------------------------------------------------------------------------
# 2. Known Auxiliary Tools (Plausible additions not in resume)
# ------------------------------------------------------------------------------
KNOWN_AUXILIARY_TOOLS: Set[str] = {
    "pytest", "unittest", "jest", "mocha", "cypress", "selenium",
    "nginx", "apache", "caddy", "envoy", "traefik",
    "datadog", "prometheus", "grafana", "new relic", "splunk", "sentry",
    "eslint", "prettier", "flake8", "black", "mypy", "pylint",
    "webpack", "babel", "esbuild", "turbopack", "parcel",
    "github actions", "gitlab ci", "jenkins", "circleci", "travis ci",
    "linux", "ubuntu", "debian", "alpine", "bash", "zsh", "powershell"
}

# ------------------------------------------------------------------------------
# 3. Conceptual Paraphrases / Category Generalizations (Aligned, NOT Conflicts)
# ------------------------------------------------------------------------------
CONCEPTUAL_PARAPHRASES: Dict[str, Set[str]] = {
    "RELATIONAL_DATABASE": {
        "relational database", "relational databases", "sql database", "sql databases",
        "rdbms", "relational store", "relational storage"
    },
    "CACHE_IN_MEMORY": {
        "cache", "caching", "in-memory cache", "in-memory store", "key-value store",
        "key-value cache", "caching tier", "caching layer"
    },
    "DOCUMENT_NOSQL": {
        "nosql", "nosql database", "document store", "document database"
    },
    "CONTAINERIZATION": {
        "container", "containers", "containerized", "containerization"
    },
    "MESSAGING": {
        "message queue", "message broker", "event stream", "pub-sub", "pub/sub", "event pipeline"
    },
    "SPATIAL_INDEXING": {
        "spatial index", "spatial indexing", "bounding box search"
    },
    "API_ARCHITECTURE": {
        "rest api", "restful api", "rest apis", "rest endpoints", "restful endpoints"
    }
}

# ------------------------------------------------------------------------------
# 4. Role Seniority Levels
# ------------------------------------------------------------------------------
ROLE_HIERARCHY: Dict[str, int] = {
    # Level 1: Intern / Student / Trainee
    "intern": 1,
    "software engineer intern": 1,
    "swe intern": 1,
    "engineering intern": 1,
    "trainee": 1,
    "student intern": 1,
    "student researcher": 1,

    # Level 2: Junior / Associate
    "junior software engineer": 2,
    "junior developer": 2,
    "associate engineer": 2,
    "associate software engineer": 2,

    # Level 3: Mid-level Engineer
    "software engineer": 3,
    "developer": 3,
    "backend developer": 3,
    "backend engineer": 3,
    "frontend developer": 3,
    "frontend engineer": 3,
    "full stack developer": 3,
    "full stack engineer": 3,
    "systems engineer": 3,

    # Level 4: Senior / Lead
    "senior software engineer": 4,
    "senior developer": 4,
    "senior backend engineer": 4,
    "tech lead": 4,
    "team lead": 4,

    # Level 5: Executive / Principal / Architect / Founder
    "lead architect": 5,
    "chief architect": 5,
    "principal architect": 5,
    "head of engineering": 5,
    "director of engineering": 5,
    "vp of engineering": 5,
    "vice president of engineering": 5,
    "chief technology officer": 5,
    "cto": 5,
    "founder": 5,
    "co-founder": 5,
}

# ------------------------------------------------------------------------------
# 5. Hypothetical and Subjunctive Patterns
# ------------------------------------------------------------------------------
HYPOTHETICAL_PATTERNS = [
    re.compile(r"(?i)\b(?:if\s+we\s+(?:had|were|could|needed)|if\s+i\s+(?:were|had)|had\s+we\s+used)\b"),
    re.compile(r"(?i)\b(?:could\s+have|would\s+have|might\s+have|should\s+have)\b"),
    re.compile(r"(?i)\b(?:alternatively|an\s+alternative\s+(?:approach|option|choice|architecture)|in\s+hindsight)\b"),
    re.compile(r"(?i)\b(?:we\s+considered|we\s+evaluated|evaluated\s+between|comparing\s+\w+\s+(?:to|with|vs))\b"),
    re.compile(r"(?i)\b(?:trade-?off|trade-?offs|pros\s+and\s+cons|for\s+future\s+scalability)\b"),
    re.compile(r"(?i)\b(?:would\s+be\s+(?:better|superior|preferable|ideal|suitable))\b"),
    re.compile(r"(?i)\b(?:in\s+a\s+different\s+architecture|hypothetically)\b"),
]


class DeterministicContradictionAnalyzer:
    """
    Deterministic knowledge-graph contradiction analyzer.
    Identifies high-confidence contradictions without confusing missing evidence,
    paraphrases, or hypothetical trade-offs with factual conflicts.
    """

    def analyze(
        self,
        candidate_answer: str,
        question: Optional[InterviewQuestion] = None,
        relationships: Optional[List[Relationship]] = None,
        claims: Optional[List[ResumeClaim]] = None,
        db: Optional[Session] = None
    ) -> ContradictionAnalysisResult:
        """
        Executes isolated deterministic contradiction verification.
        Does NOT invoke any LLM or external service.
        """
        cleaned_answer = (candidate_answer or "").strip()
        if not cleaned_answer:
            return ContradictionAnalysisResult(
                status=ContradictionStatus.NO_APPLICABLE_EVIDENCE,
                has_contradiction=False,
                explanation="Candidate answer is empty."
            )

        # 1. Resolve structured evidence records
        rels, clms = self._resolve_evidence_sources(question, relationships, claims, db)

        # If question is purely theoretical/fundamentals without grounding, skip project checks
        if self._is_purely_theoretical_question(question, rels, clms):
            return ContradictionAnalysisResult(
                status=ContradictionStatus.NO_APPLICABLE_EVIDENCE,
                has_contradiction=False,
                explanation="Theoretical or foundational question with no applicable project anchor."
            )

        # 2. Segment answer into clauses/sentences for fine-grained project anchoring
        clauses = self._split_into_clauses(cleaned_answer)

        # 3. Classify clauses (hypothetical vs factual)
        factual_clauses: List[str] = []
        hypothetical_clauses: List[str] = []

        for clause in clauses:
            if self._is_hypothetical_clause(clause):
                hypothetical_clauses.append(clause)
            else:
                factual_clauses.append(clause)

        # 4. Run deterministic rules on factual clauses
        conflicts: List[ConflictDetail] = []
        supported_claims: Set[str] = set()
        unverified_additions: Set[str] = set()
        hypothetical_claims: List[str] = []

        # Record hypothetical mentions
        for h_clause in hypothetical_clauses:
            extracted = self._extract_entities_from_text(h_clause)
            for ent, _, _ in extracted:
                hypothetical_claims.append(f"{ent} (in: '{h_clause.strip()[:60]}...')")

        # Context anchors
        target_project = question.target_project if question else None
        target_entity = question.target_entity if question else None

        # Rule 1: Technology Substitution
        sub_conflicts = self._check_technology_substitution(
            factual_clauses, target_project, target_entity, rels, clms, supported_claims, unverified_additions, question=question
        )
        conflicts.extend(sub_conflicts)

        # Rule 2: Explicit Negation
        neg_conflicts = self._check_explicit_negation(
            factual_clauses, target_project, target_entity, question, rels, clms
        )
        conflicts.extend(neg_conflicts)

        # Rule 3: Role Incompatibility
        role_conflicts = self._check_role_incompatibility(
            factual_clauses, rels
        )
        conflicts.extend(role_conflicts)

        # Check conceptual paraphrases in factual clauses only if no conflicts found
        if not conflicts:
            self._check_conceptual_paraphrases(
                factual_clauses, target_project, target_entity, rels, clms, supported_claims
            )

        # 5. Determine categorical result status
        if conflicts:
            highest_severity = ContradictionSeverity.HIGH if any(c.severity == ContradictionSeverity.HIGH for c in conflicts) else ContradictionSeverity.MEDIUM
            explanations = "; ".join([c.explanation for c in conflicts])
            return ContradictionAnalysisResult(
                status=ContradictionStatus.CONTRADICTION,
                has_contradiction=True,
                severity=highest_severity,
                conflicts=conflicts,
                supported_claims=sorted(list(supported_claims)),
                unverified_additions=sorted(list(unverified_additions)),
                hypothetical_claims=hypothetical_claims,
                explanation=f"Detected {len(conflicts)} high-confidence contradiction(s): {explanations}"
            )

        if supported_claims:
            return ContradictionAnalysisResult(
                status=ContradictionStatus.SUPPORTED,
                has_contradiction=False,
                severity=ContradictionSeverity.LOW,
                conflicts=[],
                supported_claims=sorted(list(supported_claims)),
                unverified_additions=sorted(list(unverified_additions)),
                hypothetical_claims=hypothetical_claims,
                explanation=f"Answer is supported by verified resume evidence: {', '.join(sorted(supported_claims))}."
            )

        if hypothetical_clauses and not factual_clauses:
            return ContradictionAnalysisResult(
                status=ContradictionStatus.HYPOTHETICAL_DISCUSSION,
                has_contradiction=False,
                severity=ContradictionSeverity.LOW,
                conflicts=[],
                supported_claims=[],
                unverified_additions=[],
                hypothetical_claims=hypothetical_claims,
                explanation="Answer focuses entirely on hypothetical alternatives or trade-off evaluation."
            )

        if unverified_additions:
            return ContradictionAnalysisResult(
                status=ContradictionStatus.UNVERIFIED_ADDITION,
                has_contradiction=False,
                severity=ContradictionSeverity.LOW,
                conflicts=[],
                supported_claims=[],
                unverified_additions=sorted(list(unverified_additions)),
                hypothetical_claims=hypothetical_claims,
                explanation=f"Answer introduces auxiliary details/technologies not verified in resume: {', '.join(sorted(unverified_additions))}."
            )

        return ContradictionAnalysisResult(
            status=ContradictionStatus.NO_APPLICABLE_EVIDENCE,
            has_contradiction=False,
            severity=ContradictionSeverity.LOW,
            conflicts=[],
            supported_claims=[],
            unverified_additions=[],
            hypothetical_claims=hypothetical_claims,
            explanation="No direct resume evidence was applicable or matched."
        )

    # --------------------------------------------------------------------------
    # Rule 1 Implementation: Technology Substitution on Anchored Project
    # --------------------------------------------------------------------------
    def _check_technology_substitution(
        self,
        factual_clauses: List[str],
        target_project: Optional[str],
        target_entity: Optional[str],
        rels: List[Relationship],
        clms: List[ResumeClaim],
        supported_claims: Set[str],
        unverified_additions: Set[str],
        question: Optional[InterviewQuestion] = None
    ) -> List[ConflictDetail]:
        conflicts: List[ConflictDetail] = []

        # Build project -> verified technologies map, preserving exact original project names
        project_name_map: Dict[str, str] = {}
        project_tech_map: Dict[str, Set[str]] = {}

        for r in rels:
            proj = r.context_entity or (r.source_entity if r.section == "Projects" else None)
            if proj and r.target_entity:
                p_lower = proj.lower()
                project_name_map[p_lower] = proj
                canon_target, _ = canonicalize_entity(r.target_entity)
                if canon_target:
                    project_tech_map.setdefault(p_lower, set()).add(canon_target.lower())

        for c in clms:
            if c.project and c.target_entity:
                p_lower = c.project.lower()
                project_name_map[p_lower] = c.project
                canon_target, _ = canonicalize_entity(c.target_entity)
                if canon_target:
                    project_tech_map.setdefault(p_lower, set()).add(canon_target.lower())

        if target_project:
            p_lower = target_project.lower()
            project_name_map[p_lower] = target_project
            if target_entity:
                canon_tgt, _ = canonicalize_entity(target_entity)
                if canon_tgt:
                    project_tech_map.setdefault(p_lower, set()).add(canon_tgt.lower())
            if question and question.evidence_text:
                ev_extracted = self._extract_entities_from_text(question.evidence_text)
                for ev_ent, _, _ in ev_extracted:
                    canon_ev, _ = canonicalize_entity(ev_ent)
                    if canon_ev:
                        project_tech_map.setdefault(p_lower, set()).add(canon_ev.lower())

        for clause in factual_clauses:
            extracted = self._extract_entities_from_text(clause)
            clause_lower = clause.lower()

            # Check if this clause is anchored to a specific project
            anchored_proj_lower = self._detect_clause_project_anchor(clause, target_project, project_name_map.keys())
            if not anchored_proj_lower:
                # Unanchored clause: track entities as unverified or supported across resume
                for ent, _, _ in extracted:
                    ent_norm = ent.lower()
                    if self._is_entity_anywhere_in_resume(ent_norm, rels, clms):
                        supported_claims.add(ent)
                    else:
                        unverified_additions.add(ent)
                continue

            orig_proj_name = project_name_map.get(anchored_proj_lower, anchored_proj_lower.title())
            verified_in_proj = project_tech_map.get(anchored_proj_lower, set())

            for candidate_ent, _, _ in extracted:
                cand_norm = candidate_ent.lower()

                # If the candidate entity is mentioned in an "instead of [candidate_ent]" phrase, skip treating it as verified
                if f"instead of {cand_norm}" in clause_lower or f"rather than {cand_norm}" in clause_lower:
                    continue

                # If candidate entity is verified in THIS project, it's supported
                if cand_norm in verified_in_proj:
                    supported_claims.add(candidate_ent)
                    continue

                # Check if candidate entity belongs to a mutually exclusive functional group
                cand_group = self._get_functional_group(cand_norm)
                if not cand_group:
                    # Auxiliary tool (e.g. pytest, nginx) -> unverified addition
                    unverified_additions.add(candidate_ent)
                    continue

                # Check if project has a verified technology in that EXACT functional group
                conflicting_verified = [
                    v for v in verified_in_proj
                    if v != cand_norm and self._get_functional_group(v) == cand_group
                ]

                if not conflicting_verified:
                    # No competing verified technology in that group on this project -> unverified addition
                    unverified_additions.add(candidate_ent)
                    continue

                # Check for explicit substitution signal
                is_explicit_substitution = self._is_explicit_substitution_assertion(
                    clause_lower, cand_norm, conflicting_verified[0], orig_proj_name
                )

                if is_explicit_substitution:
                    canon_v, _ = canonicalize_entity(conflicting_verified[0])
                    verified_name = canon_v if canon_v else conflicting_verified[0].title()
                    matching_rel = self._find_matching_relationship(orig_proj_name, verified_name, rels)
                    evidence_id = str(matching_rel.id) if matching_rel else None
                    evidence_text = matching_rel.evidence_text if matching_rel else None
                    if not evidence_id and question:
                        if question.evidence_ids and len(question.evidence_ids) > 0:
                            evidence_id = str(question.evidence_ids[0])
                        elif question.target_claim_id:
                            evidence_id = str(question.target_claim_id)
                    if not evidence_text and question and question.evidence_text:
                        evidence_text = question.evidence_text
                    if not evidence_text:
                        evidence_text = f"Verified {verified_name} in {orig_proj_name}."

                    conflicts.append(ConflictDetail(
                        conflict_type=ContradictionType.TECHNOLOGY_SUBSTITUTION,
                        severity=ContradictionSeverity.HIGH,
                        candidate_assertion=clause.strip(),
                        verified_fact=f"Project '{orig_proj_name}' verifies {verified_name} as primary {cand_group.replace('_', ' ').lower()}.",
                        context_project=orig_proj_name,
                        evidence_id=evidence_id,
                        evidence_text=evidence_text,
                        explanation=(
                            f"Candidate explicitly asserted using '{candidate_ent}' for project '{orig_proj_name}', "
                            f"directly substituting verified technology '{verified_name}' in the same functional role."
                        )
                    ))
                else:
                    unverified_additions.add(candidate_ent)

        return conflicts

    # --------------------------------------------------------------------------
    # Rule 2 Implementation: Explicit Negation of Verified Technology
    # --------------------------------------------------------------------------
    def _check_explicit_negation(
        self,
        factual_clauses: List[str],
        target_project: Optional[str],
        target_entity: Optional[str],
        question: Optional[InterviewQuestion],
        rels: List[Relationship],
        clms: List[ResumeClaim]
    ) -> List[ConflictDetail]:
        conflicts: List[ConflictDetail] = []

        verified_entities: Set[Tuple[str, str, Optional[str], Optional[str]]] = set()

        if target_project and target_entity:
            verified_entities.add((target_entity, target_project, "Target entity for project", None))

        for r in rels:
            proj = r.context_entity or (r.source_entity if r.section == "Projects" else None)
            if r.target_entity:
                verified_entities.add((r.target_entity, proj or "Resume", r.evidence_text, str(r.id)))

        for c in clms:
            if c.target_entity:
                verified_entities.add((c.target_entity, c.project or "Resume", c.evidence_text, c.claim_id))

        for clause in factual_clauses:
            clause_lower = clause.lower()

            for ent_name, proj_context, ev_text, ev_id in verified_entities:
                ent_norm = ent_name.lower()
                if len(ent_norm) < 3:
                    continue

                negation_pattern = rf"(?i)\b(?:did\s+not|didn['’]?t|never|avoided|without\s+using|refrained\s+from\s+using)\s+(?:use\s+|using\s+|implement\s+|deploy\s+|containerize\s+with\s+)?{re.escape(ent_norm)}\b"
                passive_negation = rf"(?i)\b{re.escape(ent_norm)}\s+(?:was\s+not|wasn['’]?t|were\s+not|weren['’]?t)\s+(?:used|implemented|deployed|configured)\b"
                no_ent_pattern = rf"(?i)\bno\s+{re.escape(ent_norm)}\s+(?:was|were)\s+(?:used|configured)\b"

                if (
                    re.search(negation_pattern, clause_lower) or
                    re.search(passive_negation, clause_lower) or
                    re.search(no_ent_pattern, clause_lower)
                ):
                    if not target_project or not proj_context or proj_context.lower() == target_project.lower() or proj_context == "Resume":
                        conflicts.append(ConflictDetail(
                            conflict_type=ContradictionType.EXPLICIT_NEGATION,
                            severity=ContradictionSeverity.HIGH,
                            candidate_assertion=clause.strip(),
                            verified_fact=f"Verified usage of {ent_name} in {proj_context}.",
                            context_project=proj_context,
                            evidence_id=ev_id,
                            evidence_text=ev_text,
                            explanation=(
                                f"Candidate explicitly negated using '{ent_name}', but verified resume evidence "
                                f"confirms its active implementation in {proj_context}."
                            )
                        ))

        return conflicts

    # --------------------------------------------------------------------------
    # Rule 3 Implementation: Role Incompatibility with Employer Context
    # --------------------------------------------------------------------------
    def _check_role_incompatibility(
        self,
        factual_clauses: List[str],
        rels: List[Relationship]
    ) -> List[ConflictDetail]:
        conflicts: List[ConflictDetail] = []

        verified_roles: Dict[str, Tuple[str, str, int, str, Optional[str]]] = {}
        for r in rels:
            if r.section == "Experience" and r.relation_type in ["WORKED_AS", "ROLE"] and r.target_entity:
                emp_orig = (r.context_entity or "").strip()
                emp_lower = emp_orig.lower()
                role_clean = r.target_entity.strip().lower()
                rank = self._get_role_rank(role_clean)
                if emp_lower and rank > 0:
                    verified_roles[emp_lower] = (emp_orig, r.target_entity, rank, r.evidence_text, str(r.id))

        if not verified_roles:
            return []

        role_claim_patterns = [
            re.compile(r"(?i)\b(?:as\s+(?:the|a)\s+([a-z\s]{3,35}?)\s+(?:at|for|with)\s+([a-z0-9\s\.\-_]{3,30}?)(?:[,\.]|\band\b|\bi\b))"),
            re.compile(r"(?i)\b(?:when\s+i\s+was\s+(?:the|a)\s+([a-z\s]{3,35}?)\s+(?:at|for|with)\s+([a-z0-9\s\.\-_]{3,30}?)(?:[,\.]|\band\b|\bi\b))"),
            re.compile(r"(?i)\b(?:i\s+was\s+(?:the|a)\s+([a-z\s]{3,35}?)\s+(?:at|for|with)\s+([a-z0-9\s\.\-_]{3,30}?)(?:[,\.]|\band\b|\bi\b))"),
        ]

        for clause in factual_clauses:
            for pat in role_claim_patterns:
                for match in pat.finditer(clause):
                    claimed_role_raw = match.group(1).strip().lower()
                    claimed_emp_raw = match.group(2).strip().lower()

                    claimed_rank = self._get_role_rank(claimed_role_raw)
                    if claimed_rank == 0:
                        continue

                    for v_emp_lower, (v_emp_orig, v_role_title, v_rank, ev_text, ev_id) in verified_roles.items():
                        if self._is_same_employer(claimed_emp_raw, v_emp_lower):
                            if (claimed_rank - v_rank) >= 3:
                                conflicts.append(ConflictDetail(
                                    conflict_type=ContradictionType.ROLE_INCOMPATIBILITY,
                                    severity=ContradictionSeverity.HIGH,
                                    candidate_assertion=clause.strip(),
                                    verified_fact=f"Verified role at {v_emp_orig} is '{v_role_title}'.",
                                    context_project=v_emp_orig,
                                    evidence_id=ev_id,
                                    evidence_text=ev_text,
                                    explanation=(
                                        f"Candidate claimed seniority role '{claimed_role_raw.title()}' at '{v_emp_orig}', "
                                        f"which contradicts verified employment record '{v_role_title}'."
                                    )
                                ))

        return conflicts

    # --------------------------------------------------------------------------
    # Conceptual Paraphrase Checker
    # --------------------------------------------------------------------------
    def _check_conceptual_paraphrases(
        self,
        factual_clauses: List[str],
        target_project: Optional[str],
        target_entity: Optional[str],
        rels: List[Relationship],
        clms: List[ResumeClaim],
        supported_claims: Set[str]
    ) -> None:
        """Finds general conceptual descriptions that match verified technologies."""
        verified_techs: Set[str] = set()
        if target_entity:
            verified_techs.add(target_entity.lower())
        for r in rels:
            if r.target_entity:
                verified_techs.add(r.target_entity.lower())
        for c in clms:
            if c.target_entity:
                verified_techs.add(c.target_entity.lower())

        for clause in factual_clauses:
            clause_lower = clause.lower()
            for group, paraphrases in CONCEPTUAL_PARAPHRASES.items():
                for paraphrase in paraphrases:
                    if paraphrase in clause_lower:
                        # Check if any verified technology falls into this group or matches
                        group_members = FUNCTIONAL_SUBSTITUTION_GROUPS.get(group, set())
                        if any(v in group_members for v in verified_techs):
                            matching_techs = [v.title() for v in verified_techs if v in group_members]
                            supported_claims.add(f"{paraphrase} (conceptually aligns with {', '.join(matching_techs)})")

    # --------------------------------------------------------------------------
    # Helper & Classification Methods
    # --------------------------------------------------------------------------
    def _is_hypothetical_clause(self, clause: str) -> bool:
        """Determines if a clause/sentence is hypothetical, speculative, or a trade-off discussion."""
        for pat in HYPOTHETICAL_PATTERNS:
            if pat.search(clause):
                return True
        return False

    def _split_into_clauses(self, text: str) -> List[str]:
        """
        Splits text into fine-grained clauses and sentences to isolate distinct project contexts.
        Splits on sentence terminals and contrastive clause conjunctions (, while / , whereas / , but in).
        """
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        clauses: List[str] = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            # Split by clause conjunctions that introduce separate project contexts
            sub_clauses = re.split(r"(?i)(?:,\s*(?:while|whereas|but\s+in|and\s+in|whereas\s+in)\s+)", s_clean)
            for sub in sub_clauses:
                if len(sub.strip()) > 3:
                    clauses.append(sub.strip())
        return clauses

    def _extract_entities_from_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extracts entities from text using ALIAS_INDEX, plus substitution groups and auxiliary tools.
        """
        lower_text = text.lower()
        candidates: List[Tuple[str, int, int, int]] = []

        # 1. Check ALIAS_INDEX
        sorted_aliases = sorted(ALIAS_INDEX.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            canonical = ALIAS_INDEX[alias]
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            for match in re.finditer(pattern, lower_text):
                candidates.append((canonical, match.start(), match.end(), match.end() - match.start()))

        # 2. Check FUNCTIONAL_SUBSTITUTION_GROUPS for entries not in ALIAS_INDEX (e.g. Oracle DBMS, Memcached)
        for group, members in FUNCTIONAL_SUBSTITUTION_GROUPS.items():
            for member in members:
                if member not in ALIAS_INDEX:
                    pattern = rf"(?<!\w){re.escape(member)}(?!\w)"
                    for match in re.finditer(pattern, lower_text):
                        canon_name = "Oracle DBMS" if "oracle" in member else member.title()
                        candidates.append((canon_name, match.start(), match.end(), match.end() - match.start()))

        # 3. Check KNOWN_AUXILIARY_TOOLS
        for tool in KNOWN_AUXILIARY_TOOLS:
            pattern = rf"(?<!\w){re.escape(tool)}(?!\w)"
            for match in re.finditer(pattern, lower_text):
                candidates.append((tool.title(), match.start(), match.end(), match.end() - match.start()))

        # Longest match first deduplication
        candidates.sort(key=lambda x: x[3], reverse=True)
        claimed_spans: List[Tuple[int, int]] = []
        resolved: List[Tuple[str, int, int]] = []

        for canon, start, end, _ in candidates:
            overlaps = any(not (end <= c_start or start >= c_end) for c_start, c_end in claimed_spans)
            if not overlaps:
                claimed_spans.append((start, end))
                resolved.append((canon, start, end))

        return sorted(resolved, key=lambda x: x[1])

    def _get_functional_group(self, entity_lower: str) -> Optional[str]:
        """Maps an entity to its mutually exclusive primary functional substitution group."""
        for group_name, members in FUNCTIONAL_SUBSTITUTION_GROUPS.items():
            if entity_lower in members:
                return group_name
            # Check prefix/alias matching (e.g. "oracle dbms" matching "oracle")
            for m in members:
                if m in entity_lower or entity_lower in m:
                    return group_name
        return None

    def _is_explicit_substitution_assertion(
        self,
        clause_lower: str,
        candidate_ent_lower: str,
        verified_ent_lower: str,
        project_name: str
    ) -> bool:
        """
        Determines whether the candidate explicitly asserts candidate_ent INSTEAD OF
        or as the primary substitute for verified_ent in project_name.
        """
        replacement_phrases = [
            f"instead of {verified_ent_lower}",
            f"rather than {verified_ent_lower}",
            f"chose {candidate_ent_lower} over {verified_ent_lower}",
            f"replaced {verified_ent_lower} with {candidate_ent_lower}",
            f"switched to {candidate_ent_lower}",
        ]
        if any(p in clause_lower for p in replacement_phrases):
            return True

        primary_patterns = [
            rf"\b(?:our\s+primary|the\s+primary|main|sole)\s+(?:database|datastore|db|framework|cloud)\s+(?:was|is)\s+{re.escape(candidate_ent_lower)}\b",
            rf"\b(?:we\s+used|built\s+with|configured)\s+{re.escape(candidate_ent_lower)}\s+(?:as\s+(?:the|our)\s+primary|instead)\b",
            rf"\b(?:in|for|on)\s+{re.escape(project_name.lower())}[^.]*?\b(?:we\s+used|our\s+database\s+was)\s+{re.escape(candidate_ent_lower)}\b",
        ]
        for pat in primary_patterns:
            if re.search(pat, clause_lower):
                return True

        return False

    def _matches_conceptual_paraphrase(self, clause_lower: str, verified_entities: Set[str]) -> bool:
        """Checks if a clause uses general conceptual terms that match a verified entity."""
        for verified_ent in verified_entities:
            group = self._get_functional_group(verified_ent)
            if group and group in CONCEPTUAL_PARAPHRASES:
                for paraphrase in CONCEPTUAL_PARAPHRASES[group]:
                    if paraphrase in clause_lower:
                        return True
        return False

    def _detect_clause_project_anchor(
        self,
        clause: str,
        target_project: Optional[str],
        known_projects: Iterable[str]
    ) -> Optional[str]:
        """
        Detects which project a clause is anchored to.
        Prioritizes explicit project names mentioned in the clause.
        """
        clause_lower = clause.lower()

        # Check if a known project name is explicitly mentioned in this clause
        for proj_lower in known_projects:
            if proj_lower and proj_lower in clause_lower:
                return proj_lower

        # If question targets a specific project, anchor if clause references it implicitly
        if target_project:
            p_lower = target_project.lower()
            if p_lower in clause_lower:
                return p_lower
            if any(ref in clause_lower for ref in ["in this project", "in our project", "for this system", "we designed", "we built", "we used"]):
                return p_lower

        return None

    def _is_entity_anywhere_in_resume(
        self,
        entity_lower: str,
        rels: List[Relationship],
        clms: List[ResumeClaim]
    ) -> bool:
        """Checks if an entity is verified anywhere in the candidate's resume."""
        for r in rels:
            if r.target_entity and r.target_entity.lower() == entity_lower:
                return True
        for c in clms:
            if c.target_entity and c.target_entity.lower() == entity_lower:
                return True
        return False

    def _is_entity_in_specific_project(
        self,
        entity_lower: str,
        project_lower: str,
        project_tech_map: Dict[str, Set[str]]
    ) -> bool:
        """Checks if an entity is verified in a specific project."""
        techs = project_tech_map.get(project_lower, set())
        return entity_lower in techs

    def _get_role_rank(self, role_lower: str) -> int:
        """Returns the hierarchical seniority rank (1-5) of a role string."""
        role_cleaned = role_lower.strip()
        if role_cleaned in ROLE_HIERARCHY:
            return ROLE_HIERARCHY[role_cleaned]
        for title, rank in ROLE_HIERARCHY.items():
            if title in role_cleaned:
                return rank
        return 0

    def _is_same_employer(self, emp_a: str, emp_b: str) -> bool:
        """Fuzzy/exact comparison of employer names."""
        a = re.sub(r"[^\w\s]", "", emp_a).strip()
        b = re.sub(r"[^\w\s]", "", emp_b).strip()
        if a == b or a in b or b in a:
            return True
        return False

    def _find_matching_relationship(
        self,
        project_name: str,
        entity_name: str,
        rels: List[Relationship]
    ) -> Optional[Relationship]:
        """Finds matching relationship record for evidence attribution."""
        p_lower = project_name.lower()
        e_lower = entity_name.lower()
        for r in rels:
            r_proj = (r.context_entity or (r.source_entity if r.section == "Projects" else "")).lower()
            if (p_lower in r_proj or r_proj in p_lower) and r.target_entity and e_lower in r.target_entity.lower():
                return r
        return None

    def _is_purely_theoretical_question(
        self,
        question: Optional[InterviewQuestion],
        rels: List[Relationship],
        clms: List[ResumeClaim]
    ) -> bool:
        """Identifies whether question is purely theoretical/foundational with no resume anchor."""
        if not question:
            return not bool(rels or clms)
        if question.category == "Fundamentals":
            return True
        if not question.target_project and not question.evidence_text and not question.targeted_claim:
            return True
        return False

    def _resolve_evidence_sources(
        self,
        question: Optional[InterviewQuestion],
        relationships: Optional[List[Relationship]],
        claims: Optional[List[ResumeClaim]],
        db: Optional[Session]
    ) -> Tuple[List[Relationship], List[ResumeClaim]]:
        """Resolves relationships and claims from direct arguments or database session."""
        rels = relationships or []
        clms = claims or []

        if not rels and db and question and question.resume_id:
            rels = db.query(Relationship).filter(Relationship.resume_id == question.resume_id).all()
        if not clms and db and question and question.resume_id:
            clms = db.query(ResumeClaim).filter(ResumeClaim.resume_id == question.resume_id).all()

        # Ensure relationships and claims belong strictly to the question's resume_id
        if question and question.resume_id:
            if rels:
                rels = [r for r in rels if not getattr(r, "resume_id", None) or r.resume_id == question.resume_id]
            if clms:
                clms = [c for c in clms if not getattr(c, "resume_id", None) or c.resume_id == question.resume_id]

        return rels, clms


# Singleton instance for service injection
contradiction_analyzer = DeterministicContradictionAnalyzer()


class ContradictionScoringPolicy:
    """
    Deterministic scoring adjustment policy for verified resume/project contradictions.

    POLICY RULES & INVARIANTS:
    1. Deterministic capping applies ONLY when a contradiction is high-confidence (severity == HIGH),
       firmly anchored to verified project/employer evidence.
    2. Adjustments are strictly restricted to 'evidence_alignment'.
    3. Other dimensions ('technical_accuracy', 'conceptual_understanding', 'completeness',
       'depth_and_specificity', 'practical_reasoning') are NEVER penalized solely for resume contradictions.
    4. If the conflict or context is ambiguous (severity == MEDIUM/LOW), findings are provided
       to the LLM prompt for contextual consideration without applying a deterministic score cap.
    5. The composite overall_score is deterministically recalculated with category weights.
    6. All adjustments are transparently recorded in feedback weaknesses and summary for auditability.
    """
    POLICY_VERSION: str = "v1.0"
    DEFAULT_HIGH_CONFIDENCE_CAP: float = 35.0  # Configurable upper ceiling for verified direct contradictions

    @classmethod
    def apply_policy(
        cls,
        contradiction_result: ContradictionAnalysisResult,
        scores: Dict[str, float],
        feedback: Dict[str, Any],
        question_category: str,
        custom_cap: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float, bool]:
        """
        Applies deterministic adjustment if high-confidence contradiction exists.

        Returns:
            (normalized_scores, feedback, recalculated_overall_score, was_adjusted)
        """
        # 1. Check if high-confidence contradiction criteria are met
        if not (
            contradiction_result.has_contradiction
            and contradiction_result.severity == ContradictionSeverity.HIGH
            and len(contradiction_result.conflicts) > 0
        ):
            overall_score, normalized_scores = calculate_composite_score(question_category, scores)
            return normalized_scores, feedback, overall_score, False

        # 2. Verify at least one high-confidence conflict with context
        high_conf_conflicts = [
            c for c in contradiction_result.conflicts
            if c.severity == ContradictionSeverity.HIGH
        ]
        if not high_conf_conflicts:
            overall_score, normalized_scores = calculate_composite_score(question_category, scores)
            return normalized_scores, feedback, overall_score, False

        # 3. Apply cap to evidence_alignment only
        cap = custom_cap if custom_cap is not None else cls.DEFAULT_HIGH_CONFIDENCE_CAP
        current_align = scores.get("evidence_alignment", 100.0)
        was_adjusted = False

        adjusted_scores = dict(scores)
        if current_align > cap:
            adjusted_scores["evidence_alignment"] = cap
            was_adjusted = True

        # 4. Deterministically recalculate composite overall_score
        new_overall, normalized_scores = calculate_composite_score(question_category, adjusted_scores)

        # 5. Explain in feedback for auditability
        adjusted_feedback = dict(feedback)
        weaknesses = list(adjusted_feedback.get("weaknesses") or [])
        for c in high_conf_conflicts:
            note = (
                f"Factual Contradiction Detected: On project '{c.context_project or 'anchored project'}', "
                f"verified architecture used '{c.verified_fact}', but candidate claimed '{c.candidate_assertion}'. "
                f"Evidence alignment score capped at {cap:.1f}/100.0."
            )
            if note not in weaknesses:
                weaknesses.append(note)
        adjusted_feedback["weaknesses"] = weaknesses

        if was_adjusted:
            summary = adjusted_feedback.get("summary") or ""
            c0 = high_conf_conflicts[0]
            audit_msg = (
                f" [Evidence Alignment Adjustment: Score adjusted from {current_align:.1f} to {cap:.1f} "
                f"due to verified contradiction on '{c0.context_project or 'project'}']."
            )
            if "[Evidence Alignment Adjustment:" not in summary:
                adjusted_feedback["summary"] = summary + audit_msg

        return normalized_scores, adjusted_feedback, new_overall, was_adjusted
