import re
import json
from typing import Dict, Any, Optional, List
from pydantic import ValidationError
from app.schemas.rag import QuestionOutputSchema
from app.services.llm.factory import get_llm_provider

CATEGORY_OBJECTIVES: Dict[str, str] = {
    "Architecture": "Examine system component boundaries, data flow across interfaces, integration contracts, separation of concerns, and architectural trade-offs.",
    "Security": "Examine authentication mechanisms, authorization boundaries, input handling, encryption, token lifecycle, and threat resilience.",
    "Performance": "Examine concurrency handling, latency mitigation, query/resource optimization, throughput constraints, and scalability limits.",
    "Implementation": "Examine concrete code-level mechanics, API contracts, state transitions, transaction handling, and error propagation.",
    "Fundamentals": "Examine foundational principles and protocol mechanics underpinning the candidate's demonstrated technology.",
    "Failure-Mode": "Examine failure domains, exception boundaries, fallback mechanisms, edge cases, and crash recovery.",
    "Claim-Verification": "Probe the exact claims stated by the candidate to verify hands-on execution depth and implementation authenticity."
}

DIFFICULTY_RUBRIC: Dict[int, str] = {
    1: "Level 1 — Fundamentals (Understand): Test understanding of core concepts, definitions, principles, and mechanisms behind a skill, technology, or project explicitly supported by the resume. Focus on 'What is X?', 'How does X fundamentally work under the hood?', or 'Why is X useful for Y?' without demanding complex trade-offs or debugging.",
    2: "Level 2 — Application (Apply): Require the candidate to apply a known concept or technology to a concrete situation or practical data flow in their documented project or experience.",
    3: "Level 3 — Implementation (Build and Debug): Require reasoning about implementation details, algorithms, data structures, debugging, boundary case testing, and technical decisions relevant to the candidate's documented work.",
    4: "Level 4 — Architecture (Design and Evaluate): Require system-level reasoning about component interactions, system boundaries, scalability, reliability, security, asynchronous data flow, and architectural trade-offs.",
    5: "Level 5 — Advanced Depth (Analyze and Defend): Require deeper reasoning under uncertainty, competing constraints, alternative approaches, failure scenarios, sensitivity to changing assumptions, and defensible engineering trade-offs."
}

class QuestionGenerationService:
    def __init__(self, force_mock: bool = False):
        self.force_mock = force_mock

    def generate(
        self,
        context: Dict[str, Any],
        category: str,
        difficulty: int,
        target_project: Optional[str] = None,
        target_entity: Optional[str] = None,
        trace_id: str = "",
        feedback_instruction: Optional[str] = None,
        previously_asked_questions: Optional[List[str]] = None,
        past_session_hints: Optional[List[str]] = None
    ) -> QuestionOutputSchema:
        try:
            provider = get_llm_provider(force_mock=self.force_mock, role="generation")
        except TypeError:
            provider = get_llm_provider(force_mock=self.force_mock)

        selected_evidence = context.get("selected_evidence_items", [])
        if not selected_evidence and not context.get("relationships") and not context.get("claims"):
            raise ValueError("Cannot generate question: No validated candidate evidence provided.")

        system_instruction = (
            "You are an expert technical interviewer for IntervAI, an adaptive mock interview platform.\n"
            "Your objective is to generate exactly ONE technical interview question derived strictly from the provided candidate resume evidence.\n\n"
            "MANDATORY OPERATING RULES:\n"
            "1. Grounding Guarantee: Every entity, technology, and project mentioned in your question MUST be explicitly present in the supplied evidence.\n"
            "2. Zero Hallucination: Do NOT assume, introduce, or hallucinate tools, frameworks, metrics, cloud providers, or design patterns absent from the evidence.\n"
            "3. Evidence Entailment & Open-Ended Phrasing (CRITICAL):\n"
            "   - If the cited evidence explicitly describes an implemented mechanism (e.g. 'built a recruiter verification workflow requiring admin approval'), you may ask directly about that documented implementation.\n"
            "   - If the cited evidence only establishes technology presence or usage (e.g. 'uses Java, Spring Boot, and PostgreSQL' or a 'HAS_TECHNOLOGY' relationship), you MUST formulate an OPEN-ENDED design question exploring candidate knowledge and architectural reasoning (e.g., 'How would you approach designing...', 'What approach would you take to...', 'What architectural trade-offs would you evaluate when...').\n"
            "   - Under technology-only evidence, you MUST NEVER presuppose, command, or assert that the candidate implemented unstated mechanisms. NEVER use phrasing like 'walk through the flow/steps of your implementation', 'walk through how you implemented/configured', 'how did you configure/manage your [mechanism]', or 'detail how [mechanism] was handled'. Always ask how they *would* approach designing or structuring it.\n"
            "   - EXAMPLES OF OPEN-ENDED VS PRESUPPOSITION PHRASING:\n"
            "     * BAD (presupposes unstated implementation): 'Walk me through how you implemented transactional error recovery in Spring Boot and PostgreSQL.'\n"
            "     * GOOD (open-ended inquiry): 'How would you approach designing transactional boundary management and error propagation between Spring Boot and PostgreSQL?'\n"
            "     * BAD (presupposes unstated configuration): 'Detail your custom configuration of Redis caching and how cache invalidation was handled.'\n"
            "     * GOOD (open-ended inquiry): 'What caching strategies and cache invalidation policies would you evaluate when integrating Redis into this architecture?'\n"
            "4. Mechanism Diversification vs Previous Questions (CRITICAL):\n"
            "   - If previous questions were asked, your question MUST probe a MATERIALLY DIFFERENT mechanism, architectural dimension, or trade-off.\n"
            "   - EXAMPLES OF MECHANISM DIVERSIFICATION:\n"
            "     * BAD (redundant mechanism): Prior question asked about connection pooling in PostgreSQL -> New candidate asks 'What trade-offs exist with PostgreSQL connection pool sizes?'\n"
            "     * GOOD (distinct mechanism): Prior question asked about connection pooling in PostgreSQL -> New candidate asks 'How would you structure database indexing and query plan optimization for high-throughput reads in PostgreSQL?'\n"
            "5. Evidence-Sparse Fallback / Fundamentals-Adjacent Mode (Part B6):\n"
            "   - When the cited evidence is only a skills list or a bare technology relationship (e.g. 'Python, PostgreSQL' or a bare HAS_TECHNOLOGY link without implementation text):\n"
            "   - Allow fundamentals-adjacent questions probing core architectural concepts, internal execution mechanics, or trade-offs of the cited technology (e.g. 'What core indexing structures does PostgreSQL support for spatial or high-throughput queries, and what are their trade-offs?'), instead of failing or producing generic design questions.\n"
            "   - Still name ONLY entities explicitly present in the candidate evidence. Keep the strict no-presupposition rule.\n"
            "6. No Shallow Fluff: Do NOT ask shallow questions (e.g. 'What is X?'). Frame questions around practical architecture, tradeoffs, and engineering depth.\n"
            "7. Citation Requirement: In the 'evidence_ids' field, cite the exact ID values from the supplied evidence (including chunk 'evidence_id', relationship 'relationship_id', or claim 'claim_id' values) that support the question.\n"
            "8. Provenance Linkage: If a claim or relationship directly supports the question, populate 'target_claim_id' and 'target_relationship_id' using provided IDs.\n"
            "9. Strict Output: You MUST return a single, valid JSON object matching the requested schema.\n"
            "10. Strict Target Entity Alignment (CRITICAL):\n"
            "   - When a Target Entity Focus is specified, the question's core technical inquiry MUST directly probe that specific technology/entity.\n"
            "   - The central question (how would you approach designing, implementing, configuring, or optimizing...) must be explicitly centered on that Target Entity.\n"
            "   - Do NOT make the Target Entity an incidental preamble mention while pivoting the actual technical inquiry to an unrelated concept or secondary technology.\n"
            "11. Modality & Temporal Status Integrity (CRITICAL):\n"
            "   - If evidence notes an in-progress certification, ongoing study, or planned tool (e.g. 'In Progress', 'pursuing', 'studying for'), formulate learning-oriented or foundational questions exploring candidate understanding. NEVER presuppose that the candidate has completed the certification or holds an earned professional credential.\n"
            "   - If evidence is an online course (e.g. Udemy, Coursera), treat it as completed coursework, not an accredited professional certification.\n"
            "12. Project Scope & Hackathon Alignment (CRITICAL):\n"
            "   - If evidence describes a hackathon prototype, rapid MVP (e.g. built in 12 hours), or competition entry, formulate questions probing rapid prototyping trade-offs, constraint handling, component prioritization, or team collaboration. NEVER presuppose enterprise-scale production infrastructure or multi-year operational maintenance for a rapid hackathon prototype.\n"
            "13. Strict Difficulty Reasoning Demand Calibration (MANDATORY):\n"
            "   - Level 1 (Fundamentals / Understand): Probe core concepts, definitions, or foundational mechanics of the cited entity (e.g. 'What is X and why is it useful for Y?'). Do NOT demand system architecture or code debugging.\n"
            "   - Level 2 (Application / Apply): Probe applying the concept to a concrete situation in their documented project or practical data flow between components.\n"
            "   - Level 3 (Implementation / Build and Debug): Probe concrete code mechanics, algorithms, data structures (e.g. node capacity, branching factor), boundary case testing, or transaction boundaries.\n"
            "   - Level 4 (Architecture / Design and Evaluate): Probe system component boundaries, scalability, reliability, security, asynchronous data flows, and architectural trade-offs.\n"
            "   - Level 5 (Advanced Depth / Analyze and Defend): Probe deeper reasoning under uncertainty, competing constraints, failure scenarios (e.g. cascading outages, network partition), sensitivity to changing assumptions, or comparing alternative strategies under shifting conditions."
        )

        category_guideline = CATEGORY_OBJECTIVES.get(category, CATEGORY_OBJECTIVES["Architecture"])
        difficulty_guideline = DIFFICULTY_RUBRIC.get(difficulty, DIFFICULTY_RUBRIC[3])

        evidence_payload = [
            {
                "evidence_id": ev.get("evidence_id"),
                "section": ev.get("section"),
                "project": ev.get("project"),
                "text": ev.get("text")
            }
            for ev in selected_evidence
        ]

        claims_payload = [
            {
                "claim_id": c.get("claim_id"),
                "project": c.get("project"),
                "text": c.get("text")
            }
            for c in context.get("claims", [])
        ]

        relationships_payload = [
            {
                "relationship_id": r.get("relationship_id"),
                "source": r.get("source"),
                "relation": r.get("relation") or r.get("rel"),
                "target": r.get("target")
            }
            for r in context.get("relationships", [])
            if r.get("relationship_id")
        ]

        target_focus_directive = ""
        if target_entity:
            target_focus_directive = (
                f"CRITICAL TARGET FOCUS MANDATE: Your question's core technical inquiry MUST be directly focused on '{target_entity}'. "
                f"Probe the design, architecture, execution mechanics, or trade-offs of '{target_entity}'. "
                f"Do not pivot the question to focus on a different secondary technology or generic concept.\n"
            )

        user_prompt = (
            f"=== TARGET PARAMETERS ===\n"
            f"Category: {category}\n"
            f"Category Directive: {category_guideline}\n"
            f"Difficulty Level: {difficulty}\n"
            f"Difficulty Directive: {difficulty_guideline}\n"
            f"Target Project Focus: {target_project or 'Infer from primary project evidence'}\n"
            f"Target Entity Focus: {target_entity or 'Infer from primary technical evidence'}\n"
            f"{target_focus_directive}\n"
            f"=== VERIFIED CANDIDATE EVIDENCE ===\n"
            f"{json.dumps(evidence_payload, indent=2)}\n\n"
            f"=== CANDIDATE CLAIMS ===\n"
            f"{json.dumps(claims_payload, indent=2)}\n\n"
            f"=== CANDIDATE GRAPH RELATIONSHIPS ===\n"
            f"{json.dumps(relationships_payload, indent=2)}\n\n"
        )

        if previously_asked_questions:
            pre_lines = [q.strip() for q in previously_asked_questions if q and q.strip()]
            if pre_lines:
                selected_pre = pre_lines if len(pre_lines) <= 15 else pre_lines[-15:]
                pre_str = "\n".join(f"- {q}" for q in selected_pre)
                user_prompt += (
                    f"=== PREVIOUSLY ASKED QUESTIONS (DO NOT DUPLICATE) ===\n"
                    f"The candidate was already asked the following questions across interviews. Your new question MUST probe a distinct technical mechanism, component, or trade-off from the verified evidence, and MUST NOT ask about or overlap with any of these:\n"
                    f"{pre_str}\n\n"
                )

        if past_session_hints:
            past_lines = [q.strip() for q in past_session_hints if q and q.strip()]
            if past_lines:
                past_str = "\n".join(f"- {q}" for q in past_lines)
                user_prompt += (
                    f"=== TOPICS EXPLORED IN PRIOR SESSIONS (SOFT HINT ONLY) ===\n"
                    f"The candidate was asked about these areas in past sessions. Prefer exploring fresh angles or different topics if available:\n"
                    f"{past_str}\n\n"
                )

        if feedback_instruction:
            user_prompt += f"CRITICAL RETRY CORRECTION: Your previous output failed validation:\n{feedback_instruction}\nRemediate this issue in your output.\n\n"

        user_prompt += (
            "=== QUESTION FORMULATION DIRECTIVE ===\n"
            "If the candidate evidence only mentions technologies/stack without explicit implementation details, you MUST start your question open-endedly: 'How would you approach designing...?' or 'What approach would you take to...?'\n"
            "Do NOT command the candidate to walk through unstated implementation flows, pipelines, or transaction mechanisms.\n"
            "ALWAYS cite one or more exact evidence ID(s) (e.g. 'chk_...', 'rel_...', 'clm_...') in 'evidence_ids'.\n\n"
            "=== JSON OUTPUT FORMAT ===\n"
            "Respond with this JSON structure only:\n"
            f"{{\n"
            f'  "question": "<The technical interview question>",\n'
            f'  "category": "{category}",\n'
            f'  "difficulty": {difficulty},\n'
            f'  "target_project": "<Exact project name from evidence or null>",\n'
            f'  "target_entity": "<Exact technology/entity from cited evidence>",\n'
            f'  "target_claim_id": "<Claim ID from provided claims or null>",\n'
            f'  "target_relationship_id": "<Relationship ID or null>",\n'
            f'  "evidence_ids": ["<One or more exact ID values from the supplied evidence blocks, relationships, or claims>"],\n'
            f'  "rationale": "<Technical rationale explaining what competency this question evaluates>"\n'
            f"}}"
        )

        llm_result = provider.generate_structured(
            prompt=user_prompt,
            system_instruction=system_instruction
        )

        data = llm_result.parsed_json
        if not isinstance(data, dict):
            raise ValueError("Candidate rejected: Model returned malformed or non-JSON output.")

        # Map available evidence by exact ID (chunks, relationships, and claims)
        evidence_by_id = {}
        for ev in evidence_payload:
            eid = ev.get("evidence_id")
            if eid:
                evidence_by_id[eid] = ev.get("text", "")
        for rel in relationships_payload:
            rid = rel.get("relationship_id")
            if rid and rid not in evidence_by_id:
                rel_type = rel.get("relation") or rel.get("rel")
                evidence_by_id[rid] = f"{rel.get('source')} {rel_type} {rel.get('target')}"
        for c in claims_payload:
            cid = c.get("claim_id")
            if cid and cid not in evidence_by_id:
                evidence_by_id[cid] = c.get("text", "")

        target_rel_id = data.get("target_relationship_id")
        target_claim_id = data.get("target_claim_id")

        # Robust extraction across provider format variations
        raw_candidates = []

        # 1. Primary field: evidence_ids
        primary_ids = data.get("evidence_ids")
        if isinstance(primary_ids, list):
            raw_candidates.extend(primary_ids)
        elif isinstance(primary_ids, str):
            raw_candidates.append(primary_ids)

        # Filter out empty/null values from primary list to know if it's truly empty
        clean_primary = [
            str(x).strip() for x in raw_candidates
            if x and str(x).strip() and str(x).strip().lower() not in ("null", "none", "n/a", "[]")
        ]

        if not clean_primary:
            raw_candidates = []
            # 2. Key variations: evidence_id, citations, citation, evidence, evidenceIds, evidenceId, sources, source_ids
            for alt_key in ("evidence_id", "citations", "citation", "evidence", "evidenceIds", "evidenceId", "sources", "source_ids"):
                alt_val = data.get(alt_key)
                if isinstance(alt_val, list):
                    raw_candidates.extend(alt_val)
                elif isinstance(alt_val, str):
                    raw_candidates.append(alt_val)
                if raw_candidates:
                    break
        else:
            raw_candidates = clean_primary

        # Sanitize and deduplicate while preserving order
        clean_cited_ids = []
        seen = set()
        for item in raw_candidates:
            if not item:
                continue
            item_str = str(item).strip()
            if not item_str or item_str.lower() in ("null", "none", "n/a", "[]"):
                continue
            if item_str not in seen:
                seen.add(item_str)
                clean_cited_ids.append(item_str)

        if not clean_cited_ids:
            raise ValueError("Candidate rejected: Model emitted no evidence citations.")

        invalid_evidence_ids = [eid for eid in clean_cited_ids if eid not in evidence_by_id]
        if invalid_evidence_ids:
            raise ValueError(f"Candidate rejected: Model cited unretrieved or invalid evidence ID(s): {invalid_evidence_ids}.")

        if target_rel_id:
            target_rel_str = str(target_rel_id).strip()
            if target_rel_str and target_rel_str.lower() not in ("null", "none"):
                valid_rel_ids = {r["relationship_id"] for r in relationships_payload if r.get("relationship_id")}
                for ev in selected_evidence:
                    if ev.get("source_type") == "relationship" and ev.get("evidence_id"):
                        valid_rel_ids.add(ev["evidence_id"])
                if target_rel_str not in valid_rel_ids:
                    raise ValueError(f"Candidate rejected: Model cited unretrieved or invalid relationship ID '{target_rel_str}'.")

        if target_claim_id:
            target_claim_str = str(target_claim_id).strip()
            if target_claim_str and target_claim_str.lower() not in ("null", "none"):
                valid_claim_ids = {c["claim_id"] for c in claims_payload if c.get("claim_id")}
                if target_claim_str not in valid_claim_ids:
                    raise ValueError(f"Candidate rejected: Model cited unretrieved or invalid claim ID '{target_claim_str}'.")

        # Resolve every cited evidence ID to its own actual evidence text
        resolved_texts = [evidence_by_id[eid] for eid in clean_cited_ids if eid in evidence_by_id]
        combined_evidence_text = " \n".join(resolved_texts)

        chosen_target_entity = data.get("target_entity")
        # Align target_entity with the question's primary technical inquiry if not explicitly pinned by caller
        if not target_entity and chosen_target_entity:
            q_text_str = data.get("question", "").strip()
            inquiry_match = re.search(r"(?:how\s+(?:would|did|do)\s+you|what\s+approach|designing|implementing|architecting)\s+(.*)", q_text_str, re.IGNORECASE)
            inquiry_part = inquiry_match.group(1).lower() if inquiry_match else q_text_str.lower()

            if chosen_target_entity.lower() not in inquiry_part:
                from app.core.canonicalizer import ALIAS_INDEX, extract_non_overlapping_entities
                inquiry_entities = [
                    canon for canon, _, _ in extract_non_overlapping_entities(inquiry_part, ALIAS_INDEX)
                    if canon.lower() in combined_evidence_text.lower()
                ]
                if inquiry_entities:
                    chosen_target_entity = inquiry_entities[0]

        try:
            return QuestionOutputSchema(
                question=data.get("question", "").strip(),
                category=category,
                difficulty=difficulty,
                target_project=data.get("target_project"),
                target_entity=chosen_target_entity,
                target_claim_id=target_claim_id if (target_claim_id and str(target_claim_id).strip().lower() not in ("null", "none")) else None,
                target_relationship=target_rel_id if (target_rel_id and str(target_rel_id).strip().lower() not in ("null", "none")) else None,
                evidence_ids=clean_cited_ids,
                evidence_text=combined_evidence_text,
                rationale=data.get("rationale", "Grounded candidate probe."),
                grounding_score=0.0,
                relevance_score=0.0,
                technical_quality_score=0.0,
                trace_id=trace_id,
                assessed_difficulty=difficulty,
                difficulty_rationale=f"Requested Level {difficulty} inquiry into {chosen_target_entity or 'core competency'}.",
                grounding_status="PASSED",
                rubric_version="v2.0-canonical-5level",
                validation_status="ACCEPTED"
            )
        except ValidationError as e:
            raise ValueError(f"LLM produced schema violation: {str(e)}")

question_generator = QuestionGenerationService()