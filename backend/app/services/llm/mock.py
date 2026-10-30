import time
import json
import re
from typing import Dict, Any
from app.services.llm.base import BaseLLMProvider, LLMGenerationResult

class MockProvider(BaseLLMProvider):
    """
    Test-only mock provider.
    Synthesizes dynamically grounded responses directly from input prompt blocks
    without hardcoded candidate names or templates.
    """
    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.2
    ) -> LLMGenerationResult:
        t0 = time.time()

        # Check if this is an Answer Evaluation prompt
        if "<untrusted_candidate_answer>" in prompt or "objective evaluator" in system_instruction.lower():
            # Extract candidate answer text
            ans_match = re.search(r"<untrusted_candidate_answer>\s*(.*?)\s*</untrusted_candidate_answer>", prompt, re.DOTALL)
            answer_text = ans_match.group(1).strip() if ans_match else ""

            # Extract category
            cat_match = re.search(r"Question Category:\s*([^\n\r]+)", prompt)
            category = cat_match.group(1).strip() if cat_match else "Architecture"

            # Dynamic score derivation based on answer content length & technical keywords
            words = len(answer_text.split())
            base_score = 75.0 if words >= 20 else (60.0 if words >= 10 else 40.0)

            # Check if prompt injection is present in the text
            from app.services.evaluation.injection_guard import scan_for_prompt_injection
            is_inj, _ = scan_for_prompt_injection(answer_text)
            if is_inj:
                # Injection attempt does not fool the evaluator into giving 100
                base_score = min(base_score, 50.0)

            parsed_eval: Dict[str, Any] = {
                "scores": {
                    "technical_accuracy": round(min(100.0, base_score + 5.0), 2),
                    "conceptual_understanding": round(min(100.0, base_score + 2.0), 2),
                    "completeness": round(base_score, 2),
                    "depth_and_specificity": round(max(0.0, base_score - 5.0), 2),
                    "practical_reasoning": round(base_score, 2),
                    "evidence_alignment": round(base_score, 2),
                },
                "feedback": {
                    "summary": "The candidate provided a structured technical response addressing key aspects of the question.",
                    "strengths": ["Addressed core architectural concepts", "Clear structured reasoning"],
                    "weaknesses": ["Could provide deeper operational and edge-case details"],
                    "missing_concepts": [],
                    "key_takeaways": "Elaborate further on trade-offs and runtime failure modes."
                }
            }

            return LLMGenerationResult(
                raw_content=json.dumps(parsed_eval),
                parsed_json=parsed_eval,
                provider="mock",
                model="mock-evaluator-v1",
                latency_ms=round((time.time() - t0) * 1000, 2),
                usage={"prompt_tokens": 150, "candidates_tokens": 80}
            )

        # 1. Extract Evidence Items from JSON payload
        ev_items = []
        ev_match = re.search(r"=== VERIFIED CANDIDATE EVIDENCE ===\s*(\[\s*[\s\S]*?\n\s*\])", prompt)
        if ev_match:
            try:
                ev_items = json.loads(ev_match.group(1))
            except Exception:
                ev_items = []

        clm_match = re.search(r"=== CANDIDATE CLAIMS ===\s*(\[\s*[\s\S]*?\n\s*\])", prompt)
        if clm_match:
            try:
                for c in json.loads(clm_match.group(1)):
                    ev_items.append({
                        "evidence_id": c.get("claim_id"),
                        "text": c.get("text", ""),
                        "project": c.get("project")
                    })
            except Exception:
                pass

        evidence_ids = [item["evidence_id"] for item in ev_items if item.get("evidence_id")]
        if not evidence_ids:
            evidence_ids = re.findall(r'"evidence_id":\s*"([^"]+)"', prompt)
        if not evidence_ids:
            evidence_ids = re.findall(r'"relationship_id":\s*"([^"]+)"', prompt)
        if not evidence_ids:
            evidence_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        if not evidence_ids:
            evidence_ids = re.findall(r"\[Evidence ID: ([\w\-]+)\]", prompt)
        primary_ev_id = evidence_ids[0] if evidence_ids else "ev_fallback"

        # 2. Extract Project (Target Project Focus header or JSON evidence block)
        project = None
        proj_focus = re.search(r"Target Project Focus:\s*([^\n\r]+)", prompt)
        if proj_focus:
            val = proj_focus.group(1).strip()
            if not any(k in val.lower() for k in ["infer", "none", "null", "your project"]):
                project = val

        if not project:
            proj_json = re.search(r'"project":\s*"([^"]+)"', prompt)
            if proj_json:
                val = proj_json.group(1).strip()
                if not any(k in val.lower() for k in ["none", "null", "your project"]):
                    project = val

        # 3. Extract Target Entity
        entity = None
        entity_focus = re.search(r"Target Entity Focus:\s*([^\n\r]+)", prompt)
        if entity_focus:
            val = entity_focus.group(1).strip()
            if not any(k in val.lower() for k in ["infer", "none", "null", "any"]):
                entity = val

        if not entity:
            text_match = re.search(r'"text":\s*"([^"]+)"', prompt)
            evidence_text = text_match.group(1) if text_match else prompt
            tech_match = re.search(r'\b([A-Z][a-zA-Z0-9_\-\.]{1,})\b', evidence_text)
            entity = tech_match.group(1) if tech_match else "Core Architecture"

        # Align cited evidence ID with target project and entity if possible
        if ev_items:
            found = False
            for item in ev_items:
                text = (item.get("text") or "").lower()
                proj_name = (item.get("project") or "").lower()
                matches_proj = (project.lower() in text or project.lower() in proj_name) if project else True
                matches_ent = (entity.lower() in text) if entity else True
                if matches_proj and matches_ent:
                    primary_ev_id = item.get("evidence_id")
                    found = True
                    break
            if not found and project:
                for item in ev_items:
                    text = (item.get("text") or "").lower()
                    proj_name = (item.get("project") or "").lower()
                    if project.lower() in text or project.lower() in proj_name:
                        primary_ev_id = item.get("evidence_id")
                        found = True
                        break
            if not found and entity:
                for item in ev_items:
                    text = (item.get("text") or "").lower()
                    if entity.lower() in text:
                        primary_ev_id = item.get("evidence_id")
                        break

        # 4. Extract Category and Difficulty
        cat_match = re.search(r"Category:\s*([^\n\r]+)", prompt)
        category = cat_match.group(1).strip() if cat_match else "Architecture"

        diff_match = re.search(r"Difficulty Level:\s*(\d)", prompt)
        difficulty = int(diff_match.group(1)) if diff_match else 3

        # 5. Extract Claim ID
        claims = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        claim_id = claims[0] if (claims and category == "Claim-Verification") else None

        # Build dynamic question text with difficulty-specific markers and variation to avoid duplicates
        context_prefix = f"In {project}, " if project else "Based on your technical work, "
        prev_section = re.search(r"=== PREVIOUSLY ASKED QUESTIONS \(DO NOT DUPLICATE\) ===\s*(.*?)(?=\n===|\Z)", prompt, re.DOTALL)
        if prev_section:
            prev_questions = [line.strip("- ").strip() for line in prev_section.group(1).splitlines() if line.strip().startswith("-")]
        else:
            prev_questions = []
        prev_count = len(prev_questions)
        retry_count = prompt.count("CRITICAL RETRY CORRECTION")
        retry_offset = retry_count * 7
        mechanisms = [
            "routing decorators and endpoint handlers",
            "Pydantic data validation and request schemas",
            "dependency injection yield lifetimes",
            "asynchronous background tasks dispatch",
            "middleware request-response pipeline hooks",
            "custom exception handlers and HTTP status codes",
            "connection pooling and database session lifespans",
            "authentication security scopes and token parsing",
            "caching layer invalidation and memory buffers",
            "serialization performance and json response streaming",
            "database rollback protocols and transaction logs",
            "event emission and distributed message queues",
            "rate limiting algorithms and request throttling",
            "health check probes and graceful shutdown signals",
            "thread safety primitives and mutex locks",
            "websocket handshake state and bidirectional streaming",
            "database migration versioning and schema alteration locks",
            "circuit breaker trip thresholds and failure recovery",
            "distributed tracing headers and span propagation",
            "composite B-tree indexing and query execution plans",
            "dead letter queue processing and poison message handling",
            "asynchronous event loops and task starvation",
            "read replica lag and replica routing policies",
            "idempotency key stores and replay attack prevention",
            "garbage collection pause times and memory profiling",
            "optimistic concurrency control and version column checks",
            "JWT expiration validation and token revocation blacklists",
            "bulkhead resource isolation and worker thread pools",
            "cursor-based pagination and keyset pagination queries",
            "distributed lock leases and heartbeat renewal mechanisms",
            "gRPC protocol buffer serialization and HTTP2 framing",
            "connection draining and zero-downtime rolling deploys",
            "two-phase commit coordination and distributed transaction managers",
            "horizontal database sharding and shard key distribution",
            "content delivery network cache headers and cache stampede protection",
            "asynchronous file upload streaming and chunked multipart handling",
            "structured logging formatting and correlated request identifiers",
            "SLA breach alerts and real-time latency metric aggregation",
            "database query timeouts and statement cancellation hooks",
            "cross-origin resource sharing preflight validation and security headers",
            "write-ahead log archival and point-in-time recovery",
            "exponential backoff retry policies with randomized jitter",
            "message deduplication windows and exactly-once delivery semantics",
            "connection keep-alive timeouts and socket reuse policies",
            "dynamic configuration reloading without worker process restarts",
            "in-memory cache eviction policies like LRU and LFU",
            "distributed rate limiter synchronization across cluster nodes",
            "database lock contention and deadlock detection graphs",
            "API versioning deprecation lifecycles and backward compatibility gates",
            "SSL/TLS termination handshakes and cipher suite negotiation"
        ]
        ctx_hash = sum(ord(c) * (i + 1) for i, c in enumerate(f"{project}:{entity}:{category}"))
        m_idx = (prev_count * 13 + retry_offset * 7 + ctx_hash) % len(mechanisms)
        m = mechanisms[m_idx]

        def get_templates(cand_m: str) -> list[str]:
            if difficulty == 1:
                return [
                    f"{context_prefix}what is the basic syntax of {cand_m} in {entity}?",
                    f"{context_prefix}define the elementary purpose of {cand_m} in {entity}.",
                    f"{context_prefix}which keyword specifies {cand_m} in {entity}?",
                    f"{context_prefix}what are the core rules governing {cand_m} in {entity}?",
                    f"{context_prefix}name the basic concept behind {cand_m} in {entity}.",
                    f"{context_prefix}list the elementary prerequisites for {cand_m} in {entity}.",
                    f"{context_prefix}what is the standard definition of {cand_m} when utilizing {entity}?",
                    f"{context_prefix}which keyword controls the activation of {cand_m} in {entity}?",
                    f"{context_prefix}define the distinction between baseline execution and {cand_m} in {entity}.",
                    f"{context_prefix}what is the elementary role of {cand_m} within {entity}?",
                    f"{context_prefix}what are the fundamental requirements to configure {cand_m} in {entity}?",
                    f"{context_prefix}name the elementary return types comprising {cand_m} in {entity}.",
                    f"{context_prefix}which keyword initiates {cand_m} during runtime in {entity}?",
                    f"{context_prefix}define the elementary boundaries of {cand_m} in {entity}.",
                    f"{context_prefix}what is the primary purpose of registering {cand_m} in {entity}."
                ]
            elif difficulty == 2:
                return [
                    f"{context_prefix}describe how {entity} integrates {cand_m} into practical application.",
                    f"{context_prefix}explain how {cand_m} is structured within the data flow of {entity}.",
                    f"{context_prefix}what role did {cand_m} play in the typical operational usage of {entity}?",
                    f"{context_prefix}describe how developers structure {cand_m} when utilizing {entity}.",
                    f"{context_prefix}explain the practical purpose of configuring {cand_m} in {entity}.",
                    f"{context_prefix}describe how {cand_m} affects the data flow and execution of {entity}.",
                    f"{context_prefix}what role does {cand_m} serve during typical operations in {entity}?",
                    f"{context_prefix}explain how {entity} coordinates with {cand_m} to maintain structured processing.",
                    f"{context_prefix}describe the steps to integrate {cand_m} into {entity}.",
                    f"{context_prefix}explain how {cand_m} coordinates with the runtime of {entity}.",
                    f"{context_prefix}what role did {cand_m} fulfill in the practical deployment of {entity}?",
                    f"{context_prefix}describe how to configure {cand_m} for typical operational usage in {entity}.",
                    f"{context_prefix}explain how {cand_m} is integrated across components in {entity}.",
                    f"{context_prefix}describe how {entity} facilitates practical usage of {cand_m}.",
                    f"{context_prefix}explain how {cand_m} participates in the execution flow of {entity}."
                ]
            elif difficulty == 3:
                return [
                    f"{context_prefix}how would you approach designing the transaction boundary for {cand_m} in {entity}?",
                    f"{context_prefix}how did you handle state transition and throughput in {cand_m} using {entity}?",
                    f"{context_prefix}how would you design error propagation mechanics for {cand_m} in {entity}?",
                    f"{context_prefix}how did you optimize lifecycle and throughput for {cand_m} in {entity}?",
                    f"{context_prefix}how would you approach designing cache invalidation for {cand_m} in {entity}?",
                    f"{context_prefix}how would you handle transaction boundary conditions during {cand_m} in {entity}?",
                    f"{context_prefix}how did you structure state transition handling for {cand_m} in {entity}?",
                    f"{context_prefix}how would you design non-blocking concurrency for {cand_m} in {entity}?",
                    f"{context_prefix}how did you implement query plan and throughput tuning for {cand_m} in {entity}?",
                    f"{context_prefix}how would you approach designing lifecycle state management for {cand_m} in {entity}?",
                    f"{context_prefix}how did you handle transaction boundary failures in {cand_m} using {entity}?",
                    f"{context_prefix}how would you design error propagation and retry limits for {cand_m} in {entity}?",
                    f"{context_prefix}how did you manage throughput optimization and buffer thresholds for {cand_m} in {entity}?",
                    f"{context_prefix}how would you approach designing cache invalidation intervals for {cand_m} in {entity}?",
                    f"{context_prefix}how did you resolve state transition conflicts for {cand_m} in {entity}?"
                ]
            elif difficulty == 4:
                return [
                    f"{context_prefix}what trade-off did you evaluate regarding horizontal auto-scaling limits for {cand_m} in {entity}?",
                    f"{context_prefix}how did you decouple architectural boundary constraints when configuring {cand_m} in {entity}?",
                    f"{context_prefix}what failover recovery bottlenecks did you encounter while scaling {cand_m} in {entity}?",
                    f"{context_prefix}what architectural trade-off guided your reactive concurrency model for {cand_m} in {entity}?",
                    f"{context_prefix}how did you resolve scaling constraint bottlenecks for {cand_m} in {entity}?",
                    f"{context_prefix}what trade-off did you make between interface contract simplicity and horizontal scale for {cand_m} in {entity}?",
                    f"{context_prefix}how did you decouple component boundaries during failover events in {cand_m} with {entity}?",
                    f"{context_prefix}what scaling constraint influenced your horizontal scale strategy for {cand_m} in {entity}?",
                    f"{context_prefix}how did you resolve concurrency model bottlenecks during high load in {cand_m} with {entity}?",
                    f"{context_prefix}what trade-off did you accept between architectural boundary isolation and latency for {cand_m} in {entity}?",
                    f"{context_prefix}how did you handle failover synchronization across horizontal scale instances of {cand_m} in {entity}?",
                    f"{context_prefix}what scaling constraint arose when tuning interface contracts for {cand_m} in {entity}?",
                    f"{context_prefix}how did you decouple bottleneck dependencies across component boundaries for {cand_m} in {entity}?",
                    f"{context_prefix}what trade-off did you evaluate between backpressure and horizontal scale for {cand_m} in {entity}?",
                    f"{context_prefix}how did you design failover routing to mitigate component boundaries bottlenecks in {cand_m} using {entity}?"
                ]
            else:  # difficulty == 5
                return [
                    f"{context_prefix}how did you mitigate cascading failure mode under extreme load for {cand_m} in {entity}?",
                    f"{context_prefix}what approach resolved split-brain state inconsistencies in {cand_m} using {entity}?",
                    f"{context_prefix}what consensus protocol and linearizability guarantees protected {cand_m} under write contention in {entity}?",
                    f"{context_prefix}how did you diagnose and prevent byzantine deadlock detection stalls in {cand_m} using {entity}?",
                    f"{context_prefix}how did you ensure distributed consistency during network partition degradation for {cand_m} in {entity}?",
                    f"{context_prefix}how did you eliminate multi-region race condition hazards during adverse network degradation for {cand_m} in {entity}?",
                    f"{context_prefix}what threat vector mitigations protected {cand_m} against adversarial input attacks in {entity}?",
                    f"{context_prefix}how did you design leader election failover without split-brain corruption in {cand_m} with {entity}?",
                    f"{context_prefix}what consensus protocol invariants prevented data loss under extreme load for {cand_m} in {entity}?",
                    f"{context_prefix}how did you maintain strict linearizability when resolving distributed consistency conflicts in {cand_m} with {entity}?",
                    f"{context_prefix}what architectural boundaries contained blast radius during cascading node outages in {cand_m} with {entity}?",
                    f"{context_prefix}how did you resolve distributed deadlock detection cycles under extreme load in {cand_m} using {entity}?",
                    f"{context_prefix}how did you ensure partition tolerance while mitigating adversarial degradation in {cand_m} with {entity}?",
                    f"{context_prefix}what threat vector analysis informed your distributed consistency controls for {cand_m} in {entity}?",
                    f"{context_prefix}how did you prevent silent data corruption during adverse partition degradation in {cand_m} with {entity}?"
                ]

        selected_text = None
        for m_shift in range(len(mechanisms)):
            cand_m = mechanisms[(m_idx + m_shift * 7) % len(mechanisms)]
            cand_templates = get_templates(cand_m)
            t_base = (prev_count * 11 + retry_count * 3 + ctx_hash) % len(cand_templates)
            for shift in range(len(cand_templates)):
                cand_t = cand_templates[(t_base + shift) % len(cand_templates)]
                cand_words = set(re.findall(r"\b\w{3,}\b", cand_t.lower()))
                is_lex_dup = False
                for pq in prev_questions:
                    pq_words = set(re.findall(r"\b\w{3,}\b", pq.lower()))
                    if cand_words and pq_words:
                        overlap = (2.0 * len(cand_words.intersection(pq_words))) / (len(cand_words) + len(pq_words))
                        if overlap >= 0.45:
                            is_lex_dup = True
                            break
                if not is_lex_dup:
                    selected_text = cand_t
                    break
            if selected_text:
                break

        question_text = selected_text or get_templates(mechanisms[m_idx])[0]

        parsed: Dict[str, Any] = {
            "question": question_text,
            "category": category,
            "difficulty": difficulty,
            "target_project": project,
            "target_entity": entity,
            "target_claim_id": claim_id,
            "target_relationship_id": None,
            "evidence_ids": [primary_ev_id],
            "rationale": f"Directly verifies candidate application of {entity}.",
            "assessed_difficulty": difficulty,
            "difficulty_rationale": f"Mock Level {difficulty} assessment of {entity}.",
            "grounding_status": "PASSED",
            "rubric_version": "v2.0-canonical-5level",
            "validation_status": "ACCEPTED"
        }

        return LLMGenerationResult(
            raw_content=json.dumps(parsed),
            parsed_json=parsed,
            provider="mock",
            model="mock-grounded-test",
            latency_ms=round((time.time() - t0) * 1000, 2),
            usage={"prompt_tokens": 120, "candidates_tokens": 40}
        )