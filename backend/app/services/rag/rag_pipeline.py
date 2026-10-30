import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from sqlalchemy.orm import Session
from fastapi import HTTPException

logger = logging.getLogger(__name__)

from app.models.resume import Resume
from app.models.relationship import Relationship
from app.models.interview import InterviewQuestion, InterviewSession
from app.models.rag import ResumeChunk, ResumeClaim, RetrievalTrace
from app.services.evaluation.scorecard import scorecard_aggregator
from app.schemas.rag import QuestionGenerationRequest, QuestionGenerationResponse, QuestionOutputSchema
from app.services.rag.embedding_service import embedding_service
from app.services.rag.chunking_service import chunking_service
from app.services.rag.claim_service import claim_service
from app.services.rag.retrievers import (
    VectorRetriever, GraphRetriever, BM25LexicalRetriever, HybridRetriever,
    resume_retrieval_cache, get_or_build_resume_cache
)
from app.services.rag.reranker import reranker
from app.services.rag.context_assembler import context_assembler
from app.services.rag.generator import question_generator
from app.services.rag.validator import grounding_validator

from app.core.config import settings
MAX_ATTEMPTS = getattr(settings, "MAX_INTERNAL_RAG_ATTEMPTS", 2)

from app.services.privacy_guard import privacy_guard

class RAGPipeline:
    def index_resume(self, resume_id, db: Session) -> Dict[str, Any]:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume record not found.")

        # 1. Defensive re-sanitization: ensure stored raw_text is completely sanitized
        sanitized_text, meta = privacy_guard.sanitize_text(resume.raw_text)
        if meta["is_sanitized"] and sanitized_text != resume.raw_text:
            resume.raw_text = sanitized_text
            db.flush()

        # 2. Extract and sanitize chunks
        chunks = chunking_service.chunk_resume(str(resume_id), resume.raw_text)
        sanitized_chunk_texts = []
        for c in chunks:
            clean_t, _ = privacy_guard.sanitize_text(c.text)
            sanitized_chunk_texts.append(clean_t)

        embeddings = embedding_service.embed_texts(sanitized_chunk_texts)

        db.query(ResumeChunk).filter(ResumeChunk.resume_id == resume_id).delete()
        for c, clean_t, emb in zip(chunks, sanitized_chunk_texts, embeddings):
            db.add(ResumeChunk(
                resume_id=resume_id,
                chunk_id=c.chunk_id,
                section=c.section,
                project=c.project,
                chunk_text=clean_t,
                embedding=emb,
                embedding_model=embedding_service.model_name,
                char_start=c.char_start,
                char_end=c.char_end
            ))

        # 3. Synchronized relationship & claim extraction:
        # Extract verified relationships if not already present so claims can be built
        if db.query(Relationship).filter(Relationship.resume_id == resume_id).count() == 0:
            try:
                from app.services.relationship_service import extract_and_store_relationships_advanced
                extract_and_store_relationships_advanced(resume_id, db)
            except Exception as rel_err:
                logger.warning("Relationship extraction warning during resume indexing: %s", rel_err)

        claims = claim_service.extract_and_store_claims(resume_id, db)

        # 4. Invalidate all dependent caches
        grounding_validator.clear_cache_for_resume(resume_id)
        resume_retrieval_cache.invalidate(resume_id)
        db.commit()

        return {
            "resume_id": resume_id,
            "total_chunks": len(chunks),
            "total_claims": len(claims),
            "embedding_model": embedding_service.model_name,
            "status": "INDEXED"
        }

    def generate_grounded_question(
        self,
        resume_id,
        req: QuestionGenerationRequest,
        db: Session,
        deadline_seconds: Optional[float] = None,
        t_global_start: Optional[float] = None,
        persist: bool = True
    ) -> QuestionGenerationResponse:
        if t_global_start is None:
            t_global_start = time.perf_counter()
        if deadline_seconds is None:
            deadline_seconds = getattr(settings, "TOTAL_QUESTION_DEADLINE_SECONDS", 12.0)
        min_llm_budget = getattr(settings, "MIN_LLM_CALL_BUDGET", 2.5)

        trace_id = f"trc_{uuid.uuid4().hex[:12]}"
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume record not found.")

        session_id = getattr(req, "session_id", None)
        if session_id is not None:
            session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Interview session not found.")
            if session.resume_id != resume_id:
                raise HTTPException(
                    status_code=400,
                    detail="Question does not belong to the candidate resume linked to this session."
                )

        # Preload request-scoped entities and metadata using cached resume data (Part A3)
        t_pre_start = time.perf_counter()
        if db.query(ResumeChunk).filter(ResumeChunk.resume_id == resume_id).count() == 0:
            self.index_resume(resume_id, db)

        cached_resume = get_or_build_resume_cache(resume_id, db)
        all_resume_questions = db.query(InterviewQuestion).filter(InterviewQuestion.resume_id == resume_id).all()
        if session_id is not None:
            # B3: Build existing_questions and prev_asked_texts from CURRENT SESSION only for hard duplicate rejection
            existing_questions = [q for q in all_resume_questions if q.session_id == session_id]
            past_session_questions = [q for q in all_resume_questions if q.session_id != session_id]
        else:
            existing_questions = all_resume_questions
            past_session_questions = []

        existing_chunks = cached_resume.chunks
        allowed_projects = cached_resume.allowed_projects
        existing_relationships = cached_resume.relationships
        allowed_entities = cached_resume.allowed_entities
        claims_map = cached_resume.claims_map
        rels_map = cached_resume.rels_map
        t_db_preload = (time.perf_counter() - t_pre_start) * 1000

        category = req.category or "Architecture"
        difficulty = req.difficulty if 1 <= req.difficulty <= 5 else 3

        attempts_log = []
        accepted_question: Optional[QuestionOutputSchema] = None
        final_context: Dict[str, Any] = {}
        resolved_evidence_items: Dict[str, Dict[str, Any]] = {}
        last_failure_reason = "No candidate evidence available."
        feedback_instruction: Optional[str] = None

        current_project = req.target_project
        current_entity = req.target_entity

        # Track previously asked questions to inject proactive anti-duplication constraints (Session-scoped)
        prev_asked_texts = [q.question_text for q in existing_questions if q.question_text]
        # Include past session questions only as soft hints (last ~5), never as rejection criteria (B3)
        past_session_hints = [q.question_text for q in past_session_questions if q.question_text][-5:]

        # Validate parent-question reference if provided
        parent_q = None
        if getattr(req, "parent_question_id", None):
            parent_q = db.query(InterviewQuestion).filter(InterviewQuestion.id == req.parent_question_id).first()
            if not parent_q:
                raise HTTPException(status_code=400, detail=f"Parent question '{req.parent_question_id}' not found.")
            if parent_q.resume_id != resume_id:
                raise HTTPException(status_code=400, detail="Parent question does not belong to the candidate resume.")
            if session_id is not None and parent_q.session_id is not None and parent_q.session_id != session_id:
                raise HTTPException(status_code=400, detail="Parent question belongs to a different interview session.")
            if parent_q.question_text and parent_q.question_text not in prev_asked_texts:
                prev_asked_texts.append(parent_q.question_text)

        # Validate target claim if explicitly requested
        if getattr(req, "target_claim_id", None):
            if req.target_claim_id not in claims_map:
                return QuestionGenerationResponse(
                    resume_id=resume_id,
                    request_id=request_id,
                    trace_id=trace_id,
                    question=None,
                    validation_status="REJECTED",
                    error_code="invalid_claim_reference",
                    rubric_version="v2.0-canonical-5level",
                    message=f"Target claim '{req.target_claim_id}' is not supported by candidate resume.",
                    attempts_count=0,
                    latencies_ms={"total_ms": 0.0}
                )
            target_claim_obj = claims_map[req.target_claim_id]
            if not current_project and target_claim_obj.project:
                current_project = target_claim_obj.project
            if not current_entity and target_claim_obj.target_entity:
                current_entity = target_claim_obj.target_entity

        # Validate target relationship if explicitly requested
        if getattr(req, "target_relationship", None):
            rel_str = str(req.target_relationship).strip()
            valid_rel = (rel_str in rels_map) or any(
                rel_str == f"{r.source_entity} {r.relation_type} {r.target_entity}"
                for r in existing_relationships
            )
            if not valid_rel:
                return QuestionGenerationResponse(
                    resume_id=resume_id,
                    request_id=request_id,
                    trace_id=trace_id,
                    question=None,
                    validation_status="REJECTED",
                    error_code="invalid_relationship_reference",
                    rubric_version="v2.0-canonical-5level",
                    message=f"Target relationship '{rel_str}' is not supported by candidate resume.",
                    attempts_count=0,
                    latencies_ms={"total_ms": 0.0}
                )

        # Validate target project if explicitly requested
        all_candidate_projects = (
            set(p.lower() for p in allowed_projects if p)
            | set(str(c.project).lower() for c in claims_map.values() if getattr(c, "project", None))
            | set(str(r.context_entity).lower() for r in existing_relationships if getattr(r, "context_entity", None))
        )
        if current_project and all_candidate_projects and current_project.lower() not in all_candidate_projects:
            return QuestionGenerationResponse(
                resume_id=resume_id,
                request_id=request_id,
                trace_id=trace_id,
                question=None,
                validation_status="REJECTED",
                error_code="invalid_project_reference",
                rubric_version="v2.0-canonical-5level",
                message=f"Target project '{current_project}' is not supported by candidate resume.",
                attempts_count=0,
                latencies_ms={"total_ms": 0.0}
            )

        if getattr(req, "follow_up_reason", None) and parent_q:
            pq_context = f" Target parent question: \"{parent_q.question_text}\"." if parent_q.question_text else ""
            followup_instruction = (
                f"TARGETED FOLLOW-UP OBJECTIVE:{pq_context}\n"
                f"Objective: {req.follow_up_reason}\n"
                f"CRITICAL RULES: Do NOT repeat the previous question verbatim or probe identical mechanics. Focus specifically on this targeted follow-up objective."
            )
            feedback_instruction = (
                f"{feedback_instruction}\n\n{followup_instruction}" if feedback_instruction else followup_instruction
            )

        # Request-scoped retrieval cache to avoid re-querying and re-embedding identical scopes
        retrieval_cache: Dict[Tuple[Optional[str], Optional[str]], Tuple[List[Any], Dict[str, Any], Dict[str, Dict[str, Any]]]] = {}
        attempted_projects: Set[str] = set()
        if current_project:
            attempted_projects.add(current_project)

        total_retrieval_ms = 0.0
        total_rerank_ms = 0.0
        total_context_assembly_ms = 0.0
        total_llm_ms = 0.0
        total_val_ms = 0.0
        total_dup_ms = 0.0
        llm_call_latencies: List[float] = []
        last_llm_duration_s = 0.0
        req_excluded = list(getattr(req, "excluded_question_texts", None) or [])

        for attempt in range(1, MAX_ATTEMPTS + 1):
            elapsed_s = time.perf_counter() - t_global_start
            remaining_s = deadline_seconds - elapsed_s
            # Enforce bounded per-target latency: abort if remaining budget < one LLM call
            if attempt > 1:
                estimated_needed = max(min_llm_budget, last_llm_duration_s)
                if remaining_s < estimated_needed:
                    logger.warning(
                        f"Aborting RAG attempt {attempt}: remaining budget {remaining_s:.2f}s is less than needed budget {estimated_needed:.2f}s."
                    )
                    break

            stage_timings: Dict[str, float] = {}

            # Reuse cached retrieval context if scope (project + entity) did not change
            cache_key = (current_project, current_entity)
            if cache_key in retrieval_cache:
                reranked, final_context, resolved_evidence_items = retrieval_cache[cache_key]
                t_retrieval = 0.0
                t_ctx = 0.0
            else:
                t_ret_start = time.perf_counter()

                # Align project filter with current_entity evidence:
                active_project_filter = current_project
                if current_entity and current_project:
                    p_low = current_project.lower()
                    e_low = current_entity.lower()
                    proj_has_entity = any(
                        chk.project and p_low in chk.project.lower() and e_low in chk.chunk_text.lower()
                        for chk in cached_resume.chunks
                    )
                    if not proj_has_entity:
                        proj_has_entity = any(
                            rel.context_entity and p_low in rel.context_entity.lower()
                            and (e_low in (rel.target_entity or "").lower() or e_low in (rel.source_entity or "").lower())
                            for rel in cached_resume.relationships
                        )
                    if not proj_has_entity:
                        active_project_filter = None

                query_text = f"{active_project_filter or ''} {current_entity or ''} {category}".strip()
                query_terms = query_text.split()
                if current_entity and current_entity.lower() not in [t.lower() for t in query_terms]:
                    query_terms.append(current_entity)

                q_emb = embedding_service.embed_query(query_text)
                vec_res = VectorRetriever.retrieve(resume_id, q_emb, db, top_k=5, project_filter=active_project_filter)
                graph_res = GraphRetriever.retrieve(resume_id, current_entity, active_project_filter, db=db, top_k=8)
                lex_res = BM25LexicalRetriever.retrieve(resume_id, query_terms, db=db, top_k=5, project_filter=active_project_filter)

                fused = HybridRetriever.fuse_and_rank(vec_res, graph_res, lex_res, top_k_final=6, target_entity=current_entity)

                # Dynamic evidence alignment & recovery for target_entity:
                if current_entity:
                    e_low = current_entity.lower()
                    has_target_in_fused = any(e_low in it.text.lower() for it in fused)
                    if not has_target_in_fused:
                        from app.core.canonicalizer import ALIAS_INDEX, CANONICAL_TAXONOMY
                        from app.schemas.rag import EvidenceItem
                        canon_entity = ALIAS_INDEX.get(e_low, current_entity)
                        entity_aliases = [a.lower() for a in CANONICAL_TAXONOMY.get(canon_entity, {}).get("aliases", [])] + [canon_entity.lower(), e_low]

                        for chk in cached_resume.chunks:
                            chk_lower = chk.chunk_text.lower()
                            if any(a in chk_lower for a in entity_aliases):
                                fused.append(EvidenceItem(
                                    evidence_id=chk.chunk_id,
                                    text=chk.chunk_text,
                                    section=chk.section or "Technical",
                                    project=chk.project,
                                    source_type="chunk",
                                    score=0.88,
                                    provenance="entity_alignment_recovery"
                                ))
                                break
                        else:
                            for rel in cached_resume.relationships:
                                rel_tgt = (rel.target_entity or "").lower()
                                rel_src = (rel.source_entity or "").lower()
                                rel_ev = (rel.evidence_text or "").lower()
                                if any(a in rel_tgt or a in rel_src or a in rel_ev for a in entity_aliases):
                                    fused.append(EvidenceItem(
                                        evidence_id=f"rel_{str(rel.id)[:8]}",
                                        text=f"{rel.source_entity} {rel.relation_type} {rel.target_entity} (Evidence: {rel.evidence_text})",
                                        section=rel.section or "Technical",
                                        project=rel.context_entity,
                                        source_type="relationship",
                                        score=0.90,
                                        provenance="entity_alignment_recovery"
                                    ))
                                    break

                        # If current_entity genuinely has zero evidence anywhere on resume, rotate to supported candidate
                        if not any(e_low in it.text.lower() for it in fused):
                            alt_candidates = [
                                e for e in cached_resume.allowed_entities
                                if e and e.lower() != e_low and e not in prev_asked_texts
                            ]
                            if alt_candidates:
                                logger.info(
                                    "Target entity '%s' has no evidence on resume; rotating to supported entity '%s'",
                                    current_entity, alt_candidates[0]
                                )
                                current_entity = alt_candidates[0]
                                continue

                t_retrieval_raw = (time.perf_counter() - t_ret_start) * 1000
                t_rr_s = time.perf_counter()
                reranked = reranker.rerank(query_text, fused, target_entity=current_entity, target_project=active_project_filter)
                t_rerank = (time.perf_counter() - t_rr_s) * 1000
                total_rerank_ms += t_rerank
                t_retrieval = t_retrieval_raw

                # Permanent failure fast-exit: If no evidence exists under general query, further retries cannot succeed
                if not reranked:
                    if current_project is not None:
                        # Fallback once to untargeted project retrieval before declaring failure
                        current_project = None
                        current_entity = None
                        continue
                    else:
                        attempts_log.append({
                            "attempt": attempt,
                            "status": "REJECTED",
                            "reason": "Evidence failed structural relevance filter. No suitable candidate evidence found.",
                            "stage_latencies_ms": {"retrieval_ms": round(t_retrieval, 2)}
                        })
                        last_failure_reason = "No candidate evidence available on candidate resume."
                        break

                t_ctx_start = time.perf_counter()
                claim_dicts = [{"claim_id": c.claim_id, "project": c.project, "target_entity": c.target_entity, "text": c.claim_text} for c in claims_map.values()]
                rel_dicts = [
                    {
                        "relationship_id": rel_id,
                        "source": r.source_entity,
                        "rel": r.relation_type,
                        "target": r.target_entity
                    }
                    for rel_id, r in rels_map.items()
                ]

                final_context = context_assembler.assemble(str(resume_id), query_text, reranked, claim_dicts, rel_dicts)
                resolved_evidence_items = {e["evidence_id"]: e for e in final_context.get("selected_evidence_items", [])}
                for rel in final_context.get("relationships", []):
                    rid = rel.get("relationship_id")
                    if rid and rid not in resolved_evidence_items:
                        resolved_evidence_items[rid] = {
                            "evidence_id": rid,
                            "text": f"{rel.get('source')} {rel.get('rel')} {rel.get('target')}",
                            "project": rel.get("source") if rel.get("source") in allowed_projects else (rel.get("target") if rel.get("target") in allowed_projects else None),
                            "source_type": "relationship"
                        }
                for clm in final_context.get("claims", []):
                    cid = clm.get("claim_id")
                    if cid and cid not in resolved_evidence_items:
                        resolved_evidence_items[cid] = {
                            "evidence_id": cid,
                            "text": clm.get("text", ""),
                            "project": clm.get("project"),
                            "source_type": "claim"
                        }
                t_ctx = (time.perf_counter() - t_ctx_start) * 1000

                # Cache this scope's assembled evidence
                retrieval_cache[cache_key] = (reranked, final_context, resolved_evidence_items)

            total_retrieval_ms += t_retrieval
            total_context_assembly_ms += t_ctx
            stage_timings["retrieval_ms"] = round(t_retrieval, 2)
            stage_timings["context_assembly_ms"] = round(t_ctx, 2)

            t_llm_start = time.perf_counter()
            try:
                candidate_q = question_generator.generate(
                    final_context,
                    category=category,
                    difficulty=difficulty,
                    target_project=current_project,
                    target_entity=current_entity,
                    trace_id=trace_id,
                    feedback_instruction=feedback_instruction,
                    previously_asked_questions=prev_asked_texts + req_excluded + [a["question"] for a in attempts_log if a.get("question") and a.get("scores", {}).get("code") == "semantic_duplicate"],
                    past_session_hints=past_session_hints
                )
                t_llm = (time.perf_counter() - t_llm_start) * 1000
                total_llm_ms += t_llm
                llm_call_latencies.append(t_llm)
                last_llm_duration_s = t_llm / 1000.0
                stage_timings["llm_ms"] = round(t_llm, 2)
            except ValueError as ve:
                t_llm = (time.perf_counter() - t_llm_start) * 1000
                total_llm_ms += t_llm
                llm_call_latencies.append(t_llm)
                last_llm_duration_s = t_llm / 1000.0
                reason = str(ve)
                stage_timings["llm_ms"] = round(t_llm, 2)
                attempts_log.append({
                    "attempt": attempt,
                    "status": "REJECTED",
                    "reason": reason,
                    "stage_latencies_ms": stage_timings
                })
                last_failure_reason = reason
                if "no evidence citations" in reason.lower():
                    feedback_instruction = "Your previous output cited no evidence. You MUST include one or more exact IDs from the verified evidence, relationships, or claims in the 'evidence_ids' field."
                else:
                    feedback_instruction = f"Generation rejected: {reason}. You must cite only exact valid IDs provided in the context."
                continue
            except Exception as e:
                t_llm = (time.perf_counter() - t_llm_start) * 1000
                total_llm_ms += t_llm
                err_msg = f"Provider execution failure: {str(e)}"
                stage_timings["llm_ms"] = round(t_llm, 2)
                attempts_log.append({
                    "attempt": attempt,
                    "status": "ERROR",
                    "reason": err_msg,
                    "stage_latencies_ms": stage_timings
                })
                last_failure_reason = err_msg
                feedback_instruction = f"Provider failure: {str(e)}. Simplify and return valid JSON."
                continue

            t_val_start = time.perf_counter()
            is_valid, reason, val_details = grounding_validator.validate(
                question=candidate_q,
                allowed_projects=allowed_projects,
                allowed_entities=allowed_entities,
                resolved_evidence_items=resolved_evidence_items,
                existing_questions=existing_questions,
                available_claims=claims_map,
                available_relationships=rels_map,
                db=db,
                extra_excluded_texts=req_excluded
            )
            t_val = (time.perf_counter() - t_val_start) * 1000
            total_val_ms += t_val
            stage_timings["validation_ms"] = round(t_val, 2)

            attempts_log.append({
                "attempt": attempt,
                "question": candidate_q.question,
                "target_project": candidate_q.target_project,
                "target_entity": candidate_q.target_entity,
                "evidence_ids": candidate_q.evidence_ids,
                "evidence_text": candidate_q.evidence_text[:200] + "..." if len(candidate_q.evidence_text) > 200 else candidate_q.evidence_text,
                "status": "ACCEPTED" if is_valid else "REJECTED",
                "reason": reason,
                "scores": val_details,
                "stage_latencies_ms": stage_timings
            })

            if is_valid:
                accepted_question = candidate_q
                if "val_details" in locals() and val_details:
                    if val_details.get("assessed_difficulty"):
                        accepted_question.assessed_difficulty = val_details["assessed_difficulty"]
                    if val_details.get("difficulty_rationale"):
                        accepted_question.difficulty_rationale = val_details["difficulty_rationale"]
                    if val_details.get("grounding_status"):
                        accepted_question.grounding_status = val_details["grounding_status"]
                    accepted_question.rubric_version = val_details.get("rubric_version", "v2.0-canonical-5level")
                    accepted_question.validation_status = "ACCEPTED"
                break
            else:
                last_failure_reason = reason
                if val_details.get("code") == "unsupported_implementation_presupposition":
                    feedback_instruction = (
                        "CRITICAL RETRY CORRECTION: Your previous question was rejected because it presupposed unverified implementation "
                        "details (e.g., asking to 'walk through' an unstated transaction flow, error-handling pipeline, or custom configuration). "
                        "Because the cited evidence only establishes technology usage, you MUST formulate an OPEN-ENDED design question "
                        "(e.g., 'How would you approach designing...?' or 'What architectural trade-offs would you evaluate when...?'). "
                        "You MUST preserve and cite only exact valid evidence_id values supplied in the context."
                    )
                elif val_details.get("code") == "semantic_duplicate":
                    dup_text = val_details.get("duplicate_question_text") or "an existing question in the interview"
                    sim = val_details.get("similarity")
                    sim_str = f" (cosine similarity: {sim})" if sim else ""

                    if req.target_project:
                        proj_evidence_snippets = [
                            item.get("text", "").strip()
                            for item in resolved_evidence_items.values()
                            if item.get("text") and (not req.target_project or req.target_project.lower() in item.get("text", "").lower() or item.get("project") == req.target_project)
                        ]
                        evidence_guidance = ""
                        if proj_evidence_snippets:
                            lines_pool = [
                                l.strip("-•●* ").strip()
                                for t in proj_evidence_snippets
                                for l in t.split("\n")
                                if len(l.strip()) > 15
                            ]
                            if lines_pool:
                                evidence_guidance = f" (Verified evidence aspects include: {'; '.join(lines_pool[:2])})"

                        feedback_instruction = (
                            f"CRITICAL RETRY CORRECTION (DUPLICATE REJECTION):\n"
                            f"Your candidate question was rejected because it is semantically duplicate with an already asked question{sim_str}:\n"
                            f"Existing Duplicate Question: \"{dup_text}\"\n\n"
                            f"ACTIONABLE REQUIREMENTS FOR THIS RETRY:\n"
                            f"1. You MUST NOT ask about the same topic, mechanism, or concepts as the existing question above (do not merely rephrase or change wording).\n"
                            f"2. Probe a MATERIALLY DIFFERENT architectural dimension, component boundary, or technical trade-off supported by the verified evidence for '{req.target_project}'{evidence_guidance}.\n"
                            f"3. Frame the question OPEN-ENDEDLY ('How would you approach designing...?' or 'What trade-offs would you evaluate when...?').\n"
                            f"4. Cite ONLY verified evidence ID(s) provided in this prompt in 'evidence_ids'. Do NOT invent features or citations."
                        )
                    else:
                        untargeted_projects = [p for p in allowed_projects if p not in attempted_projects]
                        if untargeted_projects:
                            current_project = untargeted_projects[0]
                            attempted_projects.add(current_project)
                            current_entity = None
                            feedback_instruction = (
                                f"CRITICAL RETRY CORRECTION (DUPLICATE REJECTION):\n"
                                f"Your candidate question was rejected because it is semantically duplicate with an already asked question{sim_str}:\n"
                                f"Existing Duplicate Question: \"{dup_text}\"\n\n"
                                f"ACTIONABLE REQUIREMENTS FOR THIS RETRY:\n"
                                f"1. You MUST rotate to target project '{current_project}'.\n"
                                f"2. Probe a distinct architectural dimension from the verified evidence for '{current_project}'.\n"
                                f"3. Frame the question OPEN-ENDEDLY ('How would you approach designing...?').\n"
                                f"4. Cite ONLY verified evidence ID(s) provided in this prompt in 'evidence_ids'."
                            )
                        else:
                            feedback_instruction = (
                                f"CRITICAL RETRY CORRECTION (DUPLICATE REJECTION):\n"
                                f"Your candidate question was rejected because it is semantically duplicate with an already asked question{sim_str}:\n"
                                f"Existing Duplicate Question: \"{dup_text}\"\n\n"
                                f"ACTIONABLE REQUIREMENTS FOR THIS RETRY:\n"
                                f"1. You MUST probe a MATERIALLY DIFFERENT architectural dimension, failure mode, or technical trade-off from the verified evidence.\n"
                                f"2. Do NOT reword or ask about the same topic as the existing question above.\n"
                                f"3. Frame the question OPEN-ENDEDLY ('How would you approach designing...?').\n"
                                f"4. Cite ONLY verified evidence ID(s) provided in this prompt in 'evidence_ids'."
                            )
                elif val_details.get("code") == "target_entity_focus_mismatch":
                    actual_focus = val_details.get("actual_focus") or "another technology"
                    feedback_instruction = (
                        f"CRITICAL RETRY CORRECTION (TARGET FOCUS MISMATCH):\n"
                        f"Your candidate question was rejected because its primary inquiry focused on '{actual_focus}' instead of the assigned target '{current_entity}'.\n"
                        f"ACTIONABLE REQUIREMENTS FOR THIS RETRY:\n"
                        f"1. You MUST formulate the question with '{current_entity}' as the PRIMARY technical focus.\n"
                        f"2. Directly probe the design, architecture, internal execution mechanics, or trade-offs of '{current_entity}'.\n"
                        f"3. Frame the question OPEN-ENDEDLY ('How would you approach designing...?' or 'What trade-offs would you evaluate when configuring...?').\n"
                        f"4. Cite valid evidence IDs from the context in 'evidence_ids'."
                    )
                elif val_details.get("code") == "difficulty_mismatch":
                    req_lvl = val_details.get("requested_difficulty") or difficulty
                    ass_lvl = val_details.get("assessed_difficulty")
                    guidance = {
                        1: "You MUST formulate a foundational Level 1 question: ask 'What is X?' or 'Why is X useful for Y?' or define fundamental principles. Do NOT ask for concrete implementation details, boundary tests, or multi-component architecture.",
                        2: "You MUST formulate a Level 2 question: ask how a concept is applied in standard workflows or practical data flows. Do NOT ask for debugging algorithms or system trade-offs.",
                        3: "You MUST formulate a Level 3 question: probe concrete code mechanics, algorithms, data structures, boundary case testing, or transaction boundaries.",
                        4: "You MUST formulate a Level 4 question: probe system component boundaries, horizontal scalability, reliability, security isolation, or architectural trade-offs.",
                        5: "You MUST formulate a Level 5 question: probe deep reasoning under uncertainty, competing constraints, changing assumptions, alternative strategies, or cascading distributed failure modes."
                    }.get(req_lvl, "Align reasoning demand with the canonical 5-level rubric.")
                    feedback_instruction = (
                        f"CRITICAL RETRY CORRECTION (DIFFICULTY MISMATCH):\n"
                        f"Your previous question failed difficulty calibration: Requested Level {req_lvl}, but reasoning demanded Level {ass_lvl}.\n"
                        f"MANDATE FOR THIS RETRY:\n{guidance}\n"
                        f"Cite valid evidence IDs from the context in 'evidence_ids'."
                    )
                else:
                    feedback_instruction = reason

        t_persist_start = time.perf_counter()
        total_latency = round((time.perf_counter() - t_global_start) * 1000, 2)
        latencies = {
            # Canonical Task 5 keys
            "db_preload": round(t_db_preload, 2),
            "retrieval": round(total_retrieval_ms, 2),
            "rerank": round(total_rerank_ms, 2),
            "context_assembly": round(total_context_assembly_ms, 2),
            "llm_call_1": round(llm_call_latencies[0], 2) if len(llm_call_latencies) > 0 else 0.0,
            "llm_call_2": round(llm_call_latencies[1], 2) if len(llm_call_latencies) > 1 else 0.0,
            "validation": round(total_val_ms, 2),
            "total": total_latency,

            # Legacy backward-compatible keys
            "total_ms": total_latency,
            "db_preload_ms": round(t_db_preload, 2),
            "retrieval_ms": round(total_retrieval_ms, 2),
            "rerank_ms": round(total_rerank_ms, 2),
            "context_assembly_ms": round(total_context_assembly_ms, 2),
            "llm_ms": round(total_llm_ms, 2),
            "validation_ms": round(total_val_ms, 2)
        }

        trace_rec = RetrievalTrace(
            trace_id=trace_id,
            request_id=request_id,
            resume_id=resume_id,
            query_intent=f"{category} (difficulty: {difficulty})",
            retrieval_config={"category": category, "difficulty": difficulty, "max_attempts": MAX_ATTEMPTS},
            retrieval_scores={"attempts": attempts_log},
            selected_evidence=final_context.get("selected_evidence_items", []),
            discarded_evidence=final_context.get("discarded_evidence", []),
            final_context=final_context,
            latencies_ms=latencies,
            validation_result={
                "status": "ACCEPTED" if accepted_question else "INSUFFICIENT_EVIDENCE",
                "final_reason": "Passed all verification checks." if accepted_question else last_failure_reason
            }
        )
        db.add(trace_rec)

        if accepted_question:
            final_rationale = accepted_question.rationale
            if parent_q and getattr(req, "follow_up_reason", None) and not final_rationale.startswith("[Follow-up"):
                final_rationale = f"[Follow-up: {req.follow_up_reason}] {final_rationale}"

            # Retrieve precomputed embedding from validator details or compute once (Part A2)
            q_emb = val_details.get("embedding") if ("val_details" in locals() and val_details) else None
            if not q_emb and accepted_question.question:
                try:
                    q_emb = embedding_service.embed_query(accepted_question.question)
                except Exception as emb_err:
                    logger.warning(f"Could not compute embedding for saved question: {emb_err}")
                    q_emb = None

            q_id = uuid.uuid4()
            assessed_diff = accepted_question.assessed_difficulty or (val_details.get("assessed_difficulty") if ("val_details" in locals() and val_details) else None)
            rubric_ver = accepted_question.rubric_version or "v2.0-canonical-5level"

            if persist:
                q_rec = InterviewQuestion(
                    id=q_id,
                    resume_id=resume_id,
                    session_id=session_id,
                    parent_question_id=parent_q.id if parent_q else None,
                    question_text=accepted_question.question,
                    category=accepted_question.category,
                    difficulty=accepted_question.difficulty,
                    targeted_claim=accepted_question.target_claim_id or getattr(req, "target_claim_id", None) or "Evidence-Grounded Question",
                    target_project=accepted_question.target_project or getattr(req, "target_project", None),
                    target_entity=accepted_question.target_entity or getattr(req, "target_entity", None),
                    target_claim_id=accepted_question.target_claim_id or getattr(req, "target_claim_id", None),
                    target_relationship=accepted_question.target_relationship or getattr(req, "target_relationship", None),
                    evidence_ids=accepted_question.evidence_ids,
                    evidence_text=accepted_question.evidence_text,
                    rationale=final_rationale,
                    grounding_score=accepted_question.grounding_score,
                    relevance_score=accepted_question.relevance_score,
                    technical_quality_score=accepted_question.technical_quality_score,
                    retrieval_trace_id=trace_id,
                    embedding=q_emb,
                    embedding_model=embedding_service.model_name if q_emb else None,
                    assessed_difficulty=assessed_diff,
                    rubric_version=rubric_ver,
                    validation_details=val_details if ("val_details" in locals() and val_details) else None
                )
                db.add(q_rec)

            try:
                db.commit()
                if persist and "q_rec" in locals():
                    db.refresh(q_rec)
                    q_id = q_rec.id
                    assessed_diff = q_rec.assessed_difficulty
                    rubric_ver = q_rec.rubric_version
            except Exception as e:
                db.rollback()
                logger.error(f"Database persistence failure in RAG question generation: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Database persistence failure during question creation: {str(e)}")

            # Part A4: Scorecard recalculation off the hot path (computed via endpoint or background task)
            t_scorecard = 0.0

            t_persist = (time.perf_counter() - t_persist_start) * 1000
            latencies["scorecard_recalc_ms"] = round(t_scorecard, 2)
            latencies["persistence_ms"] = round(t_persist, 2)
            latencies["total_ms"] = round((time.perf_counter() - t_global_start) * 1000, 2)
            latencies["total"] = latencies["total_ms"]
            all_rejected_texts = list(req_excluded) + [a["question"] for a in attempts_log if a.get("question") and a.get("status") != "ACCEPTED"]

            return QuestionGenerationResponse(
                resume_id=resume_id,
                request_id=request_id,
                trace_id=trace_id,
                question_id=q_id,
                question=accepted_question,
                validation_status="ACCEPTED",
                message="Question successfully generated and verified.",
                attempts_count=len(attempts_log),
                latencies_ms=latencies,
                assessed_difficulty=assessed_diff,
                rubric_version=rubric_ver,
                rejected_candidates=all_rejected_texts
            )
        else:
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"Failed to commit retrieval trace: {e}")
            t_persist = (time.perf_counter() - t_persist_start) * 1000
            latencies["persistence_ms"] = round(t_persist, 2)
            latencies["total_ms"] = round((time.perf_counter() - t_global_start) * 1000, 2)
            latencies["total"] = latencies["total_ms"]

            err_code = "insufficient_evidence"
            reason_str = str(last_failure_reason or "").lower()
            if "difficulty" in reason_str:
                err_code = "difficulty_mismatch"
            elif "duplicate" in reason_str or "redundant" in reason_str or any(a.get("scores", {}).get("code") == "semantic_duplicate" for a in attempts_log):
                err_code = "redundancy_rejection"
            elif "focus" in reason_str or "presupposition" in reason_str or "grounding" in reason_str:
                err_code = "grounding_failure"
            elif "provider" in reason_str or "timeout" in reason_str:
                err_code = "provider_error"

            all_rejected_texts = list(req_excluded) + [a["question"] for a in attempts_log if a.get("question") and a.get("status") != "ACCEPTED"]

            return QuestionGenerationResponse(
                resume_id=resume_id,
                request_id=request_id,
                trace_id=trace_id,
                question=None,
                validation_status="INSUFFICIENT_EVIDENCE",
                error_code=err_code,
                message=f"Could not generate a valid question for category '{category}'. Reason: {last_failure_reason}",
                attempts_count=len(attempts_log),
                latencies_ms=latencies,
                rubric_version="v2.0-canonical-5level",
                rejected_candidates=all_rejected_texts
            )

rag_pipeline = RAGPipeline()