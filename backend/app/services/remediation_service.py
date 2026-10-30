import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.resume import Resume
from app.models.relationship import Relationship
from app.models.rag import ResumeChunk, ResumeClaim
from app.services.privacy_guard import privacy_guard
from app.services.entity_service import extract_and_store_resume_entities
from app.services.relationship_service import extract_and_store_relationships_advanced
from app.services.rag.rag_pipeline import rag_pipeline
from app.services.rag.retrievers import resume_retrieval_cache
from app.services.rag.validator import grounding_validator

logger = logging.getLogger(__name__)


def audit_pii_in_database(db: Session) -> Dict[str, Any]:
    """
    Audits the current PostgreSQL database records across resumes, claims,
    relationships, and chunks to identify records containing PII.
    Never logs or exposes raw PII values.
    """
    resumes = db.query(Resume).all()
    affected_resumes = []
    pii_categories_found = set()

    for r in resumes:
        scr = privacy_guard.screen_text(r.raw_text or "")
        if scr.has_violations:
            affected_resumes.append(str(r.id))
            pii_categories_found.update(scr.violation_categories)

    claims = db.query(ResumeClaim).all()
    affected_claims = []
    for c in claims:
        txt = f"{c.claim_text} {c.evidence_text or ''}"
        scr = privacy_guard.screen_text(txt)
        if scr.has_violations:
            affected_claims.append(c.claim_id)
            pii_categories_found.update(scr.violation_categories)

    relationships = db.query(Relationship).all()
    affected_relationships = []
    for rel in relationships:
        txt = f"{rel.source_entity} {rel.target_entity} {rel.evidence_text or ''}"
        scr = privacy_guard.screen_text(txt)
        if scr.has_violations:
            affected_relationships.append(str(rel.id))
            pii_categories_found.update(scr.violation_categories)

    chunks = db.query(ResumeChunk).all()
    affected_chunks = []
    for ch in chunks:
        scr = privacy_guard.screen_text(ch.chunk_text or "")
        if scr.has_violations:
            affected_chunks.append(ch.chunk_id)
            pii_categories_found.update(scr.violation_categories)

    return {
        "total_resumes": len(resumes),
        "affected_resumes_count": len(affected_resumes),
        "affected_resume_ids": affected_resumes,
        "total_claims": len(claims),
        "affected_claims_count": len(affected_claims),
        "total_relationships": len(relationships),
        "affected_relationships_count": len(affected_relationships),
        "total_chunks": len(chunks),
        "affected_chunks_count": len(affected_chunks),
        "categories_found": sorted(list(pii_categories_found))
    }


def remediate_database_records(
    db: Session,
    dry_run: bool = False,
    target_resume_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    Safely sanitizes previously stored resumes and rebuilds derived knowledge graphs,
    entities, chunks, and claims from sanitized source content.
    Preserves all user ownership, interview sessions, and scorecards.
    Does NOT drop tables, reset database, or delete unrelated records.
    """
    logger.info("Starting database PII remediation (dry_run=%s, target_resume_id=%s)", dry_run, target_resume_id)

    query = db.query(Resume)
    if target_resume_id:
        query = query.filter(Resume.id == target_resume_id)
    resumes = query.all()

    remediated_resumes_count = 0
    remediated_resume_ids: List[str] = []
    categories_cleaned = set()

    for r in resumes:
        scr = privacy_guard.screen_text(r.raw_text or "")
        needs_remediation = scr.has_violations

        # Also verify if any existing derived records for this resume contain PII
        if not needs_remediation:
            rel_violation = any(
                privacy_guard.screen_text(f"{rel.source_entity} {rel.target_entity} {rel.evidence_text or ''}").has_violations
                for rel in r.relationships
            )
            chunk_violation = any(
                privacy_guard.screen_text(ch.chunk_text or "").has_violations
                for ch in db.query(ResumeChunk).filter(ResumeChunk.resume_id == r.id).all()
            )
            needs_remediation = rel_violation or chunk_violation

        if needs_remediation:
            categories_cleaned.update(scr.violation_categories)
            if not dry_run:
                # 1. Sanitize stored resume raw_text
                sanitized_text, meta = privacy_guard.sanitize_text(r.raw_text or "")
                r.raw_text = sanitized_text
                db.flush()

                # 2. Re-extract entities and replace stale skills
                extract_and_store_resume_entities(r.id, db)

                # 3. Re-map relationships and replace stale relationship edges
                extract_and_store_relationships_advanced(r.id, db)

                # 4. Re-index chunks, embeddings, and claims
                rag_pipeline.index_resume(r.id, db)

            remediated_resumes_count += 1
            remediated_resume_ids.append(str(r.id))

    if not dry_run and remediated_resumes_count > 0:
        db.commit()
        # Invalidate memory retrieval caches
        try:
            resume_retrieval_cache.cache.clear()
        except Exception:
            pass

    post_audit = audit_pii_in_database(db)

    logger.info(
        "PII remediation complete. Remediated: %d resumes. Post-audit affected resumes: %d, chunks: %d, claims: %d",
        remediated_resumes_count,
        post_audit["affected_resumes_count"],
        post_audit["affected_chunks_count"],
        post_audit["affected_claims_count"]
    )

    return {
        "dry_run": dry_run,
        "remediated_resumes_count": remediated_resumes_count,
        "remediated_resume_ids": remediated_resume_ids,
        "categories_cleaned": sorted(list(categories_cleaned)),
        "post_remediation_audit": post_audit
    }


def remediate_resume_intelligence(
    db: Session,
    target_resume_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    Safely reprocesses stored resumes to repair section detection, project extraction,
    relationship mapping, and claims indexing using the corrected document intelligence pipeline.
    Preserves all user ownership, interview sessions, and scorecards.
    """
    from app.services.resume_service import detect_sections

    query = db.query(Resume)
    if target_resume_id:
        query = query.filter(Resume.id == target_resume_id)
    resumes = query.all()

    reprocessed = []
    for r in resumes:
        # 1. Re-extract entities
        extract_and_store_resume_entities(r.id, db)

        # 3. Re-map relationships
        extract_and_store_relationships_advanced(r.id, db)

        # 4. Re-index chunks and claims
        rag_pipeline.index_resume(r.id, db)

        reprocessed.append(str(r.id))

    db.commit()

    # Clear retrieval and validator caches
    try:
        resume_retrieval_cache.cache.clear()
        grounding_validator.clear_caches()
    except Exception:
        pass

    logger.info("Resume intelligence remediation complete for %d resumes.", len(reprocessed))
    return {
        "status": "success",
        "reprocessed_resumes_count": len(reprocessed),
        "resume_ids": reprocessed
    }

