"""
Resanitization Script and Utility for Database Records

Cleans existing records in:
- resume_chunks: chunk_text (and drops any leftover header/contact chunks)
- resume_claims: claim_text, evidence_text
- relationships: source_entity, target_entity, evidence_text
- interview_questions: evidence_text, question_text

Guarantees no raw PII remains persisted.
"""
import os
import sys
import logging
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import dotenv
dotenv.load_dotenv(os.path.join(backend_dir, ".env"))

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.resume import Resume
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.relationship import Relationship
from app.models.interview import InterviewQuestion
from app.services.privacy_guard import privacy_guard, ALL_REDACTION_MARKERS

logger = logging.getLogger(__name__)


def resanitize_existing_records(db: Session) -> Dict[str, int]:
    """
    Scans and re-sanitizes all existing database rows across resumes, chunks,
    claims, relationships, and interview questions.
    """
    stats = {
        "resumes_sanitized": 0,
        "chunks_dropped": 0,
        "chunks_sanitized": 0,
        "claims_sanitized": 0,
        "relationships_sanitized": 0,
        "questions_sanitized": 0,
    }

    # 1. Resumes: raw_text
    resumes = db.query(Resume).all()
    for res in resumes:
        if not res.raw_text:
            continue
        sanitized, meta = privacy_guard.sanitize_text(res.raw_text)
        if meta["is_sanitized"] and sanitized != res.raw_text:
            res.raw_text = sanitized
            stats["resumes_sanitized"] += 1

    # 2. ResumeChunks: drop header/contact chunks, sanitize remainder
    chunks = db.query(ResumeChunk).all()
    for chunk in chunks:
        sec = (chunk.section or "").strip().lower()
        if sec in ("header", "contact"):
            db.delete(chunk)
            stats["chunks_dropped"] += 1
            continue

        if chunk.chunk_text:
            sanitized, meta = privacy_guard.sanitize_text(chunk.chunk_text)
            if sanitized != chunk.chunk_text:
                chunk.chunk_text = sanitized
                stats["chunks_sanitized"] += 1

    # 3. ResumeClaims: claim_text and evidence_text
    claims = db.query(ResumeClaim).all()
    for claim in claims:
        changed = False
        if claim.claim_text:
            sanitized_c, _ = privacy_guard.sanitize_text(claim.claim_text)
            if sanitized_c != claim.claim_text:
                claim.claim_text = sanitized_c
                changed = True

        if claim.evidence_text:
            sanitized_e, _ = privacy_guard.sanitize_text(claim.evidence_text)
            if sanitized_e != claim.evidence_text:
                claim.evidence_text = sanitized_e
                changed = True

        if changed:
            stats["claims_sanitized"] += 1

    # 4. Relationships: evidence_text, target_entity, source_entity
    relationships = db.query(Relationship).all()
    for rel in relationships:
        changed = False
        if rel.evidence_text:
            sanitized_ev, _ = privacy_guard.sanitize_text(rel.evidence_text)
            if sanitized_ev != rel.evidence_text:
                rel.evidence_text = sanitized_ev
                changed = True

        if rel.target_entity:
            sanitized_tgt, _ = privacy_guard.sanitize_text(rel.target_entity)
            if sanitized_tgt != rel.target_entity:
                rel.target_entity = sanitized_tgt
                changed = True

        if rel.source_entity:
            sanitized_src, _ = privacy_guard.sanitize_text(rel.source_entity)
            if sanitized_src != rel.source_entity:
                rel.source_entity = sanitized_src
                changed = True

        if changed:
            stats["relationships_sanitized"] += 1

    # 5. InterviewQuestions: evidence_text and question_text
    questions = db.query(InterviewQuestion).all()
    for q in questions:
        changed = False
        if q.evidence_text:
            sanitized_ev, _ = privacy_guard.sanitize_text(q.evidence_text)
            if sanitized_ev != q.evidence_text:
                q.evidence_text = sanitized_ev
                changed = True

        if q.question_text:
            sanitized_qt, _ = privacy_guard.sanitize_text(q.question_text)
            if sanitized_qt != q.question_text:
                q.question_text = sanitized_qt
                changed = True

        if changed:
            stats["questions_sanitized"] += 1

    # 6. InterviewSessions: clean session_state knowledge_state topics
    from app.models.interview import InterviewSession
    from app.services.knowledge_state_service import JUNK_TOPIC_BLOCKLIST, is_assessable_topic
    from sqlalchemy.orm.attributes import flag_modified

    def is_junk_topic(topic_name: str) -> bool:
        t_lower = topic_name.strip().lower()
        if t_lower in JUNK_TOPIC_BLOCKLIST:
            return True
        if len(t_lower) > 35 or len(t_lower.split()) > 4:
            return True
        if any(b in t_lower for b in ["institute", "university", "college", "b.tech", "m.tech", "jee mains", "fest", "captained", "extracurricular", "all india rank"]):
            return True
        if any(l in t_lower for l in ["telugu", "hindi", "tamil", "fluent in"]):
            return True
        return False

    sessions = db.query(InterviewSession).all()
    for s in sessions:
        if not s.session_state or not isinstance(s.session_state, dict):
            continue
        k_state = s.session_state.get("knowledge_state")
        if not k_state or not isinstance(k_state, dict):
            continue
        topics = k_state.get("topics", {})
        if not topics:
            continue
        changed_session = False
        junk_keys = [
            t for t in list(topics.keys())
            if is_junk_topic(t) or not is_assessable_topic(t)
        ]
        if junk_keys:
            for jk in junk_keys:
                del topics[jk]
            changed_session = True

        for t_k, t_v in topics.items():
            if not t_v.get("is_assessable"):
                t_v["is_assessable"] = True
                changed_session = True

        # Recompute coverage over assessable topics only
        assessable = {k: v for k, v in topics.items() if v.get("is_assessable", True)}
        assessed_cnt = sum(1 for v in assessable.values() if v.get("assessment_status") in ("ASSESSED", "PARTIALLY_ASSESSED") or v.get("questions_asked_count", 0) > 0)
        cov_pct = round((assessed_cnt / len(assessable) * 100.0), 2) if assessable else 0.0

        cov = k_state.setdefault("coverage", {})
        cov["total_assessable_topics_count"] = len(assessable)
        cov["assessed_topics_count"] = assessed_cnt
        cov["coverage_percentage"] = cov_pct

        if changed_session:
            flag_modified(s, "session_state")
            stats["sessions_cleaned"] = stats.get("sessions_cleaned", 0) + 1

    db.commit()
    return stats


if __name__ == "__main__":
    import os
    import sys
    import dotenv
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    dotenv.load_dotenv(os.path.join(backend_dir, ".env"))

    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        print("Starting database re-sanitization...")
        res = resanitize_existing_records(db)
        print("Re-sanitization completed successfully:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    finally:
        db.close()
