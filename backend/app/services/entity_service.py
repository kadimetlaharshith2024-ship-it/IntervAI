import re
from typing import List, Dict
import spacy
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.resume import Resume
from app.models.skill import Skill, ResumeSkill
from app.schemas.skill import ExtractedEntity

# Load lightweight tokenizer
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Authoritative unified technical taxonomy derived from CANONICAL_TAXONOMY
from app.core.canonicalizer import (
    CANONICAL_TAXONOMY,
    ALIAS_INDEX,
    canonicalize_entity,
    strip_formatting_artifacts,
    extract_non_overlapping_entities
)
from app.services.resume_service import segment_resume_sections

# Backward-compatible TECH_TAXONOMY mapping lower-case alias to category
TECH_TAXONOMY: Dict[str, str] = {}
for canonical, meta in CANONICAL_TAXONOMY.items():
    TECH_TAXONOMY[canonical.lower()] = meta["category"]
    for alias in meta.get("aliases", []):
        TECH_TAXONOMY[alias.lower().strip()] = meta["category"]

from app.services.privacy_guard import privacy_guard, ALL_REDACTION_MARKERS

AMBIGUOUS_SHORT_TERMS = {"c", "r", "go"}

def is_contextually_valid_term(term: str, text: str, match_start: int, match_end: int) -> bool:
    """Disambiguates short acronyms like 'C', 'R', 'Go' from English words and grades."""
    if term not in AMBIGUOUS_SHORT_TERMS:
        return True

    window_start = max(0, match_start - 30)
    window_end = min(len(text), match_end + 30)
    context_window = text[window_start:window_end].lower()

    if term == "c":
        # Must be in programming context: e.g. "C, C++", "C/C++", "C programming", "C language", "embedded C"
        if re.search(r"\b(?:c\+\+|cpp|c#|embedded\s+c|c\s+programming|c\s+language|ansi\s+c)\b", context_window):
            return True
        if re.search(r"[,/]\s*c\s*[,/]", context_window) or re.search(r"\bc\s*,\s*c\+\+", context_window):
            return True
        return False
    elif term == "go":
        # Must be in programming context: e.g. "Golang", "Go programming", "Go and Rust", "in Go and"
        if re.search(r"\b(?:golang|go\s+lang|go\s+language|go\s+programming)\b", context_window):
            return True
        if re.search(r"\b(?:python|rust|java|c\+\+|typescript)\s*,\s*go\b", context_window):
            return True
        if re.search(r"\b(?:go\s+(?:and|or|,)\s*(?:rust|python|java|c\+\+|typescript)|(?:rust|python|java|c\+\+|typescript)\s+(?:and|or)\s+go)\b", context_window):
            return True
        if re.search(r"\b(?:in|using|with)\s+go\s+(?:and|or|,|\.)\b", context_window):
            return True
        return False
    elif term == "r":
        if re.search(r"\b(?:r\s+programming|r\s+language|r\s+script|r-package)\b", context_window):
            return True
        return False

    return True

def extract_entities_from_text(text: str) -> List[ExtractedEntity]:
    """
    Scans text against authoritative taxonomy using a hybrid structure-aware strategy:
    1. Category-structured skills section parsing with declared categories.
    2. Contextually disambiguated full-text scanning across projects and experience.
    3. Graceful preservation of un-aliased candidate technologies.
    """
    if not text:
        return []

    # Ensure text is free of PII
    clean_text, _ = privacy_guard.sanitize_text(text)
    found_entities: Dict[str, ExtractedEntity] = {}
    lower_text = clean_text.lower()

    # =========================================================================
    # PHASE 1: STRUCTURE-AWARE PARSING OF TECHNICAL SKILLS SECTION
    # =========================================================================
    segmented = segment_resume_sections(clean_text)
    skills_text = segmented.get("skills", "")

    if skills_text:
        for line in skills_text.split("\n"):
            line_str = line.strip()
            if not line_str:
                continue

            # Check for labeled category line (e.g. "Languages: Java, Python, C, C++, SQL")
            cat_match = re.match(r"^([^:]+):\s*(.*)$", line_str)
            if cat_match:
                declared_cat = cat_match.group(1).strip()
                items_str = cat_match.group(2).strip()
                # Split on commas, semicolons, bullets, and pipes
                items = [i.strip() for i in re.split(r"[,;•|]+", items_str) if i.strip()]
                for raw_item in items:
                    raw_clean = strip_formatting_artifacts(raw_item).strip()
                    if not raw_clean or len(raw_clean) < 1 or len(raw_clean) > 40:
                        continue
                    if any(rm.lower() in raw_clean.lower() for rm in ALL_REDACTION_MARKERS) or "redact" in raw_clean.lower():
                        continue

                    canon, tax_cat = canonicalize_entity(raw_clean)
                    if canon and canon.lower() not in ["unknown", "general", "redacted"]:
                        display_name = canon
                        # Use taxonomy category if recognized, otherwise declared
                        final_category = tax_cat if tax_cat != "General" else declared_cat
                        confidence = 0.98
                    else:
                        display_name = raw_clean
                        final_category = declared_cat
                        confidence = 0.85

                    found_entities[display_name] = ExtractedEntity(
                        name=display_name,
                        category=final_category,
                        confidence=confidence
                    )
            else:
                # Comma/bullet separated line without category prefix
                items = [i.strip() for i in re.split(r"[,;•|]+", line_str) if i.strip()]
                for raw_item in items:
                    raw_clean = strip_formatting_artifacts(raw_item).strip()
                    if not raw_clean or len(raw_clean) < 1 or len(raw_clean) > 40:
                        continue
                    canon, tax_cat = canonicalize_entity(raw_clean)
                    if canon and canon.lower() not in ["unknown", "general", "redacted"]:
                        found_entities[canon] = ExtractedEntity(
                            name=canon,
                            category=tax_cat,
                            confidence=0.96
                        )

    # =========================================================================
    # PHASE 2: FULL-TEXT SCANNING WITH NON-OVERLAPPING SPAN DISAMBIGUATION
    # =========================================================================
    non_overlapping = extract_non_overlapping_entities(clean_text, ALIAS_INDEX)
    entity_counts: Dict[str, int] = {}
    for canon, start_pos, end_pos in non_overlapping:
        # Check contextual validity for ambiguous short terms (e.g. 'c', 'r', 'go')
        low_alias = clean_text[start_pos:end_pos].lower().strip()
        if low_alias in AMBIGUOUS_SHORT_TERMS and not is_contextually_valid_term(low_alias, clean_text, start_pos, end_pos):
            continue
        entity_counts[canon] = entity_counts.get(canon, 0) + 1

    for canon, count in entity_counts.items():
        if any(rm.lower() in canon.lower() for rm in ALL_REDACTION_MARKERS) or "redact" in canon.lower():
            continue
        canon_meta = CANONICAL_TAXONOMY.get(canon, {})
        category = canon_meta.get("category", "General")

        if canon in found_entities:
            # Boost confidence for entities confirmed across multiple sections
            found_entities[canon].confidence = 1.0
        else:
            conf = min(0.85 + (count * 0.05), 0.95)
            found_entities[canon] = ExtractedEntity(
                name=canon,
                category=category,
                confidence=round(conf, 2)
            )

    return list(found_entities.values())

def extract_and_store_resume_entities(resume_id, db: Session) -> List[ExtractedEntity]:
    """Finds resume, defensively sanitizes text, extracts entities, and replaces stale skills."""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")

    # 1. Defensive re-sanitization: ensure stored raw_text is completely sanitized
    sanitized_text, meta = privacy_guard.sanitize_text(resume.raw_text)
    if meta["is_sanitized"] and sanitized_text != resume.raw_text:
        resume.raw_text = sanitized_text
        db.flush()

    extracted = extract_entities_from_text(resume.raw_text)

    # 2. Re-extract idempotency & stale record replacement:
    # Identify target skill records
    active_skill_ids = set()
    for entity in extracted:
        skill = db.query(Skill).filter(Skill.name.ilike(entity.name)).first()
        if not skill:
            skill = Skill(name=entity.name, category=entity.category)
            db.add(skill)
            db.flush()
        active_skill_ids.add(skill.id)

        existing_link = db.query(ResumeSkill).filter(
            ResumeSkill.resume_id == resume.id,
            ResumeSkill.skill_id == skill.id
        ).first()

        if not existing_link:
            resume_skill = ResumeSkill(
                resume_id=resume.id,
                skill_id=skill.id,
                extraction_confidence=entity.confidence
            )
            db.add(resume_skill)
        else:
            existing_link.extraction_confidence = entity.confidence

    # Remove stale skills belonging to this resume that are no longer present in extracted content
    db.query(ResumeSkill).filter(
        ResumeSkill.resume_id == resume.id,
        ResumeSkill.skill_id.not_in(active_skill_ids) if active_skill_ids else True
    ).delete(synchronize_session=False)

    db.commit()
    return extracted