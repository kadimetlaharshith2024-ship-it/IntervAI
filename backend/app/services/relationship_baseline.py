"""
Phase 3/4 Baseline Extractor preserved intentionally for benchmarking.
Uses window co-occurrence and heuristic phrase matching.
"""
import re
from typing import List, Dict
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.resume import Resume
from app.models.relationship import Relationship
from app.schemas.relationship import RelationshipTriplet
from app.services.resume_service import segment_resume_sections

def parse_baseline_relations(text: str, known_skills: Dict[str, str], section_name: str) -> List[RelationshipTriplet]:
    triplets = []
    lines = [l.strip() for l in re.split(r"[\n•\-\|]+", text) if len(l.strip()) > 15]

    for line in lines:
        line_lower = line.lower()
        matched = []
        for term, canonical in known_skills.items():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", line_lower):
                matched.append(canonical)

        matched = list(dict.fromkeys(matched))

        # Flawed baseline logic: adjacent pairs chained on keywords
        if len(matched) >= 2 and any(k in line_lower for k in ["using", "with", "built", "implemented", "stack"]):
            for i in range(len(matched) - 1):
                src = matched[i]
                tgt = matched[i + 1]
                if src != tgt:
                    triplets.append(RelationshipTriplet(
                        source_entity=src,
                        relation_type="IMPLEMENTED_WITH",
                        target_entity=tgt,
                        section=section_name.title(),
                        context_entity=None,
                        evidence_text=line[:500],
                        confidence=0.88,
                        extraction_method="baseline_rule"
                    ))
    return triplets

def extract_baseline_relationships(resume_id, db: Session) -> List[RelationshipTriplet]:
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")

    known_skills = {rs.skill.name.lower(): rs.skill.name for rs in resume.skills if rs.skill}
    segmented = segment_resume_sections(resume.raw_text)

    all_triplets: List[RelationshipTriplet] = []
    for sec_name, sec_text in segmented.items():
        all_triplets.extend(parse_baseline_relations(sec_text, known_skills, sec_name))

    return all_triplets