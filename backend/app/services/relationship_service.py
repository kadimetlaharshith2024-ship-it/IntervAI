import re
from typing import List, Dict, Optional, Tuple, Set
import spacy
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.resume import Resume
from app.models.relationship import Relationship
from app.schemas.relationship import RelationshipTriplet
from app.core.canonicalizer import (
    canonicalize_entity,
    is_same_entity,
    strip_formatting_artifacts,
    sanitize_entity_label,
    extract_non_overlapping_entities,
    clean_project_title,
    CANONICAL_TAXONOMY,
    ALIAS_INDEX,
    DEPLOYMENT_TARGETS,
    PROJECT_TITLE_BLACKLIST
)
from app.services.resume_service import segment_resume_sections
from app.services.privacy_guard import privacy_guard, ALL_REDACTION_MARKERS

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import logging
    logging.getLogger(__name__).warning("spaCy 'en_core_web_sm' model not found. Using blank 'en' pipeline; install during setup to enable full dependency parsing.")
    nlp = spacy.blank("en")

TECH_STACK_PREFIXES = [
    r"(?i)^tech(?:nology)?\s*stack\s*:",
    r"(?i)^technologies\s*(?:used)?\s*:",
    r"(?i)^tools\s*(?:&|and)?\s*technologies\s*:",
    r"(?i)^environment\s*:",
    r"(?i)^core\s*stack\s*:"
]

RELATION_HIERARCHY = {
    "DEVELOPED": 110,
    "SECURED_WITH": 100,
    "CONTAINERIZED_WITH": 95,
    "DEPLOYED_ON": 90,
    "BUILT_WITH": 80,
    "IMPLEMENTED_WITH": 75,
    "INTEGRATES_WITH": 60,
    "USES": 50,
    "HAS_TECHNOLOGY": 40,
    "HAS_SKILL": 30
}

def calculate_calibrated_confidence(
    method: str,
    has_dep_path: bool,
    sentence_len: int,
    evidence_specificity: float
) -> float:
    base = 0.72
    if method == "dependency_parsed":
        base += 0.16
    elif method == "tech_stack_enumeration":
        base += 0.20
    elif method == "section_structural":
        base += 0.22

    if has_dep_path:
        base += 0.05

    if sentence_len > 250:
        base -= 0.05
    elif sentence_len < 120:
        base += 0.03

    score = base * evidence_specificity
    return round(min(max(score, 0.40), 0.99), 2)


def chunk_project_blocks(section_text: str) -> List[Tuple[Optional[str], Optional[str], List[str]]]:
    """
    Returns list of tuples: (project_title, header_evidence_line, list_of_body_lines)
    Strictly isolates genuine project headers from wrapped sentence lines.
    Preserves intra-word hyphens in project names (e.g. Real-Time, Event-Triggered).
    """
    lines = [l.strip() for l in section_text.split("\n") if l.strip()]
    project_blocks: List[Tuple[Optional[str], Optional[str], List[str]]] = []

    current_title: Optional[str] = None
    current_header_line: Optional[str] = None
    current_lines: List[str] = []

    for i, line in enumerate(lines):
        raw_stripped = line.strip()
        is_bullet = bool(re.match(r"^[\s●•*–]\s*|^\s*-\s+", raw_stripped))
        is_tech_stack = any(re.search(pref, raw_stripped) for pref in TECH_STACK_PREFIXES)

        potential_title = None
        # Bullets and sentence fragments cannot be project titles
        if not is_bullet and not is_tech_stack:
            # Case 1: Divided by pipe, en/em dash, or spaced hyphen/double-hyphen (e.g., "CampusBridge | React, Spring Boot | 2026")
            if any(sep in raw_stripped for sep in ["|", "–", "—", " -- "]) or re.search(r"\s+-\s+", raw_stripped):
                parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*[\–\—]\s*|\s+--\s+|\s+-\s+", raw_stripped) if p.strip()]
                for p in parts:
                    candidate = clean_project_title(p)
                    if candidate:
                        potential_title = candidate
                        break

            # Case 2: Standalone title followed immediately by a Tech Stack line
            elif i + 1 < len(lines) and any(re.search(pref, lines[i + 1]) for pref in TECH_STACK_PREFIXES):
                potential_title = clean_project_title(raw_stripped)

            # Case 3: Standalone project heading
            else:
                potential_title = clean_project_title(raw_stripped)

        if potential_title and potential_title != current_title:
            if current_lines:
                project_blocks.append((current_title, current_header_line, current_lines))
                current_lines = []
            current_title = potential_title
            current_header_line = raw_stripped
        else:
            current_lines.append(line)

    if current_lines:
        project_blocks.append((current_title, current_header_line, current_lines))

    return project_blocks

def parse_experience_section(section_text: str, known_skills: Dict[str, str]) -> List[RelationshipTriplet]:
    """
    Extracts employment facts from Experience section without confusing employers with projects.
    - Anchors: Candidate -[WORKED_AT]-> Employer and Candidate -[WORKED_AS]-> Job Title
    - Accomplishments: Candidate -[BUILT_WITH]-> Technology in context of Employer
    """
    lines = [l.strip() for l in section_text.split("\n") if l.strip()]
    triplets: List[RelationshipTriplet] = []

    current_employer: Optional[str] = None
    current_role: Optional[str] = None

    for i, line in enumerate(lines):
        is_bullet = bool(re.match(r"^[\s●•*–]\s*|^\s*-\s+", line))
        is_tech = any(re.search(pref, line) for pref in TECH_STACK_PREFIXES)

        # Check if line is an employment header (e.g. "SpeechLogix Technologies -- Software Engineer Intern | May 2026 - June 2026")
        if not is_bullet and not is_tech and (any(sep in line for sep in ["|", "–", "—", " -- "]) or re.search(r"\s+-\s+", line)):
            parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*[\–\—]\s*|\s+--\s+|\s+-\s+", line) if p.strip()]
            employer_cand = None
            role_cand = None

            for part in parts:
                clean_part = strip_formatting_artifacts(part).strip()
                if not clean_part:
                    continue
                # Date parts
                if re.search(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|present)\b", clean_part):
                    continue
                # Role parts
                if re.search(r"(?i)\b(?:intern|engineer|developer|lead|architect|manager|consultant|analyst|specialist|officer|associate|programmer)\b", clean_part):
                    role_cand = clean_part
                else:
                    clean_emp = re.sub(r"[\|\–\—\-]+$", "", clean_part).strip()
                    if 2 <= len(clean_emp) <= 60 and len(clean_emp.split()) <= 7:
                        employer_cand = clean_emp

            if employer_cand or role_cand:
                current_employer = employer_cand or "Employer"
                current_role = role_cand

                if employer_cand:
                    triplets.append(RelationshipTriplet(
                        source_entity="Candidate",
                        relation_type="WORKED_AT",
                        target_entity=employer_cand,
                        section="Experience",
                        context_entity=employer_cand,
                        evidence_text=line,
                        confidence=0.98,
                        extraction_method="section_structural"
                    ))

                if role_cand:
                    triplets.append(RelationshipTriplet(
                        source_entity="Candidate",
                        relation_type="WORKED_AS",
                        target_entity=role_cand,
                        section="Experience",
                        context_entity=current_employer,
                        evidence_text=line,
                        confidence=0.96,
                        extraction_method="section_structural"
                    ))
                continue

        # Body bullet lines within this employment entry
        if current_employer:
            if is_tech:
                tech_triplets = parse_tech_stack_line(line, "Candidate", "Experience")
                for t in tech_triplets:
                    t.context_entity = current_employer
                triplets.extend(tech_triplets)
            else:
                ling_triplets = parse_linguistic_sentence(line, context_name=None, section="Experience", known_skills=known_skills)
                for t in ling_triplets:
                    t.context_entity = current_employer
                triplets.extend(ling_triplets)

    return triplets

def parse_tech_stack_line(line: str, context_name: Optional[str], section: str) -> List[RelationshipTriplet]:
    triplets = []
    content = line
    for pref in TECH_STACK_PREFIXES:
        content = re.sub(pref, "", content).strip()

    tokens = re.split(r"[,;/•\|\(\)]+", content)
    root_node = context_name if context_name else "Candidate"
    rel_type = "HAS_TECHNOLOGY" if context_name else "HAS_SKILL"

    seen_in_line = set()
    for raw in tokens:
        raw_clean = strip_formatting_artifacts(raw)
        if not raw_clean or len(raw_clean) > 35:
            continue
        canon, cat = canonicalize_entity(raw_clean)
        if not canon or canon.lower() in seen_in_line or canon.lower() in ["unknown", "redacted"]:
            continue
        if cat in ["Unknown", "Redacted"]:
            continue
        from app.services.knowledge_state_service import JUNK_TOPIC_BLOCKLIST
        if canon.lower() in JUNK_TOPIC_BLOCKLIST or raw_clean.lower() in JUNK_TOPIC_BLOCKLIST:
            continue
        seen_in_line.add(canon.lower())

        triplets.append(RelationshipTriplet(
            source_entity=root_node,
            relation_type=rel_type,
            target_entity=canon,
            section=section,
            context_entity=context_name,
            evidence_text=line.strip()[:400],
            confidence=calculate_calibrated_confidence("tech_stack_enumeration", True, len(line), 0.96),
            extraction_method="tech_stack_enumeration"
        ))

    return triplets

def parse_linguistic_sentence(
    sentence: str,
    context_name: Optional[str],
    section: str,
    known_skills: Dict[str, str]
) -> List[RelationshipTriplet]:
    triplets = []
    sent_text = strip_formatting_artifacts(sentence)
    lower_sent = sent_text.lower()
    root_context = context_name if context_name else "Candidate"

    matched_entities = extract_non_overlapping_entities(sent_text, known_skills)

    # Dynamic entity extraction for unseen technologies in engineering phrases:
    # (e.g. "using DuckDB", "built with Svelte", "integrated with NATS", "backed by ChromaDB")
    claimed_spans = [(m[1], m[2]) for m in matched_entities]
    dynamic_verbal_pattern = re.compile(
        r"(?i)\b(?:built\s+with|developed\s+with|using|implemented\s+with|integrated\s+with|deployed\s+on|powered\s+by|backed\s+by)\s+([A-Za-z0-9+#_.-]+(?:\s+[A-Za-z0-9+#_.-]+)?)"
    )
    DYNAMIC_STOPWORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "various", "multiple",
        "several", "modern", "scalable", "custom", "distributed", "internal", "external",
        "new", "different", "complex", "simple", "efficient", "robust", "high", "low",
        "large", "small", "real", "time", "real-time", "production", "enterprise",
        "vanilla", "scratch", "standard", "pure", "raw", "clean", "best"
    }
    for dyn_match in dynamic_verbal_pattern.finditer(sent_text):
        cand_raw = dyn_match.group(1).strip()
        cand_low = cand_raw.lower()
        if cand_low not in DYNAMIC_STOPWORDS and len(cand_raw) >= 2 and not cand_raw.isdigit():
            sp_start, sp_end = dyn_match.start(1), dyn_match.end(1)
            if not any(not (sp_end <= c_s or sp_start >= c_e) for c_s, c_e in claimed_spans):
                canon_cand, _ = canonicalize_entity(cand_raw)
                if canon_cand and canon_cand.lower() not in ["unknown"]:
                    matched_entities.append((canon_cand, sp_start, sp_end))
                    claimed_spans.append((sp_start, sp_end))

    matched_entities.sort(key=lambda x: x[1])
    if not matched_entities:
        return []

    # 1. Specialized Semantic Classifiers for Docker, Deployment & Security
    if "docker" in lower_sent and any(w in lower_sent for w in ["container", "dockerized", "containerized"]):
        triplets.append(RelationshipTriplet(
            source_entity=root_context,
            relation_type="CONTAINERIZED_WITH",
            target_entity="Docker",
            section=section,
            context_entity=context_name,
            evidence_text=sent_text[:400],
            confidence=0.96,
            extraction_method="dependency_parsed"
        ))

    for deploy_alias, deploy_canon in DEPLOYMENT_TARGETS.items():
        if re.search(rf"\b(?:deployed|hosted|running)\b.*?\b{deploy_alias}\b", lower_sent) or \
           re.search(rf"\b{deploy_alias}\b.*?\b(?:deployment|hosting)\b", lower_sent):
            triplets.append(RelationshipTriplet(
                source_entity=root_context,
                relation_type="DEPLOYED_ON",
                target_entity=deploy_canon,
                section=section,
                context_entity=context_name,
                evidence_text=sent_text[:400],
                confidence=0.95,
                extraction_method="dependency_parsed"
            ))

    if any(w in lower_sent for w in ["jwt", "spring security", "oauth", "auth", "stateless"]):
        for entity_name, _, _ in matched_entities:
            if entity_name in ["JWT", "Spring Security", "OAuth2"]:
                triplets.append(RelationshipTriplet(
                    source_entity=root_context,
                    relation_type="SECURED_WITH",
                    target_entity=entity_name,
                    section=section,
                    context_entity=context_name,
                    evidence_text=sent_text[:400],
                    confidence=0.95,
                    extraction_method="dependency_parsed"
                ))

    # 2. General Construction / Engineering Accomplishments
    if context_name or any(w in lower_sent for w in [
        "built", "developed", "architected", "engineered", "created", "using", "used",
        "implemented", "utilizing", "replaced", "leveraged", "integrated", "processed",
        "expanded", "optimized", "designed", "powered", "linking", "bridging"
    ]):
        for entity_name, _, _ in matched_entities:
            if entity_name in ["Docker", "Vercel", "Render", "JWT", "Spring Security"]:
                continue
            triplets.append(RelationshipTriplet(
                source_entity=root_context,
                relation_type="BUILT_WITH",
                target_entity=entity_name,
                section=section,
                context_entity=context_name,
                evidence_text=sent_text[:400],
                confidence=0.92,
                extraction_method="dependency_parsed"
            ))

    # 3. Inter-Entity Dependencies (True architectural integration only)
    if len(matched_entities) >= 2:
        NON_SOURCE_ENTITIES = {
            "r-tree indexing", "geojson", "shapely", "numpy", "pandas",
            "pydantic", "pydantic v2", "jwt", "rest apis"
        }
        for i in range(len(matched_entities)):
            for j in range(len(matched_entities)):
                if i == j:
                    continue
                src_entity = matched_entities[i][0]
                tgt_entity = matched_entities[j][0]

                if is_same_entity(src_entity, tgt_entity):
                    continue

                if src_entity.lower() in NON_SOURCE_ENTITIES:
                    continue

                src_pos = matched_entities[i][2]
                tgt_pos = matched_entities[j][1]

                if 0 < (tgt_pos - src_pos) < 60:
                    span_between = lower_sent[src_pos:tgt_pos]
                    rel = None
                    if re.search(r"\b(?:integrated\s+with|integrating\s+with|linking\s+to|bridging\s+with|connected\s+to)\b", span_between):
                        rel = "INTEGRATES_WITH"
                    elif re.search(r"\b(?:backed\s+by|paired\s+with|persisted\s+in|using|uses)\b", span_between):
                        rel = "USES"
                    elif re.search(r"\b(?:built\s+with|developed\s+with)\b", span_between) and not re.search(r"\b(?:in|over|via)\b", span_between):
                        rel = "BUILT_WITH"

                    if rel:
                        triplets.append(RelationshipTriplet(
                            source_entity=src_entity,
                            relation_type=rel,
                            target_entity=tgt_entity,
                            section=section,
                            context_entity=context_name,
                            evidence_text=sent_text[:400],
                            confidence=0.93,
                            extraction_method="dependency_parsed"
                        ))

    return triplets

def parse_education_section(text: str) -> List[RelationshipTriplet]:
    triplets = []
    lines = [strip_formatting_artifacts(l) for l in text.split("\n") if l.strip()]

    degree_pattern = r"(?i)\b(?:Bachelor\s+of\s+[A-Za-z\s]+|B\.?Tech|B\.?E\.?|Master\s+of\s+[A-Za-z\s]+|M\.?Tech|B\.?S\.?|M\.?S\.?)\b"
    major_pattern = r"(?i)(?:in|,)\s+([A-Za-z\s&]+?)(?=\s*\||\s*\d{4}|$)"
    inst_pattern = r"(?i)\b(Vellore Institute of Technology|VIT\s+[A-Za-z]+|VIT|IIT\s+[A-Za-z]+|NIT\s+[A-Za-z]+|University|College|Institute)\b"

    for line in lines:
        cleaned_line = strip_formatting_artifacts(line)
        deg_match = re.search(degree_pattern, cleaned_line)
        inst_match = re.search(inst_pattern, cleaned_line)

        if deg_match:
            degree_str = deg_match.group(0).strip()
            major_match = re.search(major_pattern, cleaned_line)
            major = major_match.group(1).strip() if major_match else degree_str

            if len(major) > 3 and major.lower() not in ["technology", "engineering", "science"]:
                triplets.append(RelationshipTriplet(
                    source_entity="Candidate",
                    relation_type="STUDIED",
                    target_entity=major.title(),
                    section="Education",
                    context_entity=None,
                    evidence_text=line[:400],
                    confidence=0.96,
                    extraction_method="section_structural"
                ))

        if inst_match:
            triplets.append(RelationshipTriplet(
                source_entity="Candidate",
                relation_type="ATTENDED",
                target_entity=inst_match.group(0).strip(),
                section="Education",
                context_entity=None,
                evidence_text=line[:400],
                confidence=0.98,
                extraction_method="section_structural"
            ))

    return triplets

def parse_language_section(text: str) -> List[RelationshipTriplet]:
    triplets = []
    KNOWN_LANGUAGES = {
        "english", "telugu", "hindi", "tamil", "kannada", "malayalam", 
        "spanish", "french", "german", "mandarin", "japanese"
    }
    tokens = re.split(r"[\n,\|;•]+", text)
    for tok in tokens:
        clean = strip_formatting_artifacts(re.sub(r"\(.*?\)", "", tok))
        if clean.lower() in KNOWN_LANGUAGES:
            triplets.append(RelationshipTriplet(
                source_entity="Candidate",
                relation_type="SPEAKS",
                target_entity=clean.title(),
                section="Languages",
                context_entity=None,
                evidence_text=tok.strip()[:200],
                confidence=0.99,
                extraction_method="section_structural"
            ))
    return triplets

def parse_certifications_section(text: str) -> List[RelationshipTriplet]:
    triplets = []
    lines = [strip_formatting_artifacts(l) for l in text.split("\n") if len(l.strip()) > 3]
    for line in lines:
        clean = re.sub(r"(?i)\(view[-_\s]*certificate\)", "", line).strip()
        # A single line might list multiple certifications or courses separated by pipes or semicolons
        sub_items = [s.strip() for s in re.split(r"\s*\|\s*|\s*;\s*", clean) if s.strip()]
        for item in sub_items:
            norm_item = item.lower()
            # 1. In-Progress Certifications
            if re.search(r"(?i)\b(?:in\s+progress|pursuing|studying|scheduled)\b", norm_item):
                clean_name = re.sub(r"(?i)\s*[\(\[]?(?:in\s+progress|pursuing|studying|scheduled)[\)\]]?", "", item).strip()
                clean_name = strip_formatting_artifacts(clean_name).strip(" -—–|")
                if len(clean_name) >= 3:
                    formatted_name = re.sub(r"(?i)\baws\b", "AWS", clean_name[:80].title())
                    formatted_name = re.sub(r"(?i)\bgcp\b", "GCP", formatted_name)
                    triplets.append(RelationshipTriplet(
                        source_entity="Candidate",
                        relation_type="PURSUING_CERTIFICATION",
                        target_entity=formatted_name,
                        section="Certifications",
                        context_entity=None,
                        evidence_text=item[:400],
                        confidence=0.95,
                        extraction_method="section_structural"
                    ))
            # 2. Online Courses (Udemy, Coursera, etc.)
            elif re.search(r"(?i)\b(?:udemy|coursera|edx|pluralsight|course|masterclass|bootcamp)\b", norm_item):
                clean_name = strip_formatting_artifacts(item).strip(" -—–|")
                if len(clean_name) >= 3:
                    triplets.append(RelationshipTriplet(
                        source_entity="Candidate",
                        relation_type="COMPLETED_COURSE",
                        target_entity=clean_name[:80],
                        section="Certifications",
                        context_entity=None,
                        evidence_text=item[:400],
                        confidence=0.95,
                        extraction_method="section_structural"
                    ))
            # 3. Earned Accredited Certifications
            else:
                parts = re.split(r"\s*[—–\-]\s*", item)
                cert_name = strip_formatting_artifacts(parts[0]).strip()
                if len(cert_name) >= 3:
                    triplets.append(RelationshipTriplet(
                        source_entity="Candidate",
                        relation_type="EARNED_CERTIFICATION",
                        target_entity=cert_name[:80].title(),
                        section="Certifications",
                        context_entity=None,
                        evidence_text=item[:400],
                        confidence=0.95,
                        extraction_method="section_structural"
                    ))
    return triplets

def parse_achievements_section(text: str) -> List[RelationshipTriplet]:
    triplets = []
    lines = [strip_formatting_artifacts(l) for l in text.split("\n") if len(l.strip()) > 5]
    for line in lines:
        if any(lang in line.lower() for lang in ["english", "telugu", "hindi", "native", "conversational"]):
            continue

        clean = strip_formatting_artifacts(line)
        norm_line = clean.lower()

        # Hackathon / Competition Event Detection
        event_match = re.search(r"(?i)\b([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)?\s+Hackathon)\b", clean)
        event_name = event_match.group(1).title() if event_match else None

        # Hackathon Participation (e.g. 'Hackathon Participant (SDG-focused MVP) - Celestia')
        if not event_name and "hackathon" in norm_line:
            trailing_event = re.search(r"[-–—]\s*([A-Za-z0-9\s]+?)\s*$", clean)
            if trailing_event and len(trailing_event.group(1).strip()) >= 3:
                cand_name = trailing_event.group(1).strip()
                if "hackathon" not in cand_name.lower():
                    event_name = f"{cand_name} Hackathon"
                else:
                    event_name = cand_name
            else:
                event_name = "Hackathon"

        # Award / Winner Recognition (e.g. 'Winner - Best Freshers Award')
        award_match = re.search(
            r"(?i)\b(?:Winner\s*[-–:]\s*|awarded\s+|won\s+|secured\s+)([A-Za-z0-9\s&]+?)(?:;|\.|\b(?:ranking|led|built|among|in\s+\d+)\b|$)",
            clean
        )
        if award_match:
            award_title = award_match.group(1).strip().title()
            triplets.append(RelationshipTriplet(
                source_entity="Candidate",
                relation_type="WON",
                target_entity=award_title[:60],
                section="Achievements",
                context_entity=event_name,
                evidence_text=clean[:400],
                confidence=0.96,
                extraction_method="section_structural"
            ))

        # Event Participation Link
        if event_name:
            triplets.append(RelationshipTriplet(
                source_entity="Candidate",
                relation_type="PARTICIPATED_IN",
                target_entity=event_name,
                section="Achievements",
                context_entity=event_name,
                evidence_text=clean[:400],
                confidence=0.95,
                extraction_method="section_structural"
            ))

        # Rank / Placement Recognition (e.g. '1st among 32 teams' or '1st place')
        rank_match = re.search(r"(?i)\b(\d+(?:st|nd|rd|th)\s+(?:place|among\s+\d+\s+teams))\b", clean)
        if rank_match and not award_match:
            target_entity = rank_match.group(1).title()
            triplets.append(RelationshipTriplet(
                source_entity="Candidate",
                relation_type="ACHIEVED",
                target_entity=target_entity[:60],
                section="Achievements",
                context_entity=event_name,
                evidence_text=clean[:400],
                confidence=0.92,
                extraction_method="section_structural"
            ))
        elif not event_name and not award_match:
            # Fallback for generic achievement lines
            head = re.split(r"[,;\|–—]", clean)[0].strip()
            target_entity = re.sub(r"(?i)^(?:secured|won|awarded|achieved)\s+", "", head).strip().title()
            if len(target_entity) >= 3:
                triplets.append(RelationshipTriplet(
                    source_entity="Candidate",
                    relation_type="ACHIEVED",
                    target_entity=target_entity[:60],
                    section="Achievements",
                    context_entity=None,
                    evidence_text=clean[:400],
                    confidence=0.92,
                    extraction_method="section_structural"
                ))
    return triplets

def validate_evidence(triplet: RelationshipTriplet) -> bool:
    if not triplet.evidence_text or len(triplet.evidence_text.strip()) < 5:
        return False
    if not triplet.target_entity or len(triplet.target_entity.strip()) < 2:
        return False

    t_norm = triplet.target_entity.lower().strip()
    ev_norm = triplet.evidence_text.lower().strip()

    target_tokens = [t for t in re.split(r"[\s\-_]+", t_norm) if len(t) > 2]
    has_target_mention = any(tok in ev_norm for tok in target_tokens) or (t_norm in ev_norm)
    if has_target_mention:
        return True

    canon_meta = CANONICAL_TAXONOMY.get(triplet.target_entity)
    if canon_meta:
        return any(alias.lower() in ev_norm for alias in canon_meta.get("aliases", []))

    return False

def deduplicate_and_resolve_conflicts(triplets: List[RelationshipTriplet]) -> List[RelationshipTriplet]:
    """
    Deduplicates semantic relationships by canonical entity aliases while strictly preserving:
    1. Exact relation types (BUILT_WITH, IMPLEMENTED_WITH, USES, HAS_TECHNOLOGY, etc. remain distinct).
    2. Meaningful context scopes (relationships anchored in different projects/contexts remain distinct).
    3. Maximum confidence and combined evidence provenance across duplicate alias mentions.
    """
    edge_map: Dict[Tuple[str, str, str, str, str], RelationshipTriplet] = {}

    for t in triplets:
        canon_src, _ = canonicalize_entity(t.source_entity)
        canon_tgt, _ = canonicalize_entity(t.target_entity)

        if not canon_src or not canon_tgt or canon_src.lower() == canon_tgt.lower():
            continue

        if not validate_evidence(t):
            continue

        # Context scope normalization
        ctx_norm = (t.context_entity or "").strip().lower()
        if not ctx_norm and canon_src.lower() != "candidate":
            ctx_norm = canon_src.lower()

        sec_norm = (t.section or "").strip().lower()
        rel_type = t.relation_type.strip().upper()

        # Strict deduplication key: preserves exact relation type and context scope
        key = (
            canon_src.lower(),
            rel_type,
            canon_tgt.lower(),
            ctx_norm,
            sec_norm
        )

        if key in edge_map:
            existing = edge_map[key]

            # Select higher confidence representative
            is_new_higher = t.confidence > existing.confidence
            winner = t if is_new_higher else existing
            other = existing if is_new_higher else t

            conf = max(existing.confidence, t.confidence)

            # Preserve evidence provenance: combine non-redundant evidence texts
            primary_ev = winner.evidence_text.strip()
            secondary_ev = other.evidence_text.strip()
            if secondary_ev and secondary_ev.lower() not in primary_ev.lower():
                merged_evidence = f"{primary_ev} | {secondary_ev}"
            else:
                merged_evidence = primary_ev

            merged_context = winner.context_entity if winner.context_entity else other.context_entity

            edge_map[key] = RelationshipTriplet(
                source_entity=canon_src,
                relation_type=t.relation_type,
                target_entity=canon_tgt,
                section=winner.section,
                context_entity=merged_context,
                evidence_text=merged_evidence,
                confidence=conf,
                extraction_method=winner.extraction_method
            )
        else:
            edge_map[key] = RelationshipTriplet(
                source_entity=canon_src,
                relation_type=t.relation_type,
                target_entity=canon_tgt,
                section=t.section,
                context_entity=t.context_entity,
                evidence_text=t.evidence_text.strip(),
                confidence=t.confidence,
                extraction_method=t.extraction_method
            )

    return list(edge_map.values())


def extract_and_store_relationships_advanced(resume_id, db: Session) -> List[RelationshipTriplet]:
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found.")

    # Defensive re-sanitization: ensure stored raw_text is completely sanitized
    sanitized_text, meta = privacy_guard.sanitize_text(resume.raw_text)
    if meta["is_sanitized"] and sanitized_text != resume.raw_text:
        resume.raw_text = sanitized_text
        db.flush()

    known_skills = {}
    for alias_lower, canonical in ALIAS_INDEX.items():
        known_skills[alias_lower] = canonical

    # Defensive pre-sanitization before segmenting and relationship extraction
    clean_raw_text, _ = privacy_guard.sanitize_text(resume.raw_text or "")
    segmented = segment_resume_sections(clean_raw_text)
    raw_triplets: List[RelationshipTriplet] = []

    if "education" in segmented:
        raw_triplets.extend(parse_education_section(segmented["education"]))

    if "certifications" in segmented:
        raw_triplets.extend(parse_certifications_section(segmented["certifications"]))

    if "achievements" in segmented:
        raw_triplets.extend(parse_achievements_section(segmented["achievements"]))

    for sec_name, sec_text in segmented.items():
        if "lang" in sec_name or any(l in sec_text.lower() for l in ["telugu", "hindi", "native"]):
            raw_triplets.extend(parse_language_section(sec_text))

    # 1. Projects Section (Engineering Projects)
    if "projects" in segmented:
        sec_text = segmented["projects"]
        project_blocks = chunk_project_blocks(sec_text)

        for context_title, header_line, lines in project_blocks:
            # Anchor top-level Candidate -> DEVELOPED -> Project edge
            if context_title and "redact" not in context_title.lower():
                raw_triplets.append(RelationshipTriplet(
                    source_entity="Candidate",
                    relation_type="DEVELOPED",
                    target_entity=context_title,
                    section="Projects",
                    context_entity=context_title,
                    evidence_text=header_line if header_line else (lines[0] if lines else context_title),
                    confidence=0.98,
                    extraction_method="section_structural"
                ))

            # Extract technologies explicitly enumerated in the project's header line
            if header_line and context_title:
                # 1. Structural delimited parts
                parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*[\–\—]\s*|\s+--\s+|\s+-\s+", header_line) if p.strip()]
                for part in parts:
                    if is_same_entity(part, context_title) or clean_project_title(part):
                        continue
                    if re.search(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|patent|published)\b", part):
                        continue
                    raw_triplets.extend(parse_tech_stack_line(part, context_title, "Projects"))

                # 2. Non-overlapping entity extraction across the full header line (e.g., "Landing Page HTML5, CSS3, SVG")
                header_entities = extract_non_overlapping_entities(header_line, known_skills)
                for canon_ent, _, _ in header_entities:
                    if (
                        not is_same_entity(canon_ent, context_title)
                        and canon_ent.lower() not in ["github", "gitlab", "general", "unknown", "redacted"]
                        and not clean_project_title(canon_ent)
                    ):
                        raw_triplets.append(RelationshipTriplet(
                            source_entity=context_title,
                            relation_type="BUILT_WITH",
                            target_entity=canon_ent,
                            section="Projects",
                            context_entity=context_title,
                            evidence_text=header_line.strip()[:400],
                            confidence=0.98,
                            extraction_method="section_structural"
                        ))

            LINK_OR_REPO_LINE_PATTERN = re.compile(
                r"^(?:github|gitlab|bitbucket|live\s+demo|demo|view\s+code|repo|repository|deployed|website|link|url)\b|github\s+live\s+demo",
                flags=re.IGNORECASE
            )

            for line in lines:
                clean_l = line.strip()
                if not clean_l or LINK_OR_REPO_LINE_PATTERN.search(clean_l):
                    continue
                if any(re.search(pref, line) for pref in TECH_STACK_PREFIXES):
                    raw_triplets.extend(parse_tech_stack_line(line, context_title, "Projects"))
                else:
                    raw_triplets.extend(parse_linguistic_sentence(line, context_title, "Projects", known_skills))

    # 2. Experience Section (Employment History & Work Accomplishments)
    if "experience" in segmented:
        raw_triplets.extend(parse_experience_section(segmented["experience"], known_skills))

    final_triplets = deduplicate_and_resolve_conflicts(raw_triplets)

    # Filter out redaction markers and sanitize all relationship evidence
    clean_triplets: List[RelationshipTriplet] = []
    for trip in final_triplets:
        src_lower = trip.source_entity.lower()
        tgt_lower = trip.target_entity.lower()
        if any(rm.lower() in src_lower or rm.lower() in tgt_lower for rm in ALL_REDACTION_MARKERS):
            continue
        if "redact" in src_lower or "redact" in tgt_lower:
            continue
        clean_ev, _ = privacy_guard.sanitize_text(trip.evidence_text or "")
        clean_triplets.append(RelationshipTriplet(
            source_entity=trip.source_entity,
            relation_type=trip.relation_type,
            target_entity=trip.target_entity,
            section=trip.section,
            context_entity=trip.context_entity,
            evidence_text=clean_ev.strip(),
            confidence=trip.confidence,
            extraction_method=trip.extraction_method
        ))

    db.query(Relationship).filter(
        Relationship.resume_id == resume.id,
        Relationship.extraction_method != "baseline_rule"
    ).delete(synchronize_session=False)

    for trip in clean_triplets:
        db.add(Relationship(
            resume_id=resume.id,
            source_entity=trip.source_entity,
            relation_type=trip.relation_type,
            target_entity=trip.target_entity,
            section=trip.section,
            context_entity=trip.context_entity,
            evidence_text=trip.evidence_text,
            confidence=trip.confidence,
            extraction_method=trip.extraction_method
        ))

    db.commit()
    return clean_triplets