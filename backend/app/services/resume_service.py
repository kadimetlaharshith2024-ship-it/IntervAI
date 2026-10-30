import io
import re
from typing import Dict, Tuple, Optional, List, Any
from uuid import UUID
import pdfplumber
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
from app.models.resume import Resume

SECTION_PATTERNS = {
    "education": r"(?i)^(?:education|academic\s+background|qualifications|academic\s+credentials|academics)\b",
    "experience": r"(?i)^(?:work\s+experience|professional\s+experience|experience|employment\s+history|internships?|work\s+history)\b",
    "projects": r"(?i)^(?:projects|academic\s+projects|personal\s+projects|key\s+projects|technical\s+projects|featured\s+projects)\b",
    "skills": r"(?i)^(?:technical\s+skills|skills\s*&?\s*tools|core\s+competencies|technologies|skills\s*&?\s*abilities|skills)\b",
    "certifications": r"(?i)^(?:certifications?|licenses\s*&?\s*certifications?|courses|professional\s+certifications?)\b",
    "achievements": r"(?i)^(?:achievements?|honors?\s*&?\s*awards?|awards?|hackathons?|competitions?|extracurricular\s+activities)\b",
    "summary": r"(?i)^(?:professional\s+summary|summary|profile|about\s+me|objective|career\s+objective)\b",
    "publications": r"(?i)^(?:publications?|research|patents?|papers?)\b",
    "languages": r"(?i)^(?:languages?|language\s+proficiency)\b",
    "leadership": r"(?i)^(?:leadership|positions\s+of\s+responsibility)\b",
}

COMBINED_SECTION_PATTERNS = [
    (
        r"(?i)\b(?:leadership|honors?|awards?|achievements?)\b.*?(?:&|and|,)\s*.*?\b(?:certifications?|courses?|licenses?)\b",
        ["achievements", "certifications"]
    ),
    (
        r"(?i)\b(?:achievements?|awards?|honors?)\b.*?(?:&|and|,)\s*.*?\b(?:certifications?|licenses?)\b",
        ["achievements", "certifications"]
    ),
    (
        r"(?i)\b(?:leadership)\b.*?(?:&|and|,)\s*.*?\b(?:achievements?|awards?|extracurricular)\b",
        ["achievements"]
    ),
    (
        r"(?i)\b(?:certifications?)\b.*?(?:&|and|,)\s*.*?\b(?:courses?|training)\b",
        ["certifications"]
    )
]

SKILL_SUBCATEGORY_PREFIXES = [
    r"(?i)^languages?\s*:",
    r"(?i)^programming\s+languages?\s*:",
    r"(?i)^frontend\s*(?:&|\/|and)?\s*(?:technologies|frameworks|tools)?\s*:",
    r"(?i)^backend\s*(?:&|\/|and)?\s*(?:technologies|frameworks|tools)?\s*:",
    r"(?i)^databases?\s*(?:&|\/|and)?\s*(?:storage)?\s*:",
    r"(?i)^ai\s*(?:\/|&)?\s*ml\s*:",
    r"(?i)^tools?\s*(?:\/|&|and)?\s*platforms?\s*:",
    r"(?i)^core\s+cs\s+concepts?\s*:",
    r"(?i)^cloud\s*(?:&|\/|and)?\s*devops\s*:",
    r"(?i)^frameworks?\s*(?:&|\/|and)?\s*libraries?\s*:",
    r"(?i)^web\s+technologies\s*:"
]

def clean_and_normalize_text(text: str) -> str:
    """Removes irregular whitespace, non-breaking spaces, and formatting artifacts while normalizing typography."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    # Unicode ligatures
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬃ", "ffi").replace("ﬄ", "ffl")
    # Dashes and hyphens (em-dash, en-dash, minus, fullwidth) normalized with clear boundary separation
    text = re.sub(r"[\u2013\u2014\u2015\u2212\uff0d]", " — ", text)
    # Bullet points to canonical bullet
    text = re.sub(r"[\u2022\u25cf\u25cb\u25aa\u25ab\u25a0\u2043\u2219]", "• ", text)
    # Quotation marks
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Separate attached tokens at layout boundary transitions (e.g., 'PageHTML5' -> 'Page HTML5')
    text = re.sub(r"([a-z])(HTML5|CSS3|JavaScript|React|Python|Java|SQL|Node|Vue|AWS|GCP|SVG)\b", r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def segment_resume_sections(raw_text: str) -> Dict[str, str]:
    """
    Splits full resume text into classified section blocks.
    Accurately detects single and combined headings (e.g. 'LEADERSHIP, ACHIEVEMENTS & CERTIFICATIONS')
    and semantically partitions child entries into their respective categories.
    Prevents category-label lines (e.g. 'Languages: Java, Python') inside Technical Skills
    from hijacking the section parser.
    """
    lines = raw_text.split("\n")
    sections: Dict[str, list] = {"header": []}
    current_section = "header"
    active_combined_targets: Optional[List[str]] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched_section = None
        matched_combined = None

        # Check for section heading if line is short and not a bullet point
        is_bullet = bool(re.match(r"^[\s●•*–—]\s*|^\s*-\s+", stripped))
        word_count = len(stripped.split())

        # Sub-category lines inside skills or lines with a colon followed by content
        # (e.g., "Languages: Java, Python, C, C++, SQL") are category content, NEVER section headings!
        has_colon_content = bool(re.search(r":\s*\S+", stripped))
        is_skill_subcat = any(re.match(pref, stripped) for pref in SKILL_SUBCATEGORY_PREFIXES)

        is_heading_candidate = (
            not is_bullet
            and word_count <= 7
            and not has_colon_content
            and not (current_section == "skills" and is_skill_subcat)
        )

        if is_heading_candidate:
            # 1. Check combined heading patterns first
            for comb_pat, targets in COMBINED_SECTION_PATTERNS:
                if re.search(comb_pat, stripped):
                    matched_combined = targets
                    break

            # 2. Check canonical single section patterns
            if not matched_combined:
                for sec_name, pattern in SECTION_PATTERNS.items():
                    if re.match(pattern, stripped):
                        # If current section is skills, do not let a generic "languages" match break out
                        if current_section == "skills" and sec_name == "languages":
                            continue
                        matched_section = sec_name
                        break

        if matched_combined:
            active_combined_targets = matched_combined
            current_section = "_combined_" + "_".join(matched_combined)
            for tgt in matched_combined:
                if tgt not in sections:
                    sections[tgt] = []
            if current_section not in sections:
                sections[current_section] = []
        elif matched_section:
            active_combined_targets = None
            current_section = matched_section
            if current_section not in sections:
                sections[current_section] = []
        else:
            # Route line to active section(s)
            if active_combined_targets:
                norm_line = stripped.lower()
                is_cert_line = bool(re.search(
                    r"(?i)\b(?:certif|certified|in progress|udemy|coursera|license|architect|associate|masterclass|course|credential)\b",
                    norm_line
                ))
                is_achieve_line = bool(re.search(
                    r"(?i)\b(?:hackathon|award|winner|won|ranking|ranked|1st|2nd|3rd|place|team of|participant|mvp|competition|contest|freshers)\b",
                    norm_line
                ))

                if is_cert_line and not is_achieve_line:
                    if "certifications" in sections:
                        sections["certifications"].append(stripped)
                elif is_achieve_line and not is_cert_line:
                    if "achievements" in sections:
                        sections["achievements"].append(stripped)
                else:
                    for tgt in active_combined_targets:
                        if tgt in sections:
                            sections[tgt].append(stripped)

                if current_section in sections:
                    sections[current_section].append(stripped)
            else:
                sections[current_section].append(stripped)

    return {k: "\n".join(v) for k, v in sections.items() if v}

def detect_sections(raw_text: str) -> Dict[str, bool]:
    """Returns boolean flags indicating which sections were detected."""
    segmented = segment_resume_sections(raw_text)
    return {
        "skills": "skills" in segmented and bool(segmented["skills"].strip()),
        "experience": "experience" in segmented and bool(segmented["experience"].strip()),
        "projects": "projects" in segmented and bool(segmented["projects"].strip()),
        "education": "education" in segmented and bool(segmented["education"].strip()),
        "certifications": "certifications" in segmented and bool(segmented["certifications"].strip()),
        "achievements": "achievements" in segmented and bool(segmented["achievements"].strip()),
    }

from app.services.privacy_guard import privacy_guard

def extract_pdf_pages_with_provenance(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts page text with page number, character offset, and layout provenance.
    Employs adaptive x_tolerance to prevent font kerning text-merging and provides OCR fallback.
    """
    pages_data = []
    current_offset = 0
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                raise HTTPException(status_code=400, detail="The uploaded PDF contains no readable pages.")
            for idx, page in enumerate(pdf.pages):
                # 1. Primary extraction with x_tolerance=2 for natural word boundary preservation
                page_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                
                # Check for word-spacing quality: if average word length is suspiciously high (>15)
                # or space ratio is too low, fallback to other extraction modes
                words = page_text.split()
                if words and (sum(len(w) for w in words) / len(words) > 15):
                    fallback_text = page.extract_text(layout=True) or page.extract_text() or ""
                    if fallback_text:
                        page_text = fallback_text

                # 2. Optional OCR fallback for scanned/empty text layers
                if not page_text.strip() or len(page_text.strip()) < 20:
                    try:
                        import pytesseract
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(file_bytes, first_page=idx+1, last_page=idx+1)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0])
                            if ocr_text.strip():
                                page_text = ocr_text
                    except Exception:
                        pass  # Keep OCR optional and gracefully fall back

                cleaned_page = clean_and_normalize_text(page_text)
                if cleaned_page:
                    start_char = current_offset
                    end_char = start_char + len(cleaned_page)
                    pages_data.append({
                        "page_number": idx + 1,
                        "text": cleaned_page,
                        "char_start": start_char,
                        "char_end": end_char,
                        "line_count": len(cleaned_page.split("\n"))
                    })
                    current_offset = end_char + 2  # account for \n\n delimiter
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {str(exc)}")

    if not pages_data:
        raise HTTPException(
            status_code=400,
            detail="Could not extract readable text. The PDF might be scanned or image-only."
        )

    return pages_data

def extract_pdf_content(file_bytes: bytes) -> str:
    """Extracts and concatenates page text using pdfplumber."""
    pages_data = extract_pdf_pages_with_provenance(file_bytes)
    return "\n\n".join(p["text"] for p in pages_data)

async def process_and_store_resume(
    file: UploadFile,
    db: Session,
    user_id: Optional[UUID] = None
) -> Tuple[Resume, Dict[str, bool]]:
    """Validates file constraints, screens privacy, extracts text, detects sections, and saves to PostgreSQL."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files (.pdf) are allowed.")

    content = await file.read()
    max_size_bytes = 10 * 1024 * 1024

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > max_size_bytes:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

    raw_text = extract_pdf_content(content)

    # =========================================================================
    # PHASE 1: PRIVACY SANITIZATION GATE
    # Sanitize extracted text immediately before database persistence,
    # section detection, chunking, embedding, entity extraction, or LLM calls.
    # Replaces sensitive personal identifiers (emails, phone numbers,
    # LinkedIn URLs, campus IDs) with safe standardized redaction markers.
    # Accepts valid resumes by continuing normal processing on sanitized text.
    # =========================================================================
    sanitized_text, privacy_meta = privacy_guard.sanitize_text(raw_text)

    detected = detect_sections(sanitized_text)

    new_resume = Resume(
        filename=file.filename,
        raw_text=sanitized_text,
        user_id=user_id
    )
    db.add(new_resume)
    db.commit()
    db.refresh(new_resume)

    return new_resume, detected