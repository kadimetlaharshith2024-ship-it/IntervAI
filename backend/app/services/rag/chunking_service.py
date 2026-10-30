import re
import hashlib
from typing import List, Optional
from app.services.resume_service import segment_resume_sections
from app.core.canonicalizer import strip_formatting_artifacts, clean_project_title
from app.services.privacy_guard import privacy_guard

TECH_STACK_PREFIXES = [
    r"(?i)^tech(?:nology)?\s*stack\s*:",
    r"(?i)^technologies\s*(?:used)?\s*:",
    r"(?i)^tools\s*(?:&|and)?\s*technologies\s*:",
    r"(?i)^environment\s*:",
    r"(?i)^core\s*stack\s*:"
]

class ChunkItem:
    def __init__(self, chunk_id: str, section: str, project: Optional[str], text: str, char_start: int, char_end: int):
        self.chunk_id = chunk_id
        self.section = section
        self.project = project
        self.text = text
        self.char_start = char_start
        self.char_end = char_end

class ChunkingService:
    @staticmethod
    def generate_chunk_id(resume_id: str, section: str, text: str, index: int) -> str:
        h = hashlib.sha256(f"{resume_id}_{section}_{index}_{text[:50]}".encode()).hexdigest()[:12]
        clean_sec = re.sub(r"\W+", "_", section.lower())
        return f"chk_{clean_sec}_{h}"

    def chunk_resume(self, resume_id: str, raw_text: str) -> List[ChunkItem]:
        # Sanitize raw resume text at start of processing before segmenting and chunking
        clean_raw_text, _ = privacy_guard.sanitize_text(raw_text or "")
        segmented = segment_resume_sections(clean_raw_text)
        chunks: List[ChunkItem] = []
        global_idx = 0

        for sec_name, sec_text in segmented.items():
            # Drop header and contact blocks entirely from chunk extraction
            if sec_name.lower() in ("header", "contact") or sec_name.lower().startswith("_combined_"):
                continue

            sec_title = sec_name.title()

            if sec_name == "projects":
                lines = [l.strip() for l in sec_text.split("\n") if l.strip()]
                current_project: Optional[str] = None
                current_block_lines: List[str] = []

                for i, line in enumerate(lines):
                    is_bullet = bool(re.match(r"^[\s●•*–]\s*|^\s*-\s+", line))
                    is_tech = any(re.search(pref, line) for pref in TECH_STACK_PREFIXES)

                    potential_proj = None
                    if not is_bullet and not is_tech:
                        # Split strictly on pipes, en/em dashes, double hyphens, or spaced hyphens.
                        # NEVER split on single intra-word hyphens (e.g. Real-Time, Event-Triggered).
                        if any(sep in line for sep in ["|", "–", "—", " -- "]) or re.search(r"\s+-\s+", line):
                            parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*[\–\—]\s*|\s+--\s+|\s+-\s+", line) if p.strip()]
                            for p in parts:
                                c = clean_project_title(p)
                                if c:
                                    potential_proj = c
                                    break
                        elif i + 1 < len(lines) and any(re.search(pref, lines[i + 1]) for pref in TECH_STACK_PREFIXES):
                            potential_proj = clean_project_title(line)
                        else:
                            potential_proj = clean_project_title(line)

                    if potential_proj and potential_proj != current_project:
                        if current_block_lines:
                            block_content = "\n".join(current_block_lines)
                            cid = self.generate_chunk_id(resume_id, sec_title, block_content, global_idx)
                            chunks.append(ChunkItem(cid, sec_title, current_project, block_content, 0, len(block_content)))
                            global_idx += 1
                            current_block_lines = []
                        current_project = potential_proj
                        current_block_lines.append(line)
                    else:
                        current_block_lines.append(line)

                if current_block_lines:
                    block_content = "\n".join(current_block_lines)
                    cid = self.generate_chunk_id(resume_id, sec_title, block_content, global_idx)
                    chunks.append(ChunkItem(cid, sec_title, current_project, block_content, 0, len(block_content)))
                    global_idx += 1

            elif sec_name == "experience":
                # Experience section represents employment history, NOT projects.
                # Project attribute must remain None so employers are not classified as candidate projects.
                lines = [l.strip() for l in sec_text.split("\n") if l.strip()]
                current_block_lines: List[str] = []

                for i, line in enumerate(lines):
                    is_bullet = bool(re.match(r"^[\s●•*–]\s*|^\s*-\s+", line))
                    is_new_role = (
                        not is_bullet and
                        (any(sep in line for sep in ["|", "–", "—", " -- "]) or re.search(r"\s+-\s+", line)) and
                        bool(re.search(r"(?i)\b(?:intern|engineer|developer|lead|architect|manager|consultant|analyst|specialist|\d{4}|present)\b", line))
                    )

                    if is_new_role and current_block_lines:
                        block_content = "\n".join(current_block_lines)
                        cid = self.generate_chunk_id(resume_id, sec_title, block_content, global_idx)
                        chunks.append(ChunkItem(cid, sec_title, None, block_content, 0, len(block_content)))
                        global_idx += 1
                        current_block_lines = [line]
                    else:
                        current_block_lines.append(line)

                if current_block_lines:
                    block_content = "\n".join(current_block_lines)
                    cid = self.generate_chunk_id(resume_id, sec_title, block_content, global_idx)
                    chunks.append(ChunkItem(cid, sec_title, None, block_content, 0, len(block_content)))
                    global_idx += 1

            else:
                items = [l.strip() for l in sec_text.split("\n") if len(l.strip()) > 8]
                if items:
                    for item in items:
                        cleaned = strip_formatting_artifacts(item)
                        cid = self.generate_chunk_id(resume_id, sec_title, cleaned, global_idx)
                        chunks.append(ChunkItem(cid, sec_title, None, cleaned, 0, len(cleaned)))
                        global_idx += 1

        return chunks

chunking_service = ChunkingService()