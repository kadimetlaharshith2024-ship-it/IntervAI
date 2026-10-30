from typing import List, Dict, Any, Set
from app.schemas.rag import EvidenceItem

try:
    import tiktoken
except ImportError:
    tiktoken = None

class ContextAssembler:
    def __init__(self, token_budget: int = 2048):
        self.token_budget = token_budget
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def estimate_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return max(1, int(len(text) / 4))

    def assemble(
        self,
        resume_id: str,
        query: str,
        evidence_items: List[EvidenceItem],
        claims: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        allocated_tokens = 0
        selected_evidence: List[EvidenceItem] = []
        discarded_evidence: List[Dict[str, str]] = []

        # 1. Budget-bound selection of verified evidence blocks
        for item in evidence_items:
            tokens = self.estimate_tokens(item.text)
            if allocated_tokens + tokens <= self.token_budget:
                selected_evidence.append(item)
                allocated_tokens += tokens
            else:
                discarded_evidence.append({
                    "evidence_id": item.evidence_id,
                    "reason": "token_budget_exceeded"
                })

        # 2. Extract entity and project footprints from selected evidence
        active_footprint: Set[str] = set()
        for ev in selected_evidence:
            if ev.project:
                active_footprint.add(ev.project.lower())
            for token in ev.text.split():
                clean_t = token.strip(".,;:()[]{}").lower()
                if len(clean_t) >= 3:
                    active_footprint.add(clean_t)

        # 3. Filter claims: only retain claims whose entity, project, or text intersect with active footprint
        relevant_claims: List[Dict[str, Any]] = []
        for c in claims:
            c_text_lower = c.get("text", "").lower()
            c_proj = (c.get("project") or "").lower()
            c_target = (c.get("target_entity") or "").lower()

            if (c_proj and c_proj in active_footprint) or \
               (c_target and c_target in active_footprint) or \
               any(term in active_footprint for term in c_text_lower.split() if len(term) >= 4):
                relevant_claims.append(c)

        # 4. Filter relationships: only retain relationships anchored in active footprint
        relevant_relationships: List[Dict[str, Any]] = []
        active_projects = {ev.project.lower() for ev in selected_evidence if ev.project}
        for r in relationships:
            src = r.get("source", "").lower()
            tgt = r.get("target", "").lower()
            if src in active_footprint or tgt in active_footprint or any(p in src or p in tgt for p in active_projects):
                relevant_relationships.append(r)

        # Prioritize project-specific relationships over generic candidate personal attributes
        def _rel_rank(rel_dict: Dict[str, Any]) -> int:
            s = rel_dict.get("source", "").lower()
            t = rel_dict.get("target", "").lower()
            if any(p in s or p in t for p in active_projects):
                return 2
            if s != "candidate":
                return 1
            return 0

        relevant_relationships.sort(key=_rel_rank, reverse=True)

        return {
            "resume_id": str(resume_id),
            "query": query,
            "selected_evidence_items": [ev.model_dump() for ev in selected_evidence],
            "discarded_evidence": discarded_evidence,
            "claims": relevant_claims[:8],
            "relationships": relevant_relationships[:15],
            "token_count": allocated_tokens
        }

context_assembler = ContextAssembler()