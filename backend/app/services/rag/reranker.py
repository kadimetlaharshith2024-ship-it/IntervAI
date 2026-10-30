from typing import List, Optional
from app.schemas.rag import EvidenceItem
from app.services.rag.retrievers import TECHNICAL_RELATION_TYPES, NON_TECHNICAL_RELATIONS

SECTION_IMPORTANCE_MAP = {
    "projects": 0.45,
    "experience": 0.40,
    "work experience": 0.40,
    "technical skills": 0.30,
    "skills": 0.30,
    "certifications": 0.15,
    "education": 0.15,
    "achievements": 0.10,
    "languages": -0.90,
    "hobbies": -0.90,
    "personal": -0.90
}

class DynamicStructuralRelevanceReranker:
    """
    Reranks multi-modal candidates by combining retrieval fusion scores with
    structural section hierarchy, graph topology, and target alignment.
    """
    @staticmethod
    def calculate_interview_relevance_score(
        item: EvidenceItem,
        target_entity: Optional[str] = None,
        target_project: Optional[str] = None
    ) -> float:
        score = item.score
        sec_lower = item.section.lower().strip()
        text_lower = item.text.lower()

        # Disqualify non-technical personal relationship nodes
        if item.source_type == "relationship":
            if any(f" {non_tech} " in item.text for non_tech in NON_TECHNICAL_RELATIONS):
                return 0.0

        for sec_name, weight in SECTION_IMPORTANCE_MAP.items():
            if sec_name in sec_lower:
                score += weight
                break

        for tech_rel in TECHNICAL_RELATION_TYPES:
            if tech_rel in item.text:
                score += 0.25
                break

        if target_project and item.project and target_project.lower() in item.project.lower():
            score += 0.30

        if target_entity:
            t_low = target_entity.lower()
            if t_low in text_lower:
                score += 0.35
            else:
                from app.core.canonicalizer import ALIAS_INDEX, CANONICAL_TAXONOMY
                canon = ALIAS_INDEX.get(t_low, target_entity)
                aliases = CANONICAL_TAXONOMY.get(canon, {}).get("aliases", [])
                if any(a.lower() in text_lower for a in aliases):
                    score += 0.35

        if len(item.text.strip()) < 20:
            score -= 0.20

        return round(max(0.0, min(score, 1.0)), 4)

    def rerank(
        self,
        query: str,
        items: List[EvidenceItem],
        target_entity: Optional[str] = None,
        target_project: Optional[str] = None
    ) -> List[EvidenceItem]:
        filtered_items: List[EvidenceItem] = []

        for item in items:
            rel_score = self.calculate_interview_relevance_score(
                item, target_entity=target_entity, target_project=target_project
            )
            item.interview_relevance_score = rel_score
            if rel_score >= 0.20:
                filtered_items.append(item)

        filtered_items.sort(key=lambda x: x.interview_relevance_score, reverse=True)
        return filtered_items

reranker = DynamicStructuralRelevanceReranker()