import math
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, OrderedDict
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import or_, not_
from app.models.rag import ResumeChunk, ResumeClaim
from app.models.relationship import Relationship
from app.schemas.rag import EvidenceItem
from app.services.rag.embedding_service import embedding_service

TECHNICAL_RELATION_TYPES: Set[str] = {
    "SECURED_WITH", "AUTHENTICATED_WITH", "CONTAINERIZED_WITH",
    "DEPLOYED_ON", "BUILT_WITH", "DEVELOPED", "IMPLEMENTED_WITH",
    "INTEGRATES_WITH", "USES", "HAS_TECHNOLOGY", "CONNECTS_TO",
    "OPTIMIZED_WITH", "MONITORS", "ANALYZES", "COMPILED_WITH"
}

NON_TECHNICAL_RELATIONS: Set[str] = {
    "SPEAKS", "HOBBY", "INTEREST", "PERSONAL", "LOCATED_IN"
}

NON_TECHNICAL_SECTIONS: List[str] = ["language", "hobb", "personal", "interest", "contact"]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

from dataclasses import dataclass

@dataclass
class CachedResumeClaimDTO:
    """Plain data transfer object representing a verified resume claim free of ORM session attachment."""
    claim_id: str
    resume_id: Any = None
    claim_text: str = ""
    project: Optional[str] = None
    target_entity: Optional[str] = None
    source_section: Optional[str] = None
    confidence: Optional[float] = None
    evidence_chunk_id: Optional[str] = None

    @classmethod
    def from_orm(cls, obj: Any) -> "CachedResumeClaimDTO":
        if isinstance(obj, cls):
            return obj
        return cls(
            claim_id=str(getattr(obj, "claim_id", "") or ""),
            resume_id=getattr(obj, "resume_id", None),
            claim_text=str(getattr(obj, "claim_text", "") or ""),
            project=getattr(obj, "project", None),
            target_entity=getattr(obj, "target_entity", None),
            source_section=getattr(obj, "source_section", None),
            confidence=getattr(obj, "confidence", None),
            evidence_chunk_id=getattr(obj, "evidence_chunk_id", None),
        )


@dataclass
class CachedRelationshipDTO:
    """Plain data transfer object representing a verified graph relationship free of ORM session attachment."""
    id: Any = None
    resume_id: Any = None
    source_entity: str = ""
    relation_type: str = ""
    target_entity: str = ""
    context_entity: Optional[str] = None
    section: Optional[str] = None
    evidence_text: Optional[str] = None
    confidence: float = 0.5
    extraction_method: Optional[str] = None

    @classmethod
    def from_orm(cls, obj: Any) -> "CachedRelationshipDTO":
        if isinstance(obj, cls):
            return obj
        raw_conf = getattr(obj, "confidence", 0.5)
        try:
            conf_val = float(raw_conf) if raw_conf is not None else 0.5
        except (ValueError, TypeError):
            conf_val = 0.5
        return cls(
            id=getattr(obj, "id", None),
            resume_id=getattr(obj, "resume_id", None),
            source_entity=str(getattr(obj, "source_entity", "") or ""),
            relation_type=str(getattr(obj, "relation_type", "") or ""),
            target_entity=str(getattr(obj, "target_entity", "") or ""),
            context_entity=getattr(obj, "context_entity", None),
            section=getattr(obj, "section", None),
            evidence_text=getattr(obj, "evidence_text", None),
            confidence=conf_val,
            extraction_method=getattr(obj, "extraction_method", None),
        )


@dataclass
class CachedResumeChunkDTO:
    """Plain data transfer object representing a parsed resume text chunk free of ORM session attachment."""
    id: Any = None
    resume_id: Any = None
    chunk_id: str = ""
    chunk_text: str = ""
    section: Optional[str] = None
    project: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    @classmethod
    def from_orm(cls, obj: Any) -> "CachedResumeChunkDTO":
        if isinstance(obj, cls):
            return obj
        return cls(
            id=getattr(obj, "id", None),
            resume_id=getattr(obj, "resume_id", None),
            chunk_id=str(getattr(obj, "chunk_id", "") or ""),
            chunk_text=str(getattr(obj, "chunk_text", "") or ""),
            section=getattr(obj, "section", None),
            project=getattr(obj, "project", None),
            embedding=getattr(obj, "embedding", None),
            embedding_model=getattr(obj, "embedding_model", None),
        )


class CachedResumeData:
    """Stores in-memory pre-processed retrieval artifacts for a single resume."""
    def __init__(
        self,
        resume_id: Any,
        chunks: List[Any],
        relationships: List[Any],
        claims: List[Any],
        tokenized_docs: List[List[str]],
        doc_term_counts: List[Counter],
        doc_lens: List[int],
        df_counts: Dict[str, int],
        avgdl: float,
        chunk_embeddings_matrix: Optional[np.ndarray],
        chunk_norms: Optional[np.ndarray]
    ):
        self.resume_id = str(resume_id)
        self.created_at = time.time()
        self.chunks: List[CachedResumeChunkDTO] = [CachedResumeChunkDTO.from_orm(c) for c in chunks] if chunks else []
        self.relationships: List[CachedRelationshipDTO] = [CachedRelationshipDTO.from_orm(r) for r in relationships] if relationships else []
        self.claims: List[CachedResumeClaimDTO] = [CachedResumeClaimDTO.from_orm(c) for c in claims] if claims else []
        self.allowed_projects = list(set([c.project for c in self.chunks if getattr(c, "project", None)]))
        allowed_ents = set()
        for r in self.relationships:
            if getattr(r, "target_entity", None):
                allowed_ents.add(r.target_entity)
            if getattr(r, "source_entity", None) and str(r.source_entity).lower() != "candidate":
                allowed_ents.add(r.source_entity)
        for c in self.claims:
            if getattr(c, "target_entity", None):
                allowed_ents.add(c.target_entity)
        self.allowed_entities = sorted(list(allowed_ents))
        self.claims_map = {c.claim_id: c for c in self.claims if getattr(c, "claim_id", None)}
        self.rels_map = {f"rel_{str(r.id)[:8]}": r for r in self.relationships if getattr(r, "id", None)}

        self.tokenized_docs = tokenized_docs
        self.doc_term_counts = doc_term_counts
        self.doc_lens = doc_lens
        self.df_counts = df_counts
        self.avgdl = avgdl
        self.chunk_embeddings_matrix = chunk_embeddings_matrix
        self.chunk_norms = chunk_norms

class ResumeRetrievalCache:
    """
    In-memory bounded LRU cache with TTL for resume retrieval structures.
    Caches tokenized BM25 statistics, numpy matrix embeddings, and graph entities.
    """
    def __init__(self, max_size: int = 50, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CachedResumeData] = OrderedDict()

    def get(self, resume_id: Any) -> Optional[CachedResumeData]:
        key = str(resume_id)
        if key not in self._cache:
            return None
        data = self._cache[key]
        if (time.time() - data.created_at) > self.ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return data

    def set(self, resume_id: Any, data: CachedResumeData) -> None:
        key = str(resume_id)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = data
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self, resume_id: Any) -> None:
        key = str(resume_id)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

resume_retrieval_cache = ResumeRetrievalCache()

def _is_non_technical(section: Optional[str]) -> bool:
    if not section:
        return False
    sec_lower = str(section).lower()
    return any(s in sec_lower for s in NON_TECHNICAL_SECTIONS)

def get_or_build_resume_cache(resume_id: Any, db: Session) -> CachedResumeData:
    cached = resume_retrieval_cache.get(resume_id)
    if cached is not None:
        return cached

    chunks = db.query(ResumeChunk).filter(ResumeChunk.resume_id == resume_id).all()
    relationships = db.query(Relationship).filter(Relationship.resume_id == resume_id).all()
    claims = db.query(ResumeClaim).filter(ResumeClaim.resume_id == resume_id).all()

    # Pre-tokenize docs and compute BM25 corpus statistics
    tokenized_docs = [BM25LexicalRetriever.tokenize(c.chunk_text or "") for c in chunks]
    doc_lens = [len(d) for d in tokenized_docs]
    doc_term_counts = [Counter(d) for d in tokenized_docs]
    df_counts: Dict[str, int] = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            df_counts[term] += 1
    avgdl = sum(doc_lens) / max(len(chunks), 1)

    # Pre-assemble float32 numpy matrix and norms for vector cosine similarity
    matrix = None
    norms = None
    if chunks:
        try:
            valid_rows = []
            for c in chunks:
                if getattr(c, "embedding", None) and isinstance(c.embedding, list):
                    valid_rows.append(c.embedding)
                else:
                    valid_rows.append([0.0] * 384)
            matrix = np.array(valid_rows, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1)
        except Exception:
            matrix = None
            norms = None

    data = CachedResumeData(
        resume_id=resume_id,
        chunks=chunks,
        relationships=relationships,
        claims=claims,
        tokenized_docs=tokenized_docs,
        doc_term_counts=doc_term_counts,
        doc_lens=doc_lens,
        df_counts=df_counts,
        avgdl=avgdl,
        chunk_embeddings_matrix=matrix,
        chunk_norms=norms
    )
    resume_retrieval_cache.set(resume_id, data)
    return data

class VectorRetriever:
    @staticmethod
    def retrieve(
        resume_id,
        query_embedding: List[float],
        db: Session,
        top_k: int = 5,
        project_filter: Optional[str] = None
    ) -> List[Tuple[ResumeChunk, float]]:
        cached = get_or_build_resume_cache(resume_id, db)
        chunks = cached.chunks
        if not chunks:
            return []

        # Model consistency check: Ensure stored chunks have valid and matching embedding model metadata
        incompatible_chunks = [
            chk for chk in chunks
            if not getattr(chk, "embedding_model", None) or str(chk.embedding_model).strip() != embedding_service.model_name
        ]
        if incompatible_chunks:
            found_identifiers = sorted(list({
                str(chk.embedding_model).strip() if (getattr(chk, "embedding_model", None) and str(chk.embedding_model).strip()) else "<unspecified/missing>"
                for chk in incompatible_chunks
            }))
            models_str = ", ".join(found_identifiers)
            raise ValueError(
                f"Resume '{resume_id}' contains chunks with incompatible or missing embedding model metadata: [{models_str}]. "
                f"Active embedding model is '{embedding_service.model_name}'. "
                f"Please re-index this resume (POST /api/v1/rag/{resume_id}/index) before running retrieval."
            )

        # Filter out non-technical sections
        tech_indices = [i for i, c in enumerate(chunks) if not _is_non_technical(c.section)]
        if not tech_indices:
            return []

        # Apply project filter if requested
        candidate_indices = tech_indices
        if project_filter:
            p_filter_lower = project_filter.lower()
            proj_indices = [
                i for i in tech_indices
                if getattr(chunks[i], "project", None) and p_filter_lower in str(chunks[i].project).lower()
            ]
            if proj_indices:
                candidate_indices = proj_indices

        # Vectorized cosine similarity using numpy matrix dot product (Part A3)
        if cached.chunk_embeddings_matrix is not None and cached.chunk_norms is not None:
            try:
                q_vec = np.array(query_embedding, dtype=np.float32)
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm == 0.0:
                    return [(chunks[i], 0.0) for i in candidate_indices[:top_k]]

                sub_matrix = cached.chunk_embeddings_matrix[candidate_indices]
                sub_norms = cached.chunk_norms[candidate_indices]

                # Dot products in one matrix multiply
                dots = np.dot(sub_matrix, q_vec)
                denoms = sub_norms * q_norm
                # Avoid zero division
                sims = np.where(denoms > 0, dots / denoms, 0.0)

                scored = [(chunks[candidate_indices[idx]], float(sims[idx])) for idx in range(len(candidate_indices))]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
            except Exception:
                pass

        # Fallback to standard Python loop if numpy path fails
        scored = []
        for i in candidate_indices:
            chk = chunks[i]
            if chk.embedding:
                sim = cosine_similarity(query_embedding, chk.embedding)
                scored.append((chk, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class GraphRetriever:
    @staticmethod
    def retrieve(
        resume_id,
        target_entity: Optional[str],
        target_project: Optional[str],
        db: Session,
        top_k: int = 8
    ) -> List[Tuple[Relationship, float]]:
        cached = get_or_build_resume_cache(resume_id, db)
        rels = [r for r in cached.relationships if r.relation_type not in NON_TECHNICAL_RELATIONS]
        if not rels:
            return []

        matched = []
        if target_project and target_entity:
            p_low = target_project.lower()
            e_low = target_entity.lower()
            matched = [
                r for r in rels
                if (p_low in str(r.context_entity or "").lower() or p_low in str(r.source_entity or "").lower())
                and (e_low in str(r.target_entity or "").lower() or e_low in str(r.source_entity or "").lower())
            ]

        if not matched and (target_project or target_entity):
            if target_entity:
                e_low = target_entity.lower()
                matched = [
                    r for r in rels
                    if (e_low in str(r.target_entity or "").lower() or e_low in str(r.source_entity or "").lower())
                ]
            if not matched and target_project:
                p_low = target_project.lower()
                matched = [
                    r for r in rels
                    if (p_low in str(r.context_entity or "").lower() or p_low in str(r.source_entity or "").lower())
                ]

        if not matched:
            matched = [r for r in rels if r.relation_type in TECHNICAL_RELATION_TYPES]

        scored = []
        for r in matched:
            weight = r.confidence
            if r.relation_type in TECHNICAL_RELATION_TYPES:
                weight += 0.15
            if target_project and r.context_entity and target_project.lower() in str(r.context_entity).lower():
                weight += 0.10
            scored.append((r, min(weight, 1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class BM25LexicalRetriever:
    """
    Implements Okapi BM25 lexical ranking with inverse document frequency
    and document length normalization over resume text chunks.
    Precomputed corpus statistics and tokenization are cached per resume.
    """
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b\w{2,}\b", text or "")]

    @classmethod
    def retrieve(
        cls,
        resume_id,
        query_terms: List[str],
        db: Session,
        top_k: int = 5,
        project_filter: Optional[str] = None
    ) -> List[Tuple[ResumeChunk, float]]:
        cached = get_or_build_resume_cache(resume_id, db)
        chunks = cached.chunks
        if not chunks or not query_terms:
            return [(c, 0.1) for c in chunks[:top_k]]

        tech_indices = [i for i, c in enumerate(chunks) if not _is_non_technical(c.section)]
        if not tech_indices:
            return []

        candidate_indices = tech_indices
        if project_filter:
            p_filter_lower = project_filter.lower()
            proj_indices = [
                i for i in tech_indices
                if getattr(chunks[i], "project", None) and p_filter_lower in str(chunks[i].project).lower()
            ]
            if proj_indices:
                candidate_indices = proj_indices

        scored = []
        clean_q = [t.lower() for t in query_terms if len(t) >= 2]
        N = cached.avgdl > 0 and len(chunks) or 1
        avgdl = cached.avgdl

        for idx in candidate_indices:
            chk = chunks[idx]
            doc_len = cached.doc_lens[idx]
            doc_term_counts = cached.doc_term_counts[idx]
            bm25_score = 0.0

            for q_term in clean_q:
                if q_term not in doc_term_counts:
                    continue
                tf = doc_term_counts[q_term]
                df = cached.df_counts.get(q_term, 0)
                # Robertson-Spärck Jones IDF
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (cls.k1 + 1.0)
                denominator = tf + cls.k1 * (1.0 - cls.b + cls.b * (doc_len / max(avgdl, 1.0)))
                bm25_score += idf * (numerator / denominator)

            # Normalize to 0..1 interval via sigmoid-like scaling
            normalized_score = round(bm25_score / (bm25_score + 10.0), 4) if bm25_score > 0 else 0.0
            if normalized_score > 0.0:
                scored.append((chk, normalized_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class HybridRetriever:
    @staticmethod
    def fuse_and_rank(
        vector_res: List[Tuple[ResumeChunk, float]],
        graph_res: List[Tuple[Relationship, float]],
        lexical_res: List[Tuple[ResumeChunk, float]],
        top_k_final: int = 6,
        target_entity: Optional[str] = None
    ) -> List[EvidenceItem]:
        evidence_dict: Dict[str, EvidenceItem] = {}

        for rank, (chk, score) in enumerate(vector_res):
            rrf = 1.0 / (60 + rank + 1)
            eid = chk.chunk_id
            if eid not in evidence_dict:
                evidence_dict[eid] = EvidenceItem(
                    evidence_id=eid,
                    text=chk.chunk_text,
                    section=chk.section,
                    project=chk.project,
                    source_type="chunk",
                    score=round(score * 0.4 + rrf * 0.6, 4),
                    provenance="vector_retrieval"
                )

        for rank, (rel, score) in enumerate(graph_res):
            if rel.relation_type in NON_TECHNICAL_RELATIONS:
                continue
            rrf = 1.0 / (60 + rank + 1)
            eid = f"rel_{str(rel.id)[:8]}"
            if eid not in evidence_dict:
                evidence_dict[eid] = EvidenceItem(
                    evidence_id=eid,
                    text=f"{rel.source_entity} {rel.relation_type} {rel.target_entity} (Evidence: {rel.evidence_text})",
                    section=rel.section,
                    project=rel.context_entity,
                    source_type="relationship",
                    score=round(score * 0.5 + rrf * 0.5, 4),
                    provenance=rel.extraction_method
                )

        for rank, (chk, score) in enumerate(lexical_res):
            rrf = 1.0 / (60 + rank + 1)
            eid = chk.chunk_id
            if eid in evidence_dict:
                evidence_dict[eid].score += round(score * 0.2, 4)
            else:
                evidence_dict[eid] = EvidenceItem(
                    evidence_id=eid,
                    text=chk.chunk_text,
                    section=chk.section,
                    project=chk.project,
                    source_type="chunk",
                    score=round(score * 0.3 + rrf * 0.7, 4),
                    provenance="bm25_lexical"
                )

        items = list(evidence_dict.values())
        items.sort(key=lambda x: x.score, reverse=True)

        if target_entity and len(items) > top_k_final:
            t_low = target_entity.lower()
            top_matches = any(t_low in it.text.lower() for it in items[:top_k_final])
            if not top_matches:
                for idx in range(top_k_final, len(items)):
                    if t_low in items[idx].text.lower():
                        match_item = items.pop(idx)
                        items.insert(min(2, top_k_final - 1), match_item)
                        break

        return items[:top_k_final]