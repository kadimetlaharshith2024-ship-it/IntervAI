import os
import math
from typing import List
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.encoder = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        try:
            from fastembed import TextEmbedding
            self.encoder = TextEmbedding(model_name=self.model_name)
        except Exception:
            self.encoder = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if not self.encoder:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' is unavailable. "
                "Ensure required embedding dependencies (e.g. fastembed) are installed and the model is loaded. "
                "Deterministic pseudo-embedding fallbacks are disabled in production."
            )

        try:
            # fastembed returns a generator of unit-normalized numpy arrays
            embeddings = list(self.encoder.embed(texts, batch_size=self.batch_size))
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {str(exc)}") from exc

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

embedding_service = EmbeddingService()