"""In-memory vector store with hybrid scoring (dense MiniLM when available + TF-IDF keywords).

No Chroma/torch DLLs are needed (the earlier version avoided Chroma for that reason): plain numpy cosine.
`search()` returns [{"source", "page", "chunk", "score"}]; extra `score` lets callers apply a minimum.
"""
from typing import Any, Dict, List

import numpy as np

from .embedder import build_embedders

DENSE_WEIGHT = 0.6


class KnowledgeVectorStore:
    def __init__(self):
        self.dense, self.sparse = build_embedders()
        self.chunks: List[Dict[str, Any]] = []
        self._dense_vecs = None
        self._sparse_vecs = None

    @property
    def embedder_name(self) -> str:
        return f"{self.dense.name}+tfidf" if self.dense else "tfidf"

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks. Re-fits the keyword model over everything indexed so far."""
        if not chunks:
            return
        self.chunks.extend(chunks)
        texts = [c["text"] for c in self.chunks]
        self.sparse.fit(texts)
        self._sparse_vecs = self.sparse.encode(texts)
        if self.dense:
            self._dense_vecs = self.dense.encode(texts)

    def __len__(self):
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        scores = self._sparse_vecs @ self.sparse.encode([query])[0]
        if self.dense is not None:
            dense = self._dense_vecs @ self.dense.encode([query])[0]
            scores = DENSE_WEIGHT * dense + (1 - DENSE_WEIGHT) * scores
        order = np.argsort(-scores)[: min(top_k, len(self.chunks))]
        return [{"source": self.chunks[i]["source"], "page": self.chunks[i]["page"],
                 "chunk": self.chunks[i]["text"], "score": round(float(scores[i]), 4)} for i in order]
