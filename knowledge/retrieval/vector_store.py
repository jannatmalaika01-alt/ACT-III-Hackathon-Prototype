"""In-memory vector store with hybrid scoring (dense MiniLM when available + TF-IDF keywords).

No Chroma/torch DLLs are needed (the earlier version avoided Chroma for that reason): plain numpy cosine.
`search()` returns [{"source", "page", "chunk", "score"}]; extra `score` lets callers apply a minimum.
"""
from typing import Any, Dict, List

import numpy as np

import os

from .embedder import build_embedders, content_stems

DENSE_WEIGHT = 0.6
MIN_SHARED_TERMS = 2          # a hit must share at least this many meaningful words with the query ...
DENSE_ACCEPT = float(os.getenv("DENSE_ACCEPT", "0.40"))  # ... unless MiniLM itself says it is clearly related


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
        dense = None
        if self.dense is not None:
            dense = self._dense_vecs @ self.dense.encode([query])[0]
            scores = DENSE_WEIGHT * dense + (1 - DENSE_WEIGHT) * scores
        # Relevance gate: stop words such as "do" or "schedule" must not let an unrelated question match a manual page.
        q_stems = content_stems(query)
        hits = []
        for i in np.argsort(-scores):
            shared = len(q_stems & content_stems(self.chunks[i]["text"]))
            related = shared >= MIN_SHARED_TERMS or (self.dense is not None and dense[i] >= DENSE_ACCEPT)
            if related:
                hits.append({"source": self.chunks[i]["source"], "page": self.chunks[i]["page"],
                             "chunk": self.chunks[i]["text"], "score": round(float(scores[i]), 4)})
            if len(hits) == top_k:
                break
        return hits
