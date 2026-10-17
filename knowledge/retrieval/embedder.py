"""Text embedders.

* MiniLMEmbedder  - sentence-transformers all-MiniLM-L6-v2 (same model the backend's Chroma index uses).
* TfidfEmbedder   - dependency-light numpy fallback, used automatically when MiniLM cannot be loaded
                    (no internet / package missing). Retrieval still works, just less "semantic".
Select with env EMBEDDER=auto|minilm|tfidf (default auto).
"""
import math
import os
import re
from collections import Counter
from typing import List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9#]+")
_STOP = set("""the a an and or of to in on is are be for with by at as it its this that from if any must should
how do does did what when which who whom why where i we you they he she can could would not no my our your their there here than then
into over under after before between per about all each up out so too very also just me us""".split())


def content_stems(text: str) -> set:
    """Distinct meaningful word stems (stop words removed, 5-letter stems for long words)."""
    return {(t[:5] if len(t) > 4 else t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _tokens(text: str) -> List[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        out.append(t)
        if len(t) > 4:                       # cheap stemming: wear/wears, crack/cracks/cracking share a stem
            out.append(t[:5])
    return out


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vocab, self.idf = {}, None

    def fit(self, texts: List[str]):
        df = Counter()
        for t in texts:
            df.update(set(_tokens(t)))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)], dtype=np.float32)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w, c in Counter(_tokens(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    mat[r, j] = (1 + math.log(c)) * self.idf[j]
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norm, 1e-9)


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def build_embedders():
    """Return (dense_or_None, sparse). Sparse TF-IDF is always available and is mixed in as a keyword signal."""
    mode = os.getenv("EMBEDDER", "auto").lower()
    dense = None
    if mode in ("auto", "minilm"):
        try:
            dense = MiniLMEmbedder()
        except Exception as exc:  # offline, not installed, ...
            if mode == "minilm":
                raise
            print(f"[retrieval] MiniLM unavailable ({type(exc).__name__}); using TF-IDF only.")
    return dense, TfidfEmbedder()
