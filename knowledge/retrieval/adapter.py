"""Drop-in for the backend agent's retriever (backend/core.py::retrieve).

core.retrieve(query, n_results) returns [{"manual", "page", "text"}]. This returns the same shape from the REAL
manuals/SOPs in data/raw instead of the 3 hard-coded SOP strings, so the agent's citations point to real pages:

    from knowledge.retrieval.adapter import retrieve      # in place of the Chroma retrieve()
"""
from typing import List

from .knowledge_base import get_store
from .query import expand_query


def retrieve(query: str, n_results: int = 2) -> List[dict]:
    return [{"manual": h["source"], "page": h["page"], "text": h["chunk"]}
            for h in get_store().search(expand_query(query), top_k=n_results)]
