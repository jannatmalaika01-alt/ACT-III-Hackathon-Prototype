"""Citation checks: does every citation point at real text in a real source document?

    from knowledge.verify import verify_contract, verify_citation
    problems = verify_contract(contract)      # [] means every cited chunk is verbatim in its PDF page

* `verify_citation(manual, page)` - the manual exists in data/raw and has that page (used for the agent's final answer).
* `verify_evidence(evidence)`     - each chunk's text really appears on the cited page (whitespace-insensitive).
* `verify_contract(contract)`     - evidence rules + the recommended action is copied from cited evidence (or is flagged).
"""
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

from .ingestion.pdf_parser import PDFIngestor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
NOT_FOUND_MARK = "manual inspection required"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


@lru_cache(maxsize=64)
def _pages(path: str) -> Dict[int, str]:
    return {p["page"]: _norm(p["text"]) for p in PDFIngestor().parse_pdf(path)}


def _source_path(source: str, raw_dir: str) -> Optional[str]:
    name = os.path.basename(source or "")          # never follow ../ in a citation
    path = os.path.join(raw_dir, name)
    return path if name and os.path.isfile(path) and name.lower().endswith(".pdf") else None


def verify_citation(manual: str, page, raw_dir: str = RAW_DIR) -> List[str]:
    path = _source_path(manual, raw_dir)
    if path is None:
        return [f"source '{manual}' is not a document in the knowledge base"]
    try:
        page = int(page)
    except (TypeError, ValueError):
        return [f"page {page!r} is not a number"]
    if page not in _pages(path):
        return [f"{manual} has no page {page} (pages: 1-{len(_pages(path))})"]
    return []


def verify_evidence(evidence: List[dict], raw_dir: str = RAW_DIR) -> List[str]:
    problems = []
    for n, e in enumerate(evidence):
        bad = verify_citation(e.get("source"), e.get("page"), raw_dir)
        if bad:
            problems += [f"evidence[{n}]: {b}" for b in bad]
            continue
        page_text = _pages(_source_path(e["source"], raw_dir))[int(e["page"])]
        if not _norm(e.get("chunk")) or _norm(e["chunk"]) not in page_text:
            problems.append(f"evidence[{n}]: chunk text is not on {e['source']} page {e['page']}")
    return problems


def verify_contract(contract: dict, raw_dir: str = RAW_DIR) -> List[str]:
    problems = verify_evidence(contract.get("evidence") or [], raw_dir)
    defect = (contract.get("defect") or {}).get("type", "")
    rec = contract.get("recommendation") or {}
    action = _norm(rec.get("action"))
    if defect != "normal" and not contract.get("evidence") and NOT_FOUND_MARK not in action:
        problems.append("defect reported with no evidence and the recommendation does not say evidence is missing")
    if defect != "normal" and contract.get("evidence") and NOT_FOUND_MARK not in action:
        if not any(action and action in _norm(e["chunk"]) for e in contract["evidence"]):
            problems.append("recommended action is not copied from any cited evidence chunk")
    if rec.get("priority") not in ("low", "medium", "high", "critical"):
        problems.append(f"invalid priority {rec.get('priority')!r}")
    return problems
