"""Check the Evolus agent's citations against the text of our own manuals.

The agent cites like this:  document: corrosion_control_guide.txt - section: Page 2: ... - quote: "Take the line out of ..."
It writes `.txt` where our files are `.pdf`, and it sometimes changes a spelling inside a quote ("pressurized" for the manual's
"pressurised"). Every quote is looked up in the manuals of the retrieval index:

  * where to look: the document with the same stem and the cited page first, then every page of that document, then every page
    of every manual;
  * how to compare: whitespace, case and punctuation are ignored;
  * result per citation:
        verified    the quote is in the manual (after that normalisation)
        corrected   no exact match but a passage is at least 90% similar (a spelling, a missing word): the quote is REPLACED by the
                    manual's exact text and the agent's own wording is kept in `agent_quote`
        unverified  nothing is that close: the agent's text is kept and it must not be trusted.
    A citation without a quote cannot be checked and counts as unverified.

Pure standard library, so it is fast to import. The manuals are only loaded (lazily, once) when a check runs.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("citations")

THRESHOLD = 0.90                      # similarity at or above this counts as "the same passage"
VERIFIED, CORRECTED, UNVERIFIED = "verified", "corrected", "unverified"
STATUS_LABEL = {VERIFIED: "verified", CORRECTED: "corrected to source", UNVERIFIED: "unverified"}
_WORD = re.compile(r"[^\W_]+")        # letters and digits; everything else (punctuation, dashes, quotes) is a separator


def normalize(text: str) -> str:
    """lower case, punctuation and whitespace removed: 'Re-measure monthly.' -> 're measure monthly'."""
    return " ".join(m.group().lower() for m in _WORD.finditer(text or ""))


def stem(name: str) -> str:
    """'corrosion_control_guide.txt' / 'Corrosion_Control_Guide.pdf' -> 'corrosion_control_guide'."""
    return re.sub(r"\.(pdf|txt|md|docx?)$", "", (name or "").strip(), flags=re.I).lower()


# ---------------- the manuals ----------------

class Passage:
    """One page of one manual: its text and a word list that remembers where each word sits in the original text."""

    def __init__(self, source: str, page: int, text: str) -> None:
        self.source, self.page, self.text = source, page, text
        self.words = [(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text)]
        self.norm = " ".join(w for w, _, _ in self.words)

    def exact_text(self, first: int, last: int) -> str:
        """The manual's own words from word `first` to word `last` (inclusive), with its punctuation, on one line."""
        start, end = self.words[first][1], self.words[last][2]
        while end < len(self.text) and self.text[end] in ".,;:!?)\"'”":   # keep the sentence's closing punctuation
            end += 1
        return " ".join(self.text[start:end].split())


def _join_chunks(chunks: List[str]) -> str:
    """Rebuild a page from its retrieval chunks. Neighbouring chunks share one sentence (their overlap); keep it once."""
    page = ""
    for chunk in chunks:
        if page:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", chunk) if s.strip()]
            if sentences and page.rstrip().endswith(sentences[0].strip()):
                chunk = chunk[chunk.index(sentences[0]) + len(sentences[0]):].lstrip()
        page = (page + " " + chunk).strip()
    return page


def passages_from_chunks(chunks: Iterable[Dict[str, Any]]) -> List[Passage]:
    pages: Dict[Tuple[str, int], List[str]] = {}
    for c in chunks:
        pages.setdefault((c["source"], int(c["page"])), []).append(c["text"])
    return [Passage(src, page, _join_chunks(texts)) for (src, page), texts in sorted(pages.items())]


_PASSAGES: Optional[List[Passage]] = None


def load_passages() -> List[Passage]:
    """The manuals of the retrieval index (backend/core.py searches the same index), built once."""
    global _PASSAGES
    if _PASSAGES is None:
        import core  # noqa: F401  (puts the repo root on sys.path so `knowledge` can be imported)
        from knowledge.retrieval.knowledge_base import get_store
        _PASSAGES = passages_from_chunks(get_store().chunks)
    return _PASSAGES


# ---------------- reading one citation ----------------

def parse_item(item: Any) -> Dict[str, Any]:
    """{document, page, quote, raw} from the agent's citation line:  document: X - section: Page 2: Title - quote: "..." """
    if isinstance(item, dict):
        raw = " ".join(f"{k}: {v}" for k, v in item.items())
        page = item.get("page")
        return {"document": item.get("document") or item.get("source"), "page": int(page) if str(page or "").isdigit() else None,
                "quote": (item.get("quote") or None), "raw": raw}
    raw = str(item or "")
    dash = r"\s+[—–-]\s+"
    doc = re.search(r"document:\s*(.+?)(?:" + dash + r"section:|" + dash + r"quote:|$)", raw, re.I)
    section = re.search(r"section:\s*(.+?)(?:" + dash + r"quote:|$)", raw, re.I)
    quote = re.search(r"quote:\s*[\"“]?(.+?)[\"”]?\s*$", raw, re.I | re.S)
    page = re.search(r"page\s*(\d+)", (section.group(1) if section else raw), re.I)
    document = doc.group(1).strip() if doc else None
    if document is None:  # a plain "corrosion_control_guide.pdf, page 2"
        plain = re.search(r"([\w.-]+\.(?:pdf|txt))", raw, re.I)
        document = plain.group(1) if plain else None
    return {"document": document, "page": int(page.group(1)) if page else None,
            "quote": quote.group(1).strip() if quote else None, "raw": raw}


# ---------------- matching ----------------

def _best_in(passage: Passage, qnorm: str, threshold: float) -> Tuple[float, Optional[Tuple[int, int]]]:
    """(similarity, (first_word, last_word)) of the passage's best-matching stretch of words; (1.0, ...) when it contains the quote."""
    qwords = qnorm.split()
    n, words = len(qwords), [w for w, _, _ in passage.words]
    if n == 0 or not words:
        return 0.0, None
    for i in range(0, len(words) - n + 1):          # an exact stretch of words (after normalising)
        if words[i:i + n] == qwords:
            return 1.0, (i, i + n - 1)
    best, span = 0.0, None
    for length in range(max(1, n - 3), n + 4):       # a stretch of about the same length: a missing or extra word is allowed
        for i in range(0, len(words) - length + 1):
            window = " ".join(words[i:i + length])
            sm = difflib.SequenceMatcher(None, qnorm, window, autojunk=False)
            if sm.real_quick_ratio() < max(best, threshold - 0.05) or sm.quick_ratio() < max(best, threshold - 0.05):
                continue
            r = sm.ratio()
            if r > best:
                best, span = r, (i, i + length - 1)
    return best, span


def verify_citation(item: Any, passages: Optional[List[Passage]] = None, threshold: float = THRESHOLD) -> Dict[str, Any]:
    """Check ONE citation. Returns {document (.pdf name), page, quote, status, similarity, raw, agent_quote?, claimed?, note?}."""
    passages = load_passages() if passages is None else passages
    c = parse_item(item)
    claimed_doc, claimed_page = c["document"], c["page"]
    known = {stem(p.source): p.source for p in passages}
    doc_pdf = known.get(stem(claimed_doc or ""))
    out: Dict[str, Any] = {"document": doc_pdf or claimed_doc, "page": claimed_page, "quote": c["quote"], "status": UNVERIFIED,
                           "similarity": 0.0, "raw": c["raw"]}
    qnorm = normalize(c["quote"] or "")
    if not qnorm:
        out["note"] = "no quote to check"
        return out

    same_doc = [p for p in passages if doc_pdf and p.source == doc_pdf]
    stages = [[p for p in same_doc if claimed_page is not None and p.page == claimed_page],     # 1. that document, that page
              [p for p in same_doc if p.page != claimed_page],                                    # 2. that document, any other page
              [p for p in passages if p.source != doc_pdf]]                                       # 3. every other manual
    best_seen = 0.0
    for stage in stages:
        found = None
        for p in stage:
            r, span = _best_in(p, qnorm, threshold)
            best_seen = max(best_seen, r)
            if r >= threshold and (found is None or r > found[0]):
                found = (r, p, span)
            if r == 1.0:
                break
        if found:
            r, p, span = found
            out.update(document=p.source, page=p.page, similarity=round(r, 4))
            if r == 1.0:
                out["status"] = VERIFIED                                     # the agent's quote stays as it is
            else:
                out.update(status=CORRECTED, quote=p.exact_text(*span), agent_quote=c["quote"])   # the manual's exact words
            if (p.source, p.page) != (doc_pdf, claimed_page):
                out["claimed"] = {"document": claimed_doc, "page": claimed_page}
                out["note"] = (f"the agent cited {claimed_doc or 'no document'}"
                               f"{f', page {claimed_page}' if claimed_page else ''}; the text is in {p.source}, page {p.page}")
            return out
    out["similarity"] = round(best_seen, 4)
    out["note"] = "this quote is not in our manuals" if passages else "the manuals could not be loaded to check this quote"
    return out


def verify_citations(items: Iterable[Any], passages: Optional[List[Passage]] = None, threshold: float = THRESHOLD) -> List[Dict[str, Any]]:
    """Check every citation. Never raises: if the manuals cannot be loaded, every citation is left unverified (fail closed)."""
    items = list(items or [])
    try:
        passages = load_passages() if passages is None else passages
    except Exception:
        log.exception("could not load the manuals to check citations")
        out = []
        for item in items:
            c = parse_item(item)
            out.append({"document": c["document"], "page": c["page"], "quote": c["quote"], "status": UNVERIFIED, "similarity": 0.0,
                        "raw": c["raw"], "note": "the manuals could not be loaded to check this quote"})
        return out
    return [verify_citation(i, passages, threshold) for i in items]


def summarize(citations: List[Dict[str, Any]]) -> Dict[str, int]:
    return {s: sum(1 for c in citations if c["status"] == s) for s in (VERIFIED, CORRECTED, UNVERIFIED)}
