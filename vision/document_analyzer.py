"""Keyword-evidence defect finder for inspection reports / maintenance documents (already turned into text)."""
import re
from typing import Dict, List

from .labels import to_contract_label

KEYWORDS: Dict[str, List[str]] = {
    "bearing_wear": ["bearing", "vibration", "overheat", "metallic noise", "grinding", "spalling", "scoring", "lubricat", "grease"],
    "structural_crack": ["crack", "fracture", "weld seam", "fatigue", "propagat", "split", "fissure"],
    "corrosion": ["rust", "corros", "pitting", "oxid", "flaking", "scale", "wall thinning", "paint blister"],
}
_NEGATION = re.compile(r"\b(no (visible |sign|evidence|indication)|not (observed|detected|found)|free (of|from)|without|absence of)\b", re.I)


def analyze_text(text: str, name: str = "document") -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    scores = {c: 0 for c in KEYWORDS}
    first_hit: Dict[str, str] = {}
    for s in sentences:
        if _NEGATION.search(s):
            continue
        low = s.lower()
        for cls, words in KEYWORDS.items():
            hits = sum(1 for w in words if w in low)
            if hits:
                scores[cls] += hits
                first_hit.setdefault(cls, s)
    total = sum(scores.values())
    if total == 0:
        return {"input_type": "document", "defect": {
            "type": "normal", "confidence": 0.6 if len(text.strip()) > 40 else 0.3,
            "explanation": f"{name}: no defect indicators found in the text"
                           + ("." if len(text.strip()) > 40 else " (the text was empty or too short to judge).")}}
    cls = max(scores, key=scores.get)
    share = scores[cls] / total
    conf = round(min(0.95, 0.5 + 0.3 * share + 0.03 * min(scores[cls], 5)), 2)
    quote = first_hit[cls][:220]
    return {"input_type": "document", "defect": {
        "type": to_contract_label(cls), "confidence": conf,
        "explanation": f"{name} mentions {to_contract_label(cls)} indicators ({scores[cls]} keyword hits). Example: \"{quote}\""}}
