"""Turns whatever the AI agent returns into a clean DiagnosisIn - or says exactly what is wrong.

Why this exists: the database (and DiagnosisIn) are strict - severity must be one of
low/medium/high/critical, confidence must be 0..1, source_page must be a positive integer.
LLM output is messy ("Severe", 87, "p. 14"). Without this step a messy answer becomes a
500 error from PostgreSQL. With it, the problem is reported as a clear 422.

Use it inside the /diagnose endpoint:

    diag = normalize_agent_output(raw, location=body.location, image_ref=body.image_ref)
    service.attach_diagnosis(case_id, diag)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import DiagnosisIn

SEVERITY_ALIASES = {
    "low": "low", "minor": "low", "negligible": "low",
    "medium": "medium", "moderate": "medium", "mid": "medium",
    "high": "high", "major": "high", "severe": "high",
    "critical": "critical", "urgent": "critical", "emergency": "critical",
}


class DiagnosisFormatError(ValueError):
    """The agent output could not be turned into a valid diagnosis."""

    def __init__(self, problems: List[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _first(raw: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if raw.get(k) not in (None, ""):
            return raw[k]
    return None


def _severity(value: Any) -> Optional[str]:
    if value is None:
        return None
    return SEVERITY_ALIASES.get(str(value).strip().lower())


def _confidence(value: Any) -> Optional[float]:
    """0.87 -> 0.87,  87 -> 0.87,  '87%' -> 0.87,  None -> None.  Anything else -> ValueError."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    is_percent = text.endswith("%")
    number = float(text.rstrip("%").strip())
    if is_percent or 1 < number <= 100:
        number /= 100
    if not 0 <= number <= 1:
        raise ValueError(f"confidence {value!r} is out of range")
    return round(number, 3)  # the database column is numeric(4,3)


def _page(value: Any) -> Optional[int]:
    """14 -> 14,  'p. 14' -> 14,  'Page 14-15' -> 14.  No digits -> None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    match = re.search(r"\d+", str(value))
    page = int(match.group()) if match else None
    return page if page and page > 0 else None


def normalize_agent_output(raw: Dict[str, Any], *, location: Optional[str] = None,
                           image_ref: Optional[str] = None) -> DiagnosisIn:
    problems: List[str] = []

    defect = _first(raw, "defect", "defect_type")
    explanation = _first(raw, "explanation", "reasoning")
    action = _first(raw, "recommended_action", "recommendation", "action")
    manual = _first(raw, "source_manual", "manual", "source")
    if not defect:
        problems.append("missing defect")
    if not explanation:
        problems.append("missing explanation")
    if not action:
        problems.append("missing recommended_action")
    if not manual:
        problems.append("missing source_manual (a manual citation is required)")

    severity = _severity(_first(raw, "severity"))
    if severity is None:
        problems.append(f"severity {raw.get('severity')!r} is not one of low/medium/high/critical")

    page = _page(_first(raw, "source_page", "page"))
    if page is None:
        problems.append("missing or invalid source_page (a page citation is required)")

    try:
        confidence = _confidence(_first(raw, "confidence"))
    except (ValueError, TypeError):
        confidence = None
        problems.append(f"confidence {raw.get('confidence')!r} is not a number between 0 and 1 (or 0-100)")

    if problems:
        raise DiagnosisFormatError(problems)

    # keep everything the agent said, plus where/what it looked at, for debugging
    stored_raw = {**raw, "location": location, "image_ref": image_ref}
    return DiagnosisIn(
        defect=str(defect), severity=severity, explanation=str(explanation),
        recommended_action=str(action), source_manual=str(manual), source_page=page,
        confidence=confidence, raw=stored_raw,
    )
