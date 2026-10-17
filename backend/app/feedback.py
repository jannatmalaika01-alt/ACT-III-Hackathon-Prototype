"""Feedback store: when a PERSON approves a diagnosis for an image case, keep the photo and the approved label so a
candidate CNN can be retrained later (scripts/retrain_from_feedback.py).

Only human-approved labels are saved, never the pipeline's own guesses: nothing is written on reject or on a pending case.
Layout (git-ignored):  data/feedback/<class>/<case_id>.<ext>  +  data/feedback/<class>/<case_id>.json
Saving is best effort: it never raises and never blocks the approval.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from . import dual_detector as dd

log = logging.getLogger("feedback")

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def feedback_dir() -> Path:
    return Path(os.getenv("FEEDBACK_DIR") or REPO_ROOT / "data" / "feedback")


def approved_label(case: Dict[str, Any]) -> Optional[str]:
    """The class the person approved: the defect type of the diagnosis they approved, as one of the four CNN classes."""
    raw = case.get("diagnosis_raw") or {}
    defect_type = (raw.get("defect_report") or {}).get("defect_type")
    label = dd.norm_label(defect_type)
    label = dd._ALIASES.get(label, label)
    return label if label in dd.LABELS else None


def save_approved(case: Dict[str, Any], image_bytes: Optional[bytes], approved_by: str) -> Optional[Path]:
    """Write the photo + a small JSON for an approved image case. Returns the image path, or None if nothing was saved."""
    try:
        content_type = (case.get("content_type") or "").lower()
        if content_type not in IMAGE_TYPES or not image_bytes:
            return None
        label = approved_label(case)
        if label is None:
            return None
        raw = case.get("diagnosis_raw") or {}
        folder = feedback_dir() / label
        folder.mkdir(parents=True, exist_ok=True)
        stem = str(case["id"])
        image_path = folder / f"{stem}{IMAGE_TYPES[content_type]}"
        image_path.write_bytes(image_bytes)
        (folder / f"{stem}.json").write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "case_id": stem,
            "approved_label": label,
            "approved_by": approved_by,
            # a diagnosis that was flagged UNVERIFIED and then approved: the retrain script leaves these out by default
            "diagnosis_was_unverified": bool(raw.get("needs_manual_verification")),
            "defect_source": raw.get("defect_source"),
            "detectors": raw.get("dual_detector") or (raw.get("defect_report") or {}).get("detectors"),
            "file_name": case.get("file_name"),
            "content_type": content_type,
        }, indent=2))
        return image_path
    except Exception:
        log.exception("could not save feedback for case %s", case.get("id"))
        return None
