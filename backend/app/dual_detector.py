"""Dual detector: the CNN (Titus's model) and AMD Token Factory vision model(s) look at the same image in
parallel and their answers are compared. Switched on with DUAL_DETECTOR=on (default off).

DUAL_MODE picks how the answers are combined (default `pair`, the original behaviour):
    pair         CNN + DUAL_VISION_MODEL. Same label -> automatic; different -> human review.
    unanimous    CNN + DUAL_VISION_MODEL + DUAL_VISION_MODEL_2. Automatic ONLY if all available detectors agree.
    vote         the same three; 2-of-3 wins automatically; a 3-way split -> human review.
    cnn_anchored the same three; 2-of-3 wins, EXCEPT when the two vision models agree and outvote the CNN or when
                 there is a 3-way split: then a human decides.
A detector that fails or times out is MISSING: with two left they must agree; with one left it answers alone
(`single_detector`); with none the caller falls back to the normal single-CNN path. An upload never fails because of this.

agreement: agree (all answers equal) / disagree / single_detector. needs_human_review is the mode's decision.
Confidences are NEVER averaged or merged: each detector's own number is kept next to its label.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from collections import Counter
from typing import Callable, Optional

log = logging.getLogger("dual_detector")

LABELS = ("normal", "bearing_wear", "structural_crack", "corrosion")
AGREE, DISAGREE, SINGLE = "agree", "disagree", "single_detector"

PROMPT = ("You are inspecting a photo of an industrial part for a factory quality check. Classify it as exactly "
          "ONE of these labels: normal, bearing_wear, structural_crack, corrosion.\n"
          "bearing_wear = concentric circular scoring marks; structural_crack = a thin dark linear fracture line; "
          "corrosion = scattered brown/orange rust speckles; normal = none of these.\n"
          'Reply with ONLY JSON: {"defect_type": "<label>", "confidence": <number between 0 and 1>}')


def enabled() -> bool:
    return os.getenv("DUAL_DETECTOR", "off").strip().lower() in ("on", "1", "true", "yes")


MODES = ("pair", "unanimous", "vote", "cnn_anchored")


def mode() -> str:
    """DUAL_MODE, default `pair`. An unknown value falls back to `pair` (never breaks the app)."""
    m = os.getenv("DUAL_MODE", "pair").strip().lower()
    if m not in MODES:
        log.warning("unknown DUAL_MODE %r; using pair", m)
        return "pair"
    return m


def vision_model() -> str:
    return os.getenv("DUAL_VISION_MODEL", "Qwen3.8-Flash-Next").strip()


def vision_model_2() -> str:
    """The second vision model: only used by the modes that need a third detector (not `pair`)."""
    return os.getenv("DUAL_VISION_MODEL_2", "Gemma-4-31B-IT").strip()


def _timeout() -> float:
    return float(os.getenv("DUAL_VISION_TIMEOUT", "60"))


def norm_label(value) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


# ---------------- the two detectors ----------------

def run_cnn(path: str) -> dict:
    """The CNN's own answer: {"label", "confidence"}."""
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vision.defect_detector import DefectDetector
    global _cnn
    if _cnn is None:
        _cnn = DefectDetector()
    d = _cnn.predict(path)["defect"]
    return {"label": norm_label(d["type"]), "confidence": float(d["confidence"])}


_cnn = None


def run_vision_model(path: str) -> dict:
    """DUAL_VISION_MODEL's own answer: {"label", "confidence", "model"}. Raises on any failure."""
    return _call_vision(path, vision_model())


def run_vision_model_2(path: str) -> dict:
    """DUAL_VISION_MODEL_2's own answer (modes unanimous / vote / cnn_anchored only). Raises on any failure."""
    return _call_vision(path, vision_model_2())


def _call_vision(path: str, model: str) -> dict:
    import core  # backend/core.py: provider config, 429 retries and the concurrency cap are reused

    p = next((x for x in core._providers() if x["name"] == "tokenfactory"), None)
    if p is None or not p["key"]:
        raise RuntimeError("Token Factory is not configured")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    payload = {
        "model": model, "temperature": 0.2, "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": "You are a factory maintenance assistant."},
                     {"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                  {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}],
    }
    effort = os.getenv("DUAL_VISION_EFFORT", "none").strip()
    if effort and effort != "omit":
        payload["reasoning_effort"] = effort
    resp = core._post_with_retries({**p, "model": payload["model"]}, payload)
    body = resp.json()
    text = (body["choices"][0]["message"].get("content") or "").strip()
    a, b = text.find("{"), text.rfind("}")
    parsed = json.loads(text[a:b + 1]) if a != -1 and b > a else {}
    label = norm_label(parsed.get("defect_type"))
    if label not in LABELS:
        raise ValueError(f"vision model returned an unknown label: {label or 'none'}")
    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence"))))
    except (TypeError, ValueError):
        conf = None
    return {"label": label, "confidence": conf, "model": body.get("model") or payload["model"]}


# ---------------- the rules (labels only) ----------------

def decide_labels(dual_mode: str, cnn: Optional[str], v1: Optional[str], v2: Optional[str] = None) -> dict:
    """Apply a mode's rule to the labels the detectors gave (None = missing).
    Returns {"status": "auto"|"review", "label", "reason", "ref"}. `label` is the automatic answer (None on review);
    `ref` is what a review is judged against: the CNN's label if it answered, else the leading label."""
    pairs = [("cnn", cnn), ("v1", v1)] + ([("v2", v2)] if dual_mode != "pair" else [])
    present = [(k, l) for k, l in pairs if l]
    names = [l for _, l in present]

    def out(status, label, reason):
        ref = cnn or (Counter(names).most_common(1)[0][0] if names else None)
        return {"status": status, "label": label, "reason": reason, "ref": ref}

    if not present:
        return out("review", None, "no detector answered")
    top, count = Counter(names).most_common(1)[0]
    if len(present) == 1:
        return out("auto", names[0], "single detector")
    if len(present) == 2:
        return out("auto", top, "agree") if names[0] == names[1] else out("review", None, "disagree")
    # three voters
    if dual_mode == "unanimous":
        return out("auto", top, "unanimous") if count == 3 else out("review", None, "disagree")
    if count == 1:
        return out("review", None, "3-way split")
    if dual_mode == "cnn_anchored" and cnn != top:
        return out("review", None, "CNN outvoted by both vision models")
    return out("auto", top, "unanimous" if count == 3 else "2-of-3")


# ---------------- comparison ----------------

def _votes_text(names: list) -> tuple:
    """({label: n}, "2 of 3 say corrosion") so a human reviewer sees the suggestion."""
    n = len(names)
    votes = dict(Counter(names))
    if not n:
        return votes, "no detector answered"
    top, count = Counter(names).most_common(1)[0]
    if count > 1 or n == 1:
        return votes, f"{count} of {n} say {top.replace('_', ' ')}"
    return votes, "no majority: " + ", ".join(f"1 of {n} says {l.replace('_', ' ')}" for l in names)


def combine(dual_mode: str, cnn: Optional[dict], vlm: Optional[dict], vlm2: Optional[dict] = None,
            errors: Optional[dict] = None) -> Optional[dict]:
    """Combine the detectors' answers for `dual_mode`. Returns None when NO detector answered."""
    errors = errors or {}
    three = dual_mode != "pair"
    answers = [("cnn", cnn), ("vision_model", vlm)] + ([("vision_model_2", vlm2)] if three else [])
    if all(a is None for _, a in answers):
        return None

    def shown(key, a):
        if key == "cnn":
            return a if a is not None else {"label": None, "confidence": None, "error": errors.get("cnn") or "failed"}
        model = vision_model() if key == "vision_model" else vision_model_2()
        if a is not None:
            return {**a, "name": a.get("model") or model}
        return {"name": model, "label": None, "confidence": None, "error": errors.get(key) or "failed"}

    out = {"enabled": True, "mode": dual_mode}
    for key, a in answers:
        out[key] = shown(key, a)
    present = [(k, a) for k, a in answers if a is not None]
    names = [a["label"] for _, a in present]
    d = decide_labels(dual_mode, cnn["label"] if cnn else None, vlm["label"] if vlm else None,
                      vlm2["label"] if vlm2 else None)
    votes, suggestion = _votes_text(names)
    lead = d["label"] or (cnn["label"] if cnn else Counter(names).most_common(1)[0][0])
    if d["status"] == "review" and Counter(names).most_common(1)[0][1] > 1:
        lead = Counter(names).most_common(1)[0][0]          # the majority is the suggestion
    owner = next(a for _, a in present if a["label"] == lead)  # first holder in order cnn, vision, vision 2
    agreement = (SINGLE if len(present) == 1 else AGREE if len(set(names)) == 1 else DISAGREE)
    review = d["status"] == "review"
    if len(present) == 1:
        who = "CNN" if cnn else "vision model"
        note = f"Only one detector answered ({who})."
    elif review:
        note = ("Detectors disagree" if d["reason"] in ("disagree", "3-way split") else d["reason"]) + \
               ": " + ", ".join(f"{'CNN' if k == 'cnn' else out[k]['name']} says {a['label']}" for k, a in present) + "."
    elif agreement == AGREE:
        note = "All detectors report the same defect."
    else:
        note = f"Majority decides ({suggestion}); mode {dual_mode}."
    out.update(
        agreement=agreement, needs_human_review=review, decision=d["reason"], votes=votes, suggestion=suggestion,
        answer={"label": lead, "confidence": owner.get("confidence"),
                "from": next(k for k, a in present if a is owner)},
        models_answered=["CNN" if k == "cnn" else out[k]["name"] for k, _ in present], note=note)
    return out


def compare(cnn: Optional[dict], vlm: Optional[dict], cnn_error: Optional[str] = None,
            vlm_error: Optional[str] = None) -> Optional[dict]:
    """Pair comparison (the original behaviour). Returns None when BOTH detectors failed."""
    return combine("pair", cnn, vlm, None, {"cnn": cnn_error, "vision_model": vlm_error})


def run_dual(path: str, cnn_fn: Optional[Callable] = None, vlm_fn: Optional[Callable] = None,
             timeout: Optional[float] = None, dual_mode: Optional[str] = None,
             vlm2_fn: Optional[Callable] = None) -> Optional[dict]:
    """Run the detectors for the active mode in parallel on one image and combine them. A detector that fails or is
    too slow is MISSING, it never fails the request. One deadline covers all detectors (they run side by side)."""
    dual_mode, timeout = dual_mode or mode(), timeout or _timeout()
    jobs = {"cnn": cnn_fn or run_cnn, "vision_model": vlm_fn or run_vision_model}
    if dual_mode != "pair":
        jobs["vision_model_2"] = vlm2_fn or run_vision_model_2
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futures = {key: pool.submit(fn, path) for key, fn in jobs.items()}
    wait(list(futures.values()), timeout=timeout)
    results, errors = {}, {}
    for key, fut in futures.items():
        if not fut.done():
            errors[key] = f"no answer within {timeout:.0f}s"
            continue
        try:
            results[key] = fut.result()
        except Exception as exc:  # never include the request: it may carry the image or key
            errors[key] = f"{type(exc).__name__}"
            log.warning("%s detector failed: %s", key, type(exc).__name__)
    pool.shutdown(wait=False)  # a timed-out call keeps running in the background, we just stop waiting
    return combine(dual_mode, results.get("cnn"), results.get("vision_model"), results.get("vision_model_2"), errors)
