"""Dual detector: the CNN (Titus's model) and an AMD Token Factory vision model look at the same image in
parallel and their answers are compared. Switched on with DUAL_DETECTOR=on (default off).

    agree            both detectors gave the same label
    disagree         different labels  -> needs_human_review = True; BOTH results are kept and passed on
    single_detector  only one detector answered (the other failed or timed out)

The two confidences are NEVER averaged or merged: each detector's own number is kept next to its label.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
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


def vision_model() -> str:
    return os.getenv("DUAL_VISION_MODEL", "Qwen3.8-27B").strip()


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
    """The Token Factory vision model's own answer: {"label", "confidence", "model"}. Raises on any failure."""
    import core  # backend/core.py: provider config, 429 retries and the concurrency cap are reused

    p = next((x for x in core._providers() if x["name"] == "tokenfactory"), None)
    if p is None or not p["key"]:
        raise RuntimeError("Token Factory is not configured")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    payload = {
        "model": vision_model(), "temperature": 0.2, "max_tokens": 1500,
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


# ---------------- comparison ----------------

def compare(cnn: Optional[dict], vlm: Optional[dict], cnn_error: Optional[str] = None,
            vlm_error: Optional[str] = None) -> Optional[dict]:
    """Combine the two results. Returns None when BOTH detectors failed (nothing to compare)."""
    if cnn is None and vlm is None:
        return None
    out = {
        "enabled": True,
        "cnn": cnn if cnn is not None else {"label": None, "confidence": None, "error": cnn_error or "failed"},
        "vision_model": ({**vlm, "name": vlm.get("model") or vision_model()} if vlm is not None
                         else {"name": vision_model(), "label": None, "confidence": None,
                               "error": vlm_error or "failed"}),
    }
    if cnn is None or vlm is None:
        out.update(agreement=SINGLE, needs_human_review=False,
                   note=f"Only one detector answered ({'CNN' if vlm is None else 'vision model'}).")
    elif cnn["label"] == vlm["label"]:
        out.update(agreement=AGREE, needs_human_review=False, note="Both detectors report the same defect.")
    else:
        out.update(agreement=DISAGREE, needs_human_review=True,
                   note=f"Detectors disagree: CNN says {cnn['label']}, vision model says {vlm['label']}.")
    return out


def run_dual(path: str, cnn_fn: Optional[Callable] = None, vlm_fn: Optional[Callable] = None,
             timeout: Optional[float] = None) -> Optional[dict]:
    """Run both detectors in parallel on one image and compare. A detector that fails or is too slow is
    recorded as an error (single_detector), it never fails the request."""
    cnn_fn, vlm_fn, timeout = cnn_fn or run_cnn, vlm_fn or run_vision_model, timeout or _timeout()
    pool = ThreadPoolExecutor(max_workers=2)
    futures = {"cnn": pool.submit(cnn_fn, path), "vlm": pool.submit(vlm_fn, path)}
    results, errors = {}, {}
    for key, fut in futures.items():
        try:
            results[key] = fut.result(timeout=timeout)
        except FutureTimeout:
            errors[key] = f"no answer within {timeout:.0f}s"
        except Exception as exc:  # never include the request: it may carry the image or key
            errors[key] = f"{type(exc).__name__}"
            log.warning("%s detector failed: %s", key, type(exc).__name__)
    pool.shutdown(wait=False)  # a timed-out call keeps running in the background, we just stop waiting
    return compare(results.get("cnn"), results.get("vlm"), errors.get("cnn"), errors.get("vlm"))
