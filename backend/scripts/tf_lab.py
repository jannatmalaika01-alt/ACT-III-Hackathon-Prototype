"""Shared helpers for the Token Factory experiments (vision probe, rounds 1 and 2). Never prints the key.

Rate limits are shared by the whole team (30 requests/min, 8 concurrent): the limiter below keeps this
process under MAX_PER_MIN and MAX_CONCURRENT, and a 429 / daily cap stops the model instead of looping.
"""
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
REPO = HERE.parents[2]
import core  # noqa: E402  (loads .env)

MAX_PER_MIN = 20
MAX_CONCURRENT = 4
TIMEOUT = 120.0
LABELS = ["normal", "bearing_wear", "structural_crack", "corrosion"]
SYSTEM = "You are a factory maintenance assistant."


class Limiter:
    def __init__(self, per_min=MAX_PER_MIN, concurrent=MAX_CONCURRENT):
        self.gap, self.lock, self.next = 60.0 / per_min, threading.Lock(), 0.0
        self.slots = threading.BoundedSemaphore(concurrent)
        self.sent = 0

    def __enter__(self):
        with self.lock:
            now = time.time()
            wait, self.next = max(0.0, self.next - now), max(now, self.next) + self.gap
            self.sent += 1
        time.sleep(wait)
        self.slots.acquire()

    def __exit__(self, *a):
        self.slots.release()


LIMITER = Limiter()


def provider() -> dict:
    p = next((x for x in core._providers() if x["name"] == "tokenfactory"), None)
    if not p or not p["key"] or not p["base"]:
        raise SystemExit("TOKENFACTORY_BASE_URL / TOKENFACTORY_API_KEY are not set in backend/.env")
    return p


def list_models() -> list:
    p = provider()
    r = httpx.get(p["base"] + "/models", headers={"Authorization": "Bearer " + p["key"]}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def image_part(png_bytes: bytes, mime="image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(png_bytes).decode()}}


def pil_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def chat(model: str, user_content, effort="none", max_tokens=1500, json_mode=True) -> dict:
    """One chat call. Returns {ok, content, latency, status, error, error_kind, finish}. `effort` None omits it."""
    p = provider()
    body = {"model": model, "temperature": 0.2, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content}]}
    if effort:
        body["reasoning_effort"] = effort
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    out = {"ok": False, "content": "", "latency": 0.0, "status": None, "error": None, "error_kind": None,
           "finish": None, "model_reported": None}
    start = time.time()
    try:
        with LIMITER:
            start = time.time()
            r = httpx.post(p["base"] + "/chat/completions", headers={"Authorization": "Bearer " + p["key"]},
                           json=body, timeout=TIMEOUT)
        out["status"] = r.status_code
        if r.status_code >= 400:
            out["error"] = core._error_detail(r)
            out["error_kind"] = ("rate_limit" if r.status_code == 429 else "http_%d" % r.status_code)
        else:
            j = r.json()
            ch = j["choices"][0]
            out.update(content=(ch["message"].get("content") or "").strip(), finish=ch.get("finish_reason"),
                       model_reported=j.get("model"))
            out["ok"] = bool(out["content"])
            if not out["ok"]:
                out["error_kind"], out["error"] = "empty_content", f"empty content (finish_reason={ch.get('finish_reason')})"
    except httpx.TimeoutException:
        out["error_kind"], out["error"] = "timeout", f"no answer within {TIMEOUT:.0f}s"
    except Exception as exc:
        out["error_kind"], out["error"] = type(exc).__name__, str(exc)[:150]
    out["latency"] = time.time() - start
    return out


def parse_json(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return {}
    try:
        d = json.loads(text[a:b + 1])
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}
