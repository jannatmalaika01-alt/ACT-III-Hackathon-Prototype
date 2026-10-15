"""core.py - the two things the agent needs: an LLM call and SOP retrieval.

LLM: any OpenAI-compatible chat endpoint, tried in order (see _providers() below).
     AMD Token Factory is tried first; a local vLLM server, Groq and Gemini are fallbacks.
     They all speak the same protocol, so switching is a .env change, not a code change.
RAG: the REAL manuals in data/raw (Titus's knowledge/ package), CPU only. No hard-coded SOPs.
     (The shared Token Factory endpoints have no embeddings, so retrieval stays on the CPU.)
"""
import os
import random
import re
import sys
import threading
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# repo root (one level above backend/) so `knowledge` can be imported however the server starts
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------- LLM providers ----------------
# Tried in order; the first one that answers wins. A provider is skipped if its key/URL is missing.
#   1. TOKENFACTORY = AMD Token Factory.  TOKENFACTORY_BASE_URL, TOKENFACTORY_API_KEY, TOKENFACTORY_MODEL.
#   2. LOCAL        = vLLM on the AMD GPU. LOCAL_LLM_BASE_URL (e.g. http://HOST:8000/v1), LOCAL_LLM_MODEL.
#   3. GROQ         = GROQ_API_KEY, optional LLM_MODEL.
#   4. GEMINI       = GEMINI_API_KEY, optional GEMINI_MODEL.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # Groq model
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Token Factory limits are shared by the whole team: 30 requests/min, 8 concurrent.
TF_MAX_TOKENS = 1500          # thinking tokens count toward max_tokens, so keep this >= 1500
TF_RETRIES = 3                # extra attempts after a 429 / 502 / 503
TF_RETRY_STATUS = (429, 502, 503)
TF_MAX_WAIT = 60.0            # never sleep longer than this between retries
_LLM_SLOTS = threading.BoundedSemaphore(int(os.getenv("LLM_MAX_CONCURRENT", "4")))
_sleep = time.sleep           # tests replace this so they don't really wait

# JSON the agent asks for. MiMo-V2.6-Flash is the only Token Factory chat model that accepts a strict
# json_schema; every other model only accepts {"type": "json_object"}.
DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "recommended_action": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "source_index": {"type": "integer"},
        "evidence_quote": {"type": "string"},
    },
    "required": ["explanation", "recommended_action", "severity", "source_index", "evidence_quote"],
    "additionalProperties": False,
}

# provider used by the most recent successful call - the agent stores it in raw so we can state
# honestly in the demo which model (and which hardware) produced each diagnosis
LAST_LLM = {"provider": None, "model": None}


class LLMUnavailableError(RuntimeError):
    """Every configured provider failed (or none is configured)."""


class LLMConfigError(RuntimeError):
    """A provider is partly configured (for example the URL is set but the key is missing)."""


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _secret(name: str) -> str:
    """An env value usable in an HTTP header. A key pasted from a terminal can carry an invisible
    control character (for example ESC), which makes the server answer '400 Invalid HTTP request'."""
    return _CONTROL_CHARS.sub("", os.getenv(name, "")).strip()


def _providers() -> list:
    out = []
    if os.getenv("TOKENFACTORY_BASE_URL") or os.getenv("TOKENFACTORY_API_KEY"):
        out.append({"name": "tokenfactory", "base": os.getenv("TOKENFACTORY_BASE_URL", "").rstrip("/"),
                    "key": _secret("TOKENFACTORY_API_KEY"),
                    "model": os.getenv("TOKENFACTORY_MODEL", "MiMo-V2.6-Flash")})
    if os.getenv("LOCAL_LLM_BASE_URL"):
        out.append({"name": "local-vllm", "base": os.environ["LOCAL_LLM_BASE_URL"].rstrip("/"),
                    "key": os.getenv("LOCAL_LLM_API_KEY", "EMPTY"),
                    "model": os.getenv("LOCAL_LLM_MODEL", "")})
    if os.getenv("GROQ_API_KEY"):
        out.append({"name": "groq", "base": "https://api.groq.com/openai/v1",
                    "key": os.environ["GROQ_API_KEY"], "model": LLM_MODEL})
    if os.getenv("GEMINI_API_KEY"):
        out.append({"name": "gemini", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "key": os.environ["GEMINI_API_KEY"], "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash")})
    return out


def llm_status() -> dict:
    """Provider/model for /health: the one that last answered, else the first one configured."""
    if LAST_LLM.get("provider"):
        return {"provider": LAST_LLM["provider"], "model": LAST_LLM["model"], "answered": True}
    first = (_providers() or [{}])[0]
    return {"provider": first.get("name"), "model": first.get("model"), "answered": False}


def _is_mimo(model: str) -> bool:
    return "mimo-v2.6-flash" in (model or "").lower()


def _retry_delay(resp, attempt: int) -> float:
    """Seconds to wait before retrying: the Retry-After header if present, else exponential backoff + jitter."""
    try:
        return min(TF_MAX_WAIT, max(0.0, float(resp.headers.get("Retry-After"))))
    except (TypeError, ValueError):
        return min(TF_MAX_WAIT, (2 ** attempt) + random.uniform(0, 1))


def _post(p: dict, payload: dict):
    """One HTTP call, holding a concurrency slot only while the request is in flight."""
    with _LLM_SLOTS:
        return httpx.post(f"{p['base']}/chat/completions",
                          headers={"Authorization": f"Bearer {p['key']}"},
                          json=payload, timeout=LLM_TIMEOUT)


def _post_with_retries(p: dict, payload: dict):
    """Honour 429 Retry-After. Only 429/502/503 are retried (Token Factory only), at most TF_RETRIES times."""
    retries = TF_RETRIES if p["name"] == "tokenfactory" else 0
    for attempt in range(retries + 1):
        r = _post(p, payload)
        if r.status_code in TF_RETRY_STATUS and attempt < retries:
            _sleep(_retry_delay(r, attempt))
            continue
        r.raise_for_status()
        return r


def _payload(p: dict, system: str, prompt: str, json_output: bool, max_tokens: int) -> dict:
    payload = {"model": p["model"], "temperature": 0.2,
               # exactly one system message, at index 0, text-only content
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if p["name"] == "tokenfactory":
        payload["reasoning_effort"] = "low"
        payload["max_tokens"] = max_tokens
        if json_output:
            payload["response_format"] = (
                {"type": "json_schema",
                 "json_schema": {"name": "diagnosis", "strict": True, "schema": DIAGNOSIS_SCHEMA}}
                if _is_mimo(p["model"]) else {"type": "json_object"})
    return payload


def _ask(p: dict, system: str, prompt: str, json_output: bool) -> tuple:
    """Call one provider. Returns (text, model_name_from_response)."""
    if p["name"] == "tokenfactory":
        if not p["key"]:
            raise LLMConfigError("TOKENFACTORY_API_KEY is not set")
        if not p["base"]:
            raise LLMConfigError("TOKENFACTORY_BASE_URL is not set")
    max_tokens = TF_MAX_TOKENS
    for _ in range(2):  # the second pass only happens when the answer was cut off by max_tokens
        r = _post_with_retries(p, _payload(p, system, prompt, json_output, max_tokens))
        body = r.json()
        choice = body["choices"][0]
        # message.content only: message.reasoning is the model's thinking and is ignored
        text = _THINK_RE.sub("", choice["message"].get("content") or "").strip()
        if text:
            return text, body.get("model") or p["model"]
        if choice.get("finish_reason") == "length" and p["name"] == "tokenfactory":
            max_tokens *= 2  # thinking used the whole budget; retry once with more room
            continue
        break
    raise RuntimeError("empty answer")


def call_llm(prompt: str, system: str = "You are a factory maintenance assistant.",
             json_output: bool = True) -> str:
    providers = _providers()
    if not providers:
        raise LLMUnavailableError(
            "No LLM configured: set TOKENFACTORY_API_KEY (+ TOKENFACTORY_BASE_URL), GROQ_API_KEY, "
            "GEMINI_API_KEY or LOCAL_LLM_BASE_URL in .env")
    errors = []
    for p in providers:
        try:
            text, model = _ask(p, system, prompt, json_output)
        except LLMConfigError as exc:  # a clear message, never the key
            errors.append(f"{p['name']}: {exc}")
            continue
        except httpx.HTTPStatusError as exc:
            errors.append(f"{p['name']}: HTTP {exc.response.status_code}")
            continue
        except Exception as exc:  # try the next provider; never log the key
            errors.append(f"{p['name']}: {type(exc).__name__}")
            continue
        LAST_LLM.update(provider=p["name"], model=model)
        return text
    raise LLMUnavailableError("All LLM providers failed (" + ", ".join(errors) + ")")


# ---------------- Test defects (agent.py's `python agent.py` demo; no real detector needed) ----------------

FAKE_DEFECTS = [
    {"defect_type": "crack", "location": "weld joint A3", "confidence": 0.91, "image_ref": "img_014.png"},
    {"defect_type": "corrosion", "location": "pipe section B7", "confidence": 0.78, "image_ref": "img_022.png"},
    {"defect_type": "misalignment", "location": "conveyor belt C2", "confidence": 0.65, "image_ref": "img_031.png"},
    {"defect_type": "loose_bolt", "location": "panel D5", "confidence": 0.95, "image_ref": "img_040.png"},
]

# ---------------- RAG over the real manuals (CPU only) ----------------

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.08"))  # below this a chunk is NOT trusted as evidence
RAG_REL_SCORE = 0.6  # also drop hits scoring under 60% of the best hit (stray pages from other manuals)


def retrieve(query: str, n_results: int = 2) -> list:
    """Return [{"manual", "page", "text"}] from the real manuals in data/raw.

    Returns [] when nothing is relevant enough, so the agent refuses to diagnose instead of
    citing an unrelated page. (The old Chroma version always returned n_results, relevant or not.)"""
    from knowledge.retrieval.knowledge_base import get_store  # lazy: built once, then cached
    from knowledge.retrieval.query import expand_query

    hits = [h for h in get_store().search(expand_query(query), top_k=n_results) if h["score"] >= RAG_MIN_SCORE]
    if not hits:
        return []
    hits = [h for h in hits if h["score"] >= RAG_REL_SCORE * hits[0]["score"]]
    return [{"manual": h["source"], "page": int(h["page"]), "text": h["chunk"]} for h in hits]