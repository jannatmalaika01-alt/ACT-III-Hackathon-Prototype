"""core.py - the two things the agent needs: an LLM call and SOP retrieval.

LLM: any OpenAI-compatible chat endpoint, tried in order (see LLM_PROVIDERS below).
     Groq, Gemini and a vLLM server on the AMD GPU all speak this protocol, so when the GPU
     arrives you only change .env - no code changes.
RAG: the REAL manuals in data/raw (Titus's knowledge/ package), CPU only. No hard-coded SOPs.
"""
import os
import re
import sys
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# repo root (one level above backend/) so `knowledge` can be imported however the server starts
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------- LLM providers ----------------
# Tried in order; the first one that answers wins. A provider is skipped if its key/URL is missing.
#   1. LOCAL  = vLLM on the AMD GPU.  Set LOCAL_LLM_BASE_URL (e.g. http://HOST:8000/v1) and LOCAL_LLM_MODEL.
#   2. GROQ   = what you use today.   GROQ_API_KEY, optional LLM_MODEL.
#   3. GEMINI = fallback.             GEMINI_API_KEY, optional GEMINI_MODEL.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # Groq model (kept: /health reads this name)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# provider used by the most recent successful call - the agent stores it in raw so we can state
# honestly in the demo which model (and which hardware) produced each diagnosis
LAST_LLM = {"provider": None, "model": None}


class LLMUnavailableError(RuntimeError):
    """Every configured provider failed (or none is configured)."""


def _providers() -> list:
    out = []
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


def call_llm(prompt: str, system: str = "You are a factory maintenance assistant.") -> str:
    providers = _providers()
    if not providers:
        raise LLMUnavailableError("No LLM configured: set GROQ_API_KEY, GEMINI_API_KEY or LOCAL_LLM_BASE_URL in .env")
    errors = []
    for p in providers:
        try:
            r = httpx.post(
                f"{p['base']}/chat/completions",
                headers={"Authorization": f"Bearer {p['key']}"},
                json={"model": p["model"], "temperature": 0.2,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                timeout=LLM_TIMEOUT,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"].get("content") or ""
            LAST_LLM.update(provider=p["name"], model=p["model"])
            # Qwen-style models print reasoning in <think> tags; strip it so JSON parsing works.
            return _THINK_RE.sub("", text).strip()
        except Exception as exc:  # try the next provider; never log the key
            errors.append(f"{p['name']}: {type(exc).__name__}")
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