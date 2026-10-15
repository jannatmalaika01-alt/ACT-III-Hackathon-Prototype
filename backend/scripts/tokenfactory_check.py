"""Live smoke test for AMD Token Factory: lists the models, then makes ONE chat call.

    PYTHONPATH=. python scripts/tokenfactory_check.py

Reads TOKENFACTORY_BASE_URL / TOKENFACTORY_API_KEY / TOKENFACTORY_MODEL from backend/.env. Never prints the key.
"""
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402  (loads .env)


def main() -> int:
    base = os.getenv("TOKENFACTORY_BASE_URL", "").rstrip("/")
    key = core._secret("TOKENFACTORY_API_KEY")
    model = os.getenv("TOKENFACTORY_MODEL", "MiMo-V2.6-Flash")
    if not base or not key:
        print("TOKENFACTORY_BASE_URL and TOKENFACTORY_API_KEY must be set in backend/.env")
        return 1
    if any(ord(c) < 32 or ord(c) == 127 for c in os.getenv("TOKENFACTORY_API_KEY", "")):
        print("WARNING: TOKENFACTORY_API_KEY in .env contains a hidden control character (bad paste); "
              "it is stripped at runtime, but re-paste the key.")
    print(f"base URL : {base}\nmodel    : {model}\nAPI key  : {'set' if key else 'MISSING'} (not shown)")

    try:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        ids = sorted(m.get("id", "?") for m in r.json().get("data", []))
        print(f"\n/models  : HTTP {r.status_code}, {len(ids)} models")
        for i in ids:
            print(f"  {'*' if i == model else ' '} {i}")
        if model not in ids:
            print(f"  WARNING: TOKENFACTORY_MODEL '{model}' is not in the live catalog")
    except Exception as exc:
        print(f"\n/models  : FAILED ({type(exc).__name__}: {exc})")

    prompt = ('Reply with ONLY a JSON object: {"explanation": "...", "recommended_action": "...", '
              '"severity": "low|medium|high|critical", "source_index": 0, "evidence_quote": "..."} '
              "for a hairline crack on weld joint A3.")
    start = time.time()
    try:
        text = core.call_llm(prompt)
    except Exception as exc:
        print(f"\nchat     : FAILED ({exc})")
        return 1
    latency = time.time() - start
    try:
        data = json.loads(text)
        parsed = isinstance(data, dict) and {"explanation", "recommended_action", "severity"} <= set(data)
    except json.JSONDecodeError:
        parsed = False
    print(f"\nchat     : provider={core.LAST_LLM['provider']} model={core.LAST_LLM['model']} "
          f"latency={latency:.1f}s JSON parsed={parsed}")
    print(f"answer   : {text[:300]}")
    return 0 if parsed else 2


if __name__ == "__main__":
    sys.exit(main())
