"""Compare AMD Token Factory models on the agent's real diagnosis prompt.

    PYTHONPATH=. python scripts/compare_models.py                 # DRY RUN: 3 cases per model (9 requests)
    PYTHONPATH=. python scripts/compare_models.py --full          # all cases (about 18 per model) -> docs/model_comparison.md
    PYTHONPATH=. python scripts/compare_models.py --models MiMo-V2.6-Flash Qwen3.8-27B --full

Uses ONLY Token Factory (the other providers are switched off so every answer is attributable) and the real
CPU retriever over data/raw. Requests are paced to at most 20/min because the quota is shared by the team.
Cases: the 4 FAKE_DEFECTS plus every non-"normal" row of data/test_cases.csv. The detection confidence is
raised to 0.90 so every case reaches the LLM (low-confidence cases skip the LLM by design).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
REPO = HERE.parents[2]

DEFAULT_MODELS = ["MiMo-V2.6-Flash", "Qwen3.8-27B", "DeepSeek-V4-Flash"]
MAX_PER_MIN = 20
DRY_RUN_CASES = 3

# the manual a correct citation must come from, per defect type
EXPECTED_MANUAL = {
    "crack": "structural_inspection_sop.pdf", "structural_crack": "structural_inspection_sop.pdf",
    "corrosion": "corrosion_control_guide.pdf",
    "loose_bolt": "fastener_alignment_sop.pdf", "misalignment": "fastener_alignment_sop.pdf",
    "bearing_wear": "bearing_maintenance_manual.pdf",
}


def build_cases(core) -> list:
    cases = [{**d, "confidence": 0.90} for d in core.FAKE_DEFECTS]
    with open(REPO / "data" / "test_cases.csv", newline="") as f:
        for row in csv.DictReader(f):
            label = row["expected_label"]
            if label == "normal":
                continue
            cases.append({"defect_type": label, "location": f"inspection {row['image']}",
                          "confidence": 0.90, "image_ref": row["image"]})
    return cases


class Pacer:
    """Keeps the whole run under MAX_PER_MIN requests/minute and counts requests and retry waits."""

    def __init__(self, core):
        self.core, self.last, self.requests, self.waits = core, 0.0, 0, 0
        self.gap = 60.0 / MAX_PER_MIN
        self._post, self._sleep = core._post, core._sleep
        core._post = self.post
        core._sleep = self.sleep

    def post(self, p, payload):
        wait = self.last + self.gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        self.requests += 1
        return self._post(p, payload)

    def sleep(self, seconds):  # a 429/502/503 backoff inside core
        self.waits += 1
        self._sleep(seconds)


def run_model(model, cases, agent, core, pacer):
    os.environ["TOKENFACTORY_MODEL"] = model
    rows = []
    for case in cases:
        before_waits = pacer.waits
        start = time.time()
        try:
            result = agent.diagnose(dict(case))
            error = None
        except Exception as exc:
            result, error = None, type(exc).__name__ + ": " + str(exc)[:120]
        latency = time.time() - start
        row = {"case": f"{case['defect_type']} / {case['location']}", "latency": latency,
               "http_retries": pacer.waits - before_waits, "error": error,
               "json_valid": False, "json_retried": False, "quote_verified": False, "manual_ok": False,
               "answered_by": None}
        if result:
            g = (result.get("raw") or {}).get("grounding") or {}
            llm = (result.get("raw") or {}).get("llm") or {}
            row.update(json_valid=bool(g.get("json_valid")), json_retried=bool(g.get("json_retried")),
                       quote_verified=bool(g.get("evidence_quote_verified")),
                       manual_ok=result["source_manual"] == EXPECTED_MANUAL.get(case["defect_type"]),
                       answered_by=llm.get("model"))
        rows.append(row)
        flag = "ok " if not error and row["json_valid"] else "BAD"
        print(f"  [{flag}] {row['case'][:44]:44} {latency:5.1f}s", flush=True)
    return rows


def summarize(model, rows):
    n = len(rows)
    pct = lambda k: 100.0 * sum(1 for r in rows if r[k]) / n  # noqa: E731
    return {"model": model, "n": n, "json": pct("json_valid"), "quote": pct("quote_verified"),
            "manual": pct("manual_ok"), "latency": sum(r["latency"] for r in rows) / n, "max_latency": max(r["latency"] for r in rows),
            "rate_limited": sum(r["http_retries"] for r in rows),
            "error_texts": sorted({r["error"] for r in rows if r["error"]}),
            "retries": sum(r["http_retries"] + r["json_retried"] for r in rows),
            "errors": sum(1 for r in rows if r["error"])}


def table(summaries):
    head = ("| Model | Cases | Valid JSON | Evidence quote verified | Correct manual cited | Avg / max latency | "
            "Retries (HTTP + JSON) | Errors |\n|---|---|---|---|---|---|---|---|\n")
    return head + "\n".join(
        f"| {s['model']} | {s['n']} | {s['json']:.0f}% | {s['quote']:.0f}% | {s['manual']:.0f}% | "
        f"{s['latency']:.1f}s / {s['max_latency']:.1f}s | {s['retries']} | {s['errors']} |" for s in summaries)


def pick_winner(summaries):
    # grounding first (verified quote, right manual), then valid JSON, fewer errors/retries, then speed
    return min(summaries, key=lambda s: (-s["quote"], -s["manual"], -s["json"], s["errors"], s["retries"], s["latency"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--full", action="store_true", help="run every case (default is a 3-case dry run)")
    ap.add_argument("--reasoning-effort", default="low", help="low (default) or none; sent to Token Factory")
    ap.add_argument("--out", default=str(REPO / "docs" / "model_comparison.md"))
    args = ap.parse_args()

    os.environ["TOKENFACTORY_REASONING_EFFORT"] = args.reasoning_effort
    import core
    for name in ("LOCAL_LLM_BASE_URL", "GROQ_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(name, None)  # Token Factory only: no silent fallback to another provider
    if not core._providers():
        print("TOKENFACTORY_BASE_URL / TOKENFACTORY_API_KEY are not set in backend/.env")
        return 1
    import agent

    cases = build_cases(core)
    if not args.full:
        cases = cases[:DRY_RUN_CASES]
    est = len(cases) * len(args.models)
    print(f"{'FULL RUN' if args.full else 'DRY RUN'}: {len(cases)} cases x {len(args.models)} models = "
          f"{est} requests, about {est * 60 / MAX_PER_MIN / 60:.1f} min at <= {MAX_PER_MIN}/min\n")

    pacer, summaries, details = Pacer(core), [], {}
    for model in args.models:
        print(model)
        rows = run_model(model, cases, agent, core, pacer)
        summaries.append(summarize(model, rows))
        details[model] = rows

    print("\n" + table(summaries))
    usable = [s for s in summaries if s["errors"] < s["n"]]
    if not usable:
        print(f"\nEvery request failed ({pacer.requests} sent): check TOKENFACTORY_API_KEY with "
              "scripts/tokenfactory_check.py. No result written.")
        return 1
    winner = pick_winner(usable)
    print(f"\nRecommended: {winner['model']}  ({pacer.requests} requests sent)")
    if args.full:
        md = ["# Token Factory model comparison", "",
              f"Run on {time.strftime('%Y-%m-%d')}: {len(cases)} cases per model ({pacer.requests} requests). "
              "The agent's real prompt and the real CPU retriever were used; confidence was set to 0.90 so every case "
              "reaches the LLM. reasoning_effort=" + args.reasoning_effort + ". 'Evidence quote verified' means the quote the model gave really appears in the cited "
              "manual chunk; 'correct manual' means the cited manual is the one that covers that defect.", "",
              table(summaries), "", f"**Recommended: {winner['model']}** (ranked by verified evidence, correct "
              "citation, valid JSON, then errors/retries and speed).", "", "## Failures", ""]
        for model, rows in details.items():
            bad = [r for r in rows if r["error"] or not (r["json_valid"] and r["quote_verified"] and r["manual_ok"])]
            md.append(f"**{model}** (reasoning_effort={args.reasoning_effort}): " + ("none" if not bad else ""))
            md += [f"- {r['case']}: " + (r["error"] or f"json={r['json_valid']} quote={r['quote_verified']} "
                                          f"manual={r['manual_ok']}") for r in bad]
            md.append("")
        Path(args.out).write_text("\n".join(md))
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
