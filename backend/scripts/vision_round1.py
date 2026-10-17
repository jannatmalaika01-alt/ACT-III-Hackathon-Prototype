"""Step 3, round 1: every image-capable model (and the CNN) classifies every image in data/dual_lab.

    PYTHONPATH=. python scripts/vision_round1.py [--models A B ...] [--limit N]

Per-image answers are saved to data/dual_lab/round1/<model>.json so round 2 (pairing with the CNN) needs
no new requests. A model is STOPPED after 3 consecutive timeouts / rate limits (no retry loop) and the
reason is recorded. Same prompt for every model; labels outside the allowed list count as wrong.
"""
import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tf_lab as lab

LAB = lab.REPO / "data" / "dual_lab"
IMG_DIR = LAB / "images"
PROMPT = ("You are inspecting a photo of an industrial part for a factory quality check. Classify it as exactly "
          "ONE of these labels: normal, bearing_wear, structural_crack, corrosion.\n"
          "bearing_wear = concentric circular scoring marks; structural_crack = a thin dark linear fracture line; "
          "corrosion = scattered brown/orange rust speckles; normal = none of these.\n"
          'Reply with ONLY JSON: {"defect_type": "<label>", "confidence": <number between 0 and 1>}')
STOP_AFTER = 3  # consecutive timeouts / rate limits


def load_manifest() -> list:
    with open(LAB / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    random.Random(7).shuffle(rows)  # so a stopped model still has a mixed sample
    return rows


def norm_label(v) -> str:
    return str(v or "").strip().lower().replace(" ", "_").replace("-", "_")


def run_cnn(rows) -> dict:
    sys.path.insert(0, str(lab.REPO))
    from PIL import Image
    from vision.defect_detector import DefectDetector
    det = DefectDetector()
    out = {}
    for r in rows:
        t = time.time()
        res = det.probabilities(Image.open(IMG_DIR / r["file"]).convert("RGB"))
        top = max(res, key=res.get)
        out[r["file"]] = {"pred": top, "confidence": round(res[top], 4), "latency": time.time() - t, "error": None}
    return out


def run_model(model: str, effort, rows, limit) -> dict:
    rows = rows[:limit] if limit else rows
    state = {"consec": 0, "stopped": None}
    results = {}

    def one(r):
        if state["stopped"]:
            return
        png = (IMG_DIR / r["file"]).read_bytes()
        res = lab.chat(model, [{"type": "text", "text": PROMPT}, lab.image_part(png)], effort=effort, max_tokens=1500)
        data = lab.parse_json(res["content"]) if res["ok"] else {}
        pred = norm_label(data.get("defect_type"))
        try:
            conf = float(data.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        results[r["file"]] = {"pred": pred if pred in lab.LABELS else (pred or None), "confidence": conf,
                              "latency": res["latency"], "error": None if res["ok"] and data else
                              (res["error_kind"] or "invalid_json"), "error_text": res["error"]}
        if res["error_kind"] in ("timeout", "rate_limit"):
            state["consec"] += 1
            if state["consec"] >= STOP_AFTER:
                state["stopped"] = f"stopped after {STOP_AFTER} consecutive {res['error_kind']}s: {res['error']}"
        else:
            state["consec"] = 0

    with ThreadPoolExecutor(lab.MAX_CONCURRENT) as ex:
        list(ex.map(one, rows))
    return {"model": model, "effort": effort, "stopped": state["stopped"], "results": results}


def mop_up(rows, only) -> int:
    """Re-ask once the images whose answer was lost to a rate limit (the server allows 20 requests/min per key)."""
    lab.LIMITER = lab.Limiter(per_min=12, concurrent=2)
    by_file = {r["file"]: r for r in rows}
    for f in sorted((LAB / "round1").glob("*.json")):
        if f.stem == "cnn" or (only and f.stem not in only):
            continue
        out = json.loads(f.read_text())
        if out.get("stopped"):
            print(f"{f.stem}: skipped (model was stopped)")
            continue
        todo = [by_file[k] for k, v in out["results"].items() if v.get("error") == "rate_limit"]
        if not todo:
            continue
        redo = run_model(out["model"], out["effort"], todo, 0)["results"]
        recovered = sum(1 for v in redo.values() if v["error"] is None)
        out["results"].update(redo)
        out["mop_up"] = {"retried": len(todo), "recovered": recovered}
        f.write_text(json.dumps(out))
        print(f"{f.stem}: retried {len(todo)} rate-limited answers, recovered {recovered}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-rate-limited", action="store_true",
                    help="ONE extra pass over answers that failed with a rate limit (12 requests/min); no loops")
    args = ap.parse_args()
    rows = load_manifest()
    (LAB / "round1").mkdir(exist_ok=True)
    if args.retry_rate_limited:
        return mop_up(rows, args.models)

    (LAB / "round1" / "cnn.json").write_text(json.dumps({"model": "CNN", "results": run_cnn(rows)}))
    caps = [c for c in json.loads((LAB / "vision_models.json").read_text()) if c.get("sees_images") == "yes"]
    if args.models:
        caps = [c for c in caps if c["model"] in args.models]
    print(f"{len(rows)} images x {len(caps)} models; CNN done", flush=True)
    for c in caps:
        t = time.time()
        out = run_model(c["model"], c.get("effort"), rows, args.limit)
        (LAB / "round1" / f"{c['model']}.json").write_text(json.dumps(out))
        ok = sum(1 for v in out["results"].values() if v["error"] is None)
        print(f"{c['model']:30} {len(out['results'])} answered, {ok} usable, {time.time() - t:.0f}s"
              + (f"  STOPPED: {out['stopped']}" if out["stopped"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
