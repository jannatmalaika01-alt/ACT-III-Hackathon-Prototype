"""Step 1: which Token Factory chat models accept images, and which reasoning_effort each one accepts.

    PYTHONPATH=. python scripts/vision_probe.py [--out PATH.json]

Sends ONE tiny image (a solid red square, 160x160 px) per chat model and asks for its colour, so "accepts images" means
the model really saw the pixels (it answers 'red'), not just that the request was not rejected.
MinerU2.5-Pro is skipped (OCR only, via /v1/ocr). Output: scripts/vision_models.json (git-ignored copy in data/).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import tf_lab as lab

SKIP = {"MinerU2.5-Pro"}
EFFORTS = ["none", "low", None]  # preferred first; None = leave the parameter out


def probe(model: str) -> dict:
    img = lab.image_part(lab.pil_png(Image.new("RGB", (160, 160), (220, 20, 20))))
    content = [{"type": "text", "text": 'What is the main colour of this image? Reply as JSON: {"color": "<one word>"}'}, img]
    row = {"model": model, "effort": "unsupported", "sees_images": "no", "detail": "", "latency": None}
    for effort in EFFORTS:
        r = lab.chat(model, content, effort=effort, max_tokens=300)
        msg = (r["error"] or "").lower()
        if r["status"] in (400, 422) and "reasoning" in msg:
            continue  # this effort value is not accepted by the model: try the next one
        row["latency"] = round(r["latency"], 1)
        row["effort"] = effort or "(omitted)"
        if r["ok"]:
            said_red = "red" in r["content"].lower()
            row.update(sees_images="yes" if said_red else "accepted, but answer is wrong",
                       detail=r["content"][:80].replace("\n", " "))
        else:
            row.update(sees_images="no" if r["status"] == 400 else "error",
                       detail=f"{r['error_kind']}: {r['error']}"[:200])
        return row
    row["detail"] = "no reasoning_effort value accepted"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lab.REPO / "data" / "dual_lab" / "vision_models.json"))
    args = ap.parse_args()
    models = sorted(m["id"] for m in lab.list_models() if m["id"] not in SKIP and "ocr" not in (m.get("output") or []))
    print(f"{len(models)} chat models (MinerU2.5-Pro skipped)")
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(probe, models))
    print(f"\n{'model':26} {'effort':10} {'sees images':28} {'latency':>8}  detail")
    for r in rows:
        print(f"{r['model']:26} {r['effort']:10} {r['sees_images']:28} {str(r['latency']):>8}  {r['detail']}")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
