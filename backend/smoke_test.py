"""Runs the full flow against the running server: upload -> diagnose -> approve, then one dual-detector upload
(step 4, only when the server runs with DUAL_DETECTOR=on).

Usage (server must already be running on port 8000; set BASE_URL to use another address):
    cd backend
    python smoke_test.py
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# a tiny 1x1 PNG, so you don't need to pick a file
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


# a real generated test photo (corrosion) so the CNN and the vision model have something to look at
PHOTO = Path(__file__).resolve().parents[1] / "data" / "test" / "test_14.png"


def dual_detector_step(client):
    """Step 4: one upload with the dual detector on. Prints the agreement, the model that answered and the time."""
    health = client.get(f"{BASE}/health").json()
    if not health.get("dual_detector"):
        print("\n=== 4. Dual detector: SKIPPED (the server has DUAL_DETECTOR off; set DUAL_DETECTOR=on in .env) ===")
        return
    data = PHOTO.read_bytes() if PHOTO.is_file() else PNG
    r = client.post(f"{BASE}/cases", files={"file": ("dual.png", data, "image/png")}, data={"uploaded_by": "malaika"})
    if r.status_code != 201:
        sys.exit("Dual detector upload failed - stop here.")
    case_id = r.json()["id"]
    start = time.time()
    r = client.post(f"{BASE}/cases/{case_id}/analyze")
    total = time.time() - start
    print(f"\n=== 4. Dual detector: HTTP {r.status_code} ===")
    if r.status_code != 200:
        print(r.text[:500])
        return
    dual = r.json().get("dual_detector") or {}
    print(f"mode               : {dual.get('mode')}")
    print(f"agreement          : {dual.get('agreement')}   (needs_human_review={dual.get('needs_human_review')})")
    for key, who in (("cnn", "CNN"), ("vision_model", "vision model"), ("vision_model_2", "vision model 2")):
        d = dual.get(key)
        if d is not None:
            name = f"{d['name']} " if d.get("name") else ""
            print(f"{who:19}: {name}-> {d.get('label')} ({d.get('confidence')}) {d.get('error') or ''}")
    print(f"votes              : {dual.get('suggestion')}")
    if dual.get("cnn_dissent"):
        print(f"CNN dissent (note) : {dual['cnn_dissent']['label']} ({dual['cnn_dissent']['confidence']}), low trust")
    for w in dual.get("warnings") or []:
        print(f"warning            : {w}")
    print(f"models that answered: {', '.join(dual.get('models_answered') or [])}")
    print(f"total time         : {total:.1f}s")


def show(title, response):
    print(f"\n=== {title}: HTTP {response.status_code} ===")
    try:
        print(json.dumps(response.json(), indent=2)[:1500])
    except Exception:
        print(response.text[:500])


def main():
    with httpx.Client(timeout=300) as client:
        r = client.post(
            f"{BASE}/cases",
            files={"file": ("test.png", PNG, "image/png")},
            data={"uploaded_by": "malaika"},
        )
        show("1. Upload", r)
        if r.status_code != 201:
            sys.exit("Upload failed - stop here.")
        case_id = r.json()["id"]

        r = client.post(
            f"{BASE}/cases/{case_id}/diagnose",
            json={
                "defect_type": "crack",
                "location": "weld joint A3",
                "confidence": 0.91,
                "image_ref": "img_014.png",
            },
        )
        show("2. Diagnose", r)
        if r.status_code != 200:
            sys.exit("Diagnose failed - stop here.")

        r = client.post(f"{BASE}/cases/{case_id}/approve", json={"decided_by": "malaika"})
        show("3. Approve", r)

        dual_detector_step(client)

    print("\nDone. Look for status 'task_created' in step 3.")


if __name__ == "__main__":
    main()