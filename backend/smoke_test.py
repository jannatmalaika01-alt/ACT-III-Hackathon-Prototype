"""Runs the full flow against the running server: upload -> diagnose -> approve.

Usage (server must already be running on port 8000):
    cd backend
    python smoke_test.py
"""
import base64
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"

# a tiny 1x1 PNG, so you don't need to pick a file
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


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

    print("\nDone. Look for status 'task_created' in step 3.")


if __name__ == "__main__":
    main()