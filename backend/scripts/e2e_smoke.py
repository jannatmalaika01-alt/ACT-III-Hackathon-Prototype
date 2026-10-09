"""End-to-end check of the full path against a RUNNING server (real Supabase or in-memory).

    1. start the server:   uvicorn app.main:create_app --factory --port 8000
    2. run this script:    python scripts/e2e_smoke.py            (BASE_URL=... to override)

Tip: start the server with EVOLUS_MOCK_FAIL_FIRST_N=1 to also exercise the
failed-task -> retry path. Exit code is 0 only if every check passes.
"""
from __future__ import annotations

import base64
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
SECRET = os.getenv("WEBHOOK_SECRET", "")

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
DIAGNOSIS = {
    "defect": "Hydraulic seal leaking", "severity": "critical",
    "explanation": "Oil film visible around the cylinder seal.",
    "recommended_action": "Replace the cylinder seal kit and check fluid level.",
    "source_manual": "Hydraulic Press SOP", "source_page": 31, "confidence": 0.88,
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    return ok


def new_diagnosed_case(c: httpx.Client) -> str:
    r = c.post("/cases", files={"file": ("press.png", PNG_1X1, "image/png")}, data={"uploaded_by": "smoke"})
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    assert c.post(f"/cases/{cid}/diagnosis", json=DIAGNOSIS).status_code == 200
    return cid


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        print(f"Testing {BASE_URL}")

        print("\n1. health")
        r = c.get("/health")
        if not check("server is up", r.status_code == 200, r.text):
            return 1

        print("\n2. upload -> diagnose -> approve -> task")
        r = c.post("/cases", files={"file": ("belt.png", PNG_1X1, "image/png")}, data={"uploaded_by": "smoke"})
        check("upload returns 201 + pending", r.status_code == 201 and r.json()["status"] == "pending", r.text)
        cid = r.json()["id"]
        check("approve before diagnosis is blocked (409)",
              c.post(f"/cases/{cid}/approve", json={"decided_by": "smoke"}).status_code == 409)
        check("diagnosis attaches", c.post(f"/cases/{cid}/diagnosis", json=DIAGNOSIS).status_code == 200)
        r = c.post(f"/cases/{cid}/approve", json={"decided_by": "smoke-supervisor"})
        status = r.json().get("status")
        check("approve returns 200", r.status_code == 200, r.text)

        if status == "task_failed":
            print("   (Evolus failed - testing retry)")
            check("failed task keeps error message", bool(r.json().get("task_error")))
            r = c.post(f"/cases/{cid}/retry-task")
            status = r.json().get("status")
        check("task created", status == "task_created" and bool(r.json().get("evolus_task_id")), r.text)

        d = c.get(f"/cases/{cid}").json()
        check("approval recorded", [a["decision"] for a in d["approvals"]] == ["approved"], str(d["approvals"]))
        check("audit trail has events", len(d["events"]) >= 3, str(d["events"]))
        check("double approve is 409", c.post(f"/cases/{cid}/approve", json={"decided_by": "x"}).status_code == 409)

        print("\n3. rejected case")
        cid2 = new_diagnosed_case(c)
        check("reject without reason is 422",
              c.post(f"/cases/{cid2}/reject", json={"decided_by": "smoke"}).status_code == 422)
        r = c.post(f"/cases/{cid2}/reject", json={"decided_by": "smoke", "reason": "Photo shows wrong machine"})
        check("reject works", r.status_code == 200 and r.json()["status"] == "rejected", r.text)
        check("rejected case can't be approved", c.post(f"/cases/{cid2}/approve", json={"decided_by": "x"}).status_code == 409)

        print("\n4. webhook")
        body = json.dumps({"event_id": "smoke-1", "case_id": cid, "status": "task_created",
                           "evolus_task_id": "X"}).encode()
        r = c.post("/webhooks/case-status", content=body,
                   headers={"X-Webhook-Signature": "sha256=bad", "Content-Type": "application/json"})
        check("bad signature is 401", r.status_code == 401, r.text)
        if SECRET:
            from app.security import sign  # run from the project root
            r = c.post("/webhooks/case-status", content=body,
                       headers={"X-Webhook-Signature": sign(SECRET, body), "Content-Type": "application/json"})
            check("signed webhook accepted (noop: task already exists)",
                  r.status_code == 200 and r.json()["result"] in {"noop", "duplicate", "applied"}, r.text)
        else:
            print("  [SKIP] signed webhook (set WEBHOOK_SECRET in .env)")

        print("\n5. list")
        r = c.get("/cases", params={"status": "rejected"})
        check("list filter works", r.status_code == 200 and any(i["id"] == cid2 for i in r.json()["items"]))

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:", ", ".join(failed), "\n-> report these to Karim with the case ids above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
