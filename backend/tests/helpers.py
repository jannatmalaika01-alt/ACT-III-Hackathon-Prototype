import json

from app.security import sign

SECRET = "test-secret"

DIAGNOSIS = {
    "defect": "Conveyor belt fraying",
    "severity": "high",
    "explanation": "Visible fraying along the belt edge.",
    "recommended_action": "Replace the belt tensioner and inspect pulley alignment.",
    "source_manual": "Conveyor Maintenance Manual",
    "source_page": 14,
    "confidence": 0.91,
}


def upload_case(client, name="belt.png", content=b"fake-image-bytes", ctype="image/png"):
    r = client.post("/cases", files={"file": (name, content, ctype)}, data={"uploaded_by": "tech1"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def diagnose(client, case_id, **overrides):
    return client.post(f"/cases/{case_id}/diagnosis", json={**DIAGNOSIS, **overrides})


def ready_case(client):
    """Uploaded + diagnosed, waiting for a human."""
    cid = upload_case(client)
    assert diagnose(client, cid).status_code == 200
    return cid


def approve(client, case_id, who="supervisor1"):
    return client.post(f"/cases/{case_id}/approve", json={"decided_by": who})


def webhook(client, payload, secret=SECRET, signature=None):
    body = json.dumps(payload).encode()
    sig = signature if signature is not None else sign(secret, body)
    return client.post("/webhooks/case-status", content=body,
                       headers={"X-Webhook-Signature": sig, "Content-Type": "application/json"})
