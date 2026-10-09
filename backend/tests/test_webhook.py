"""Webhook: signature, de-duplication, valid/invalid status changes."""
import uuid

from tests.helpers import approve, ready_case, webhook


def failed_case(client):
    client.app.state.evolus.fail_next(1)
    cid = ready_case(client)
    assert approve(client, cid).json()["status"] == "task_failed"
    return cid


def test_missing_or_bad_signature_is_401(client):
    cid = failed_case(client)
    payload = {"event_id": "e1", "case_id": cid, "status": "task_created", "evolus_task_id": "T-1"}
    assert webhook(client, payload, signature="sha256=deadbeef").status_code == 401
    assert webhook(client, payload, secret="wrong-secret").status_code == 401
    r = client.post("/webhooks/case-status", json=payload)  # no signature header at all
    assert r.status_code == 401
    assert client.get(f"/cases/{cid}").json()["status"] == "task_failed"  # untouched


def test_valid_webhook_moves_failed_case_to_task_created(client):
    cid = failed_case(client)
    r = webhook(client, {"event_id": "e1", "case_id": cid, "status": "task_created",
                         "evolus_task_id": "EVO-42"})
    assert r.status_code == 200
    assert r.json()["result"] == "applied"
    case = client.get(f"/cases/{cid}").json()
    assert case["status"] == "task_created" and case["evolus_task_id"] == "EVO-42"
    assert case["task_error"] is None
    assert case["events"][-1]["source"] == "webhook" and case["events"][-1]["event_id"] == "e1"


def test_duplicate_delivery_is_ignored(client):
    cid = failed_case(client)
    payload = {"event_id": "same-id", "case_id": cid, "status": "task_created", "evolus_task_id": "EVO-1"}
    assert webhook(client, payload).json()["result"] == "applied"
    r = webhook(client, payload)
    assert r.status_code == 200 and r.json()["result"] == "duplicate"
    events = client.get(f"/cases/{cid}").json()["events"]
    assert sum(1 for e in events if e["event_id"] == "same-id") == 1


def test_webhook_for_state_already_reached_is_noop(client):
    cid = ready_case(client)
    approve(client, cid)  # mock Evolus succeeds -> task_created
    r = webhook(client, {"event_id": "late", "case_id": cid, "status": "task_created",
                         "evolus_task_id": "EVO-9"})
    assert r.status_code == 200 and r.json()["result"] == "noop"
    assert client.get(f"/cases/{cid}").json()["evolus_task_id"].startswith("MOCK-TASK-")


def test_webhook_cannot_skip_the_approval_step(client):
    cid = ready_case(client)  # still pending
    r = webhook(client, {"event_id": "e2", "case_id": cid, "status": "task_created",
                         "evolus_task_id": "EVO-1"})
    assert r.status_code == 409
    assert client.get(f"/cases/{cid}").json()["status"] == "pending"


def test_webhook_task_failed_stores_error(client):
    cid = ready_case(client)
    approve(client, cid)  # -> task_created, so task_failed isn't a legal next step
    assert webhook(client, {"event_id": "e3", "case_id": cid, "status": "task_failed",
                            "error": "boom"}).status_code == 409

    cid2 = failed_case(client)
    r = webhook(client, {"event_id": "e4", "case_id": cid2, "status": "task_failed", "error": "still down"})
    assert r.status_code == 200
    assert client.get(f"/cases/{cid2}").json()["task_error"] == "still down"


def test_webhook_unknown_case_is_404(client):
    r = webhook(client, {"event_id": "e5", "case_id": str(uuid.uuid4()),
                         "status": "task_created", "evolus_task_id": "T"})
    assert r.status_code == 404


def test_webhook_payload_validation(client):
    cid = failed_case(client)
    # task_created without a task id
    assert webhook(client, {"event_id": "e6", "case_id": cid, "status": "task_created"}).status_code == 422
    # task_failed without an error
    assert webhook(client, {"event_id": "e7", "case_id": cid, "status": "task_failed"}).status_code == 422
    # status not allowed through the webhook
    assert webhook(client, {"event_id": "e8", "case_id": cid, "status": "approved"}).status_code == 422
    # not JSON
    from app.security import sign
    body = b"not json"
    r = client.post("/webhooks/case-status", content=body, headers={"X-Webhook-Signature": sign("test-secret", body)})
    assert r.status_code == 422
