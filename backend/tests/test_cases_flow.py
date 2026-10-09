"""Upload -> diagnose -> approve/reject -> Evolus task (success, failure, retry)."""
import uuid

from tests.helpers import approve, diagnose, ready_case, upload_case


# ---------- upload ----------
def test_upload_creates_pending_case(client):
    cid = upload_case(client)
    r = client.get(f"/cases/{cid}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["file_name"] == "belt.png"
    assert body["events"][0]["to_status"] == "pending"


def test_upload_rejects_unsupported_type(client):
    r = client.post("/cases", files={"file": ("virus.exe", b"x", "application/x-msdownload")})
    assert r.status_code == 415


def test_upload_rejects_too_large_file(client):
    r = client.post("/cases", files={"file": ("big.png", b"0" * (2 * 1024 * 1024), "image/png")})
    assert r.status_code == 413


def test_upload_rejects_empty_file(client):
    r = client.post("/cases", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_filename_is_sanitised(client):
    cid = upload_case(client, name="../../etc/pass wd.png")
    assert client.get(f"/cases/{cid}").json()["file_name"] == "pass_wd.png"


def test_unknown_or_malformed_case_id(client):
    assert client.get(f"/cases/{uuid.uuid4()}").status_code == 404
    assert client.get("/cases/not-a-uuid").status_code == 422


# ---------- approve + task creation ----------
def test_approve_creates_task(client):
    cid = ready_case(client)
    r = approve(client, cid)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "task_created"
    assert body["evolus_task_id"].startswith("MOCK-TASK-")
    assert body["task_attempts"] == 1

    detail = client.get(f"/cases/{cid}").json()
    assert [a["decision"] for a in detail["approvals"]] == ["approved"]
    assert [e["to_status"] for e in detail["events"]][-2:] == ["approved", "task_created"]
    # the task sent to Evolus carries the manual citation
    sent = client.app.state.evolus.created[0]
    assert "page 14" in sent["description"] and sent["priority"] == "high"


def test_cannot_approve_without_diagnosis(client):
    cid = upload_case(client)
    r = approve(client, cid)
    assert r.status_code == 409
    assert "missing" in r.json()["detail"]
    assert client.get(f"/cases/{cid}").json()["status"] == "pending"


def test_diagnosis_without_citation_is_rejected(client):
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnosis", json={
        "defect": "x", "severity": "low", "explanation": "y", "recommended_action": "z"})
    assert r.status_code == 422  # source_manual / source_page missing


def test_double_approve_is_409(client):
    cid = ready_case(client)
    assert approve(client, cid).status_code == 200
    assert approve(client, cid).status_code == 409
    assert len(client.app.state.evolus.created) == 1  # only ONE task created


# ---------- rejection ----------
def test_reject_requires_reason(client):
    cid = ready_case(client)
    assert client.post(f"/cases/{cid}/reject", json={"decided_by": "s1"}).status_code == 422
    assert client.post(f"/cases/{cid}/reject", json={"decided_by": "s1", "reason": "   "}).status_code == 422
    assert client.get(f"/cases/{cid}").json()["status"] == "pending"


def test_rejected_case_is_final_and_creates_no_task(client):
    cid = ready_case(client)
    r = client.post(f"/cases/{cid}/reject", json={"decided_by": "s1", "reason": "Wrong machine in photo"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"

    detail = client.get(f"/cases/{cid}").json()
    assert detail["approvals"][0]["decision"] == "rejected"
    assert detail["approvals"][0]["reason"] == "Wrong machine in photo"

    assert approve(client, cid).status_code == 409                          # can't approve afterwards
    assert client.post(f"/cases/{cid}/retry-task").status_code == 409      # nothing to retry
    assert client.app.state.evolus.created == []                           # Evolus never called


# ---------- Evolus failure + retry ----------
def test_task_failure_keeps_approval_and_can_be_retried(client):
    client.app.state.evolus.fail_next(1)
    cid = ready_case(client)

    r = approve(client, cid)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "task_failed"
    assert "Evolus unavailable" in body["task_error"]
    assert body["task_attempts"] == 1

    detail = client.get(f"/cases/{cid}").json()
    assert [a["decision"] for a in detail["approvals"]] == ["approved"]  # human decision kept

    r = client.post(f"/cases/{cid}/retry-task")
    assert r.status_code == 200
    assert r.json()["status"] == "task_created"
    assert r.json()["task_attempts"] == 2
    assert r.json()["task_error"] is None


def test_retry_can_fail_again(client):
    client.app.state.evolus.fail_next(2)
    cid = ready_case(client)
    assert approve(client, cid).json()["status"] == "task_failed"
    again = client.post(f"/cases/{cid}/retry-task").json()
    assert again["status"] == "task_failed" and again["task_attempts"] == 2


def test_retry_only_allowed_when_task_failed(client):
    cid = ready_case(client)
    assert client.post(f"/cases/{cid}/retry-task").status_code == 409   # still pending
    approve(client, cid)
    assert client.post(f"/cases/{cid}/retry-task").status_code == 409   # already task_created


def test_unexpected_evolus_exception_does_not_lose_approval(client):
    def boom(case):
        raise RuntimeError("connection reset")
    client.app.state.evolus.create_task = boom
    cid = ready_case(client)
    body = approve(client, cid).json()
    assert body["status"] == "task_failed" and "connection reset" in body["task_error"]


# ---------- listing ----------
def test_list_filter_by_status(client):
    a, b = ready_case(client), ready_case(client)
    approve(client, a)
    pending = client.get("/cases", params={"status": "pending"}).json()["items"]
    done = client.get("/cases", params={"status": "task_created"}).json()["items"]
    assert [c["id"] for c in pending] == [b]
    assert [c["id"] for c in done] == [a]
    assert client.get("/cases", params={"status": "bogus"}).status_code == 422
