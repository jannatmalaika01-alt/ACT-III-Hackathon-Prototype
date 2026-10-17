"""Retry resumes an unfinished Evolus execution, and a case never has two Evolus runs at once. HTTP is mocked."""
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import evolus
from app.config import Settings, get_settings
from app.db import InMemoryRepo
from app.evolus import EvolusError, RestEvolusClient
from app.main import create_app
from tests.test_evolus_rest import AGENT_REPLY, CASE, Resp, client, done, execution, http  # noqa: F401  (fixtures)


def run_state(status, ex_id="old-1"):
    return Resp(200, {**execution(status), "id": ex_id})


def wait_until(cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---------------- client: resume decision ----------------

@pytest.mark.parametrize("status", ["Running", "Pending"])
def test_a_pending_or_running_run_is_resumed_not_restarted(http, status):
    http["polls"] = [run_state(status), run_state("Running"), done()]      # 1st GET = the look at the old run
    started = []
    task = client().create_task(CASE, resume_execution_id="old-1", on_started=started.append)
    assert http["posts"] == []                                              # NO second run was started
    assert task == "old-1" and task.result["resumed"] is True and started == ["old-1"]
    assert all("/executions/old-1" in g["url"] for g in http["gets"]) and len(http["gets"]) == 3
    assert task.result["parsed"]["needs_human_review"] is False and task.result["previous_execution_id"] is None


def test_a_completed_run_is_just_read_not_repeated(http):
    http["polls"] = [Resp(200, {**execution("Completed", AGENT_REPLY), "id": "old-1"})]
    task = client().create_task(CASE, resume_execution_id="old-1")
    assert http["posts"] == [] and len(http["gets"]) == 1 and http["sleeps"] == []
    assert task == "old-1" and task.result["resumed"] is True and task.result["parsed"]["severity"] == "high"


def test_a_failed_run_is_replaced_by_a_new_one(http):
    http["polls"] = [run_state("Failed"), done()]
    started = []
    task = client().create_task(CASE, resume_execution_id="old-1", on_started=started.append)
    assert len(http["posts"]) == 1                                          # a new run, because the old one failed
    assert task == "ex-1" and started == ["ex-1"]
    assert task.result["resumed"] is False and task.result["previous_execution_id"] == "old-1"


@pytest.mark.parametrize("peek", [Resp(404, {}), Resp(500, {}), Resp(200, {"status": "SomethingNew"}), Resp(200, None),
                                  httpx.ConnectError("down")])
def test_an_unknown_old_run_is_replaced_by_a_new_one(http, peek):
    http["polls"] = [peek, done()]
    task = client().create_task(CASE, resume_execution_id="old-1")
    assert len(http["posts"]) == 1 and task == "ex-1" and task.result["previous_execution_id"] == "old-1"


def test_a_resumed_run_that_still_does_not_finish_keeps_its_id_in_the_error(http):
    http["polls"] = [run_state("Running")]
    with pytest.raises(EvolusError) as e:
        client(poll_timeout=20).create_task(CASE, resume_execution_id="old-1")
    assert e.value.execution_id == "old-1" and "within 20 s" in str(e.value) and http["posts"] == []


def test_on_started_is_not_called_when_the_start_fails(http):
    http["post_resp"] = Resp(500, {})
    started = []
    with pytest.raises(EvolusError):
        client().create_task(CASE, on_started=started.append)
    assert started == []


def test_the_run_duration_comes_from_evolus_own_timestamps(http):
    http["polls"] = [Resp(200, {**execution("Completed", AGENT_REPLY), "requestedAt": "2026-10-17T14:04:36.3615156Z",
                                "completedAt": "2026-10-17T14:08:41.8063989Z"})]
    assert client().create_task(CASE).result["evolus_seconds"] == 245.4
    http["polls"] = [done()]
    assert client().create_task(CASE).result["evolus_seconds"] is None


def test_mock_client_ignores_the_resume_arguments():
    mock = evolus.MockEvolusClient()
    assert mock.create_task({"id": "x"}, resume_execution_id="old", on_started=lambda e: None).startswith("MOCK-TASK-")
    assert mock.supports_resume is False and RestEvolusClient.supports_resume is True


# ---------------- service: a retry resumes ----------------

def make_app(evolus_client, background=False):
    settings = Settings(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s", evolus_mode="rest",
                        evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True, evolus_background=background)
    tc = TestClient(create_app(settings=settings, repo=InMemoryRepo(), evolus=evolus_client))
    case = tc.post("/cases", files={"file": ("a.png", b"x", "image/png")}, data={"uploaded_by": "t"}).json()
    tc.post(f"/cases/{case['id']}/diagnosis", json={
        "defect": "corrosion", "severity": "high", "explanation": "e", "recommended_action": "a",
        "source_manual": "m.pdf", "source_page": 1, "confidence": 0.9,
        "raw": {"defect_report": {"defect_type": "corrosion", "location": "flange"}}})
    return tc, case["id"]


def test_retry_after_a_timeout_resumes_the_same_run(http):
    tc, cid = make_app(client(poll_timeout=20))
    http["polls"] = [Resp(200, execution("Running"))]
    first = tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"}).json()
    assert first["status"] == "task_failed" and "did not finish within 20 s" in first["task_error"]
    assert first["diagnosis_raw"]["evolus"]["execution_id"] == "ex-1" and first["diagnosis_raw"]["evolus"]["state"] == "failed"
    assert len(http["posts"]) == 1

    http["polls"] = [Resp(200, {**execution("Running"), "id": "ex-1"}), done()]      # the old run is still going, then finishes
    again = tc.post(f"/cases/{cid}/retry-task").json()
    assert again["status"] == "task_created" and again["evolus_task_id"] == "ex-1"
    assert len(http["posts"]) == 1                                                    # still ONE run in Evolus: it was resumed
    ev = again["diagnosis_raw"]["evolus"]
    assert ev["resumed"] is True and ev["parsed"]["severity"] == "high" and again["task_attempts"] == 2
    notes = [e["note"] for e in tc.get(f"/cases/{cid}").json()["events"]]
    assert any("resumed an earlier run" in (n or "") for n in notes)


def test_retry_starts_a_new_run_when_the_old_one_failed(http):
    tc, cid = make_app(client(poll_timeout=20))
    http["polls"] = [Resp(200, execution("Running"))]
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    http["post_resp"] = Resp(202, {"executionId": "ex-2"}, {"X-Workflow-Execution-Id": "ex-2"})
    http["polls"] = [run_state("Failed", "ex-1"), done()]
    again = tc.post(f"/cases/{cid}/retry-task").json()
    assert len(http["posts"]) == 2 and again["status"] == "task_created" and again["evolus_task_id"] == "ex-2"
    assert again["diagnosis_raw"]["evolus"]["previous_execution_id"] == "ex-1"


def test_the_execution_id_is_saved_as_soon_as_evolus_accepts_the_run(http, monkeypatch):
    tc, cid = make_app(client())
    repo, seen = tc.app.state.repo, []
    real_get = evolus.httpx.get

    def spy(url, **kw):
        seen.append((repo.get_case(cid)["diagnosis_raw"] or {}).get("evolus"))
        return real_get(url, **kw)

    monkeypatch.setattr(evolus.httpx, "get", spy)
    http["polls"] = [done()]
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    assert seen[0]["execution_id"] == "ex-1" and seen[0]["state"] == "running" and seen[0]["started_at"]   # saved mid-run


def test_a_stalled_approved_case_can_be_retried_and_resumes_its_run(http):
    """The server restarted mid-run: the case is 'approved', nothing is running, but the execution id was saved."""
    tc, cid = make_app(client())
    tc.app.state.repo.update_fields(cid, {"status": "approved", "diagnosis_raw": {"evolus": {"execution_id": "ex-9", "state": "running"}}})
    http["polls"] = [Resp(200, {**execution("Running"), "id": "ex-9"}), done()]
    r = tc.post(f"/cases/{cid}/retry-task").json()
    assert r["status"] == "task_created" and r["evolus_task_id"] == "ex-9" and http["posts"] == []


def test_retry_is_refused_for_cases_that_are_not_failed_or_stalled(http):
    tc, cid = make_app(client())
    assert tc.post(f"/cases/{cid}/retry-task").status_code == 409            # still pending
    http["polls"] = [done()]
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "b"})
    assert tc.post(f"/cases/{cid}/retry-task").status_code == 409            # already task_created


# ---------------- the no-duplicate guard ----------------

class BlockingClient:
    """An Evolus client whose run lasts until the test releases it."""
    supports_resume = False

    def __init__(self, fail=None):
        self.started, self.release, self.calls, self.fail = threading.Event(), threading.Event(), 0, fail

    def create_task(self, case, resume_execution_id=None, on_started=None):
        self.calls += 1
        self.started.set()
        assert self.release.wait(10), "the test never released the run"
        if self.fail:
            raise self.fail
        return "TASK-1"


def test_background_approve_returns_at_once_and_a_second_run_is_refused():
    blocking = BlockingClient()
    tc, cid = make_app(blocking, background=True)
    r = tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    assert r.status_code == 200 and r.json()["status"] == "approved"           # did not wait for Evolus
    assert blocking.started.wait(5)
    detail = tc.get(f"/cases/{cid}").json()
    assert detail["evolus"]["running"] is True and detail["evolus"]["started_at"]

    dup = tc.post(f"/cases/{cid}/retry-task")
    assert dup.status_code == 409 and "already in progress" in dup.json()["detail"]
    assert tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"}).status_code == 409
    assert blocking.calls == 1                                                  # exactly one run

    blocking.release.set()
    tc.app.state.service.wait_for_background(5)
    done_case = tc.get(f"/cases/{cid}").json()
    assert done_case["status"] == "task_created" and done_case["evolus_task_id"] == "TASK-1"
    assert done_case["evolus"]["running"] is False and blocking.calls == 1


def test_a_synchronous_run_also_refuses_a_second_one_at_the_same_time():
    blocking = BlockingClient()
    tc, cid = make_app(blocking, background=False)
    result = {}
    t = threading.Thread(target=lambda: result.update(r=tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})))
    t.start()
    assert blocking.started.wait(5) and wait_until(lambda: cid in tc.app.state.service._running)
    dup = tc.post(f"/cases/{cid}/retry-task")
    assert dup.status_code == 409 and "already in progress" in dup.json()["detail"]
    blocking.release.set()
    t.join(5)
    assert result["r"].status_code == 200 and result["r"].json()["status"] == "task_created" and blocking.calls == 1


def test_the_claim_is_released_after_a_failure_so_retry_works():
    failing = BlockingClient(fail=EvolusError("Evolus did not finish within 600 s"))
    failing.release.set()
    tc, cid = make_app(failing, background=True)
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    tc.app.state.service.wait_for_background(5)
    detail = tc.get(f"/cases/{cid}").json()
    assert detail["status"] == "task_failed" and detail["evolus"]["running"] is False
    failing.fail = None
    assert tc.post(f"/cases/{cid}/retry-task").status_code == 200
    tc.app.state.service.wait_for_background(5)
    assert tc.get(f"/cases/{cid}").json()["status"] == "task_created"


def test_a_crash_in_the_background_thread_never_leaves_the_case_claimed():
    crashing = BlockingClient(fail=RuntimeError("boom"))
    crashing.release.set()
    tc, cid = make_app(crashing, background=True)
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    tc.app.state.service.wait_for_background(5)
    detail = tc.get(f"/cases/{cid}").json()
    assert detail["evolus"]["running"] is False and detail["status"] == "task_failed" and "unexpected error" in detail["task_error"]


def test_two_threads_cannot_claim_the_same_case():
    from app.service import CaseService, InvalidTransition
    svc = CaseService(InMemoryRepo(), evolus.MockEvolusClient())
    svc._begin("c1")
    with pytest.raises(InvalidTransition, match="already in progress"):
        svc._begin("c1")
    svc._begin("c2")                       # a different case is independent
    svc._end("c1")
    svc._begin("c1")                       # released, so it can be claimed again


# ---------------- settings ----------------

def test_background_only_applies_to_rest_mode():
    base = dict(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s", evolus_mock_fail_first_n=0,
                max_upload_mb=1, use_memory_db=True)
    flags = {}
    for mode, bg in (("mock", True), ("rest", True), ("rest", False)):
        app = create_app(settings=Settings(evolus_mode=mode, evolus_background=bg, **base), repo=InMemoryRepo(),
                         evolus=evolus.MockEvolusClient())
        flags[(mode, bg)] = app.state.service.background
    assert flags == {("mock", True): False, ("rest", True): True, ("rest", False): False}


def test_poll_timeout_defaults_to_600_and_can_be_changed(monkeypatch):
    for k in ("EVOLUS_POLL_TIMEOUT", "EVOLUS_POLL_INTERVAL", "EVOLUS_BACKGROUND"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()
    s = get_settings()
    assert (s.evolus_poll_timeout, s.evolus_poll_interval, s.evolus_background) == (600.0, 5.0, True)
    monkeypatch.setenv("EVOLUS_POLL_TIMEOUT", "900")
    monkeypatch.setenv("EVOLUS_BACKGROUND", "false")
    get_settings.cache_clear()
    assert get_settings().evolus_poll_timeout == 900.0 and get_settings().evolus_background is False
    get_settings.cache_clear()
