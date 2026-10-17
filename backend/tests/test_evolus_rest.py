"""RestEvolusClient: start the workflow, poll for the agent's answer, parse it. The HTTP layer is mocked: the real
Evolus API is never called."""
import json

import httpx
import pytest

from app import evolus
from app.config import Settings
from app.evolus import EvolusError, RestEvolusClient, TaskId, build_evolus_client, build_evolus_input, parse_agent_reply

AGENT_REPLY = """**defect:** Corrosion on pipe flange
**severity:** High
**probable_cause:** Moisture ingress and a failed gasket.
**safety_precautions:**
- Isolate and depressurise the line
- Wear gloves and eye protection
**recommended_steps:**
1. Inspect the flange face for pitting
2. Replace the gasket and re-torque the bolts
**citation:** corrosion_control_guide.pdf, page 2
**confidence:** 0.85
**needs_human_review:** false
**summary:** Moderate flange corrosion; repair at the next stop."""

CASE = {"id": "c1", "severity": "high", "defect": "corrosion at pipe flange", "explanation": "Rust on the flange.",
        "recommended_action": "Inspect it.", "source_manual": "corrosion_control_guide.pdf", "source_page": 2,
        "diagnosis_raw": {"defect_report": {"defect_type": "corrosion", "location": "pipe flange, boiler room line 3"},
                          "dual_detector": {"cnn": {"label": "structural_crack", "confidence": 0.82},
                                            "vision_model": {"name": "Qwen3.8-Flash-Next", "label": "corrosion", "confidence": 0.99},
                                            "vision_model_2": {"name": "Gemma-4-31B-IT", "label": "corrosion", "confidence": 0.98},
                                            "suggestion": "2 of 3 say corrosion",
                                            "cnn_dissent": {"label": "structural_crack", "confidence": 0.82, "trust": "low"},
                                            "warnings": []}}}


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self.headers, self._body = status, headers or {}, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def execution(status, output=None, error=None):
    return {"id": "ex-1", "status": status, "output": output, "error": error, "stepOutputs": {}}


@pytest.fixture
def http(monkeypatch):
    """post() answers 202; get() answers the scripted polls in order (a list of Resp / exceptions)."""
    state = {"posts": [], "gets": [], "polls": [], "clock": 0.0, "sleeps": []}
    state["post_resp"] = Resp(202, {"jobId": "j1", "executionId": "ex-1", "enqueuedAt": "t", "traceId": "tr"},
                              {"X-Workflow-Execution-Id": "ex-1"})

    def post(url, headers=None, files=None, timeout=None):
        state["posts"].append({"url": url, "headers": headers, "files": files})
        if isinstance(state["post_resp"], Exception):
            raise state["post_resp"]
        return state["post_resp"]

    def get(url, headers=None, timeout=None):
        state["gets"].append({"url": url, "headers": headers})
        item = state["polls"].pop(0) if len(state["polls"]) > 1 else state["polls"][0]
        if isinstance(item, Exception):
            raise item
        return item

    def sleep(seconds):                       # a fake clock: sleeping advances time, nothing really waits
        state["sleeps"].append(seconds)
        state["clock"] += seconds

    monkeypatch.setattr(evolus.httpx, "post", post)
    monkeypatch.setattr(evolus.httpx, "get", get)
    monkeypatch.setattr(evolus, "_sleep", sleep)
    monkeypatch.setattr(evolus, "_monotonic", lambda: state["clock"])
    return state


def client(**kw):
    args = dict(base_url="https://api.evolus.ai/api/v1", api_key="secret-key", application_id="app-1", workflow_id="wf-9")
    return RestEvolusClient(**{**args, **kw})


def done(output=AGENT_REPLY):
    return Resp(200, execution("Completed", output))


# ---------------- the request ----------------

def test_the_request_sends_the_defect_as_the_input_field(http):
    http["polls"] = [done()]
    client().create_task(CASE)
    p = http["posts"][0]
    assert p["url"] == "https://api.evolus.ai/api/v1/workflows/wf-9/execute"
    assert p["headers"] == {"X-ApiKey": "secret-key", "X-Application-Id": "app-1"}
    assert set(p["files"]) == {"input"}                                      # only the `input` form field
    text = p["files"]["input"][1]
    for line in ("Defect: corrosion", "Location: pipe flange, boiler room line 3", "Case ID: c1",
                 "CNN (small image model trained on drawings, low trust on real photos): structural_crack (confidence 0.82)",
                 "- Qwen3.8-Flash-Next: corrosion (confidence 0.99)", "- Gemma-4-31B-IT: corrosion (confidence 0.98)",
                 "Votes: 2 of 3 say corrosion", "CNN note (low trust): the CNN said structural_crack with confidence 0.82",
                 "Manual cited by our pipeline: corrosion_control_guide.pdf, page 2"):
        assert line in text


def test_provider_prefixes_are_dropped_from_model_names_in_the_input():
    case = {"id": "x", "diagnosis_raw": {"dual_detector": {"vision_model": {"name": "self-dploy/Qwen3.8-Flash-Next",
                                                                            "label": "corrosion", "confidence": 0.95}}}}
    assert "- Qwen3.8-Flash-Next: corrosion (confidence 0.95)" in build_evolus_input(case)


def test_input_text_without_detector_details_still_reads_well():
    text = build_evolus_input({"id": "x", "defect": "crack at weld A3", "confidence": 0.91, "severity": "high"})
    assert "Defect: crack at weld A3" in text and "Detection confidence: 0.91" in text and "Detector results" not in text


# ---------------- polling ----------------

def test_polls_every_5_seconds_until_completed_and_returns_the_parsed_answer(http):
    http["polls"] = [Resp(200, execution("Pending")), Resp(200, execution("Running")), Resp(200, execution("Running")), done()]
    task = client().create_task(CASE)
    assert isinstance(task, str) and task == "ex-1" and isinstance(task, TaskId)      # still just the execution id
    assert http["sleeps"] == [5.0, 5.0, 5.0] and len(http["gets"]) == 4
    assert http["gets"][0]["url"] == "https://api.evolus.ai/api/v1/workflows/wf-9/executions/ex-1"
    assert http["gets"][0]["headers"] == {"X-ApiKey": "secret-key", "X-Application-Id": "app-1"}
    r = task.result
    assert r["execution_id"] == "ex-1" and r["status"] == "Completed" and r["raw_text"] == AGENT_REPLY
    assert r["parsed"]["needs_human_review"] is False and r["parsed"]["defect"] == "Corrosion on pipe flange"
    assert r["duration_seconds"] == 15.0


def test_times_out_after_600_seconds_with_a_clear_error(http):
    http["polls"] = [Resp(200, execution("Running"))]
    with pytest.raises(EvolusError) as e:
        client().create_task(CASE)
    assert "did not finish within 600 s" in str(e.value) and "ex-1" in str(e.value) and "Running" in str(e.value)
    assert "Retry will pick that run up again" in str(e.value) and e.value.execution_id == "ex-1"
    assert http["clock"] == 600.0 and len(http["gets"]) == 121         # 0, 5, ... 600


def test_the_timeout_and_interval_are_configurable(http):
    http["polls"] = [Resp(200, execution("Running"))]
    with pytest.raises(EvolusError, match="within 20 s"):
        client(poll_interval=10, poll_timeout=20).create_task(CASE)
    assert http["sleeps"] == [10, 10]


def test_a_failed_run_is_an_error_with_its_reason(http):
    http["polls"] = [Resp(200, execution("Running")), Resp(200, execution("Failed", error="agent step crashed"))]
    with pytest.raises(EvolusError, match="failed: agent step crashed"):
        client().create_task(CASE)


@pytest.mark.parametrize("status", ["Cancelled", "Errored", "TimedOut"])
def test_other_failure_statuses_are_errors(http, status):
    http["polls"] = [Resp(200, execution(status))]
    with pytest.raises(EvolusError):
        client().create_task(CASE)


@pytest.mark.parametrize("code,text", [(401, "credentials"), (403, "credentials"), (404, "does not know execution"),
                                       (400, "HTTP 400")])
def test_http_errors_while_polling(http, code, text):
    http["polls"] = [Resp(code, {})]
    with pytest.raises(EvolusError, match=text):
        client().create_task(CASE)
    assert len(http["gets"]) == 1                                      # fatal errors are not retried


def test_a_few_transient_poll_errors_are_tolerated(http):
    http["polls"] = [httpx.ConnectError("boom"), Resp(502, {}), Resp(429, {}), done()]
    assert client().create_task(CASE).result["status"] == "Completed"
    assert http["sleeps"] == [5.0, 5.0, 5.0]


def test_too_many_transient_errors_give_up(http):
    http["polls"] = [httpx.ReadTimeout("slow")]
    with pytest.raises(EvolusError, match="stopped answering.*ReadTimeout"):
        client().create_task(CASE)
    assert len(http["gets"]) == 4


def test_unknown_statuses_keep_polling_until_the_timeout(http):
    http["polls"] = [Resp(200, execution("Queued")), done()]
    assert client().create_task(CASE) == "ex-1"


def test_completed_without_text_is_a_success_with_a_note(http):
    http["polls"] = [Resp(200, execution("Completed", None))]
    r = client().create_task(CASE).result
    assert r["raw_text"] == "" and r["notes"][0] == "Evolus finished but returned no answer text"
    assert r["parsed"]["needs_human_review"] is True


@pytest.mark.parametrize("output", [{"text": AGENT_REPLY}, [AGENT_REPLY], {"output": {"content": AGENT_REPLY}}])
def test_output_shapes(http, output):
    http["polls"] = [Resp(200, execution("Completed", output))]
    assert client().create_task(CASE).result["parsed"]["severity"] == "high"


def test_the_answer_is_taken_from_the_last_step_if_output_is_empty():
    ex = {"output": None, "stepOutputs": {"1": "", "2": AGENT_REPLY, "3": ""}}
    assert evolus.output_text(ex) == AGENT_REPLY


# ---------------- start errors (unchanged behaviour) ----------------

def test_id_from_header_wins_and_other_shapes_are_tolerated(http):
    http["polls"] = [done()]
    http["post_resp"] = Resp(202, None, {"X-Workflow-Execution-Id": "hdr-1"})
    assert client().create_task(CASE) == "hdr-1"
    http["post_resp"] = Resp(202, {"data": {"execution_id": 77}})
    assert client().create_task(CASE) == "77"


def test_missing_id_is_an_error(http):
    http["post_resp"] = Resp(202, {"status": "ok"})
    with pytest.raises(EvolusError, match="no execution id"):
        client().create_task(CASE)


@pytest.mark.parametrize("status,text", [(409, "switched off"), (401, "credentials"), (500, "HTTP 500")])
def test_http_errors_when_starting(http, status, text):
    http["post_resp"] = Resp(status, {})
    with pytest.raises(EvolusError, match=text):
        client().create_task(CASE)
    assert http["gets"] == []


def test_network_errors_never_leak_the_key(http):
    http["post_resp"] = httpx.ConnectError("boom secret-key")
    with pytest.raises(EvolusError) as e:
        client().create_task(CASE)
    assert "secret-key" not in str(e.value)
    http["post_resp"] = Resp(202, {"executionId": "ex-1"})
    http["polls"] = [httpx.ReadTimeout("secret-key")]
    with pytest.raises(EvolusError) as e:
        client().create_task(CASE)
    assert "secret-key" not in str(e.value)


def test_missing_config_is_reported_without_a_request(http):
    with pytest.raises(EvolusError, match="EVOLUS_API_KEY, EVOLUS_WORKFLOW_ID"):
        client(api_key="", workflow_id="").create_task(CASE)
    assert http["posts"] == []


def test_mode_selection_and_polling_settings():
    base = dict(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s",
                evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True)
    c = build_evolus_client(Settings(evolus_mode="rest", evolus_poll_interval=2.0, evolus_poll_timeout=30.0, **base))
    assert isinstance(c, RestEvolusClient) and (c.poll_interval, c.poll_timeout) == (2.0, 30.0)
    default = build_evolus_client(Settings(evolus_mode="rest", **base))
    assert (default.poll_interval, default.poll_timeout) == (5.0, 600.0)
    assert Settings(evolus_mode="rest", **base).evolus_background is True       # rest mode runs the Evolus step in the background
    assert isinstance(build_evolus_client(Settings(evolus_mode="mock", **base)), evolus.MockEvolusClient)


# ---------------- the case flow: result stored, failures use the retry path ----------------

@pytest.fixture
def app_client(http):
    from fastapi.testclient import TestClient
    from app.db import InMemoryRepo
    from app.main import create_app
    settings = Settings(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s", evolus_mode="rest",
                        evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True,
                        evolus_background=False)   # these tests check the outcome of approve itself, so run it in the request
    tc = TestClient(create_app(settings=settings, repo=InMemoryRepo(), evolus=client()))
    case = tc.post("/cases", files={"file": ("a.png", b"x", "image/png")}, data={"uploaded_by": "t"}).json()
    tc.post(f"/cases/{case['id']}/diagnosis", json={
        "defect": "corrosion", "severity": "high", "explanation": "e", "recommended_action": "a",
        "source_manual": "m.pdf", "source_page": 1, "confidence": 0.9,
        "raw": {"defect_report": {"defect_type": "corrosion", "location": "flange"}}})
    return tc, case["id"]


def test_approving_stores_the_agents_answer_in_the_case(app_client, http):
    tc, cid = app_client
    http["polls"] = [Resp(200, execution("Running")), done()]
    r = tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"}).json()
    assert r["status"] == "task_created" and r["evolus_task_id"] == "ex-1"
    ev = r["diagnosis_raw"]["evolus"]
    assert ev["raw_text"] == AGENT_REPLY and ev["parsed"]["recommended_steps"][0].startswith("Inspect the flange")
    assert r["diagnosis_raw"]["defect_report"]["defect_type"] == "corrosion"          # the original raw data is kept
    notes = [e["note"] for e in tc.get(f"/cases/{cid}").json()["events"]]
    assert any("agent answered" in (n or "") and "needs_human_review=False" in n for n in notes)


def test_a_timeout_marks_the_case_task_failed_and_retry_works(app_client, http):
    tc, cid = app_client
    http["polls"] = [Resp(200, execution("Running"))]
    r = tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    assert r.status_code == 200                                         # never crashes the request
    body = r.json()
    assert body["status"] == "task_failed" and "did not finish within 600 s" in body["task_error"] and body["task_attempts"] == 1
    http["polls"] = [done()]
    again = tc.post(f"/cases/{cid}/retry-task").json()
    assert again["status"] == "task_created" and again["diagnosis_raw"]["evolus"]["parsed"]["severity"] == "high"


def test_a_failed_run_and_an_unreachable_evolus_use_the_same_retry_path(app_client, http):
    tc, cid = app_client
    http["polls"] = [Resp(200, execution("Failed", error="boom"))]
    assert tc.post(f"/cases/{cid}/approve", json={"decided_by": "b"}).json()["status"] == "task_failed"
    http["post_resp"] = httpx.ConnectError("down")
    assert tc.post(f"/cases/{cid}/retry-task").json()["status"] == "task_failed"


def test_mock_mode_is_unchanged(client):
    cid = client.post("/cases", files={"file": ("a.png", b"x", "image/png")}, data={"uploaded_by": "t"}).json()["id"]
    client.post(f"/cases/{cid}/diagnosis", json={"defect": "d", "severity": "low", "explanation": "e",
                                                 "recommended_action": "a", "source_manual": "m", "source_page": 1,
                                                 "confidence": 0.5})
    r = client.post(f"/cases/{cid}/approve", json={"decided_by": "b"}).json()
    assert r["status"] == "task_created" and r["evolus_task_id"].startswith("MOCK-TASK-")
    assert "evolus" not in (r.get("diagnosis_raw") or {})
