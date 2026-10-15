"""RestEvolusClient with the HTTP layer mocked. The real Evolus API is never called."""
import json

import httpx
import pytest

from app import evolus
from app.config import Settings
from app.evolus import EvolusError, RestEvolusClient, build_evolus_client

CASE = {"id": "c1", "severity": "high", "defect": "crack at weld A3", "explanation": "why",
        "recommended_action": "inspect", "source_manual": "m.pdf", "source_page": 3}


class Resp:
    def __init__(self, status=202, body=None, headers=None):
        self.status_code, self.headers, self._body = status, headers or {}, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture
def post(monkeypatch):
    calls = []
    box = {"resp": Resp(body={"jobId": "j1", "executionId": "ex-42"})}

    def fake(url, headers=None, files=None, timeout=None):
        calls.append({"url": url, "headers": headers, "files": files})
        if isinstance(box["resp"], Exception):
            raise box["resp"]
        return box["resp"]

    monkeypatch.setattr(evolus.httpx, "post", fake)
    return calls, box


def client(**kw):
    args = dict(base_url="https://api.evolus.ai/api/v1", api_key="secret-key", application_id="app-1",
                workflow_id="wf-9")
    return RestEvolusClient(**{**args, **kw})


def test_success_builds_the_documented_request(post):
    calls, _ = post
    assert client().create_task(CASE) == "ex-42"
    c = calls[0]
    assert c["url"] == "https://api.evolus.ai/api/v1/workflows/wf-9/execute"
    assert c["headers"] == {"X-ApiKey": "secret-key", "X-Application-Id": "app-1"}
    params = json.loads(c["files"]["parameters"][1])
    assert params["external_ref"] == "c1" and params["priority"] == "high"
    assert "Recommended action: inspect" in c["files"]["input"][1]


def test_id_from_header_wins_and_other_shapes_are_tolerated(post):
    _, box = post
    box["resp"] = Resp(body=None, headers={"X-Workflow-Execution-Id": "hdr-1"})
    assert client().create_task(CASE) == "hdr-1"
    box["resp"] = Resp(body={"data": {"execution_id": 77}})
    assert client().create_task(CASE) == "77"


def test_missing_id_is_an_error(post):
    _, box = post
    box["resp"] = Resp(body={"status": "ok"})
    with pytest.raises(EvolusError, match="no execution id"):
        client().create_task(CASE)


@pytest.mark.parametrize("status,text", [(409, "switched off"), (401, "credentials"), (500, "HTTP 500")])
def test_http_errors(post, status, text):
    _, box = post
    box["resp"] = Resp(status=status, body={})
    with pytest.raises(EvolusError, match=text):
        client().create_task(CASE)


def test_network_error_never_leaks_the_key(post):
    _, box = post
    box["resp"] = httpx.ConnectError("boom secret-key")
    with pytest.raises(EvolusError) as e:
        client().create_task(CASE)
    assert "secret-key" not in str(e.value)


def test_missing_config_is_reported_without_a_request(post):
    calls, _ = post
    with pytest.raises(EvolusError, match="EVOLUS_API_KEY, EVOLUS_WORKFLOW_ID"):
        client(api_key="", workflow_id="").create_task(CASE)
    assert calls == []


def test_mode_selection():
    base = dict(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s",
                evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True)
    assert isinstance(build_evolus_client(Settings(evolus_mode="rest", **base)), RestEvolusClient)
    assert isinstance(build_evolus_client(Settings(evolus_mode="mock", **base)), evolus.MockEvolusClient)


def test_failed_rest_task_marks_case_task_failed_and_retry_works(post, client_factory=None):
    """The service keeps its task_failed / retry behaviour with the real client."""
    from fastapi.testclient import TestClient
    from app.db import InMemoryRepo
    from app.main import create_app

    calls, box = post
    box["resp"] = Resp(status=500, body={})
    settings = Settings(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s",
                        evolus_mode="rest", evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True)
    app = create_app(settings=settings, repo=InMemoryRepo(), evolus=client())
    tc = TestClient(app)
    case = tc.post("/cases", files={"file": ("a.txt", b"x", "text/plain")}, data={"uploaded_by": "t"}).json()
    tc.post(f"/cases/{case['id']}/diagnosis", json={
        "defect": "crack", "severity": "high", "explanation": "e", "recommended_action": "a",
        "source_manual": "m.pdf", "source_page": 1, "confidence": 0.9})
    r = tc.post(f"/cases/{case['id']}/approve", json={"decided_by": "boss"}).json()
    assert r["status"] == "task_failed"
    box["resp"] = Resp(body={"executionId": "ex-1"})
    assert tc.post(f"/cases/{case['id']}/retry-task").json()["status"] == "task_created"
