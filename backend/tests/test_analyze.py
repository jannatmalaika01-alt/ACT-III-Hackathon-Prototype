"""Tests for POST /cases/{id}/analyze and the /health analyzer fields.
A fake `agent` module is used, so no Groq key or AI packages are needed."""
import sys
import types
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import InMemoryRepo
from app.main import SIMULATED_DEFECTS, create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


class FakeNoSOP(Exception):
    pass


@pytest.fixture
def agent(monkeypatch):
    """Installs a fake `agent` module. Returns an object with .calls and .error."""
    state = types.SimpleNamespace(calls=[], error=None)

    def fake_diagnose(report, rejection_feedback=None):
        state.calls.append(report)
        if state.error is not None:
            raise state.error
        return {
            "defect": f"{report['defect_type']} at {report['location']}",
            "severity": "high",
            "explanation": "Fake explanation for tests.",
            "recommended_action": "Fake action for tests.",
            "source_manual": "Test SOP",
            "source_page": 3,
            "confidence": report["confidence"],
            "raw": {"severity_verified": True},
        }

    fake = types.ModuleType("agent")
    fake.diagnose = fake_diagnose
    fake.NoRelevantSOPError = FakeNoSOP
    monkeypatch.setitem(sys.modules, "agent", fake)
    return state


@pytest.fixture
def client(agent):
    app = create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=object())
    return TestClient(app)


def _upload(client):
    r = client.post("/cases", files={"file": ("t.png", PNG, "image/png")}, data={"uploaded_by": "tester"})
    assert r.status_code == 201
    return r.json()["id"]


def test_analyze_without_body_uses_labelled_simulated_defect(client, agent):
    case_id = _upload(client)
    r = client.post(f"/cases/{case_id}/analyze")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["source_manual"] == "Test SOP" and body["source_page"] == 3
    assert body["diagnosis_raw"]["defect_source"] == "simulated"
    assert body["diagnosis_raw"]["severity_verified"] is True  # the agent's own flags are kept
    assert agent.calls[0] in SIMULATED_DEFECTS


def test_analyze_with_body_uses_the_real_detector_output(client, agent):
    case_id = _upload(client)
    report = {"defect_type": "corrosion", "location": "pipe B7", "confidence": 0.8, "image_ref": "x.png"}
    r = client.post(f"/cases/{case_id}/analyze", json=report)
    assert r.status_code == 200
    assert r.json()["diagnosis_raw"]["defect_source"] == "vision"
    assert agent.calls[0]["defect_type"] == "corrosion"


def test_analyze_unknown_case_is_404(client):
    assert client.post(f"/cases/{uuid4()}/analyze").status_code == 404


def test_analyze_no_relevant_sop_is_422(client, agent):
    case_id = _upload(client)
    agent.error = FakeNoSOP()
    assert client.post(f"/cases/{case_id}/analyze").status_code == 422


def test_analyze_agent_failure_is_502_and_hides_details(client, agent):
    case_id = _upload(client)
    agent.error = RuntimeError("secret groq key problem")
    r = client.post(f"/cases/{case_id}/analyze")
    assert r.status_code == 502
    assert "secret" not in r.text



def test_health_reports_analyzer_fields(client):
    from app import main

    body = client.get("/health").json()
    real_detector = (main._REPO_ROOT / "vision" / "weights" / "defect_cnn.pt").is_file()

    assert body["status"] == "ok"
    assert body["analyzer_mode"] == ("vision" if real_detector else "simulated")
    assert body["detector"] == ("defect_cnn" if real_detector else "simulated")
    assert body["llm_provider"] == "groq"
