"""Dual detector: compare() / run_dual(), the /analyze integration behind DUAL_DETECTOR, and the agent routing."""
import sys
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from app import dual_detector as dd
from app.config import get_settings
from app.db import InMemoryRepo
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
CNN = {"label": "corrosion", "confidence": 0.93}
VLM = {"label": "corrosion", "confidence": 0.8, "model": "Qwen3.8-27B"}


# ---------------- compare() ----------------

def test_agree():
    r = dd.compare(CNN, VLM)
    assert r["agreement"] == "agree" and r["needs_human_review"] is False


def test_disagree_flags_human_review_and_keeps_both_results():
    r = dd.compare(CNN, {**VLM, "label": "normal", "confidence": 0.99})
    assert r["agreement"] == "disagree" and r["needs_human_review"] is True
    assert (r["cnn"]["label"], r["cnn"]["confidence"]) == ("corrosion", 0.93)
    assert (r["vision_model"]["label"], r["vision_model"]["confidence"]) == ("normal", 0.99)


def test_confidences_are_never_merged():
    r = dd.compare(CNN, {**VLM, "label": "normal", "confidence": 0.5})
    assert "confidence" not in r  # no combined/averaged number anywhere at the top level
    assert r["cnn"]["confidence"] == 0.93 and r["vision_model"]["confidence"] == 0.5


@pytest.mark.parametrize("cnn,vlm,errored", [(None, VLM, "cnn"), (CNN, None, "vision_model")])
def test_single_detector_when_one_fails(cnn, vlm, errored):
    r = dd.compare(cnn, vlm, cnn_error="boom", vlm_error="timeout")
    assert r["agreement"] == "single_detector" and r["needs_human_review"] is False
    assert r[errored]["label"] is None and r[errored]["error"]


def test_both_failed_returns_none():
    assert dd.compare(None, None) is None


# ---------------- run_dual(): parallel, timeouts, failures ----------------

def test_detectors_run_in_parallel():
    barrier = threading.Barrier(2, timeout=5)  # only passes if BOTH detectors are running at the same time

    def cnn(path):
        barrier.wait()
        return CNN

    def vlm(path):
        barrier.wait()
        return VLM

    assert dd.run_dual("x.png", cnn, vlm, timeout=5)["agreement"] == "agree"


def test_slow_detector_times_out_to_single_detector():
    def slow(path):
        time.sleep(2)
        return VLM

    start = time.time()
    r = dd.run_dual("x.png", lambda p: CNN, slow, timeout=0.2)
    assert r["agreement"] == "single_detector" and "no answer within" in r["vision_model"]["error"]
    assert time.time() - start < 1.5  # did not wait for the slow one


def test_exception_in_a_detector_is_single_detector_and_hides_details():
    def boom(path):
        raise RuntimeError("secret-token-in-message")

    r = dd.run_dual("x.png", boom, lambda p: VLM, timeout=2)
    assert r["agreement"] == "single_detector"
    assert "secret" not in str(r)


# ---------------- /analyze integration ----------------

@pytest.fixture
def agent(monkeypatch):
    state = types.SimpleNamespace(calls=[])

    def fake_diagnose(report, rejection_feedback=None):
        state.calls.append(report)
        return {"defect": f"{report['defect_type']} at {report['location']}", "severity": "high",
                "explanation": "e", "recommended_action": "a", "source_manual": "Test SOP", "source_page": 3,
                "confidence": report["confidence"], "raw": {}}

    fake = types.ModuleType("agent")
    fake.diagnose, fake.NoRelevantSOPError = fake_diagnose, type("NoSOP", (Exception,), {})
    monkeypatch.setitem(sys.modules, "agent", fake)
    return state


@pytest.fixture
def client(agent):
    return TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=object()))


def _case(client):
    return client.post("/cases", files={"file": ("t.png", PNG, "image/png")}, data={"uploaded_by": "t"}).json()["id"]


def test_off_by_default_never_calls_the_vision_model(client, agent, monkeypatch):
    monkeypatch.delenv("DUAL_DETECTOR", raising=False)
    monkeypatch.setattr(dd, "run_vision_model", lambda p: pytest.fail("vision model must not be called"))
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert "dual_detector" not in body


def test_analyze_agree(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setattr(dd, "run_cnn", lambda p: CNN)
    monkeypatch.setattr(dd, "run_vision_model", lambda p: VLM)
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert body["dual_detector"]["agreement"] == "agree"
    assert body["diagnosis_raw"]["dual_detector"]["cnn"]["confidence"] == 0.93  # both visible in the stored case
    assert agent.calls[0]["needs_human_review"] is False
    assert agent.calls[0]["confidence"] == 0.93  # the CNN's own number, not a blend


def test_analyze_disagree_sets_needs_human_review_and_passes_both_results(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setattr(dd, "run_cnn", lambda p: CNN)
    monkeypatch.setattr(dd, "run_vision_model", lambda p: {**VLM, "label": "structural_crack", "confidence": 0.7})
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert body["dual_detector"]["agreement"] == "disagree" and body["dual_detector"]["needs_human_review"]
    sent = agent.calls[0]
    assert sent["needs_human_review"] is True
    assert sent["detectors"]["cnn"]["label"] == "corrosion"
    assert sent["detectors"]["vision_model"]["label"] == "structural_crack"


def test_analyze_single_detector_when_the_vision_model_fails(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setattr(dd, "run_cnn", lambda p: CNN)

    def fail(p):
        raise TimeoutError()

    monkeypatch.setattr(dd, "run_vision_model", fail)
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert body["dual_detector"]["agreement"] == "single_detector"
    assert body["dual_detector"]["vision_model"]["error"]
    assert agent.calls[0]["defect_type"] == "corrosion"


def test_analyze_single_detector_when_only_the_vision_model_answers(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")

    def fail(p):
        raise RuntimeError()

    monkeypatch.setattr(dd, "run_cnn", fail)
    monkeypatch.setattr(dd, "run_vision_model", lambda p: VLM)
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert body["dual_detector"]["agreement"] == "single_detector"
    assert agent.calls[0]["confidence"] == 0.8  # the vision model's own confidence


def test_health_shows_dual_detector_state(client, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    h = client.get("/health").json()
    assert h["dual_detector"] is True and h["vision_model"]


# ---------------- the real agent: disagreement skips the LLM and asks a human ----------------

def test_agent_routes_a_disagreement_to_human_review(monkeypatch):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent as real_agent
    monkeypatch.setattr(real_agent, "call_llm", lambda *a, **k: pytest.fail("the LLM must not run on a disagreement"))
    report = {"defect_type": "corrosion", "location": "pipe B7", "confidence": 0.93, "needs_human_review": True,
              "detectors": {"cnn": {"label": "corrosion", "confidence": 0.93},
                            "vision_model": {"name": "Qwen3.8-27B", "label": "normal", "confidence": 0.9}}}
    d = real_agent.diagnose(report)
    assert d["raw"]["needs_manual_verification"] is True
    assert "DISAGREE" in d["explanation"] and "normal" in d["explanation"] and "corrosion" in d["explanation"]
    assert d["source_manual"]  # still cited from the real manuals
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
