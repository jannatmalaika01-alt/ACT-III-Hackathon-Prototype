"""vision_primary is the default whenever DUAL_DETECTOR is on; the other modes stay selectable with DUAL_MODE."""
import sys
import types

import pytest
from fastapi.testclient import TestClient

from app import dual_detector as dd
from app.config import get_settings
from app.db import InMemoryRepo
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
C, N = "corrosion", "normal"


@pytest.fixture(autouse=True)
def real_default(monkeypatch):
    monkeypatch.delenv("DUAL_MODE", raising=False)        # conftest pins "pair" for the older tests; here we want the real default


def det(label, conf, model=None):
    return {"label": label, "confidence": conf, **({"model": model} if model else {})}


def make_client(monkeypatch):
    def fake(report, rejection_feedback=None):
        return {"defect": f"{report['defect_type']} at {report['location']}", "severity": "high", "explanation": "e",
                "recommended_action": "a", "source_manual": "SOP", "source_page": 1, "confidence": report["confidence"], "raw": {}}

    mod = types.ModuleType("agent")
    mod.diagnose, mod.NoRelevantSOPError = fake, type("NoSOP", (Exception,), {})
    monkeypatch.setitem(sys.modules, "agent", mod)
    return TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=object()))


def test_the_default_mode_is_vision_primary(monkeypatch):
    assert dd.DEFAULT_MODE == "vision_primary" and dd.mode() == "vision_primary"


@pytest.mark.parametrize("mode", ["pair", "unanimous", "vote", "cnn_anchored", "vision_primary"])
def test_every_other_mode_is_still_selectable(monkeypatch, mode):
    monkeypatch.setenv("DUAL_MODE", mode)
    assert dd.mode() == mode


@pytest.mark.parametrize("value", ["", "banana", "pair2"])
def test_an_empty_or_unknown_value_gives_the_default(monkeypatch, value):
    monkeypatch.setenv("DUAL_MODE", value)
    assert dd.mode() == "vision_primary"


def test_detector_on_without_a_mode_runs_vision_primary_with_both_vision_models(monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    called = []
    r = dd.run_dual("x.png", lambda p: det(N, 0.5), lambda p: (called.append("flash"), det(C, 0.95, "Flash"))[1], timeout=2,
                    vlm2_fn=lambda p: (called.append("gemma"), det(C, 0.98, "Gemma"))[1])
    assert sorted(called) == ["flash", "gemma"]
    assert r["mode"] == "vision_primary" and r["answer"]["label"] == C and r["cnn_dissent"]["trust"] == "low"


def test_health_shows_the_mode_only_when_the_detector_is_on(monkeypatch):
    c = make_client(monkeypatch)
    monkeypatch.delenv("DUAL_DETECTOR", raising=False)
    off = c.get("/health").json()
    assert off["dual_detector"] is False and off["dual_mode"] is None and off["vision_model_2"] is None
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    on = c.get("/health").json()
    assert (on["dual_detector"], on["dual_mode"]) == (True, "vision_primary")
    assert on["vision_model"] == "Qwen3.8-Flash-Next" and on["vision_model_2"] == "Gemma-4-31B-IT"
    monkeypatch.setenv("DUAL_MODE", "vote")
    assert c.get("/health").json()["dual_mode"] == "vote"


def test_analyze_uses_vision_primary_by_default(monkeypatch):
    c = make_client(monkeypatch)
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setattr(dd, "run_cnn", lambda p: det("bearing_wear", 0.47))
    monkeypatch.setattr(dd, "run_vision_model", lambda p: det(C, 0.95, "Flash"))
    monkeypatch.setattr(dd, "run_vision_model_2", lambda p: det(C, 0.98, "Gemma"))
    cid = c.post("/cases", files={"file": ("t.png", PNG, "image/png")}, data={"uploaded_by": "t"}).json()["id"]
    d = c.post(f"/cases/{cid}/analyze").json()["dual_detector"]
    assert d["mode"] == "vision_primary" and d["needs_human_review"] is False and d["answer"]["label"] == C
    assert d["cnn_dissent"]["label"] == "bearing_wear"        # the CNN is a low-trust note, not a vote
