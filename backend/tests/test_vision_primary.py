"""DUAL_MODE=vision_primary: the two vision models decide, the CNN is advisory. Fake detectors only."""
import sys
import types

import pytest
from fastapi.testclient import TestClient

from app import dual_detector as dd
from app.config import get_settings
from app.db import InMemoryRepo
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
C, N, B, S = "corrosion", "normal", "bearing_wear", "structural_crack"


def det(label, conf=0.9, model=None):
    return None if label is None else {"label": label, "confidence": conf, **({"model": model} if model else {})}


def run(cnn, flash, gemma):
    def fn(value):
        def call(path):
            if value is None:
                raise RuntimeError("failed")
            return value
        return call
    return dd.run_dual("x.png", fn(det(cnn, 0.47) if cnn else None), fn(det(flash, 0.95, "Flash") if flash else None),
                       timeout=2, dual_mode="vision_primary", vlm2_fn=fn(det(gemma, 0.85, "Gemma") if gemma else None))


# ---------------- both vision models answer ----------------

@pytest.mark.parametrize("cnn", [C, N, B, None])
def test_vision_models_agree_so_that_label_wins_whatever_the_cnn_says(cnn):
    r = run(cnn, C, C)
    assert (r["agreement"], r["needs_human_review"], r["answer"]["label"]) == ("agree", False, C)
    assert r["answer"]["from"] == "vision_model" and r["answer"]["confidence"] == 0.95  # a vision model's own number


def test_cnn_dissent_is_a_note_not_a_review():
    r = run(B, C, C)
    assert r["needs_human_review"] is False and r["agreement"] == "agree"
    d = r["cnn_dissent"]
    assert (d["label"], d["confidence"], d["trust"]) == (B, 0.47, "low") and "not a review" in d["note"]
    assert r["votes"] == {C: 2, B: 1} and r["suggestion"] == "2 of 3 say corrosion"
    assert r["cnn"]["label"] == B and r["cnn"]["confidence"] == 0.47          # shown, never merged
    assert r["vision_model"]["confidence"] == 0.95 and r["vision_model_2"]["confidence"] == 0.85
    assert "CNN dissent noted" in r["note"]


def test_no_dissent_when_the_cnn_agrees():
    assert run(C, C, C)["cnn_dissent"] is None
    assert run(None, C, C)["cnn_dissent"] is None


@pytest.mark.parametrize("flash,gemma,expected", [(C, N, C), (C, N, N), (B, C, B), (B, C, C)])
def test_vision_models_disagree_and_the_cnn_breaks_the_tie(flash, gemma, expected):
    r = run(expected, flash, gemma)
    assert (r["needs_human_review"], r["answer"]["label"], r["agreement"]) == (False, expected, "disagree")
    assert r["decision"] == "CNN breaks the tie between the vision models"
    assert r["cnn_dissent"] is None and r["suggestion"] == "2 of 3 say " + expected.replace("_", " ")


def test_three_way_split_needs_a_human_and_shows_every_vote():
    r = run(B, C, N)
    assert (r["needs_human_review"], r["agreement"], r["decision"]) == (True, "disagree", "3-way split")
    assert set(r["votes"]) == {B, C, N} and r["suggestion"].startswith("no majority")
    assert r["answer"]["label"] == C and r["answer"]["from"] == "vision_model"  # the suggestion is a vision label
    assert [r[k]["confidence"] for k in ("cnn", "vision_model", "vision_model_2")] == [0.47, 0.95, 0.85]


def test_vision_models_disagree_and_no_cnn_is_a_review():
    r = run(None, C, N)
    assert r["needs_human_review"] is True and "no CNN to break the tie" in r["decision"]


# ---------------- a vision model fails ----------------

@pytest.mark.parametrize("failed", ["flash", "gemma"])
def test_one_vision_model_fails_the_other_plus_the_cnn_decide(failed):
    flash, gemma = (None, C) if failed == "flash" else (C, None)
    agree = run(C, flash, gemma)
    assert (agree["needs_human_review"], agree["answer"]["label"], agree["agreement"]) == (False, C, "agree")
    assert len(agree["warnings"]) == 1 and "failed" in agree["warnings"][0]
    assert agree["warnings"][0].startswith("Qwen3.8-Flash-Next" if failed == "flash" else "Gemma-4-31B-IT")
    disagree = run(N, flash, gemma)
    assert (disagree["needs_human_review"], disagree["agreement"]) == (True, "disagree")
    assert len(disagree["warnings"]) == 1


def test_one_vision_model_fails_and_the_cnn_is_missing_too_asks_a_human():
    r = run(None, C, None)
    assert r["needs_human_review"] is True and r["agreement"] == "single_detector" and r["warnings"]


# ---------------- both vision models fail ----------------

def test_both_vision_models_fail_cnn_alone_and_unverified():
    r = run(C, None, None)
    assert (r["needs_human_review"], r["agreement"]) == (True, "single_detector")
    assert r["decision"].startswith("both vision models failed") and r["cnn"]["label"] == C
    assert r["answer"]["label"] == C and r["answer"]["from"] == "cnn"
    assert len(r["warnings"]) == 3 and "low trust" in r["warnings"][-1]


def test_everything_fails_returns_none_for_the_caller_to_fall_back():
    assert run(None, None, None) is None


def test_other_modes_do_not_know_about_cnn_dissent():
    r = dd.run_dual("x.png", lambda p: det(B), lambda p: det(C), timeout=2, dual_mode="vote", vlm2_fn=lambda p: det(C))
    assert r["cnn_dissent"] is None and r["warnings"] == []


def test_mode_is_selectable_and_the_default_is_unchanged(monkeypatch):
    monkeypatch.setenv("DUAL_MODE", "vision_primary")
    assert dd.mode() == "vision_primary"
    monkeypatch.delenv("DUAL_MODE")
    assert dd.mode() == "pair" and dd.enabled() is False


# ---------------- the endpoint and the real agent ----------------

@pytest.fixture
def agent(monkeypatch):
    state = types.SimpleNamespace(calls=[])

    def fake(report, rejection_feedback=None):
        state.calls.append(report)
        return {"defect": f"{report['defect_type']} at {report['location']}", "severity": "high", "explanation": "e",
                "recommended_action": "a", "source_manual": "Test SOP", "source_page": 3,
                "confidence": report["confidence"], "raw": {}}

    mod = types.ModuleType("agent")
    mod.diagnose, mod.NoRelevantSOPError = fake, type("NoSOP", (Exception,), {})
    monkeypatch.setitem(sys.modules, "agent", mod)
    return state


@pytest.fixture
def client(agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "vision_primary")
    return TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=object()))


def _case(client):
    return client.post("/cases", files={"file": ("t.png", PNG, "image/png")}, data={"uploaded_by": "t"}).json()["id"]


def _patch(monkeypatch, cnn, flash, gemma):
    monkeypatch.setattr(dd, "run_cnn", lambda p: det(cnn, 0.47))
    monkeypatch.setattr(dd, "run_vision_model", lambda p: det(flash, 0.95, "Flash"))
    monkeypatch.setattr(dd, "run_vision_model_2", lambda p: det(gemma, 0.85, "Gemma"))


def test_endpoint_passes_the_dissent_and_all_detectors_to_the_agent(client, agent, monkeypatch):
    _patch(monkeypatch, B, C, C)
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    d = body["dual_detector"]
    assert (d["mode"], d["needs_human_review"], d["suggestion"]) == ("vision_primary", False, "2 of 3 say corrosion")
    sent = agent.calls[0]
    assert sent["defect_type"] == "corrosion" and sent["confidence"] == 0.95 and sent["needs_human_review"] is False
    assert sent["cnn_dissent"]["label"] == B and sent["detectors"]["cnn"]["confidence"] == 0.47
    assert sent["detectors"]["vision_model_2"]["confidence"] == 0.85
    h = client.get("/health").json()
    assert (h["dual_mode"], h["vision_model_2"]) == ("vision_primary", "Gemma-4-31B-IT")


def test_endpoint_survives_a_vision_failure(client, agent, monkeypatch):
    monkeypatch.setattr(dd, "run_cnn", lambda p: det(C, 0.9))

    def boom(p):
        raise TimeoutError()

    monkeypatch.setattr(dd, "run_vision_model", boom)
    monkeypatch.setattr(dd, "run_vision_model_2", boom)
    r = client.post(f"/cases/{_case(client)}/analyze")
    assert r.status_code == 200 and r.json()["dual_detector"]["needs_human_review"] is True
    assert agent.calls[0]["needs_human_review"] is True and len(agent.calls[0]["warnings"]) == 3


def _real_agent(monkeypatch, llm):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent as real
    import core
    top = core.retrieve("corrosion at x")[0]
    quote = " ".join(top["text"].split()[5:15])
    monkeypatch.setattr(real, "call_llm", lambda *a, **k: llm(quote))
    return real


def test_real_agent_adds_the_cnn_dissent_note_to_the_explanation(monkeypatch):
    import json
    real = _real_agent(monkeypatch, lambda q: json.dumps({"explanation": "Rust on the plate.", "recommended_action": "Treat it.",
                                                          "severity": "high", "source_index": 0, "evidence_quote": q}))
    report = {"defect_type": "corrosion", "location": "plate", "confidence": 0.95, "needs_human_review": False,
              "cnn_dissent": {"label": B, "confidence": 0.47, "trust": "low"},
              "warnings": ["Gemma-4-31B-IT failed (timeout)"], "detectors": {}}
    ex = real.diagnose(report)["explanation"]
    assert ex.startswith("Rust on the plate.") and "CNN dissent, low trust" in ex and "bearing_wear (confidence 0.47)" in ex
    assert "Warning: Gemma-4-31B-IT failed (timeout)" in ex
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


def test_real_agent_three_way_split_is_unverified_and_states_everything(monkeypatch):
    real = _real_agent(monkeypatch, lambda q: pytest.fail("the LLM must not run on a review"))
    report = {"defect_type": "corrosion", "location": "plate", "confidence": 0.95, "needs_human_review": True,
              "detectors": {"decision": "3-way split", "suggestion": "no majority: 1 of 3 says corrosion, 1 of 3 says normal",
                            "cnn": {"label": B, "confidence": 0.47},
                            "vision_model": {"name": "Flash", "label": C, "confidence": 0.95},
                            "vision_model_2": {"name": "Gemma", "label": N, "confidence": 0.85}}}
    d = real.diagnose(report)
    assert d["explanation"].startswith("UNVERIFIED") and d["raw"]["needs_manual_verification"] is True
    for text in ("CNN says bearing_wear (confidence 0.47)", "Flash says corrosion", "Gemma says normal", "Votes: no majority"):
        assert text in d["explanation"]
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


def test_real_agent_both_vision_failed_says_so(monkeypatch):
    real = _real_agent(monkeypatch, lambda q: pytest.fail("no LLM"))
    report = {"defect_type": "corrosion", "location": "plate", "confidence": 0.9, "needs_human_review": True,
              "detectors": {"decision": "both vision models failed; CNN alone (low trust)", "cnn": {"label": C, "confidence": 0.9},
                            "vision_model": {"name": "Flash", "label": None, "error": "timeout"},
                            "vision_model_2": {"name": "Gemma", "label": None, "error": "timeout"},
                            "warnings": ["Flash failed (timeout)", "Gemma failed (timeout)"]}}
    ex = real.diagnose(report)["explanation"]
    assert ex.startswith("UNVERIFIED") and "BOTH VISION MODELS FAILED" in ex and "Flash gave no answer" in ex
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
