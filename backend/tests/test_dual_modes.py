"""DUAL_MODE: pair (default), unanimous, vote, cnn_anchored. Fake detectors only: no real model is ever called."""
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
C, N, B = "corrosion", "normal", "bearing_wear"


def det(label, conf=0.9, model=None):
    return None if label is None else {"label": label, "confidence": conf, **({"model": model} if model else {})}


def run(mode, cnn, v1, v2, timeout=2.0):
    def fn(value):
        def call(path):
            if value is None:
                raise RuntimeError("detector failed")
            return value
        return call
    return dd.run_dual("x.png", fn(det(cnn, 0.91) if cnn else None), fn(det(v1, 0.81, "Flash") if v1 else None),
                       timeout=timeout, dual_mode=mode, vlm2_fn=fn(det(v2, 0.71, "Gemma") if v2 else None))


# ---------------- settings ----------------

def test_detector_is_off_and_models_have_defaults(monkeypatch):
    monkeypatch.delenv("DUAL_DETECTOR", raising=False)
    monkeypatch.delenv("DUAL_VISION_MODEL", raising=False)
    monkeypatch.delenv("DUAL_VISION_MODEL_2", raising=False)
    assert dd.enabled() is False                      # the mode default is checked in test_dual_default_mode.py
    assert dd.vision_model() == "Qwen3.8-Flash-Next" and dd.vision_model_2() == "Gemma-4-31B-IT"


def test_unknown_mode_falls_back_to_the_default_and_names_are_forgiving(monkeypatch):
    monkeypatch.setenv("DUAL_MODE", "banana")
    assert dd.mode() == "vision_primary"
    monkeypatch.setenv("DUAL_MODE", " Vote ")
    assert dd.mode() == "vote"


def test_pair_mode_never_calls_the_second_vision_model():
    r = dd.run_dual("x.png", lambda p: det(C), lambda p: det(C, 0.8, "Flash"), timeout=2, dual_mode="pair",
                    vlm2_fn=lambda p: pytest.fail("pair mode must not call the second vision model"))
    assert r["mode"] == "pair" and "vision_model_2" not in r


# ---------------- all agree ----------------

@pytest.mark.parametrize("mode", ["unanimous", "vote", "cnn_anchored"])
def test_all_three_agree(mode):
    r = run(mode, C, C, C)
    assert (r["agreement"], r["needs_human_review"], r["decision"]) == ("agree", False, "unanimous")
    assert r["suggestion"] == "3 of 3 say corrosion" and r["votes"] == {C: 3}
    assert r["models_answered"] == ["CNN", "Flash", "Gemma"]
    assert r["answer"] == {"label": C, "confidence": 0.91, "from": "cnn"}


def test_pair_agree():
    r = run("pair", C, C, None)
    assert (r["agreement"], r["needs_human_review"], r["suggestion"]) == ("agree", False, "2 of 2 say corrosion")


# ---------------- 2-of-3 ----------------

def test_two_of_three_cnn_in_the_majority():
    for mode, review in (("unanimous", True), ("vote", False), ("cnn_anchored", False)):
        r = run(mode, C, C, N)
        assert r["needs_human_review"] is review, mode
        assert r["suggestion"] == "2 of 3 say corrosion" and r["votes"] == {C: 2, N: 1}
        assert r["agreement"] == "disagree"  # the detectors do not all agree, whatever the mode decides
        assert r["answer"]["label"] == C


# ---------------- three-way split ----------------

@pytest.mark.parametrize("mode", ["unanimous", "vote", "cnn_anchored"])
def test_three_way_split_goes_to_a_human_with_all_results(mode):
    r = run(mode, C, N, B)
    assert (r["needs_human_review"], r["agreement"], r["decision"]) == (True, "disagree", r["decision"])
    assert r["suggestion"].startswith("no majority") and set(r["votes"]) == {C, N, B}
    assert r["answer"]["label"] == C and r["answer"]["from"] == "cnn"  # no majority: the CNN's label is the suggestion
    assert r["cnn"]["confidence"] == 0.91 and r["vision_model"]["confidence"] == 0.81 and r["vision_model_2"]["confidence"] == 0.71


# ---------------- CNN outvoted ----------------

def test_cnn_outvoted_by_both_vision_models():
    u, v, c = run("unanimous", C, N, N), run("vote", C, N, N), run("cnn_anchored", C, N, N)
    assert u["needs_human_review"] is True
    assert v["needs_human_review"] is False and v["answer"]["label"] == N and v["answer"]["from"] == "vision_model"
    assert c["needs_human_review"] is True and c["decision"] == "CNN outvoted by both vision models"
    assert c["suggestion"] == "2 of 3 say normal" and c["answer"]["label"] == N  # the suggestion a human sees
    assert c["cnn"]["label"] == C  # and the CNN's own answer is still shown


# ---------------- missing detectors ----------------

@pytest.mark.parametrize("mode", ["unanimous", "vote", "cnn_anchored"])
def test_one_detector_missing_two_left_must_agree(mode):
    ok = run(mode, C, C, None)
    assert (ok["needs_human_review"], ok["agreement"]) == (False, "agree")
    assert ok["models_answered"] == ["CNN", "Flash"] and ok["vision_model_2"]["error"]
    bad = run(mode, C, None, N)
    assert (bad["needs_human_review"], bad["agreement"]) == (True, "disagree")
    assert bad["cnn"]["label"] == C and bad["vision_model_2"]["label"] == N
    assert bad["vision_model"]["label"] is None and bad["vision_model"]["error"] == "RuntimeError"
    both_vision = run(mode, None, N, N)  # the CNN is the missing one
    assert both_vision["needs_human_review"] is False and both_vision["cnn"]["error"]


@pytest.mark.parametrize("mode", ["unanimous", "vote", "cnn_anchored"])
def test_two_detectors_missing_the_last_one_answers_alone(mode):
    r = run(mode, C, None, None)
    assert (r["agreement"], r["needs_human_review"]) == ("single_detector", False)
    assert r["models_answered"] == ["CNN"] and r["answer"]["label"] == C
    r = run(mode, None, None, N)
    assert r["agreement"] == "single_detector" and r["answer"]["from"] == "vision_model_2"


@pytest.mark.parametrize("mode", ["pair", "unanimous", "vote", "cnn_anchored"])
def test_all_missing_returns_none_so_the_caller_falls_back(mode):
    assert run(mode, None, None, None) is None


def test_pair_single_detector_unchanged():
    r = run("pair", C, None, None)
    assert (r["agreement"], r["needs_human_review"]) == ("single_detector", False)


# ---------------- timeouts ----------------

@pytest.mark.parametrize("mode", ["pair", "unanimous", "vote", "cnn_anchored"])
def test_a_slow_vision_model_times_out_and_the_others_still_answer(mode):
    def slow(path):
        time.sleep(3)
        return det(C, 0.8, "Slow")

    start = time.time()
    r = dd.run_dual("x.png", lambda p: det(C), slow, timeout=0.3, dual_mode=mode, vlm2_fn=lambda p: det(C, 0.7, "Gemma"))
    assert time.time() - start < 1.5
    assert r["vision_model"]["error"].startswith("no answer within")
    assert r["needs_human_review"] is False and r["cnn"]["label"] == C


def test_vision_calls_run_in_parallel_with_one_shared_deadline():
    barrier = threading.Barrier(3, timeout=5)  # passes only if CNN and BOTH vision models run at the same time

    def call(label):
        def f(path):
            barrier.wait()
            return det(label, 0.8, "m")
        return f

    r = dd.run_dual("x.png", call(C), call(C), timeout=5, dual_mode="vote", vlm2_fn=call(C))
    assert r["agreement"] == "agree"


def test_two_slow_vision_models_still_return_within_one_timeout():
    def slow(path):
        time.sleep(3)
        return det(C)

    start = time.time()
    r = dd.run_dual("x.png", lambda p: det(C), slow, timeout=0.3, dual_mode="vote", vlm2_fn=slow)
    assert time.time() - start < 1.0  # one deadline for everyone, not 0.3 s per detector one after the other
    assert r["agreement"] == "single_detector" and r["models_answered"] == ["CNN"]


# ---------------- the endpoint: default behaviour, modes, and the agent ----------------

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
def client(agent):
    return TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=object()))


def _case(client):
    return client.post("/cases", files={"file": ("t.png", PNG, "image/png")}, data={"uploaded_by": "t"}).json()["id"]


def _patch(monkeypatch, cnn, v1, v2):
    monkeypatch.setattr(dd, "run_cnn", lambda p: det(cnn, 0.93))
    monkeypatch.setattr(dd, "run_vision_model", lambda p: det(v1, 0.8, "Flash"))
    monkeypatch.setattr(dd, "run_vision_model_2", lambda p: det(v2, 0.7, "Gemma"))


def test_off_by_default_no_detector_runs(client, agent, monkeypatch):
    for name in ("DUAL_DETECTOR", "DUAL_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dd, "run_vision_model", lambda p: pytest.fail("off by default"))
    monkeypatch.setattr(dd, "run_vision_model_2", lambda p: pytest.fail("off by default"))
    assert "dual_detector" not in client.post(f"/cases/{_case(client)}/analyze").json()


def test_pair_can_still_be_chosen_and_never_uses_the_second_vision_model(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "pair")
    _patch(monkeypatch, C, C, N)
    monkeypatch.setattr(dd, "run_vision_model_2", lambda p: pytest.fail("pair must not use the second vision model"))
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    assert body["dual_detector"]["mode"] == "pair" and body["dual_detector"]["agreement"] == "agree"
    assert "vision_model_2" not in body["dual_detector"]
    assert client.get("/health").json()["dual_mode"] == "pair"


def test_vote_mode_endpoint_majority_decides_and_everything_is_shown(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "vote")
    _patch(monkeypatch, C, N, N)
    body = client.post(f"/cases/{_case(client)}/analyze").json()
    d = body["dual_detector"]
    assert (d["mode"], d["needs_human_review"], d["suggestion"]) == ("vote", False, "2 of 3 say normal")
    assert body["diagnosis_raw"]["dual_detector"]["vision_model_2"]["label"] == N
    sent = agent.calls[0]
    assert sent["defect_type"] == "normal" and sent["confidence"] == 0.8  # the owning detector's own confidence
    assert sent["detectors"]["cnn"]["confidence"] == 0.93 and sent["detectors"]["vision_model_2"]["confidence"] == 0.7
    h = client.get("/health").json()
    assert (h["dual_mode"], h["vision_model_2"]) == ("vote", "Gemma-4-31B-IT")


def test_cnn_anchored_endpoint_review_on_outvote(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "cnn_anchored")
    _patch(monkeypatch, C, N, N)
    d = client.post(f"/cases/{_case(client)}/analyze").json()["dual_detector"]
    assert d["needs_human_review"] is True and d["suggestion"] == "2 of 3 say normal"
    assert agent.calls[0]["needs_human_review"] is True


def test_upload_survives_when_vision_models_fail(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "unanimous")
    monkeypatch.setattr(dd, "run_cnn", lambda p: det(C, 0.9))

    def boom(p):
        raise TimeoutError()

    monkeypatch.setattr(dd, "run_vision_model", boom)
    monkeypatch.setattr(dd, "run_vision_model_2", boom)
    r = client.post(f"/cases/{_case(client)}/analyze")
    assert r.status_code == 200 and r.json()["dual_detector"]["agreement"] == "single_detector"


def test_every_detector_failing_falls_back_to_the_normal_path(client, agent, monkeypatch):
    monkeypatch.setenv("DUAL_DETECTOR", "on")
    monkeypatch.setenv("DUAL_MODE", "vote")

    def boom(p):
        raise RuntimeError()

    for name in ("run_cnn", "run_vision_model", "run_vision_model_2"):
        monkeypatch.setattr(dd, name, boom)
    r = client.post(f"/cases/{_case(client)}/analyze")
    assert r.status_code == 200 and "dual_detector" not in r.json()  # the single-CNN / simulated path took over


# ---------------- the real agent: any review skips the LLM and states every detector ----------------

def test_agent_states_all_three_detectors_and_the_votes(monkeypatch):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent as real
    monkeypatch.setattr(real, "call_llm", lambda *a, **k: pytest.fail("the LLM must not run on a review"))
    report = {"defect_type": "normal", "location": "pipe B7", "confidence": 0.8, "needs_human_review": True,
              "detectors": {"cnn": {"label": C, "confidence": 0.93},
                            "vision_model": {"name": "Flash", "label": N, "confidence": 0.8},
                            "vision_model_2": {"name": "Gemma", "label": N, "confidence": 0.7},
                            "suggestion": "2 of 3 say normal"}}
    d = real.diagnose(report)
    ex = d["explanation"]
    assert d["raw"]["needs_manual_verification"] is True and ex.startswith("UNVERIFIED")
    for text in ("DISAGREE", "CNN says corrosion (confidence 0.93)", "Flash says normal (confidence 0.8)",
                 "Gemma says normal (confidence 0.7)", "Votes: 2 of 3 say normal", "not combined"):
        assert text in ex
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


def test_agent_names_a_detector_that_gave_no_answer(monkeypatch):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent as real
    monkeypatch.setattr(real, "call_llm", lambda *a, **k: pytest.fail("no LLM on a review"))
    report = {"defect_type": "corrosion", "location": "x", "confidence": 0.9, "needs_human_review": True,
              "detectors": {"cnn": {"label": C, "confidence": 0.9},
                            "vision_model": {"name": "Flash", "label": N, "confidence": 0.8},
                            "vision_model_2": {"name": "Gemma", "label": None, "confidence": None, "error": "timeout"}}}
    assert "Gemma gave no answer" in real.diagnose(report)["explanation"]
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
