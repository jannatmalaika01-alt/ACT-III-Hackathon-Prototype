"""Tests for the AI agent and POST /cases/{id}/diagnose.

Groq and Chroma are replaced by a fake `core` module, so these run offline with no API key."""
import sys
import types

import pytest

from tests.helpers import approve, diagnose as post_diagnosis, upload_case

DEFECT = {"defect_type": "crack", "location": "weld joint A3", "confidence": 0.91, "image_ref": "img_014.png"}
GOOD_REPLY = (
    '<think>hmm</think>```json\n{"explanation": "Crack in weld", "recommended_action": '
    '"Flag for manual inspection", "severity": "CRITICAL", "source_index": 0}\n```'
)


@pytest.fixture
def fake_core(monkeypatch):
    mod = types.ModuleType("core")
    mod.FAKE_DEFECTS = []
    mod.docs = [
        {"manual": "Weld Inspection SOP", "page": 12, "text": "weld text"},
        {"manual": "Mechanical Alignment SOP", "page": 4, "text": "align text"},
    ]
    mod.reply = GOOD_REPLY
    mod.prompts = []

    def call_llm(prompt, system=""):
        mod.prompts.append(prompt)
        return mod.reply

    mod.call_llm = call_llm
    mod.retrieve = lambda query, n_results=2: mod.docs
    monkeypatch.setitem(sys.modules, "core", mod)
    monkeypatch.delitem(sys.modules, "agent", raising=False)  # force agent.py to re-import with the fake core
    yield mod
    sys.modules.pop("agent", None)


# ---------------- agent ----------------

def test_high_confidence_uses_llm_and_real_citation(fake_core):
    import agent
    d = agent.diagnose(DEFECT)
    assert d["severity"] == "critical"
    assert (d["source_manual"], d["source_page"]) == ("Weld Inspection SOP", 12)
    assert d["confidence"] == 0.91
    assert d["raw"]["severity_verified"] is True


def test_low_confidence_skips_llm_and_is_flagged_unverified(fake_core):
    import agent
    d = agent.diagnose({**DEFECT, "confidence": 0.5})
    assert fake_core.prompts == []
    assert d["explanation"].startswith("UNVERIFIED")
    assert d["raw"]["needs_manual_verification"] is True and d["raw"]["severity_verified"] is False
    assert "Manually verify" in d["recommended_action"]


def test_bad_llm_output_falls_back_and_never_invents_a_citation(fake_core):
    import agent
    fake_core.reply = "I think it is bad"
    d = agent.diagnose(DEFECT)
    assert d["severity"] == "medium" and d["explanation"] == "I think it is bad"
    fake_core.reply = '{"explanation": "x", "recommended_action": "y", "severity": "low", "source_index": 99}'
    assert agent.diagnose(DEFECT)["source_manual"] == "Weld Inspection SOP"  # bad index -> first retrieved doc


def test_empty_retrieval_raises_before_calling_llm(fake_core):
    import agent
    fake_core.docs = []
    with pytest.raises(agent.NoRelevantSOPError):
        agent.diagnose(DEFECT)
    assert fake_core.prompts == []


def test_rejection_feedback_reaches_the_prompt(fake_core):
    import agent
    agent.diagnose(DEFECT, rejection_feedback="this pipe carries pressurized fluid")
    assert "pressurized fluid" in fake_core.prompts[0]


# ---------------- endpoint ----------------

def test_diagnose_stores_diagnosis_and_case_stays_pending(client, fake_core):
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnose", json=DEFECT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["source_manual"] == "Weld Inspection SOP" and body["source_page"] == 12
    assert approve(client, cid).json()["status"] == "task_created"  # a human can still approve it


def test_diagnose_passes_rejection_feedback(client, fake_core):
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnose", json={**DEFECT, "rejection_feedback": "shut the line instead"})
    assert r.status_code == 200
    assert "shut the line instead" in fake_core.prompts[0]


def test_diagnose_no_sop_returns_422_and_leaves_case_undiagnosed(client, fake_core):
    fake_core.docs = []
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnose", json=DEFECT)
    assert r.status_code == 422
    assert client.get(f"/cases/{cid}").json()["defect"] is None


def test_diagnose_llm_failure_returns_generic_message(client, fake_core):
    def boom(prompt, system=""):
        raise RuntimeError("secret-api-key-123")
    fake_core.call_llm = boom
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnose", json=DEFECT)
    assert r.status_code == 502
    assert r.json()["detail"] == "AI diagnosis failed."
    assert "secret-api-key-123" not in r.text


def test_diagnose_agent_not_installed_returns_503(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent", None)  # makes `from agent import ...` raise ImportError
    cid = upload_case(client)
    r = client.post(f"/cases/{cid}/diagnose", json=DEFECT)
    assert r.status_code == 503 and r.json()["detail"] == "AI agent is not available."


def test_diagnose_validation_and_state_errors(client, fake_core):
    assert client.post("/cases/00000000-0000-0000-0000-000000000000/diagnose", json=DEFECT).status_code == 404
    cid = upload_case(client)
    assert client.post(f"/cases/{cid}/diagnose", json={**DEFECT, "confidence": 3}).status_code == 422
    post_diagnosis(client, cid)
    approve(client, cid)
    assert client.post(f"/cases/{cid}/diagnose", json=DEFECT).status_code == 409  # no longer pending
