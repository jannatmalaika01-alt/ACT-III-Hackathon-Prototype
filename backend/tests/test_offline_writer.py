"""With NO LLM key the app still works: the recommendation is the manual's own 'Action Required' step, clearly labelled."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import InMemoryRepo
from app.evolus import MockEvolusClient
from app.main import create_app

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "test" / "test_14.png"      # a corrosion image that ships in the repo


@pytest.fixture
def ag(monkeypatch):
    monkeypatch.setenv("OFFLINE_WRITER", "auto")                  # conftest switches it off for the older tests
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent
    import core

    def no_llm(*a, **k):
        raise AssertionError("the offline writer must not call any LLM")

    monkeypatch.setattr(agent, "call_llm", no_llm)
    monkeypatch.setattr(agent, "call_llm_backup", no_llm)
    yield agent, core
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


def report(defect, conf=0.9):
    return {"defect_type": defect, "location": "pipe flange", "confidence": conf}


@pytest.mark.parametrize("defect,manual,severity", [("corrosion", "corrosion_control_guide.pdf", "medium"),
                                                    ("crack", "structural_inspection_sop.pdf", "critical"),
                                                    ("bearing_wear", "bearing_maintenance_manual.pdf", "high")])
def test_the_action_step_is_copied_from_the_manual(ag, defect, manual, severity):
    agent, core = ag
    d = agent.diagnose(report(defect))
    assert d["source_manual"] == manual and d["source_page"] >= 1
    assert d["severity"] == severity                                           # taken from the manual's own "Priority"
    assert d["recommended_action"] and "Recommendation unavailable" not in d["recommended_action"]
    top = [x for x in core.retrieve(f"{defect} at pipe flange") if x["manual"] == manual and x["page"] == d["source_page"]][0]
    assert " ".join(d["recommended_action"].split()[:6]) in " ".join(top["text"].split())     # really copied from that page


def test_it_is_labelled_as_a_template_everywhere(ag):
    agent, _ = ag
    d = agent.diagnose(report("corrosion"))
    assert "Template text, not written by an AI model" in d["explanation"]
    assert d["explanation"].endswith("(written by an offline template from the manual (no LLM is configured))")
    w = d["raw"]["grounding"]["writer"]
    assert (w["role"], w["model"]) == ("offline", None) and "offline template" in w["written_by"]
    assert d["raw"]["grounding"]["evidence_quote_verified"] is True and d["raw"]["llm"] is None
    assert d["raw"]["needs_manual_verification"] is False


def test_a_normal_part_gets_no_repair_step(ag):
    agent, _ = ag
    d = agent.diagnose(report("normal"))
    assert d["recommended_action"].startswith("No defect was detected") and d["severity"] == "low"
    assert "no repair step is proposed" in d["explanation"] and d["raw"]["needs_manual_verification"] is False
    assert d["source_manual"]                                                     # a citation is still attached


def test_a_passage_without_an_action_step_flags_the_case(ag, monkeypatch):
    agent, core = ag
    monkeypatch.setattr(agent, "retrieve", lambda q, n_results=2: [{"manual": "m.pdf", "page": 7, "text": "Background reading about corrosion."}])
    d = agent.diagnose(report("corrosion"))
    assert d["raw"]["needs_manual_verification"] is True and d["recommended_action"].startswith("Review the cited manual page")
    assert d["raw"]["grounding"]["evidence_quote_verified"] is False and (d["source_manual"], d["source_page"]) == ("m.pdf", 7)


def test_the_detector_notes_still_appear(ag):
    agent, _ = ag
    d = agent.diagnose({**report("corrosion"), "cnn_dissent": {"label": "structural_crack", "confidence": 0.82}, "warnings": ["Gemma failed"]})
    assert "CNN dissent, low trust" in d["explanation"] and "Warning: Gemma failed" in d["explanation"]


def test_low_confidence_still_goes_to_a_person_without_any_writer(ag):
    agent, _ = ag
    assert agent.diagnose(report("corrosion", conf=0.4))["explanation"].startswith("UNVERIFIED")


def test_it_can_be_switched_off_and_a_key_turns_it_off(ag, monkeypatch):
    agent, core = ag
    assert core.offline_writer_active() is True
    monkeypatch.setenv("OFFLINE_WRITER", "off")
    assert core.offline_writer_active() is False
    assert agent.diagnose(report("corrosion"))["explanation"].startswith("RECOMMENDATION UNAVAILABLE")   # the old behaviour
    monkeypatch.setenv("OFFLINE_WRITER", "auto")
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TOKENFACTORY_BASE_URL", "https://tf.example/v1")
    assert core.offline_writer_active() is False                                # a configured provider means real writers


def test_health_says_which_writer_is_in_use(monkeypatch):
    monkeypatch.setenv("OFFLINE_WRITER", "auto")
    c = TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=MockEvolusClient()))
    h = c.get("/health").json()
    assert h["llm_provider"] == "offline" and "no LLM key" in h["reasoner"] and h["evolus_mode"] == "mock"


def test_no_keys_end_to_end_through_the_real_pipeline(monkeypatch):
    """Upload a photo, analyse it with the real CNN and the real manuals, approve it with the mock Evolus: no key anywhere."""
    monkeypatch.setenv("OFFLINE_WRITER", "auto")
    monkeypatch.delenv("DUAL_DETECTOR", raising=False)
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    try:
        c = TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=MockEvolusClient()))
        cid = c.post("/cases", files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "image/png")}, data={"uploaded_by": "judge"}).json()["id"]
        r = c.post(f"/cases/{cid}/analyze")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["status"] == "pending" and d["defect"].startswith("corrosion") and d["source_manual"] == "corrosion_control_guide.pdf"
        assert d["recommended_action"].startswith("Scrape loose rust") and "Template text" in d["explanation"]
        assert d["diagnosis_raw"]["defect_source"] == "vision"
        a = c.post(f"/cases/{cid}/approve", json={"decided_by": "judge"}).json()
        assert a["status"] == "task_created" and a["evolus_task_id"].startswith("MOCK-TASK-")
    finally:
        for name in ("core", "agent"):
            sys.modules.pop(name, None)
