"""Grounding tests with the REAL manuals (data/raw) and a scripted LLM - CPU only, no API key needed.

They check the things judges will ask: is every citation a real page, is off-topic input refused,
and does a made-up evidence quote get caught?"""
import json
import sys

import pytest

DEFECTS = [  # (defect report, manual the citation must come from)
    ({"defect_type": "crack", "location": "weld joint A3", "confidence": 0.91}, "structural_inspection_sop.pdf"),
    ({"defect_type": "corrosion", "location": "pipe section B7", "confidence": 0.88}, "corrosion_control_guide.pdf"),
    ({"defect_type": "loose_bolt", "location": "panel D5", "confidence": 0.95}, "fastener_alignment_sop.pdf"),
    ({"defect_type": "misalignment", "location": "conveyor belt C2", "confidence": 0.9}, "fastener_alignment_sop.pdf"),
    ({"defect_type": "bearing_wear", "location": "motor M2", "confidence": 0.9}, "bearing_maintenance_manual.pdf"),
]


@pytest.fixture
def agent_mod(monkeypatch):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent
    prompts = []
    agent.prompts = prompts
    agent.reply = {"explanation": "x", "recommended_action": "y", "severity": "high", "source_index": 0}

    def fake_llm(prompt, system=""):
        prompts.append(prompt)
        r = agent.reply
        return r if isinstance(r, str) else json.dumps(r)

    monkeypatch.setattr(agent, "call_llm", fake_llm)

    def no_backup(prompt, system=""):  # the backup text model is never reached for real in these tests
        raise RuntimeError("backup unavailable")

    monkeypatch.setattr(agent, "call_llm_backup", no_backup)
    yield agent
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("report,manual", DEFECTS)
def test_citation_is_the_right_real_manual(agent_mod, report, manual):
    d = agent_mod.diagnose(report)
    assert d["source_manual"] == manual
    assert d["source_page"] >= 1


def test_off_topic_report_is_refused_before_the_llm(agent_mod):
    with pytest.raises(agent_mod.NoRelevantSOPError):
        agent_mod.diagnose({"defect_type": "banana smoothie", "location": "kitchen", "confidence": 0.9})
    assert agent_mod.prompts == []


def test_real_quote_is_verified_and_fake_quote_is_flagged(agent_mod):
    import core
    top = core.retrieve("crack at weld joint A3")[0]
    words = top["text"].split()
    real_quote = " ".join(words[5:15])

    agent_mod.reply = {"explanation": "x", "recommended_action": "y", "severity": "high",
                       "source_index": 0, "evidence_quote": real_quote}
    g = agent_mod.diagnose(DEFECTS[0][0])["raw"]["grounding"]
    assert g["evidence_quote_verified"] is True

    agent_mod.reply = {"explanation": "x", "recommended_action": "y", "severity": "high",
                       "source_index": 0, "evidence_quote": "always replace the entire factory immediately"}
    g = agent_mod.diagnose(DEFECTS[0][0])["raw"]["grounding"]
    assert g["evidence_quote_verified"] is False

    agent_mod.reply = {"explanation": "x", "recommended_action": "y", "severity": "high", "source_index": 0}
    assert agent_mod.diagnose(DEFECTS[0][0])["raw"]["grounding"]["evidence_quote_verified"] is False  # no quote = unverified


def test_invalid_json_is_retried_once(agent_mod):
    agent_mod.reply = "not json at all"
    agent_mod.diagnose(DEFECTS[0][0])
    assert len(agent_mod.prompts) == 2  # first try + one retry, then it falls back (never loops)

def test_no_llm_configured_raises_a_controlled_error(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "LOCAL_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import core
    with pytest.raises(core.LLMUnavailableError):
        core.call_llm("hello")
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
