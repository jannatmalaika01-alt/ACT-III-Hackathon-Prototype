"""Writer step: primary MiMo-V2.6-Flash, ONE try with the backup DeepSeek-V4-Flash (effort none), else the
'recommendation unavailable' case. The real manuals and the real agent are used; every model call is mocked."""
import json
import sys

import httpx
import pytest


@pytest.fixture
def ag(monkeypatch):
    for name in ("core", "agent"):
        sys.modules.pop(name, None)
    import agent
    import core
    top = core.retrieve("crack at weld joint A3")[0]
    agent.good_quote = " ".join(top["text"].split()[5:15])
    agent.source = top
    agent.calls = {"primary": 0, "backup": 0}
    agent.primary_script, agent.backup_script = [], []

    def make(role, script):
        def call(prompt, system=""):
            agent.calls[role] += 1
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item(agent.good_quote) if callable(item) else item
        return call

    monkeypatch.setattr(agent, "call_llm", make("primary", agent.primary_script))
    monkeypatch.setattr(agent, "call_llm_backup", make("backup", agent.backup_script))
    yield agent
    for name in ("core", "agent"):
        sys.modules.pop(name, None)


def reply(severity="high", action="Inspect the weld.", quote=None):
    def build(good):
        return json.dumps({"explanation": "A crack.", "recommended_action": action, "severity": severity,
                           "source_index": 0, "evidence_quote": good if quote is None else quote})
    return build


REPORT = {"defect_type": "crack", "location": "weld joint A3", "confidence": 0.91,
          "detectors": {"cnn": {"label": "structural_crack", "confidence": 0.5},
                        "vision_model": {"name": "Flash", "label": "structural_crack", "confidence": 0.9}}}


def test_primary_ok_the_backup_is_never_called(ag):
    ag.primary_script.append(reply())
    d = ag.diagnose(dict(REPORT))
    assert ag.calls == {"primary": 1, "backup": 0}
    w = d["raw"]["grounding"]["writer"]
    assert (w["role"], w["written_by"]) == ("primary", "written by MiMo-V2.6-Flash")
    assert d["explanation"].endswith("(written by MiMo-V2.6-Flash)") and d["recommended_action"] == "Inspect the weld."


@pytest.mark.parametrize("why,primary", [
    ("timeout", [httpx.ReadTimeout("slow")]),
    ("http 5xx / 429 after retries", [RuntimeError("All LLM providers failed (tokenfactory: HTTP 503)")]),
    ("empty response", ["", ""]),
    ("invalid json", ["not json at all", "still not json"]),
    ("bad citation", [reply(quote="always replace the entire factory immediately")]),
    ("no citation", [lambda good: json.dumps({"explanation": "x", "recommended_action": "y", "severity": "low"})]),
])
def test_primary_fails_then_the_backup_writes_it(ag, why, primary):
    ag.primary_script.extend(primary)
    ag.backup_script.append(reply(severity="critical", action="Backup action."))
    d = ag.diagnose(dict(REPORT))
    assert ag.calls["backup"] == 1, why
    w = d["raw"]["grounding"]["writer"]
    assert (w["role"], w["model"], w["written_by"]) == ("fallback", "DeepSeek-V4-Flash", "written by DeepSeek-V4-Flash (fallback)")
    assert d["recommended_action"] == "Backup action." and d["severity"] == "critical"
    assert d["explanation"].endswith("(written by DeepSeek-V4-Flash (fallback))")
    assert w["attempts"][0]["role"] == "primary" and w["attempts"][0]["problem"]
    assert d["raw"]["grounding"]["evidence_quote_verified"] is True
    assert (d["source_manual"], d["source_page"]) == (ag.source["manual"], ag.source["page"])


def test_invalid_json_keeps_the_existing_single_retry_on_the_primary_before_switching(ag):
    ag.primary_script.extend(["nope", reply()])  # second primary reply is fine: no backup needed
    ag.diagnose(dict(REPORT))
    assert ag.calls == {"primary": 2, "backup": 0}


def test_both_fail_returns_the_unavailable_case_with_detectors_and_citation(ag):
    ag.primary_script.append(reply(quote="this quote is not in the manual at all"))
    ag.backup_script.append(RuntimeError("secret-api-key-123"))
    d = ag.diagnose(dict(REPORT))                      # does not raise
    assert ag.calls["backup"] == 1                     # ONE try, no loop
    assert d["explanation"].startswith("RECOMMENDATION UNAVAILABLE")
    assert "CNN says structural_crack (confidence 0.5)" in d["explanation"]     # detector results still shown
    assert "Flash says structural_crack (confidence 0.9)" in d["explanation"]
    assert "citation check failed" in d["explanation"] and "secret-api-key-123" not in json.dumps(d)
    assert d["recommended_action"].startswith("Recommendation unavailable")
    assert (d["source_manual"], d["source_page"]) == (ag.source["manual"], ag.source["page"])   # citation still shown
    assert d["severity"] == "medium" and d["raw"]["needs_manual_verification"] is True
    w = d["raw"]["grounding"]["writer"]
    assert (w["role"], w["written_by"]) == ("none", "no text model produced an answer")
    assert [a["role"] for a in w["attempts"]] == ["primary", "fallback"]


def test_both_fail_when_the_backup_also_has_a_bad_citation(ag):
    ag.primary_script.append(httpx.ReadTimeout("slow"))
    ag.backup_script.append(reply(quote="made up words that are not in the manual"))
    d = ag.diagnose(dict(REPORT))
    assert d["explanation"].startswith("RECOMMENDATION UNAVAILABLE") and ag.calls == {"primary": 1, "backup": 1}


def test_a_low_confidence_or_review_case_never_calls_either_writer(ag):
    d = ag.diagnose({**REPORT, "confidence": 0.5})
    assert ag.calls == {"primary": 0, "backup": 0} and d["explanation"].startswith("UNVERIFIED")


# ---------------- core.call_llm_backup: same URL and key, backup model, effort none ----------------

class FakeResp:
    def __init__(self, status=200, content='{"ok": true}', model="DeepSeek-V4-Flash"):
        self.status_code, self.headers = status, {}
        self._b = {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=self)


@pytest.fixture
def core_tf(monkeypatch):
    for k in ("TOKENFACTORY_BASE_URL", "TOKENFACTORY_API_KEY", "TOKENFACTORY_MODEL", "TEXT_FALLBACK_MODEL",
              "TEXT_FALLBACK_EFFORT", "TOKENFACTORY_REASONING_EFFORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TOKENFACTORY_BASE_URL", "https://tf.example/v1")
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "test-key-not-real")
    sys.modules.pop("core", None)
    import core
    core.calls, core.script = [], []
    monkeypatch.setattr(core, "_sleep", lambda s: None)

    def fake_post(url, headers=None, json=None, timeout=None):
        core.calls.append({"url": url, "headers": headers, "json": json})
        return core.script.pop(0)

    monkeypatch.setattr(core.httpx, "post", fake_post)
    yield core
    sys.modules.pop("core", None)


def test_backup_defaults_and_request(core_tf):
    core_tf.script = [FakeResp()]
    assert json.loads(core_tf.call_llm_backup("hi")) == {"ok": True}
    c = core_tf.calls[0]
    assert c["url"] == "https://tf.example/v1/chat/completions"                 # the same Token Factory base URL
    assert c["headers"]["Authorization"] == "Bearer test-key-not-real"           # the same key
    assert c["json"]["model"] == "DeepSeek-V4-Flash" and c["json"]["reasoning_effort"] == "none"
    assert c["json"]["response_format"] == {"type": "json_object"}
    assert core_tf.LAST_LLM["model"] == "DeepSeek-V4-Flash"
    assert (core_tf.text_fallback_model(), core_tf.text_fallback_effort(), core_tf.primary_text_model()) == \
        ("DeepSeek-V4-Flash", "none", "MiMo-V2.6-Flash")


def test_backup_settings_are_configurable(core_tf, monkeypatch):
    monkeypatch.setenv("TEXT_FALLBACK_MODEL", "Qwen3.8-27B")
    monkeypatch.setenv("TEXT_FALLBACK_EFFORT", "low")
    core_tf.script = [FakeResp(model="Qwen3.8-27B")]
    core_tf.call_llm_backup("hi")
    assert (core_tf.calls[0]["json"]["model"], core_tf.calls[0]["json"]["reasoning_effort"]) == ("Qwen3.8-27B", "low")


def test_backup_is_independent_of_the_primary_effort_setting(core_tf, monkeypatch):
    monkeypatch.setenv("TOKENFACTORY_REASONING_EFFORT", "low")
    core_tf.script = [FakeResp()]
    core_tf.call_llm_backup("hi")
    assert core_tf.calls[0]["json"]["reasoning_effort"] == "none"


def test_backup_failure_is_one_clean_error_without_the_key(core_tf):
    core_tf.script = [FakeResp(500)]
    with pytest.raises(core_tf.LLMUnavailableError) as e:
        core_tf.call_llm_backup("hi")
    assert "HTTP 500" in str(e.value) and "test-key-not-real" not in str(e.value) and len(core_tf.calls) == 1


def test_backup_without_token_factory_is_a_clean_error(core_tf, monkeypatch):
    monkeypatch.delenv("TOKENFACTORY_API_KEY")
    monkeypatch.delenv("TOKENFACTORY_BASE_URL")
    with pytest.raises(core_tf.LLMUnavailableError, match="not configured"):
        core_tf.call_llm_backup("hi")
    assert core_tf.calls == []
