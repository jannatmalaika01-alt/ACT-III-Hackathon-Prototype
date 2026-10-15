"""Token Factory provider in core.py, with the HTTP layer mocked. The real API is never called."""
import json
import sys

import httpx
import pytest

ENV_KEYS = ("TOKENFACTORY_BASE_URL", "TOKENFACTORY_API_KEY", "TOKENFACTORY_MODEL", "LOCAL_LLM_BASE_URL",
            "GROQ_API_KEY", "GEMINI_API_KEY")
GOOD = {"explanation": "e", "recommended_action": "a", "severity": "high", "source_index": 0,
        "evidence_quote": "q"}


class FakeResp:
    def __init__(self, status=200, content=json.dumps(GOOD), finish="stop", model="MiMo-V2.6-Flash", headers=None):
        self.status_code, self.headers = status, headers or {}
        self._body = {"model": model, "choices": [{"finish_reason": finish,
                                                   "message": {"content": content, "reasoning": "secret thoughts"}}]}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=self)


@pytest.fixture
def core(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TOKENFACTORY_BASE_URL", "https://tf.example/v1")
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TOKENFACTORY_MODEL", "MiMo-V2.6-Flash")
    sys.modules.pop("core", None)
    import core
    core.LAST_LLM.update(provider=None, model=None)
    core.sleeps = []
    monkeypatch.setattr(core, "_sleep", core.sleeps.append)
    core.calls = []
    core.script = []

    def fake_post(url, headers=None, json=None, timeout=None):
        core.calls.append({"url": url, "headers": headers, "json": json})
        return core.script.pop(0)

    monkeypatch.setattr(core.httpx, "post", fake_post)
    yield core
    sys.modules.pop("core", None)


def test_success_sends_token_factory_parameters(core):
    core.script = [FakeResp()]
    assert json.loads(core.call_llm("hi")) == GOOD
    call = core.calls[0]
    assert call["url"] == "https://tf.example/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key-not-real"
    body = call["json"]
    assert body["reasoning_effort"] == "low" and body["max_tokens"] >= 1500 and body["temperature"] == 0.2
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    # the live server rejects json_schema for MiMo (HTTP 400), so json_object is the default
    assert body["response_format"] == {"type": "json_object"}
    assert set(body) == {"model", "temperature", "messages", "reasoning_effort", "max_tokens", "response_format"}
    assert core.LAST_LLM == {"provider": "tokenfactory", "model": "MiMo-V2.6-Flash"}


def test_control_characters_in_the_key_are_removed(core, monkeypatch):
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "abc\x1bdef\n")  # pasted from a terminal
    core.script = [FakeResp()]
    core.call_llm("hi")
    assert core.calls[0]["headers"]["Authorization"] == "Bearer abcdef"


def test_other_models_use_json_object(core, monkeypatch):
    monkeypatch.setenv("TOKENFACTORY_MODEL", "Qwen3.8-27B")
    core.script = [FakeResp(model="Qwen3.8-27B")]
    core.call_llm("hi")
    assert core.calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert core.LAST_LLM["model"] == "Qwen3.8-27B"


def test_model_name_comes_from_the_response(core):
    core.script = [FakeResp(model="MiMo-V2.6-Flash-2026-10")]
    core.call_llm("hi")
    assert core.LAST_LLM["model"] == "MiMo-V2.6-Flash-2026-10"


def test_reasoning_is_ignored(core):
    core.script = [FakeResp(content="")]  # content empty, reasoning present, finish_reason stop
    with pytest.raises(core.LLMUnavailableError) as e:
        core.call_llm("hi")
    assert "secret thoughts" not in str(e.value)


def test_429_honours_retry_after_then_succeeds(core):
    core.script = [FakeResp(429, headers={"Retry-After": "7"}), FakeResp()]
    assert json.loads(core.call_llm("hi")) == GOOD
    assert core.sleeps == [7.0] and len(core.calls) == 2


def test_503_backs_off_with_jitter_and_gives_up_after_3_retries(core):
    core.script = [FakeResp(503) for _ in range(4)]
    with pytest.raises(core.LLMUnavailableError) as e:
        core.call_llm("hi")
    assert len(core.calls) == 4 and len(core.sleeps) == 3
    assert core.sleeps[0] < core.sleeps[1] < core.sleeps[2]
    assert "HTTP 503" in str(e.value)


def test_400_is_not_retried(core):
    core.script = [FakeResp(400)]
    with pytest.raises(core.LLMUnavailableError):
        core.call_llm("hi")
    assert len(core.calls) == 1 and core.sleeps == []


def test_empty_content_with_length_retries_once_with_more_tokens(core):
    core.script = [FakeResp(content="", finish="length"), FakeResp()]
    assert json.loads(core.call_llm("hi")) == GOOD
    assert core.calls[1]["json"]["max_tokens"] > core.calls[0]["json"]["max_tokens"]


def test_empty_content_with_length_twice_fails(core):
    core.script = [FakeResp(content="", finish="length"), FakeResp(content="", finish="length")]
    with pytest.raises(core.LLMUnavailableError):
        core.call_llm("hi")
    assert len(core.calls) == 2


def test_missing_key_gives_clear_error_and_no_request(core, monkeypatch):
    monkeypatch.delenv("TOKENFACTORY_API_KEY")
    with pytest.raises(core.LLMUnavailableError) as e:
        core.call_llm("hi")
    assert "TOKENFACTORY_API_KEY is not set" in str(e.value)
    assert core.calls == []


def test_falls_back_to_next_provider(core, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    core.script = [FakeResp(500), FakeResp(model="openai/gpt-oss-120b")]
    assert json.loads(core.call_llm("hi")) == GOOD
    assert core.calls[0]["url"].startswith("https://tf.example")
    assert core.calls[1]["url"].startswith("https://api.groq.com")
    assert "reasoning_effort" not in core.calls[1]["json"]  # TF-only parameters
    assert core.LAST_LLM["provider"] == "groq"


def test_token_factory_is_tried_first(core, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://gpu:8000/v1")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert [p["name"] for p in core._providers()] == ["tokenfactory", "local-vllm", "groq", "gemini"]


def test_health_reports_the_provider_that_answered(client, core):
    assert client.get("/health").json()["llm_provider"] == "tokenfactory"
    assert client.get("/health").json()["llm_answered"] is False
    core.script = [FakeResp()]
    core.call_llm("hi")
    h = client.get("/health").json()
    assert (h["llm_provider"], h["reasoner"], h["llm_answered"]) == ("tokenfactory", "MiMo-V2.6-Flash", True)


def test_json_schema_is_opt_in_and_sent_strict(core, monkeypatch):
    monkeypatch.setenv("TOKENFACTORY_JSON_SCHEMA", "true")
    core.script = [FakeResp()]
    core.call_llm("hi")
    rf = core.calls[0]["json"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert set(schema["required"]) == {"explanation", "recommended_action", "severity", "source_index",
                                       "evidence_quote"}
    assert schema["properties"]["severity"]["enum"] == ["low", "medium", "high", "critical"]


def test_server_rejecting_json_schema_falls_back_to_json_object(core, monkeypatch):
    """Regression: the live API answers 400 'Model MiMo-V2.6-Flash does not support JSON schema output mode'."""
    monkeypatch.setenv("TOKENFACTORY_JSON_SCHEMA", "true")
    bad = FakeResp(400)
    bad.json = lambda: {"error": {"message": "Model MiMo-V2.6-Flash does not support JSON schema output mode"}}
    core.script = [bad, FakeResp()]
    assert json.loads(core.call_llm("hi")) == GOOD
    assert core.calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert core.calls[1]["json"]["response_format"] == {"type": "json_object"}
    assert core.sleeps == []  # a 400 is not retried with backoff


def test_http_error_log_has_status_and_message_but_never_the_key(core, caplog):
    bad = FakeResp(400)
    bad.json = lambda: {"error": {"message": "unsupported parameter foo"}}
    core.script = [bad]
    with caplog.at_level("WARNING", logger="core"), pytest.raises(core.LLMUnavailableError) as e:
        core.call_llm("hi")
    assert "HTTP 400 unsupported parameter foo" in caplog.text
    assert "unsupported parameter foo" in str(e.value)
    assert "test-key-not-real" not in caplog.text + str(e.value)
