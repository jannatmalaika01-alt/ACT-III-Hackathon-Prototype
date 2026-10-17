import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import InMemoryRepo
from app.evolus import MockEvolusClient
from app.main import create_app

SECRET = "test-secret"

LLM_ENV = ("TOKENFACTORY_BASE_URL", "TOKENFACTORY_API_KEY", "TOKENFACTORY_MODEL", "TOKENFACTORY_JSON_SCHEMA",
           "LOCAL_LLM_BASE_URL", "GROQ_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Tests must never reach a real LLM (it spends the team's shared quota): ignore backend/.env and clear
    every provider variable. Tests that need a provider set their own (see test_tokenfactory.py)."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    # With no key the app uses the offline template writer. The older tests assert what the LLM paths do, so they switch it off;
    # tests/test_offline_writer.py turns it on.
    monkeypatch.setenv("OFFLINE_WRITER", "off")


@pytest.fixture
def settings():
    return Settings(
        supabase_url="", supabase_key="", storage_bucket="test",
        webhook_secret=SECRET, evolus_mode="mock", evolus_mock_fail_first_n=0,
        max_upload_mb=1, use_memory_db=True,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings=settings, repo=InMemoryRepo(), evolus=MockEvolusClient())
    return TestClient(app)
