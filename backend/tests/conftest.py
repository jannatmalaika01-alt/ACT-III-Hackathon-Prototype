import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import InMemoryRepo
from app.evolus import MockEvolusClient
from app.main import create_app

SECRET = "test-secret"


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
