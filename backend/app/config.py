"""Settings loaded from environment variables (.env in development)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    storage_bucket: str
    webhook_secret: str
    evolus_mode: str            # "mock" | "rest" | "mcp"
    evolus_mock_fail_first_n: int
    max_upload_mb: int
    use_memory_db: bool
    evolus_base_url: str = "https://api.evolus.ai/api/v1"
    evolus_api_key: str = ""
    evolus_application_id: str = ""
    evolus_workflow_id: str = ""
    evolus_poll_interval: float = 5.0   # seconds between result checks (EVOLUS_POLL_INTERVAL)
    evolus_poll_timeout: float = 240.0  # give up waiting for the agent after this long (EVOLUS_POLL_TIMEOUT)


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        storage_bucket=os.getenv("STORAGE_BUCKET", "case-uploads"),
        webhook_secret=os.getenv("WEBHOOK_SECRET", ""),
        evolus_mode=os.getenv("EVOLUS_MODE", "mock").strip().lower(),
        evolus_mock_fail_first_n=int(os.getenv("EVOLUS_MOCK_FAIL_FIRST_N", "0") or 0),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10") or 10),
        use_memory_db=_bool(os.getenv("USE_IN_MEMORY_DB", "false")),
        evolus_base_url=os.getenv("EVOLUS_BASE_URL", "https://api.evolus.ai/api/v1").strip().rstrip("/"),
        evolus_api_key=os.getenv("EVOLUS_API_KEY", "").strip(),
        evolus_application_id=os.getenv("EVOLUS_APPLICATION_ID", "").strip(),
        evolus_workflow_id=os.getenv("EVOLUS_WORKFLOW_ID", "").strip(),
        evolus_poll_interval=float(os.getenv("EVOLUS_POLL_INTERVAL", "5") or 5),
        evolus_poll_timeout=float(os.getenv("EVOLUS_POLL_TIMEOUT", "240") or 240),
    )
