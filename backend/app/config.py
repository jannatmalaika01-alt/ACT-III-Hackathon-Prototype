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
    evolus_mode: str            # "mock" | "mcp"
    evolus_mock_fail_first_n: int
    max_upload_mb: int
    use_memory_db: bool


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
    )
