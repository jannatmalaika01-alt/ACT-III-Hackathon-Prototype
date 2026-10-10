"""Check that the Supabase project is set up correctly. Safe: read-only, plus one tiny
upload that is deleted again (skip it with --no-write).

    python scripts/check_db.py

Needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env. Exit code 0 = all good.
"""
from __future__ import annotations

import base64
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

TABLES = {
    "cases": ["id", "status", "file_path", "file_name", "defect", "severity", "explanation",
              "recommended_action", "source_manual", "source_page", "confidence", "diagnosis_raw",
              "diagnosed_at", "evolus_task_id", "task_error", "task_attempts", "updated_at"],
    "approvals": ["id", "case_id", "decision", "decided_by", "reason", "decided_at"],
    "case_events": ["id", "case_id", "event_id", "source", "from_status", "to_status", "note"],
}


def key_role(key: str) -> str:
    """'service_role' | 'anon' | 'secret' | 'publishable' | 'unknown'  (no network, no verification)."""
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("sb_publishable_"):
        return "publishable"
    try:
        payload = key.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("role", "unknown")
    except Exception:
        return "unknown"


def run_checks(client, bucket: str = "case-uploads", write_test: bool = True):
    """Returns a list of (name, ok, detail). `client` is a supabase-py client."""
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    for table, cols in TABLES.items():
        try:
            client.table(table).select(",".join(cols)).limit(1).execute()
            add(f"table '{table}' exists with all expected columns", True)
        except Exception as exc:
            add(f"table '{table}' exists with all expected columns", False, str(exc)[:200])

    try:
        b = client.storage.get_bucket(bucket)
        public = getattr(b, "public", None)
        add(f"storage bucket '{bucket}' exists", True)
        add(f"bucket '{bucket}' is private", public is False, f"public={public}")
    except Exception as exc:
        add(f"storage bucket '{bucket}' exists", False, str(exc)[:200])

    if write_test:
        probe = "_check_db/probe.txt"
        try:
            client.storage.from_(bucket).upload(probe, b"ok", {"content-type": "text/plain", "upsert": "true"})
            client.storage.from_(bucket).remove([probe])
            add("can upload to and delete from the bucket", True)
        except Exception as exc:
            add("can upload to and delete from the bucket", False, str(exc)[:200])
    return out


def main() -> int:
    url, key = os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    bucket = os.getenv("STORAGE_BUCKET", "case-uploads")
    if not url or not key:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing in .env")
        return 1

    role = key_role(key)
    print(f"Project: {url}\nKey type: {role}")
    if role in ("anon", "publishable"):
        print("[FAIL] this is the ANON/publishable key - the backend needs the service_role (secret) key")
        return 1

    from supabase import create_client
    results = run_checks(create_client(url, key), bucket, write_test="--no-write" not in sys.argv)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    failed = [n for n, ok, _ in results if not ok]
    print("\nAll good - the database is ready." if not failed else
          "\nNot ready. If tables are missing, run supabase/migrations/001_init.sql in the Supabase SQL Editor.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
