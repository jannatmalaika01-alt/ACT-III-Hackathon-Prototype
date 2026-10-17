"""No credentials in our files, and the redactor really masks them."""
import os
import re
import subprocess

import pytest

from knowledge.redact import PATTERNS, redact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE = ("vision", "knowledge", "vk_tests", "data", "docs", "evaluate.py", "run_pipeline.py", "push_to_backend.py",
        "setup_dataset.py", "setup_manuals.py", "requirements-vision.txt")
FAKE_KEYS = ["gsk_abcdefghijklmnopqrstuvwx", "sk-abcdefghijklmnopqrstuvwxyz123456", "ghp_" + "a" * 36,
             "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoic2VydmljZSJ9.abcdefghij", "AIza" + "B" * 35]


@pytest.mark.parametrize("key", FAKE_KEYS)
def test_redact_masks_keys(key):
    out = redact(f"calling with {key} now")
    assert key not in out and "[REDACTED]" in out


def test_redact_masks_env_style_assignments_and_bearer():
    out = redact("SUPABASE_SERVICE_ROLE_KEY=abc123def456ghi\nAuthorization: Bearer abcdefghijklmnop12345")
    assert "abc123def456ghi" not in out and "abcdefghijklmnop12345" not in out


def test_redact_leaves_normal_text_alone():
    text = "bearing wear at motor M3, confidence 0.99, bearing_maintenance_manual.pdf page 2"
    assert redact(text) == text


def test_none_of_our_files_contain_credentials():
    try:
        files = subprocess.run(["git", "ls-files", *MINE], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        pytest.skip("not a git checkout")
    bad = []
    for f in files:
        if f.endswith((".png", ".pdf", ".pt")) or f.endswith("test_no_secrets.py"):
            continue
        text = open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore").read()
        bad += [f for p in PATTERNS if p.search(text)]
    assert not bad, bad
    assert not [f for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
                if re.search(r"(^|/)\.env$", f)], ".env must never be committed"
