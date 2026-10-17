"""Mask credentials before anything is printed, logged or pasted into a report."""
import re

PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),                                  # Groq
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),                                 # OpenAI-style
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),                                # Google
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),  # GitHub
    re.compile(r"sb_secret_[A-Za-z0-9_-]{10,}"),                          # Supabase
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"),  # JWT (Supabase keys, bearer tokens)
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
]
ASSIGNMENT = re.compile(r"(?im)\b([A-Z0-9_]*(?:API_KEY|SERVICE_ROLE_KEY|SECRET|TOKEN|PASSWORD|AUTHORIZATION)[A-Z0-9_]*)\s*[=:]\s*\"?([^\s\"',}]{6,})")
MASK = "[REDACTED]"


def redact(text: str) -> str:
    for p in PATTERNS:                      # known token shapes first (also catches "Bearer <token>")
        text = p.sub(lambda m: (m.group(1) + MASK) if m.lastindex else MASK, text)
    return ASSIGNMENT.sub(lambda m: f"{m.group(1)}={MASK}", text)
