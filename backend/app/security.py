"""HMAC signature for webhooks: header  X-Webhook-Signature: sha256=<hex>."""
from __future__ import annotations

import hashlib
import hmac


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(secret: str, body: bytes, header_value: str | None) -> bool:
    if not secret or not header_value:
        return False
    return hmac.compare_digest(sign(secret, body), header_value)
