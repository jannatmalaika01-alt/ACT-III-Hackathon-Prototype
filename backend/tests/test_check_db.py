import base64
import json

from scripts.check_db import key_role, run_checks


def _jwt(role):
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'HS256'})}.{enc({'role': role})}.sig"


def test_key_role_detection():
    assert key_role(_jwt("service_role")) == "service_role"
    assert key_role(_jwt("anon")) == "anon"
    assert key_role("sb_secret_abc") == "secret"
    assert key_role("sb_publishable_abc") == "publishable"
    assert key_role("garbage") == "unknown"


class _Q:
    def __init__(self, fail): self.fail = fail
    def select(self, *_): return self
    def limit(self, *_): return self
    def execute(self):
        if self.fail:
            raise RuntimeError('relation "public.cases" does not exist')


class _Bucket:
    def __init__(self, public): self.public = public


class _Storage:
    def __init__(self, public, missing=False): self.public, self.missing = public, missing
    def get_bucket(self, _):
        if self.missing:
            raise RuntimeError("Bucket not found")
        return _Bucket(self.public)
    def from_(self, _): return self
    def upload(self, *a, **k): pass
    def remove(self, *a, **k): pass


class _Client:
    def __init__(self, missing_tables=(), public=False, bucket_missing=False):
        self.missing_tables = missing_tables
        self.storage = _Storage(public, bucket_missing)
    def table(self, name): return _Q(name in self.missing_tables)


def test_everything_ok():
    assert all(ok for _, ok, _ in run_checks(_Client()))


def test_missing_table_is_reported():
    res = run_checks(_Client(missing_tables=("cases",)))
    assert [n for n, ok, _ in res if not ok] == ["table 'cases' exists with all expected columns"]


def test_public_bucket_is_flagged():
    res = run_checks(_Client(public=True))
    assert [n for n, ok, _ in res if not ok] == ["bucket 'case-uploads' is private"]


def test_missing_bucket_is_flagged():
    res = run_checks(_Client(bucket_missing=True))
    assert any("bucket" in n and not ok for n, ok, _ in res)
