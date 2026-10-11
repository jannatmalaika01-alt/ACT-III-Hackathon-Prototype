"""Compatibility with the team backend (backend/ on GitHub main): our output must be accepted by its API,
and our retriever must work inside its real LangGraph agent. Skipped if the backend or its packages are missing."""
import os
import sys
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
if not os.path.isdir(BACKEND):
    pytest.skip("backend/ not present", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from knowledge.pipeline import VisionKnowledgePipeline  # noqa: E402
from knowledge.retrieval.adapter import retrieve  # noqa: E402

PNG = os.path.join(ROOT, "data", "test", "test_02.png")


@pytest.fixture
def backend(monkeypatch):
    """The backend's real agent.py, with its `core` module replaced: LLM is a stub, retrieval is OURS."""
    monkeypatch.syspath_prepend(BACKEND)
    core = types.ModuleType("core")
    core.call_llm = lambda prompt, system="": '{"explanation": "Worn bearing.", "recommended_action": "Replace bearing.", "severity": "high", "source_index": 0}'
    core.retrieve = retrieve
    core.FAKE_DEFECTS = []
    monkeypatch.setitem(sys.modules, "core", core)
    monkeypatch.delitem(sys.modules, "agent", raising=False)
    from app.config import Settings
    from app.db import InMemoryRepo
    from app.evolus import MockEvolusClient
    from app.main import create_app
    s = Settings(supabase_url="", supabase_key="", storage_bucket="t", webhook_secret="x", evolus_mode="mock",
                 evolus_mock_fail_first_n=0, max_upload_mb=1, use_memory_db=True)
    return TestClient(create_app(settings=s, repo=InMemoryRepo(), evolus=MockEvolusClient()))


def test_defect_report_is_accepted_and_diagnosis_cites_real_manual(backend):
    from app.main import DefectReportIn
    pipe = VisionKnowledgePipeline()
    result = pipe.detector.predict(PNG)
    report = pipe.detector.defect_report(result, location="motor M3 bearing housing", image_ref="test_02.png")
    DefectReportIn(**report)  # same schema the backend validates

    case = backend.post("/cases", files={"file": ("test_02.png", open(PNG, "rb"), "image/png")}, data={"uploaded_by": "titus"}).json()
    path = "analyze" if any(r.path.endswith("/analyze") for r in backend.app.routes) else "diagnose"
    r = backend.post(f"/cases/{case['id']}/{path}", json=report)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_manual"].endswith(".pdf") and body["source_page"] >= 1  # a real manual page, not a placeholder
    assert body["status"] == "pending" and body["defect"].startswith("bearing wear")


def test_agent_refuses_without_sources_instead_of_inventing(backend):
    """If the knowledge base is empty the agent must raise NoRelevantSOPError (backend turns it into a 422)."""
    import core
    core.retrieve = lambda query, n_results=2: []
    from agent import NoRelevantSOPError, diagnose
    with pytest.raises(NoRelevantSOPError):
        diagnose({"defect_type": "bearing wear", "location": "x", "confidence": 0.9})
