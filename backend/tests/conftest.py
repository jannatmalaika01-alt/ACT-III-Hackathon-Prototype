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
    # The default dual mode is vision_primary; the older dual-detector tests were written for the pair, so they pin it here.
    # tests/test_dual_default_mode.py removes this to check the real default.
    monkeypatch.setenv("DUAL_MODE", "pair")


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


# ---- the manuals the citation checker looks quotes up in (real page texts of data/raw/corrosion_control_guide.pdf) ----
CORROSION_P2 = ("Page 2: Rust & Oxidation Treatment\nDefect Type: Surface Corrosion and Rust.\nSymptom: Brownish oxide flakes, surface pitting, "
                "or paint degradation on metallic piping.\nAction Required: Scrape loose rust off using wire brush or sandblasting. Apply "
                "chemical rust\nconverter followed by two coats of zinc-rich primer paint. Inspect wall thickness with ultrasonic\ngauge if "
                "pitting exceeds 1mm depth. Priority: Medium.")
CORROSION_P3 = ("Page 3: Pitting and Wall Thickness Loss\nDefect Type: Corrosion with wall thinning.\nSymptom: Deep pits, or ultrasonic readings "
                "showing wall loss above 20 percent of nominal\nthickness.\nAction Required: Take the line out of pressurised service. Cut "
                "out and replace the affected\npipe section when wall loss exceeds 20 percent; between 10 and 20 percent schedule repair\n"
                "within 7 days and re-measure monthly. Priority: High.")
CORROSION_P4 = ("Page 4: Protective Coatings and Cathodic Protection\nProcedure: Preventing corrosion returning.\nRe-coat repaired steel "
                "within 4 hours of surface preparation. Check sacrificial anodes every 6\nmonths and replace any that have lost more than "
                "half their mass. Record coating thickness\nreadings after each paint job.")
BEARING_P2 = ("Page 2: Bearing Wear Diagnosis & Action\nDefect Type: Bearing wear.\nAction Required: Immediately isolate machine power. "
              "Remove bearing housing and replace worn ball bearings. Priority: High.")


@pytest.fixture(autouse=True)
def manuals_for_citations(monkeypatch):
    """Evolus answers are checked against our manuals. Tests use these real pages instead of building the whole retrieval index."""
    from app import citations
    monkeypatch.setattr(citations, "_PASSAGES", [citations.Passage("corrosion_control_guide.pdf", 2, CORROSION_P2),
                                                 citations.Passage("corrosion_control_guide.pdf", 3, CORROSION_P3),
                                                 citations.Passage("corrosion_control_guide.pdf", 4, CORROSION_P4),
                                                 citations.Passage("bearing_maintenance_manual.pdf", 2, BEARING_P2)])
