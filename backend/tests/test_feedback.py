"""Feedback store: only human-approved image cases are saved, never the pipeline's own guesses."""
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import feedback
from app.config import get_settings
from app.db import InMemoryRepo
from app.main import create_app
from app.evolus import MockEvolusClient

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def fb(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path / "feedback"))
    return tmp_path / "feedback"


@pytest.fixture
def agent(monkeypatch):
    def fake(report, rejection_feedback=None):
        return {"defect": f"{report['defect_type']} at {report['location']}", "severity": "high", "explanation": "e",
                "recommended_action": "a", "source_manual": "SOP", "source_page": 3, "confidence": report["confidence"],
                "raw": {"defect_report": report, "needs_manual_verification": bool(report.get("needs_human_review")),
                        "defect_source": "vision"}}

    mod = types.ModuleType("agent")
    mod.diagnose, mod.NoRelevantSOPError = fake, type("NoSOP", (Exception,), {})
    monkeypatch.setitem(sys.modules, "agent", mod)


@pytest.fixture
def client(fb, agent):
    return TestClient(create_app(settings=get_settings(), repo=InMemoryRepo(), evolus=MockEvolusClient()))


def make_case(client, defect="corrosion", filename="rust.png", ctype="image/png", data=PNG, **extra):
    cid = client.post("/cases", files={"file": (filename, data, ctype)}, data={"uploaded_by": "t"}).json()["id"]
    body = {"defect_type": defect, "location": "plate", "confidence": 0.9, **extra}
    assert client.post(f"/cases/{cid}/diagnose", json=body).status_code == 200
    return cid


def approve(client, cid, who="boss"):
    return client.post(f"/cases/{cid}/approve", json={"decided_by": who})


def test_approve_saves_the_photo_and_a_small_json(client, fb):
    cid = make_case(client)
    assert approve(client, cid).json()["status"] == "task_created"
    img = fb / "corrosion" / f"{cid}.png"
    assert img.read_bytes() == PNG
    meta = json.loads((fb / "corrosion" / f"{cid}.json").read_text())
    assert (meta["approved_label"], meta["approved_by"], meta["case_id"]) == ("corrosion", "boss", cid)
    assert meta["timestamp"] and meta["diagnosis_was_unverified"] is False and meta["defect_source"] == "vision"
    assert meta["detectors"] is None or isinstance(meta["detectors"], dict)


def test_the_class_name_is_normalised(client, fb):
    cid = make_case(client, defect="bearing wear")
    approve(client, cid)
    assert (fb / "bearing_wear" / f"{cid}.png").exists()   # "bearing wear" -> the class bearing_wear


def test_reject_saves_nothing(client, fb):
    cid = make_case(client)
    client.post(f"/cases/{cid}/reject", json={"decided_by": "boss", "reason": "wrong"})
    assert not fb.exists() or not any(fb.rglob("*.*"))


def test_a_pending_case_and_a_failed_approval_save_nothing(client, fb):
    cid = make_case(client)            # diagnosed but not approved
    assert not fb.exists()
    assert client.post("/cases/00000000-0000-0000-0000-000000000000/approve", json={"decided_by": "b"}).status_code == 404
    assert not fb.exists()


def test_only_images_are_saved(client, fb):
    cid = make_case(client, filename="report.txt", ctype="text/plain", data=b"vibration is high")
    approve(client, cid)
    assert not fb.exists() or not any(fb.rglob("*.*"))


def test_unknown_labels_are_not_saved(client, fb):
    cid = make_case(client, defect="loose_bolt")   # not one of the four CNN classes
    assert approve(client, cid).status_code == 200
    assert not fb.exists() or not any(fb.rglob("*.*"))


def test_an_approved_unverified_diagnosis_is_marked(client, fb):
    cid = make_case(client)
    client.app.state.repo.cases[cid]["diagnosis_raw"]["needs_manual_verification"] = True  # it was flagged UNVERIFIED
    approve(client, cid)
    meta = json.loads((fb / "corrosion" / f"{cid}.json").read_text())
    assert meta["diagnosis_was_unverified"] is True


def test_a_storage_problem_never_blocks_the_approval(client, fb, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_bytes", boom)
    cid = make_case(client)
    assert approve(client, cid).json()["status"] == "task_created"


def test_the_feedback_folder_and_candidate_weights_are_git_ignored():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".gitignore").read_text()
    assert "data/feedback/" in text and "vision/weights/candidate_cnn" in text
    assert feedback.REPO_ROOT == root


def test_default_location_is_data_feedback(monkeypatch):
    monkeypatch.delenv("FEEDBACK_DIR", raising=False)
    assert feedback.feedback_dir() == feedback.REPO_ROOT / "data" / "feedback"
