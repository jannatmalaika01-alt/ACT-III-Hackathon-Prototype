"""retrain_from_feedback.py: a tiny real training run on generated images; the live weights must never change."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("retrain_from_feedback", ROOT / "scripts" / "retrain_from_feedback.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

LIVE = ROOT / "vision" / "weights" / "defect_cnn.pt"
TINY = ["--per-class", "3", "--val-per-class", "3", "--epochs", "1"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def feedback(tmp_path):
    from vision import synthetic
    from vision.labels import CLASSES
    folder = tmp_path / "feedback"
    for label in CLASSES:
        (folder / label).mkdir(parents=True)
        for i in range(8):
            case = f"{label}-{i}"
            synthetic.generate(label, 500 + i, "clean").save(folder / label / f"{case}.png")
            (folder / label / f"{case}.json").write_text(json.dumps(
                {"case_id": case, "approved_label": label, "diagnosis_was_unverified": i == 7}))
    return folder


def test_trains_a_candidate_prints_both_accuracies_and_never_touches_the_live_weights(feedback, tmp_path, capsys):
    before = sha(LIVE)
    out = tmp_path / "candidate_cnn.pt"
    assert rf.main(["--feedback", str(feedback), "--out", str(out), "--real-photos", str(tmp_path / "none"), *TINY]) == 0
    text = capsys.readouterr().out
    assert out.exists() and out.with_suffix(".json").exists()
    assert sha(LIVE) == before                                      # the live CNN is byte-for-byte unchanged
    assert "current defect_cnn.pt" in text and "candidate" in text and "generated test images" in text
    assert "feedback hold-out" in text and "was NOT changed" in text
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["feedback_photos"] > 0 and "NOT_PROMOTED" in meta


def test_refuses_to_overwrite_the_live_weights(feedback, tmp_path):
    before = sha(LIVE)
    for bad in (LIVE, tmp_path / ".." / tmp_path.name / "defect_cnn.pt", ROOT / "vision" / "weights" / ".." / "weights" / "defect_cnn.pt",
                tmp_path / "candidate.pth"):
        with pytest.raises(SystemExit, match="never touches the live weights"):
            rf.main(["--feedback", str(feedback), "--out", str(bad), *TINY])
    assert sha(LIVE) == before


def test_the_default_output_is_the_candidate_file(monkeypatch):
    import inspect
    assert "candidate_cnn.pt" in inspect.getsource(rf.main)


def test_no_feedback_yet_trains_nothing(tmp_path, capsys):
    out = tmp_path / "candidate_cnn.pt"
    assert rf.main(["--feedback", str(tmp_path / "empty"), "--out", str(out), *TINY]) == 0
    assert not out.exists() and "No approved feedback" in capsys.readouterr().out


def test_unverified_diagnoses_are_left_out_unless_asked(feedback):
    assert len(rf.load_feedback(feedback, include_unverified=False)) == 28
    assert len(rf.load_feedback(feedback, include_unverified=True)) == 32


def test_holdout_split_is_stable_and_disjoint(feedback):
    items = rf.load_feedback(feedback, True)
    t1, h1 = rf.split_feedback(items)
    t2, h2 = rf.split_feedback(list(reversed(items)))
    assert {i["case_id"] for i in h1} == {i["case_id"] for i in h2} and h1
    assert not ({i["case_id"] for i in t1} & {i["case_id"] for i in h1}) and len(t1) + len(h1) == len(items)


def test_real_photos_used_for_training_are_skipped_in_the_comparison(feedback, tmp_path, capsys):
    items, _ = rf.split_feedback(rf.load_feedback(feedback, False))
    real = tmp_path / "real"
    real.mkdir()
    (real / "same.png").write_bytes(items[0]["path"].read_bytes())          # a photo that is also in the training feedback
    from PIL import Image
    Image.effect_noise((64, 64), 40).convert("RGB").save(real / "other.png")   # a photo that was never in the feedback
    (real / "labels.csv").write_text(f"filename,label\nsame.png,{items[0]['label']}\nother.png,normal\n")
    rf.main(["--feedback", str(feedback), "--out", str(tmp_path / "c.pt"), "--real-photos", str(real), *TINY])
    out = capsys.readouterr().out
    assert "skipped 1 that were used for training" in out and "real photos (real)" in out
