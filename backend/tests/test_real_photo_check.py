"""real_photo_check.py with a tiny fake folder: no real detector or API is used."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from app import dual_detector as dd

spec = importlib.util.spec_from_file_location("real_photo_check",
                                              Path(__file__).parents[1] / "scripts" / "real_photo_check.py")
rpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpc)

# photo -> (truth, CNN label, vision label or None = vision model failed)
CASES = {
    "a_agree_ok.png": ("corrosion", "corrosion", "corrosion"),
    "b_agree_bad.png": ("normal", "corrosion", "corrosion"),
    "c_cnn_right.png": ("corrosion", "corrosion", "normal"),
    "d_vision_right.png": ("structural_crack", "normal", "structural_crack"),
    "e_both_wrong.png": ("bearing_wear", "normal", "corrosion"),
    "f_single.png": ("normal", "normal", None),
}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    for name in CASES:
        Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / name)
    (tmp_path / "labels.csv").write_text("filename,label\n" + "".join(f"{n},{t[0]}\n" for n, t in CASES.items()))
    monkeypatch.setattr(rpc, "_sleep", lambda s: None)
    monkeypatch.setattr(dd, "run_cnn", lambda p: {"label": CASES[Path(p).name][1], "confidence": 0.9})

    def vlm(p):
        label = CASES[Path(p).name][2]
        if label is None:
            raise TimeoutError()
        return {"label": label, "confidence": 0.8, "model": dd.vision_model()}

    monkeypatch.setattr(dd, "run_vision_model", vlm)
    return tmp_path


def test_table_counts_every_outcome(folder, capsys):
    assert rpc.main(["--photos", str(folder)]) == 0
    out = capsys.readouterr().out
    for line in ("| Agree, correct | 1 |", "| Agree, WRONG (both wrong the same way) | 1 |", "| Disagree, CNN right | 1 |",
                 "| Disagree, vision model right | 1 |", "| Disagree, both wrong | 1 |",
                 "| single_detector (one detector gave no answer) | 1 |"):
        assert line in out
    assert "CNN mistakes (among photos where both answered): 3" in out  # b, d and e
    assert "caught by the disagreement flag: 2 of 3" in out            # d and e; b is the shared mistake
    assert "false alarms (flagged although the CNN was right): 1" in out
    assert "missed (both agreed on a wrong label): 1" in out


def test_vision_model_flag_changes_the_model_without_editing_code(folder, capsys, monkeypatch):
    monkeypatch.setenv("DUAL_VISION_MODEL", "Qwen3.8-Flash-Next")
    rpc.main(["--photos", str(folder), "--vision-model", "Gemma-4-31B-IT"])
    assert "Vision model: Gemma-4-31B-IT" in capsys.readouterr().out


def test_pacing_between_photos(folder, monkeypatch):
    waits = []
    monkeypatch.setattr(rpc, "_sleep", waits.append)
    rpc.main(["--photos", str(folder), "--per-minute", "12"])
    assert waits == [5.0] * (len(CASES) - 1)


def test_report_can_be_written_to_a_file(folder, tmp_path):
    out = tmp_path / "report.md"
    rpc.main(["--photos", str(folder), "--out", str(out)])
    assert "Total time" in out.read_text()


def test_bad_labels_are_explained(folder):
    (folder / "labels.csv").write_text("filename,label\na_agree_ok.png,rusty\nmissing.png,normal\n")
    with pytest.raises(SystemExit) as e:
        rpc.main(["--photos", str(folder)])
    msg = str(e.value)
    assert "unknown label 'rusty'" in msg and "photo not found: missing.png" in msg


def test_missing_labels_file_and_wrong_header(folder):
    (folder / "labels.csv").unlink()
    with pytest.raises(SystemExit, match="labels file not found"):
        rpc.main(["--photos", str(folder)])
    (folder / "labels.csv").write_text("name,kind\nx,y\n")
    with pytest.raises(SystemExit, match="header: filename,label"):
        rpc.main(["--photos", str(folder)])
