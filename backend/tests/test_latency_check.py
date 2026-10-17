import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("latency_check", Path(__file__).parents[1] / "scripts" / "latency_check.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


@pytest.fixture
def folder(tmp_path):
    for i in range(1, 21):
        Image.new("RGB", (8, 8)).save(tmp_path / f"p{i:02d}.jpg")
    (tmp_path / "labels.csv").write_text("filename,label\n" + "".join(f"p{i:02d}.jpg,normal\n" for i in range(1, 21)))
    return tmp_path


def test_photos_are_spread_evenly(folder):
    assert lc.pick(folder, 10) == [f"p{i:02d}.jpg" for i in range(1, 21, 2)]
    assert len(lc.pick(folder, 50)) == 20


def test_calls_are_sequential_and_summarised(folder):
    active, overlap, seen = [0], [False], []

    def ask(fn, path):
        active[0] += 1
        overlap[0] |= active[0] > 1                       # two calls at once would set this
        seen.append((fn, Path(path).name))
        active[0] -= 1
        if fn == "run_vision_model_2" and path.endswith("p06.jpg"):
            raise TimeoutError()
        return {"label": "normal"}

    res = lc.measure(4, folder, ask)
    assert not overlap[0] and len(seen) == 8 and [s[0] for s in seen[:4]] == ["run_vision_model"] * 4
    text = lc.summary(res)
    assert "| Qwen3.8-Flash-Next | 4 |" in text and "| Gemma-4-31B-IT | 4 |" in text and text.rstrip().endswith("| 1 |")
