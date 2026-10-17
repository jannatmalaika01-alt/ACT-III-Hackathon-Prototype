"""Images are downscaled (longest side <= 1024 px, JPEG q85) before they go to a Token Factory vision model."""
import base64
import hashlib
import io

import pytest
from PIL import Image

from app import dual_detector as dd


def make(tmp_path, size, name="big.png", mode="RGB", fmt=None):
    path = tmp_path / name
    Image.new(mode, size, (200, 30, 30) if mode == "RGB" else 128).save(path, format=fmt)
    return path


def test_large_photo_is_downscaled_to_1024_jpeg(tmp_path):
    path = make(tmp_path, (3024, 4032), "big.jpg")
    data, mime = dd.prepare_image_for_vision(str(path))
    im = Image.open(io.BytesIO(data))
    assert mime == "image/jpeg" and im.format == "JPEG"
    assert max(im.size) == 1024 and im.size == (768, 1024)  # aspect ratio kept
    assert len(data) < path.stat().st_size


def test_original_file_is_untouched(tmp_path):
    path = make(tmp_path, (2500, 1800))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    dd.prepare_image_for_vision(str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert Image.open(path).size == (2500, 1800)


def test_small_images_are_not_upscaled(tmp_path):
    data, _ = dd.prepare_image_for_vision(str(make(tmp_path, (224, 224))))
    assert Image.open(io.BytesIO(data)).size == (224, 224)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_other_modes_become_rgb_jpeg(tmp_path, mode):
    path = tmp_path / f"{mode}.png"
    Image.new(mode, (1500, 900)).save(path)
    data, mime = dd.prepare_image_for_vision(str(path))
    assert Image.open(io.BytesIO(data)).mode == "RGB" and mime == "image/jpeg"


def test_exif_rotation_is_applied(tmp_path):
    path = tmp_path / "rot.jpg"
    im = Image.new("RGB", (400, 200), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees": the upright image is 200x400
    im.save(path, exif=exif)
    assert Image.open(io.BytesIO(dd.prepare_image_for_vision(str(path))[0])).size == (200, 400)


def test_undecodable_file_is_sent_unchanged(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"not an image")
    assert dd.prepare_image_for_vision(str(path)) == (b"not an image", "image/png")


def _capture_payload(monkeypatch):
    import core
    sent = {}
    monkeypatch.setenv("TOKENFACTORY_BASE_URL", "https://tf.example/v1")
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "test-key-not-real")

    class R:
        def json(self):
            return {"model": "m", "choices": [{"message": {"content": '{"defect_type": "corrosion", "confidence": 0.9}'}}]}

    def fake_post(p, payload):
        sent["payload"] = payload
        return R()

    monkeypatch.setattr(core, "_post_with_retries", fake_post)
    return sent


def test_the_vision_request_carries_the_small_jpeg_not_the_original(tmp_path, monkeypatch):
    sent = _capture_payload(monkeypatch)
    path = make(tmp_path, (3024, 4032), "huge.png")
    dd.run_vision_model(str(path))
    url = sent["payload"]["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert max(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size) == 1024


def test_the_cnn_still_reads_the_original_file(tmp_path, monkeypatch):
    seen = {}

    class FakeDetector:
        def predict(self, p):
            seen["size"] = Image.open(p).size
            return {"defect": {"type": "corrosion", "confidence": 0.9}}

    monkeypatch.setattr(dd, "_cnn", FakeDetector())
    dd.run_cnn(str(make(tmp_path, (3024, 4032), "orig.png")))
    assert seen["size"] == (3024, 4032)


def test_the_voting_script_goes_through_the_same_resize(tmp_path, monkeypatch):
    """voting_compare.py asks the vision models through dd.run_vision_model, so its photos are resized too."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("voting_compare", Path(__file__).parents[1] / "scripts" / "voting_compare.py")
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    sent = _capture_payload(monkeypatch)
    out = vc._ask(dd, "Qwen3.8-Flash-Next", str(make(tmp_path, (4000, 3000), "photo.png")))
    assert out["error"] is None
    url = sent["payload"]["messages"][1]["content"][1]["image_url"]["url"]
    assert max(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size) == 1024
