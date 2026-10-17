"""The detector must survive the degradations a real photo has (blur, darkness, rotation, noise)."""
import collections

from PIL import Image

from vision import robustness, synthetic
from vision.defect_detector import DefectDetector
from vision.labels import CLASSES, to_contract_label


def test_variants_keep_size_and_differ():
    img = synthetic.generate("corrosion", 7, "clean")
    v = robustness.hard_variants(img, 7)
    assert set(v) == {"blur", "dark", "rotate", "noise"}
    for im in v.values():
        assert isinstance(im, Image.Image) and im.size == img.size and list(im.getdata()) != list(img.getdata())


def test_generation_is_deterministic():
    a, b = synthetic.generate("structural_crack", 11, "hard"), synthetic.generate("structural_crack", 11, "hard")
    assert list(a.getdata()) == list(b.getdata())


def test_stress_accuracy_on_unseen_images():
    det = DefectDetector()
    res = collections.defaultdict(lambda: [0, 0])
    for lab in CLASSES:
        for i in range(10):
            seed = 3_000_000 + i
            base = synthetic.generate(lab, seed, "clean")
            for name, img in {"clean": base, **robustness.hard_variants(base, seed)}.items():
                res[name][1] += 1
                res[name][0] += det.predict_image(img)["defect"]["type"] == to_contract_label(lab)
    assert res["clean"][0] == res["clean"][1], dict(res)
    for name, (ok, n) in res.items():
        assert ok / n >= 0.90, (name, ok, n)   # old detector: blur 60%, dark 75%, rotate 71%
