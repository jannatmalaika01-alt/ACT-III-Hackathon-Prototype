"""Step 2: build the dual-detector test set in data/dual_lab/ (git-ignored, never committed).

    PYTHONPATH=. python scripts/build_dual_testset.py

5 fresh generated images per class (seeds the CNN never saw in training) + 4 harder copies of each
(blurred, darkened, rotated, noisy). The true label is kept in manifest.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision import synthetic  # noqa: E402
from vision.labels import CLASSES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "dual_lab" / "images"
PER_CLASS, SEED0 = 5, 900_000  # the CNN trains on small seed offsets; these were never used


def hard_variants(img: Image.Image, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32)
    noisy = Image.fromarray(np.clip(arr + rng.normal(0, 28, arr.shape), 0, 255).astype(np.uint8))
    return {
        "blur": img.filter(ImageFilter.GaussianBlur(3.0)),
        "dark": ImageEnhance.Brightness(img).enhance(0.3),
        "rotate": img.rotate(35, resample=Image.BILINEAR, fillcolor=(40, 40, 40)),
        "noise": noisy,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in CLASSES:
        for i in range(PER_CLASS):
            seed = SEED0 + i
            img = synthetic.generate(label, seed, "clean")
            for variant, im in {"clean": img, **hard_variants(img, seed)}.items():
                name = f"{label}_{i}_{variant}.png"
                im.save(OUT / name)
                rows.append({"file": name, "label": label, "variant": variant, "seed": seed})
    with open(OUT.parent / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "variant", "seed"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images in {OUT} ({PER_CLASS} per class x {len(CLASSES)} classes x 5 variants)")


if __name__ == "__main__":
    main()
