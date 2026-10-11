"""Procedural training / evaluation images for the four classes.

There is no real defect photo set yet, so the detector is trained on generated images. `style="clean"`
reproduces the look of data/test/test_01..10 (flat grey, simple strokes); `style="hard"` adds textured
backgrounds, noise, blur, lighting and varied geometry so the model cannot just memorise one picture.
To use real photos instead, put them in data/train/<class_id>/ and run `python -m vision.train`.
"""
import math
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 224


def _background(rng: random.Random, style: str) -> Image.Image:
    if style == "clean":
        g = rng.choice([200, 200, 200, 190, 210])
        return Image.new("RGB", (SIZE, SIZE), (g, g, g))
    base = np.array([rng.randint(90, 215)] * 3, dtype=np.float32) + np.array([rng.randint(-8, 8) for _ in range(3)])
    arr = np.ones((SIZE, SIZE, 3), dtype=np.float32) * base
    # soft lighting gradient
    gx, gy = rng.uniform(-35, 35), rng.uniform(-35, 35)
    xs = np.linspace(-1, 1, SIZE)
    arr += (xs[None, :, None] * gx + xs[:, None, None] * gy)
    # metal-like fine texture
    arr += np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(2, 12), (SIZE, SIZE, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _bearing_wear(d: ImageDraw.ImageDraw, rng: random.Random, style: str) -> None:
    cx = SIZE // 2 + (rng.randint(-25, 25) if style == "hard" else 0)
    cy = SIZE // 2 + (rng.randint(-25, 25) if style == "hard" else 0)
    n = rng.randint(3, 6)
    step = rng.randint(10, 18)
    start = rng.randint(12, 28)
    tone = rng.randint(40, 100)
    for i in range(n):
        r = start + i * step
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(tone, tone, tone), width=rng.randint(2, 4))
    if style == "hard" and rng.random() < 0.5:  # extra scoring arc
        r = start + rng.randint(0, n) * step
        d.arc([cx - r, cy - r, cx + r, cy + r], rng.randint(0, 360), rng.randint(0, 360), fill=(30, 30, 30), width=2)


def _structural_crack(d: ImageDraw.ImageDraw, rng: random.Random, style: str) -> None:
    x, y = rng.randint(5, 60), rng.randint(5, 60)
    pts = [(x, y)]
    ang = rng.uniform(0.3, 1.2)
    for _ in range(rng.randint(4, 8)):
        ang += rng.uniform(-0.7, 0.7)
        ln = rng.randint(20, 45)
        x = min(SIZE - 2, max(1, x + math.cos(ang) * ln))
        y = min(SIZE - 2, max(1, y + math.sin(ang) * ln))
        pts.append((x, y))
    tone = rng.randint(5, 45)
    w = rng.randint(2, 5)
    d.line(pts, fill=(tone, tone, tone), width=w)
    if style == "hard" and rng.random() < 0.6:  # branch
        bx, by = pts[len(pts) // 2]
        d.line([(bx, by), (bx + rng.randint(-40, 40), by + rng.randint(10, 45))], fill=(tone, tone, tone), width=max(1, w - 1))


def _corrosion(d: ImageDraw.ImageDraw, rng: random.Random, style: str) -> None:
    n = rng.randint(150, 450)
    cluster = style == "hard" and rng.random() < 0.6
    centers = [(rng.randint(30, 190), rng.randint(30, 190)) for _ in range(rng.randint(1, 4))]
    for _ in range(n):
        if cluster:
            cx, cy = rng.choice(centers)
            x, y = int(rng.gauss(cx, 28)), int(rng.gauss(cy, 28))
        else:
            x, y = rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1)
        col = (rng.randint(120, 185), rng.randint(55, 95), rng.randint(0, 35))
        s = rng.randint(1, 4)
        d.rectangle([x, y, x + s, y + s], fill=col)


_DRAW = {"bearing_wear": _bearing_wear, "structural_crack": _structural_crack, "corrosion": _corrosion}


def generate(label: str, seed: int, style: str = "hard") -> Image.Image:
    """Return a 224x224 RGB PIL image for `label` (deterministic for a given seed)."""
    rng = random.Random(f"{label}-{style}-{seed}")
    img = _background(rng, style)
    if label in _DRAW:
        _DRAW[label](ImageDraw.Draw(img), rng, style)
    if label == "normal" and style == "hard" and rng.random() < 0.5:  # harmless dust specks
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(3, 25)):
            x, y = rng.randint(0, SIZE - 3), rng.randint(0, SIZE - 3)
            g = rng.randint(120, 230)
            d.point((x, y), fill=(g, g, g))
    if style == "hard":
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
        arr = np.asarray(img).astype(np.float32)
        arr *= rng.uniform(0.8, 1.2)
        arr += np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(0, 7), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def batch(labels: list, n_per_class: int, seed_offset: int, style_mix: Tuple[float, float] = (0.3, 0.7)):
    """Yield (PIL image, class index-label) pairs; `style_mix` is the (clean, hard) share."""
    rng = random.Random(seed_offset)
    for lab in labels:
        for i in range(n_per_class):
            style = "clean" if rng.random() < style_mix[0] else "hard"
            yield generate(lab, seed_offset + i, style), lab
