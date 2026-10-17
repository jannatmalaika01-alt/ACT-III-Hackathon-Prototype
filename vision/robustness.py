"""Stress variants of an image: the ways a real photo differs from a clean render.

`hard_variants` applies the four degradations the team used to compare detectors (blur, dark, rotated, noisy), so
numbers from here and from the dual-detector test set mean the same thing. `random_degrade` applies random strengths
(and combinations) of the same degradations and is used to train the CNN. Evaluation seeds never overlap training seeds.
"""
import random
from typing import Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def blur(img: Image.Image, sigma: float = 3.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(sigma))


def darken(img: Image.Image, factor: float = 0.3) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def rotate(img: Image.Image, degrees: float = 35.0, fill=(40, 40, 40)) -> Image.Image:
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=fill)


def add_noise(img: Image.Image, sigma: float = 28.0, seed: int = 0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    return Image.fromarray(np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8))


def hard_variants(img: Image.Image, seed: int) -> Dict[str, Image.Image]:
    """The four standard stress variants (fixed strengths)."""
    return {"blur": blur(img), "dark": darken(img), "rotate": rotate(img), "noise": add_noise(img, 28.0, seed)}


def random_degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random mix of the same degradations at random strengths (training augmentation)."""
    if rng.random() < 0.55:
        img = rotate(img, rng.uniform(0, 360), fill=tuple([rng.randint(0, 220)] * 3))
    if rng.random() < 0.45:
        img = blur(img, rng.uniform(0.5, 4.0))
    if rng.random() < 0.5:
        img = darken(img, rng.uniform(0.2, 1.3))
    if rng.random() < 0.45:
        img = add_noise(img, rng.uniform(5, 40), rng.randint(0, 10**9))
    return img
