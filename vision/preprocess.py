import numpy as np
from PIL import Image

from .model import INPUT_SIZE, MEAN, STD


def to_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 CHW array, resized and normalised for DefectCNN."""
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return arr.transpose(2, 0, 1)
