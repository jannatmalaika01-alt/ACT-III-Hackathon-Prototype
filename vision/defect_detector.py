"""Image defect detector: a small CNN trained by `python -m vision.train`.

    from vision.defect_detector import DefectDetector
    result = DefectDetector().predict("data/test/test_02.png")      # shared-contract fragment
    report = DefectDetector().defect_report(result, location="bearing housing A")   # POST body for the backend

Confidence is the model's real softmax probability (no floor, no tricks); anything below
LOW_CONFIDENCE is for a human to double-check (the backend agent uses the same 0.70 cut-off).
"""
import json
import os
from typing import Optional

import torch
from PIL import Image

from .labels import CLASSES, NORMAL, to_contract_label
from .model import DefectCNN
from .preprocess import to_array
from .thresholds import LOW_CONFIDENCE  # noqa: F401  (re-exported; defined once in thresholds.py)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "defect_cnn.pt")

_CUES = {
    "normal": "no concentric scoring, linear fracture or oxide speckle was found",
    "bearing_wear": "concentric circular scoring marks typical of a worn bearing race",
    "structural_crack": "a thin dark linear fracture line propagating across the surface",
    "corrosion": "scattered brown/orange oxide speckling typical of rust and pitting",
}


class DefectDetector:
    def __init__(self, weights_path: Optional[str] = None):
        path = weights_path or DEFAULT_WEIGHTS
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detector weights not found at {path}. Train them with: python -m vision.train")
        self.model = DefectCNN(len(CLASSES))
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval()
        meta = path.replace(".pt", ".json")
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}

    @torch.no_grad()
    def probabilities(self, img: Image.Image) -> dict:
        x = torch.tensor(to_array(img)).unsqueeze(0)
        p = torch.softmax(self.model(x)[0], dim=0)
        return {c: float(p[i]) for i, c in enumerate(CLASSES)}

    def predict_image(self, img: Image.Image, name: str = "image") -> dict:
        probs = self.probabilities(img)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top, conf = ranked[0]
        runner, runner_p = ranked[1]
        explanation = f"{name}: {_CUES[top]} (model confidence {conf:.2f})."
        if runner_p >= 0.15:
            explanation += f" Second most likely: {to_contract_label(runner)} ({runner_p:.2f})."
        if conf < LOW_CONFIDENCE:
            explanation += " Confidence is low - a person should verify visually."
        return {
            "case_id": "C-001",
            "input_type": "image",
            "defect": {"type": to_contract_label(top), "confidence": round(conf, 2), "explanation": explanation},
        }

    def predict(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        with Image.open(image_path) as img:
            return self.predict_image(img, os.path.basename(image_path))

    @staticmethod
    def defect_report(result: dict, location: Optional[str] = None, image_ref: Optional[str] = None) -> dict:
        """Shape the backend's POST /cases/{id}/analyze (or /diagnose) body expects (DefectReportIn)."""
        d = result["defect"]
        return {
            "defect_type": d["type"],
            "location": location or "location not specified in upload",
            "confidence": d["confidence"],
            "image_ref": image_ref,
        }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "test", "test_02.png")
    print(json.dumps(DefectDetector().predict(path), indent=2))
