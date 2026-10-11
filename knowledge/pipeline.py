"""Vision/Knowledge pipeline: input file -> defect -> cited manual chunks -> shared-contract JSON.

    from knowledge.pipeline import VisionKnowledgePipeline
    contract = VisionKnowledgePipeline().process_case("data/test/test_02.png", case_id="C-101")

Output is EXACTLY the team contract (case_id, input_type, defect, evidence, recommendation, status, task_id).
"""
import json
import os
import re
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from knowledge.ingestion.loader import extract_text  # noqa: E402
from knowledge.retrieval.knowledge_base import get_store  # noqa: E402
from knowledge.retrieval.query import expand_query  # noqa: E402
from vision import document_analyzer, sensor_analyzer  # noqa: E402
from vision.labels import NORMAL, from_contract_label  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
SENSOR_EXT = (".csv", ".json")
DOC_EXT = (".pdf", ".txt", ".md")
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.08"))  # below this a chunk is not trusted as evidence
PRIORITIES = ("low", "medium", "high", "critical")
REL_SCORE = 0.6  # keep hits scoring at least this fraction of the best hit (drops stray pages from other manuals)



def detect_input_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in SENSOR_EXT:
        return "sensor"
    if ext in DOC_EXT:
        return "document"
    raise ValueError(f"Unsupported file type: {ext or path}")


class VisionKnowledgePipeline:
    def __init__(self, detector=None, store=None):
        self._detector = detector
        self.store = store or get_store()

    @property
    def detector(self):
        if self._detector is None:  # lazy: sensor/document cases and retrieval work without torch
            from vision.defect_detector import DefectDetector
            self._detector = DefectDetector()
        return self._detector

    # ---- step 2: detect and read ----
    def detect(self, path: str, input_type: Optional[str] = None) -> dict:
        input_type = input_type or detect_input_type(path)
        name = os.path.basename(path)
        if input_type == "image":
            return self.detector.predict(path)
        if input_type == "sensor":
            return sensor_analyzer.analyze(path)
        return document_analyzer.analyze_text(extract_text(path), name)

    # ---- step 3: find the fix ----
    def retrieve_evidence(self, defect_label: str, top_k: int = 3, explanation: str = "") -> List[dict]:
        if from_contract_label(defect_label) == NORMAL:
            query = "general inspection overview routine rounds daily checks"
        else:
            # defect vocabulary + what the detector actually saw (sensor channel names, quoted report text, ...)
            query = f"{expand_query(defect_label)} {explanation[:300]}"
        hits = [h for h in self.store.search(query, top_k=top_k) if h["score"] >= MIN_SCORE]
        return [h for h in hits if h["score"] >= REL_SCORE * hits[0]["score"]]

    @staticmethod
    def recommend(defect_label: str, confidence: float, hits: List[dict]) -> dict:
        if from_contract_label(defect_label) == NORMAL:
            return {"action": "No defect detected - continue routine inspection rounds.", "priority": "low"}
        for h in hits:  # first retrieved chunk that actually contains a procedure
            m = re.search(r"Action Required:\s*(.+?)(?:\s*Priority:\s*(\w+)\.?)?\s*$", h["chunk"], re.S)
            if m:
                action = " ".join(m.group(1).split())
                prio = (m.group(2) or "").lower()
                return {"action": action, "priority": prio if prio in PRIORITIES else "high"}
        return {"action": f"No matching procedure found in the manuals for '{defect_label}' - manual inspection required.",
                "priority": "medium"}

    def process_case(self, path: str, case_id: str = "C-001", input_type: Optional[str] = None) -> dict:
        input_type = input_type or detect_input_type(path)
        detected = self.detect(path, input_type)
        defect = detected["defect"]
        hits = self.retrieve_evidence(defect["type"], explanation=defect.get("explanation", ""))
        self.last_scores = [h["score"] for h in hits]
        return {
            "case_id": case_id,
            "input_type": input_type,
            "defect": defect,
            "evidence": [{"source": h["source"], "page": h["page"], "chunk": h["chunk"]} for h in hits],
            "recommendation": self.recommend(defect["type"], defect["confidence"], hits),
            "status": "pending",
            "task_id": None,
        }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "test", "test_02.png")
    print(json.dumps(VisionKnowledgePipeline().process_case(target, case_id="C-101"), indent=2))
