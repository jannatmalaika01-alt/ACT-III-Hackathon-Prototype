"""Tests for the vision + knowledge modules.   python -m pytest vk_tests -q"""
import csv
import os

import pytest
from PIL import Image

from knowledge.chunking.chunker import DocumentChunker
from knowledge.ingestion.pdf_parser import PDFIngestor, ocr_available
from knowledge.pipeline import VisionKnowledgePipeline, detect_input_type
from vision import document_analyzer, sensor_analyzer, synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CONTRACT_KEYS = {"case_id", "input_type", "defect", "evidence", "recommendation", "status", "task_id"}


@pytest.fixture(scope="module")
def pipe():
    return VisionKnowledgePipeline()


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---- ingestion / chunking ----
def test_pdf_pages_and_text():
    pages = PDFIngestor().parse_pdf(os.path.join(DATA, "raw", "bearing_maintenance_manual.pdf"))
    assert [p["page"] for p in pages] == [1, 2, 3, 4, 5]
    assert "BRG-9920" in pages[1]["text"] and not pages[1]["is_ocr"]


def test_missing_pdf_raises():
    with pytest.raises(FileNotFoundError):
        PDFIngestor().parse_pdf("nope.pdf")


@pytest.mark.skipif(not ocr_available(), reason="Tesseract OCR not installed")
def test_scanned_pdf_is_read_by_ocr():
    pages = PDFIngestor().parse_pdf(os.path.join(DATA, "test_reports", "d05_scanned_crack_report.pdf"))
    assert pages[0]["is_ocr"] and "crack" in pages[0]["text"].lower()


def test_no_fake_ocr_text_when_ocr_missing(monkeypatch, tmp_path):
    import knowledge.ingestion.pdf_parser as pp
    monkeypatch.setattr(pp, "ocr_available", lambda: False)
    pages = pp.PDFIngestor().parse_pdf(os.path.join(DATA, "test_reports", "d05_scanned_crack_report.pdf"))
    assert pages[0]["text"] == "" and pages[0]["ocr_unavailable"]


def test_chunks_keep_page_and_never_cross_pages():
    pages = [{"source": "m.pdf", "page": 7, "text": "Sentence one is here. " * 60}, {"source": "m.pdf", "page": 8, "text": "Short."}]
    chunks = DocumentChunker(chunk_size=200).chunk_pages(pages)
    assert len(chunks) > 2 and {c["page"] for c in chunks} == {7, 8}
    assert all(len(c["text"]) <= 260 for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


# ---- retrieval ----
def test_retrieval_questions_find_right_page(pipe):
    rows = _rows(os.path.join(DATA, "retrieval_eval.csv"))
    top3 = sum((r["expected_source"], int(r["expected_page"])) in
               [(h["source"], h["page"]) for h in pipe.store.search(r["question"], 3)] for r in rows)
    assert top3 == len(rows)


def test_empty_store_returns_nothing():
    from knowledge.retrieval.vector_store import KnowledgeVectorStore
    assert KnowledgeVectorStore().search("anything") == []


def test_adapter_matches_backend_retrieve_shape():
    from knowledge.retrieval.adapter import retrieve
    hits = retrieve("worn bearing replacement", 2)
    assert len(hits) == 2 and set(hits[0]) == {"manual", "page", "text"} and isinstance(hits[0]["page"], int)


# ---- detection ----
def test_all_test_images_detected():
    from vision.defect_detector import DefectDetector
    det = DefectDetector()
    for r in _rows(os.path.join(DATA, "test_cases.csv")):
        got = det.predict(os.path.join(DATA, "test", r["image"]))["defect"]["type"].replace(" ", "_")
        assert got == r["expected_label"], r["image"]


def test_confidence_is_real_not_floored():
    from vision.defect_detector import DefectDetector
    det = DefectDetector()
    noise = Image.effect_noise((224, 224), 80).convert("RGB")
    assert 0.25 <= det.predict_image(noise)["defect"]["confidence"] <= 1.0
    probs = det.probabilities(synthetic.generate("corrosion", 5))
    assert abs(sum(probs.values()) - 1) < 1e-4


def test_missing_image_raises():
    from vision.defect_detector import DefectDetector
    with pytest.raises(FileNotFoundError):
        DefectDetector().predict("missing.png")


def test_missing_weights_message():
    from vision.defect_detector import DefectDetector
    with pytest.raises(FileNotFoundError, match="vision.train"):
        DefectDetector("/nonexistent/w.pt")


def test_sensor_and_document_manifest():
    pipe = VisionKnowledgePipeline(detector=object())  # detector must not be needed for these
    for r in _rows(os.path.join(DATA, "test_reports", "manifest.csv")):
        if "scanned" in r["file"] and not ocr_available():
            continue
        c = pipe.process_case(os.path.join(DATA, "test_reports", r["file"]), input_type=r["input_type"])
        assert c["defect"]["type"].replace(" ", "_") == r["expected_label"], r["file"]


def test_negated_sentences_are_not_defects():
    out = document_analyzer.analyze_text("No visible cracks were found and the pipe is free of rust. Everything is within limits.")
    assert out["defect"]["type"] == "normal"


def test_sensor_unknown_channels_are_not_a_confident_normal(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("foo,bar\n1,2\n")
    assert sensor_analyzer.analyze(str(f))["defect"]["confidence"] <= 0.5


def test_unsupported_type():
    with pytest.raises(ValueError):
        detect_input_type("movie.mp4")


# ---- contract ----
def test_contract_shape_for_image(pipe):
    c = pipe.process_case(os.path.join(DATA, "test", "test_02.png"), case_id="C-101")
    assert set(c) == CONTRACT_KEYS
    assert c["case_id"] == "C-101" and c["input_type"] == "image" and c["status"] == "pending" and c["task_id"] is None
    assert set(c["defect"]) == {"type", "confidence", "explanation"} and 0 <= c["defect"]["confidence"] <= 1
    assert c["defect"]["type"] == "bearing wear"
    assert c["evidence"] and set(c["evidence"][0]) == {"source", "page", "chunk"}
    assert c["evidence"][0]["source"].endswith(".pdf") and c["evidence"][0]["page"] >= 1
    assert set(c["recommendation"]) == {"action", "priority"} and c["recommendation"]["priority"] in ("low", "medium", "high", "critical")
    assert "BRG-9920" in c["recommendation"]["action"]


def test_recommendation_without_manual_support_is_flagged(pipe):
    rec = pipe.recommend("corrosion", 0.9, [])
    assert "manual inspection required" in rec["action"]
