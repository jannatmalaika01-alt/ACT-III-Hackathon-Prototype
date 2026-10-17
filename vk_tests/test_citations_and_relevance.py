"""Citations must point at real text in real documents; unrelated questions must get no evidence."""
import copy
import csv
import os

import pytest

from knowledge.pipeline import VisionKnowledgePipeline
from knowledge.retrieval.knowledge_base import get_store
from knowledge.retrieval.query import expand_query
from knowledge.verify import verify_citation, verify_contract, verify_evidence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def pipe():
    return VisionKnowledgePipeline(detector=object())


def _contract(pipe):
    return pipe.process_case(os.path.join(DATA, "test_reports", "s01_motor_vibration.csv"), input_type="sensor")


def test_real_citation_passes():
    assert verify_citation("bearing_maintenance_manual.pdf", 2) == []


@pytest.mark.parametrize("manual,page", [("made_up_manual.pdf", 1), ("bearing_maintenance_manual.pdf", 99),
                                          ("bearing_maintenance_manual.pdf", 0), ("bearing_maintenance_manual.pdf", "x"),
                                          ("../README.md", 1), ("", 1), (None, 1)])
def test_bad_citations_are_rejected(manual, page):
    assert verify_citation(manual, page)


def test_citation_cannot_escape_the_manuals_folder():
    assert verify_citation("../../etc/passwd.pdf", 1)


def test_every_fixture_output_is_fully_traceable(pipe):
    names = [f for f in os.listdir(os.path.join(DATA, "test_reports")) if f != "manifest.csv" and "scanned" not in f]
    for f in names:
        kind = "sensor" if f.startswith("s") else "document"
        c = pipe.process_case(os.path.join(DATA, "test_reports", f), input_type=kind)
        assert verify_contract(c) == [], f


def test_tampered_evidence_is_caught(pipe):
    c = copy.deepcopy(_contract(pipe))
    c["evidence"][0]["chunk"] = "Replace the bearing with part #FAKE-0000 immediately."
    assert any("not on" in p for p in verify_evidence(c["evidence"]))


def test_wrong_page_is_caught(pipe):
    c = copy.deepcopy(_contract(pipe))
    c["evidence"][0]["page"] = 4 if c["evidence"][0]["page"] != 4 else 3
    assert verify_contract(c)


def test_invented_action_is_caught(pipe):
    c = copy.deepcopy(_contract(pipe))
    c["recommendation"]["action"] = "Paint the machine blue."
    assert any("not copied" in p for p in verify_contract(c))


def test_defect_without_evidence_must_say_so(pipe):
    c = copy.deepcopy(_contract(pipe))
    c["evidence"] = []
    c["recommendation"] = {"action": "Replace the bearing.", "priority": "high"}
    assert verify_contract(c)
    c["recommendation"] = pipe.recommend("bearing wear", 0.9, [])
    assert verify_contract(c) == []


def test_off_topic_questions_get_no_evidence():
    store = get_store()
    with open(os.path.join(DATA, "retrieval_negative.csv"), newline="") as f:
        for row in csv.DictReader(f):
            assert store.search(expand_query(row["question"]), 3) == [], row["question"]


def test_backend_style_queries_still_find_the_procedure():
    store = get_store()
    for q, manual in [("corrosion at pipe section B7", "corrosion_control_guide.pdf"),
                      ("crack at weld joint A3", "structural_inspection_sop.pdf"),
                      ("loose_bolt at panel D5", "fastener_alignment_sop.pdf"),
                      ("misalignment at conveyor belt C2", "fastener_alignment_sop.pdf"),
                      ("bearing wear at motor M3", "bearing_maintenance_manual.pdf")]:
        hits = store.search(expand_query(q), 3)
        assert hits and hits[0]["source"] == manual, q


def test_all_retrieval_questions_cite_verifiable_text():
    store = get_store()
    with open(os.path.join(DATA, "retrieval_eval.csv"), newline="") as f:
        for row in csv.DictReader(f):
            ev = [{"source": h["source"], "page": h["page"], "chunk": h["chunk"]} for h in store.search(row["question"], 3)]
            assert ev and verify_evidence(ev) == [], row["question"]


def test_normal_machine_gets_routine_inspection_guidance_not_an_error():
    hits = get_store().search(expand_query("normal at motor M3"), 3)
    assert hits and verify_evidence([{"source": h["source"], "page": h["page"], "chunk": h["chunk"]} for h in hits]) == []
