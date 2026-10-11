"""Run the full test set and report results.      python evaluate.py [--json results.json]

1. Image detection      data/test_cases.csv            (10 original + 10 held-out harder images)
2. Sensor/document      data/test_reports/manifest.csv (OCR case is skipped, and said so, if Tesseract is missing)
3. Retrieval            data/retrieval_eval.csv        correct manual page found in top-1 / top-3
4. End to end           detected defect -> top evidence chunk is the right defect procedure
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from knowledge.ingestion.pdf_parser import ocr_available  # noqa: E402
from knowledge.pipeline import VisionKnowledgePipeline  # noqa: E402
from vision.labels import from_contract_label  # noqa: E402

# which manual page documents the procedure for each defect class
# (manual, pages that carry a "Defect Type ... Action Required" procedure for it)
EXPECTED_PAGE = {"bearing_wear": ("bearing_maintenance_manual.pdf", {2, 3}),
                 "structural_crack": ("structural_inspection_sop.pdf", {2}),
                 "corrosion": ("corrosion_control_guide.pdf", {2, 3})}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    pipe = VisionKnowledgePipeline()
    print(f"embedder: {pipe.store.embedder_name}   chunks indexed: {len(pipe.store)}   OCR available: {ocr_available()}\n")
    out = {"embedder": pipe.store.embedder_name}

    # 1 + 2: detection ------------------------------------------------------------------------------
    cases = [(os.path.join(ROOT, "data", "test", r["image"]), "image", r["expected_label"]) for r in rows(os.path.join(ROOT, "data", "test_cases.csv"))]
    cases += [(os.path.join(ROOT, "data", "test_reports", r["file"]), r["input_type"], r["expected_label"]) for r in rows(os.path.join(ROOT, "data", "test_reports", "manifest.csv"))]
    stats = {}
    e2e_ok = e2e_n = 0
    for path, kind, expected in cases:
        name = os.path.basename(path)
        if name.endswith(".pdf") and "scanned" in name and not ocr_available():
            print(f"  SKIP  {name}: needs Tesseract OCR (not installed here)")
            continue
        c = pipe.process_case(path, case_id=name, input_type=kind)
        got = from_contract_label(c["defect"]["type"])
        ok = got == expected
        s = stats.setdefault(kind, [0, 0])
        s[0] += ok
        s[1] += 1
        if expected != "normal":
            e2e_n += 1
            top = c["evidence"][0] if c["evidence"] else None
            manual, pages = EXPECTED_PAGE[expected]
            e2e_ok += bool(ok and top and top["source"] == manual and top["page"] in pages)
        print(f"  {'OK  ' if ok else 'FAIL'}  {kind:8s} {name:32s} expected {expected:17s} got {got:17s} conf {c['defect']['confidence']:.2f}")
    print()
    for kind, (ok, n) in stats.items():
        print(f"Detection [{kind}]: {ok} of {n} correct")
    out["detection"] = {k: {"correct": v[0], "total": v[1]} for k, v in stats.items()}

    # 3: retrieval ----------------------------------------------------------------------------------
    qs = rows(os.path.join(ROOT, "data", "retrieval_eval.csv"))
    top1 = top3 = 0
    for q in qs:
        hits = pipe.store.search(q["question"], top_k=3)
        want = (q["expected_source"], int(q["expected_page"]))
        keys = [(h["source"], h["page"]) for h in hits]
        top1 += bool(keys and keys[0] == want)
        top3 += want in keys
        if not keys or keys[0] != want:
            print(f"  retrieval miss: {q['question'][:70]!r} want {want} got {keys[:3]}")
    print(f"Retrieval: correct section found in top-1 for {top1} of {len(qs)}, in top-3 for {top3} of {len(qs)}")
    out["retrieval"] = {"top1": top1, "top3": top3, "total": len(qs)}
    print(f"End to end (defect found AND its procedure is the top cited page): {e2e_ok} of {e2e_n}")
    out["end_to_end"] = {"correct": e2e_ok, "total": e2e_n}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
