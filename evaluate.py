"""Repeatable test run for the vision + knowledge modules.

    python evaluate.py                       # print results
    python evaluate.py --write-report        # also write docs/EVALUATION.md and docs/eval_results.json
    python evaluate.py --real-dir data/real_photos   # add real photos (folder with labels.csv: filename,label)

Everything is deterministic (fixed seeds), so two runs on the same code and data give identical numbers.
 1. Image detection  data/test_cases.csv (20 images) + a stress set (blur, dark, rotated, noisy copies of unseen images)
 2. Sensor/document  data/test_reports/manifest.csv (OCR case skipped, and said so, if Tesseract is missing)
 3. Retrieval        data/retrieval_eval.csv (top-1 / top-3 page hit) + data/retrieval_negative.csv (must return nothing)
 4. Citations        every evidence chunk is found verbatim on its cited PDF page; recommended action copied from evidence
 5. End to end       detected defect -> top cited page is that defect's procedure
"""
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from knowledge.ingestion.pdf_parser import ocr_available  # noqa: E402
from knowledge.pipeline import VisionKnowledgePipeline  # noqa: E402
from knowledge.retrieval.query import expand_query  # noqa: E402
from knowledge.verify import verify_contract  # noqa: E402
from vision import robustness, synthetic  # noqa: E402
from vision.labels import CLASSES, from_contract_label, to_contract_label  # noqa: E402

# (manual, pages that carry a "Defect Type ... Action Required" procedure for it)
EXPECTED_PAGE = {"bearing_wear": ("bearing_maintenance_manual.pdf", {2, 3}),
                 "structural_crack": ("structural_inspection_sop.pdf", {2}),
                 "corrosion": ("corrosion_control_guide.pdf", {2, 3})}
STRESS_SEED, STRESS_PER_CLASS = 3_000_000, 25  # seeds never used for training (training uses 1_000.., validation 10_000_000..)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(a, b):
    return f"{a} of {b}" + (f" ({a / b:.0%})" if b else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--real-dir")
    args = ap.parse_args()
    pipe = VisionKnowledgePipeline()
    det_meta = getattr(pipe.detector, "meta", {})
    R = OrderedDict()
    R["embedder"] = pipe.store.embedder_name
    R["chunks_indexed"] = len(pipe.store)
    R["ocr_available"] = ocr_available()
    R["detector"] = {k: det_meta.get(k) for k in ("trained_on", "train_images", "degraded_share", "val_accuracy")}
    print(f"embedder: {R['embedder']}   chunks: {R['chunks_indexed']}   OCR: {R['ocr_available']}   detector: {R['detector']}\n")

    # 1+2: detection on the fixed test files, plus 4 + 5 on the same outputs -------------------------
    cases = [(os.path.join(ROOT, "data", "test", r["image"]), "image", r["expected_label"]) for r in rows(os.path.join(ROOT, "data", "test_cases.csv"))]
    cases += [(os.path.join(ROOT, "data", "test_reports", r["file"]), r["input_type"], r["expected_label"]) for r in rows(os.path.join(ROOT, "data", "test_reports", "manifest.csv"))]
    stats, details, skipped = OrderedDict(), [], []
    cite_total = cite_ok = 0
    e2e_ok = e2e_n = 0
    contracts = []
    for path, kind, expected in cases:
        name = os.path.basename(path)
        if "scanned" in name and not ocr_available():
            skipped.append(name)
            print(f"  SKIP  {name}: needs Tesseract OCR (not installed here)")
            continue
        c = pipe.process_case(path, case_id=name, input_type=kind)
        contracts.append(c)
        got = from_contract_label(c["defect"]["type"])
        ok = got == expected
        s = stats.setdefault(kind, [0, 0])
        s[0] += ok
        s[1] += 1
        problems = verify_contract(c)
        cite_total += 1
        cite_ok += not problems
        if expected != "normal":
            e2e_n += 1
            top = c["evidence"][0] if c["evidence"] else None
            manual, pages = EXPECTED_PAGE[expected]
            e2e_ok += bool(ok and top and top["source"] == manual and top["page"] in pages)
        details.append((kind, name, expected, got, c["defect"]["confidence"], ok, problems))
        print(f"  {'OK  ' if ok else 'FAIL'}  {kind:8s} {name:32s} expected {expected:17s} got {got:17s} conf {c['defect']['confidence']:.2f}"
              + (f"   CITATION PROBLEMS: {problems}" if problems else ""))
    R["detection"] = {k: {"correct": v[0], "total": v[1]} for k, v in stats.items()}
    R["skipped"] = skipped
    print()
    for kind, (ok, n) in stats.items():
        print(f"Detection [{kind}]: {pct(ok, n)}")

    # 1b: stress set -------------------------------------------------------------------------------
    stress = OrderedDict()
    for lab in CLASSES:
        for i in range(STRESS_PER_CLASS):
            seed = STRESS_SEED + i
            base = synthetic.generate(lab, seed, "clean")
            for variant, img in {"clean": base, **robustness.hard_variants(base, seed)}.items():
                got = pipe.detector.predict_image(img)["defect"]["type"]
                s = stress.setdefault(variant, [0, 0])
                s[1] += 1
                s[0] += got == to_contract_label(lab)
    R["stress"] = {k: {"correct": v[0], "total": v[1]} for k, v in stress.items()}
    hard_ok = sum(v[0] for k, v in stress.items() if k != "clean")
    hard_n = sum(v[1] for k, v in stress.items() if k != "clean")
    R["stress_hard_total"] = {"correct": hard_ok, "total": hard_n}
    print("Stress set (unseen images, " + str(STRESS_PER_CLASS * len(CLASSES)) + " per variant): "
          + ", ".join(f"{k} {pct(*v)}" for k, v in stress.items()) + f"; all degraded copies {pct(hard_ok, hard_n)}")

    # optional real photos -------------------------------------------------------------------------
    if args.real_dir:
        folder = os.path.abspath(args.real_dir)
        real = rows(os.path.join(folder, "labels.csv"))
        ok = sum(from_contract_label(pipe.detector.predict(os.path.join(folder, r["filename"]))["defect"]["type"]) == from_contract_label(r["label"]) for r in real)
        R["real_photos"] = {"correct": ok, "total": len(real)}
        print(f"Real photos ({args.real_dir}): {pct(ok, len(real))}")

    # 3: retrieval ---------------------------------------------------------------------------------
    qs = rows(os.path.join(ROOT, "data", "retrieval_eval.csv"))
    top1 = top3 = 0
    misses = []
    for q in qs:
        hits = pipe.store.search(q["question"], top_k=3)
        want = (q["expected_source"], int(q["expected_page"]))
        keys = [(h["source"], h["page"]) for h in hits]
        top1 += bool(keys and keys[0] == want)
        top3 += want in keys
        if not keys or keys[0] != want:
            misses.append((q["question"], want, keys[:3]))
            print(f"  retrieval miss: {q['question'][:70]!r} want {want} got {keys[:3]}")
    neg = rows(os.path.join(ROOT, "data", "retrieval_negative.csv"))
    leaked = [n["question"] for n in neg if pipe.store.search(expand_query(n["question"]), top_k=3)]
    for q in leaked:
        print(f"  off-topic question returned evidence: {q!r}")
    R["retrieval"] = {"top1": top1, "top3": top3, "total": len(qs), "off_topic_total": len(neg), "off_topic_wrongly_answered": len(leaked)}
    print(f"Retrieval: correct page in top-1 for {pct(top1, len(qs))}, in top-3 for {pct(top3, len(qs))}")
    print(f"Off-topic questions correctly given no evidence: {pct(len(neg) - len(leaked), len(neg))}")

    # 4: citations ---------------------------------------------------------------------------------
    R["citations"] = {"verified": cite_ok, "total": cite_total}
    print(f"Citations: {pct(cite_ok, cite_total)} outputs have every evidence chunk verbatim on its cited page and an action copied from evidence")

    # 5: end to end --------------------------------------------------------------------------------
    R["end_to_end"] = {"correct": e2e_ok, "total": e2e_n}
    print(f"End to end (right defect AND its procedure is the top cited page): {pct(e2e_ok, e2e_n)}")

    # repeatability: the same inputs must give identical JSON
    again = [pipe.process_case(os.path.join(ROOT, "data", "test", "test_0%d.png" % n)) for n in (2, 4, 5)]
    first = [pipe.process_case(os.path.join(ROOT, "data", "test", "test_0%d.png" % n)) for n in (2, 4, 5)]
    R["repeatable"] = json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)
    print(f"Repeatable (same input, same JSON): {R['repeatable']}")

    if args.write_report:
        write_report(R, details, misses, leaked)
    return R


def write_report(R, details, misses, leaked):
    out = os.path.join(ROOT, "docs")
    json.dump(R, open(os.path.join(out, "eval_results.json"), "w"), indent=2)
    d, s, r = R["detection"], R["stress"], R["retrieval"]
    L = ["# Vision and Knowledge: test results", "",
         "Generated by `python evaluate.py --write-report` (deterministic; re-running gives the same numbers).", "",
         f"- Embedder: `{R['embedder']}` ({R['chunks_indexed']} chunks indexed from `data/raw`)",
         f"- OCR available when run: {R['ocr_available']}" + (f" (skipped: {', '.join(R['skipped'])})" if R["skipped"] else ""),
         f"- Detector: {R['detector']}", "",
         "## Summary", "", "| Check | Result |", "|---|---|"]
    for kind, v in d.items():
        L.append(f"| Detection, {kind} test files | {pct(v['correct'], v['total'])} |")
    for variant, v in s.items():
        L.append(f"| Stress set, {variant} copies of unseen images | {pct(v['correct'], v['total'])} |")
    L += [f"| Retrieval: right page in top-1 | {pct(r['top1'], r['total'])} |",
          f"| Retrieval: right page in top-3 | {pct(r['top3'], r['total'])} |",
          f"| Off-topic questions that correctly return no evidence | {pct(r['off_topic_total'] - r['off_topic_wrongly_answered'], r['off_topic_total'])} |",
          f"| Citations verified verbatim against the source PDF page | {pct(R['citations']['verified'], R['citations']['total'])} |",
          f"| End to end (defect + procedure page) | {pct(R['end_to_end']['correct'], R['end_to_end']['total'])} |",
          f"| Same input gives identical JSON | {R['repeatable']} |", "",
          "## Per-file detection", "", "| Type | File | Expected | Got | Confidence | Result | Citation problems |", "|---|---|---|---|---|---|---|"]
    for kind, name, exp, got, conf, ok, problems in details:
        L.append(f"| {kind} | {name} | {exp} | {got} | {conf:.2f} | {'OK' if ok else 'FAIL'} | {'; '.join(problems) or 'none'} |")
    L += ["", "## Retrieval misses (right page not ranked first)", ""]
    L += [f"- {q!r}: wanted {w}, got {g}" for q, w, g in misses] or ["- none"]
    if leaked:
        L += ["", "## Off-topic questions that wrongly returned evidence", ""] + [f"- {q!r}" for q in leaked]
    L += ["", "## How to read this", "",
          "- All images are generated (`vision/synthetic.py`); no real factory photos were available. The stress set blurs, darkens, rotates",
          "  and adds noise to unseen images (the same four degradations as the team's dual-detector test). It measures robustness to those",
          "  changes, not accuracy on real photos. Add real photos with `python evaluate.py --real-dir <folder with labels.csv>`.",
          "- Manuals are short synthetic samples written for the prototype; the knowledge base is rebuilt from `data/raw/*.pdf` on every run.",
          "- If the embedder line says `tfidf`, semantic search (MiniLM) was not available; re-run where `sentence-transformers` can download its model."]
    open(os.path.join(out, "EVALUATION.md"), "w").write("\n".join(L) + "\n")
    print("wrote docs/EVALUATION.md and docs/eval_results.json")


if __name__ == "__main__":
    main()
