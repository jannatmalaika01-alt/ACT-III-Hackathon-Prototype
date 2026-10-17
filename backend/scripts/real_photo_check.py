"""Check the CNN and the dual detector on REAL photos with known labels.

    cd backend
    PYTHONPATH=. python scripts/real_photo_check.py --photos ../data/real_photos
    PYTHONPATH=. python scripts/real_photo_check.py --photos ../data/real_photos --vision-model Gemma-4-31B-IT

The folder holds the photos and a labels.csv (header `filename,label`), for example:

    filename,label
    pump_01.jpg,bearing_wear
    weld_07.png,structural_crack
    pipe_03.jpg,corrosion
    motor_02.jpg,normal

Allowed labels: normal, bearing_wear, structural_crack, corrosion (spaces instead of underscores are fine).
Each photo makes ONE vision-model request, paced to --per-minute (default 15; the shared key allows 20/min).
The key is read from backend/.env and never printed.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402,F401  (loads backend/.env)
from app import dual_detector as dd  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_sleep = time.sleep  # tests replace this so they do not wait


def read_labels(folder: Path, labels_file: Path) -> dict:
    if not labels_file.is_file():
        raise SystemExit(f"labels file not found: {labels_file}\nExpected a CSV with the header: filename,label")
    labels, problems = {}, []
    with open(labels_file, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or {"filename", "label"} - {h.strip() for h in reader.fieldnames}:
            raise SystemExit("labels.csv must have the header: filename,label")
        for n, row in enumerate(reader, start=2):
            name, label = (row["filename"] or "").strip(), dd.norm_label(row["label"])
            if not (folder / name).is_file():
                problems.append(f"line {n}: photo not found: {name}")
            elif Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                problems.append(f"line {n}: not an image file: {name}")
            elif label not in dd.LABELS:
                problems.append(f"line {n}: unknown label '{row['label']}' (allowed: {', '.join(dd.LABELS)})")
            else:
                labels[name] = label
    if problems:
        raise SystemExit("labels.csv problems:\n  " + "\n  ".join(problems))
    if not labels:
        raise SystemExit("labels.csv has no rows")
    return labels


def category(truth: str, dual) -> str:
    """One of the six outcomes for a photo."""
    if dual is None or dual["agreement"] == dd.SINGLE:
        return "single_detector"
    cnn, vlm = dual["cnn"]["label"], dual["vision_model"]["label"]
    if cnn == vlm:
        return "agree_correct" if cnn == truth else "agree_wrong"
    if cnn == truth:
        return "disagree_cnn_right"
    return "disagree_vision_right" if vlm == truth else "disagree_both_wrong"


def run(folder: Path, labels: dict, per_minute: float) -> list:
    rows = []
    for i, (name, truth) in enumerate(sorted(labels.items())):
        if i:
            _sleep(60.0 / per_minute)
        start = time.time()
        dual = dd.run_dual(str(folder / name))
        rows.append({"file": name, "truth": truth, "dual": dual, "category": category(truth, dual),
                     "seconds": time.time() - start})
    return rows


def render(rows: list, model: str) -> str:
    c = Counter(r["category"] for r in rows)
    both = [r for r in rows if r["category"] != "single_detector"]
    cnn_wrong = [r for r in both if r["dual"]["cnn"]["label"] != r["truth"]]
    caught = c["disagree_vision_right"] + c["disagree_both_wrong"]
    out = [f"Vision model: {model}   photos: {len(rows)}   with both detectors answering: {len(both)}", "",
           "| Outcome | Photos |", "|---|---|",
           f"| Agree, correct | {c['agree_correct']} |",
           f"| Agree, WRONG (both wrong the same way) | {c['agree_wrong']} |",
           f"| Disagree, CNN right | {c['disagree_cnn_right']} |",
           f"| Disagree, vision model right | {c['disagree_vision_right']} |",
           f"| Disagree, both wrong | {c['disagree_both_wrong']} |",
           f"| single_detector (one detector gave no answer) | {c['single_detector']} |", "",
           f"CNN mistakes (among photos where both answered): {len(cnn_wrong)}",
           f"  caught by the disagreement flag: {caught} of {len(cnn_wrong)}",
           f"  false alarms (flagged although the CNN was right): {c['disagree_cnn_right']}",
           f"  missed (both agreed on a wrong label): {c['agree_wrong']}", "", "Per photo:"]
    for r in rows:
        d = r["dual"]
        cnn = f"{d['cnn']['label']} ({d['cnn']['confidence']:.2f})" if d and d["cnn"].get("label") else "no answer"
        v = d["vision_model"] if d else {}
        vis = f"{v['label']} ({v['confidence'] if v.get('confidence') is not None else '?'})" if v.get("label") else "no answer"
        out.append(f"  {r['file']:28} truth={r['truth']:17} CNN={cnn:24} vision={vis:24} {r['category']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="folder with the photos and labels.csv")
    ap.add_argument("--labels", help="labels file (default: <photos>/labels.csv)")
    ap.add_argument("--vision-model", help="override DUAL_VISION_MODEL for this run, e.g. Gemma-4-31B-IT")
    ap.add_argument("--per-minute", type=float, default=15.0, help="max photos per minute (default 15)")
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args(argv)

    folder = Path(args.photos)
    if not folder.is_dir():
        raise SystemExit(f"photo folder not found: {folder}")
    labels = read_labels(folder, Path(args.labels) if args.labels else folder / "labels.csv")
    if args.vision_model:
        os.environ["DUAL_VISION_MODEL"] = args.vision_model
    model = dd.vision_model()
    print(f"{len(labels)} photos, vision model {model}, about {len(labels) / args.per_minute:.1f} min at "
          f"<= {args.per_minute:g}/min\n", flush=True)
    start = time.time()
    rows = run(folder, labels, args.per_minute)
    report = render(rows, model) + f"\n\nTotal time: {time.time() - start:.0f}s"
    print(report)
    if args.out:
        Path(args.out).write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
