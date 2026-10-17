"""Train a CANDIDATE CNN from human-approved feedback and compare it with the current one.

    python scripts/retrain_from_feedback.py                      # uses data/feedback/ (git-ignored)
    python scripts/retrain_from_feedback.py --real-photos data/real_photos --epochs 12

(a) Trains vision/weights/candidate_cnn.pt with the same loop as `python -m vision.train` (generated images + any
    photos in data/train/ + the approved feedback photos, x3 weight like real photos).
(b) Prints the accuracy of the candidate AND of the current vision/weights/defect_cnn.pt on held-out sets: the
    generated test images (data/test_cases.csv), the real photos you name (labels.csv), and 20% of the feedback that was
    NOT used for training. Photos that were used for training are skipped in the real-photo comparison.

It NEVER overwrites vision/weights/defect_cnn.pt (the output path is checked). Promoting the candidate is a manual
decision: copy the file yourself after looking at the numbers.
Only approved feedback is used; diagnoses that were flagged UNVERIFIED are left out unless --include-unverified.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
HOLDOUT_EVERY = 5  # 1 in 5 feedback photos (chosen by a stable hash of its case id) is held out, never trained on


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_feedback(folder: Path, include_unverified: bool) -> list:
    """[{path, label, case_id, sha}] for every approved photo that has its JSON."""
    from vision.labels import CLASSES
    items = []
    for label in CLASSES:
        for js in sorted((folder / label).glob("*.json")) if (folder / label).is_dir() else []:
            meta = json.loads(js.read_text())
            if meta.get("approved_label") != label:
                continue
            if meta.get("diagnosis_was_unverified") and not include_unverified:
                continue
            img = next((js.with_suffix(ext) for ext in IMAGE_SUFFIXES if js.with_suffix(ext).exists()), None)
            if img is not None:
                items.append({"path": img, "label": label, "case_id": meta.get("case_id") or js.stem, "sha": sha(img)})
    return items


def split_feedback(items: list) -> tuple:
    train, hold = [], []
    for it in items:
        (hold if int(hashlib.sha256(str(it["case_id"]).encode()).hexdigest()[:8], 16) % HOLDOUT_EVERY == 0 else train).append(it)
    return train, hold


def read_labelled_folder(folder: Path, labels_csv: Path) -> list:
    with open(labels_csv, newline="") as f:
        rows = [(r["filename"].strip(), r["label"].strip().lower().replace(" ", "_")) for r in csv.DictReader(f)]
    return [{"path": folder / n, "label": l} for n, l in rows if (folder / n).is_file()]


def accuracy(weights: Path, samples: list) -> tuple:
    from PIL import Image
    from vision.defect_detector import DefectDetector
    det = DefectDetector(str(weights))
    ok = 0
    for s in samples:
        with Image.open(s["path"]) as im:
            probs = det.probabilities(im.convert("RGB"))
        ok += max(probs, key=probs.get) == s["label"]
    return ok, len(samples)


def check_output_path(out: Path) -> None:
    from vision import train as vtrain
    live = Path(vtrain.WEIGHTS).resolve()
    if out.resolve() == live or out.name == "defect_cnn.pt" or out.suffix != ".pt":
        raise SystemExit(f"Refusing to write {out}: this script never touches the live weights ({live.name}). "
                         "Use a different .pt file such as vision/weights/candidate_cnn.pt.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback", default=str(REPO / "data" / "feedback"))
    ap.add_argument("--out", default=str(REPO / "vision" / "weights" / "candidate_cnn.pt"))
    ap.add_argument("--real-photos", default=None, help="folder with photos + labels.csv to compare on (default: data/real_photos if present)")
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--val-per-class", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--feedback-weight", type=int, default=3)
    ap.add_argument("--include-unverified", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    check_output_path(out)  # before any work: the live weights are never at risk

    import numpy as np
    import torch
    from PIL import Image
    from vision import train as vtrain
    from vision.labels import CLASSES
    from vision.preprocess import to_array

    items = load_feedback(Path(args.feedback), args.include_unverified)
    if not items:
        print(f"No approved feedback photos in {args.feedback} yet. Approve some image cases in the app first; "
              "nothing was trained and nothing was changed.")
        return 0
    train_items, hold_items = split_feedback(items)
    print(f"feedback: {len(items)} approved photos -> {len(train_items)} for training, {len(hold_items)} held out")

    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    xs, ys = vtrain._synthetic(args.per_class, vtrain.TRAIN_SEED)
    rx, ry = vtrain._real_images()
    xs += rx * 3
    ys += ry * 3
    for it in train_items:
        with Image.open(it["path"]) as im:
            xs += [to_array(im.convert("RGB"))] * args.feedback_weight
        ys += [CLASSES.index(it["label"])] * args.feedback_weight
    vx, vy = vtrain._synthetic(args.val_per_class, vtrain.VAL_SEED)
    model, val_acc = vtrain.fit(xs, ys, vx, vy, args.epochs, args.lr)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps({
        "classes": CLASSES, "val_accuracy": round(val_acc, 4), "train_images": len(xs), "feedback_photos": len(train_items),
        "trained_on": "synthetic + approved feedback", "NOT_PROMOTED": "copy over defect_cnn.pt by hand if it is better"}, indent=2))
    print(f"saved candidate {out}")

    # ---- compare on held-out sets ----
    live = Path(vtrain.WEIGHTS)
    train_hashes = {it["sha"] for it in train_items}
    sets = []
    tc = REPO / "data" / "test_cases.csv"
    if tc.is_file():
        rows = [{"path": REPO / "data" / "test" / r["image"], "label": r["expected_label"]} for r in csv.DictReader(open(tc))]
        sets.append(("generated test images (data/test_cases.csv)", [r for r in rows if r["path"].is_file()]))
    rp = Path(args.real_photos) if args.real_photos else REPO / "data" / "real_photos"
    if (rp / "labels.csv").is_file():
        real = read_labelled_folder(rp, rp / "labels.csv")
        fresh = [s for s in real if sha(s["path"]) not in train_hashes]
        if len(fresh) != len(real):
            print(f"real photos: skipped {len(real) - len(fresh)} that were used for training")
        sets.append((f"real photos ({rp.name})", fresh))
    if hold_items:
        sets.append(("feedback hold-out (not trained on)", [{"path": it["path"], "label": it["label"]} for it in hold_items]))

    print("\nAccuracy on held-out data (the candidate was never shown these):\n")
    print(f"{'Held-out set':48} {'photos':>6} {'current defect_cnn.pt':>22} {'candidate':>12}")
    for name, samples in sets:
        if not samples:
            continue
        a, n = accuracy(live, samples)
        b, _ = accuracy(out, samples)
        print(f"{name:48} {n:6d} {f'{a}/{n} = {100 * a / n:.0f}%':>22} {f'{b}/{n} = {100 * b / n:.0f}%':>12}")
    print(f"\n{live.name} was NOT changed. If the candidate is clearly better on the REAL sets, promote it by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
