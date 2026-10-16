"""Compare six ways of combining the detectors. The rules are the app's own (app/dual_detector.py decide_labels), so
this replay and DUAL_MODE can never drift apart.

    0  CNN alone.
    A  Pair: CNN + Qwen3.8-Flash-Next. Disagree -> human review.
    B  Independent 3-way vote: CNN, Flash-Next, Gemma-4-31B-IT. 2-of-3 wins; a 3-way split -> human review.
    C  CNN always counts: like B, but if the two vision models agree with each other and OUTVOTE the CNN, the case
       goes to human review instead of being decided silently.
    U  Unanimity: CNN, Flash-Next and Gemma. Automatic ONLY if all available detectors agree; any disagreement -> review.
    (also reported: the pair CNN + Gemma-4-31B-IT)
    A detector with no answer for an image is MISSING: with 2 voters left, agreement wins and disagreement -> review;
    with 1 left it answers alone (like single_detector in the app); with 0 the case goes to review.

Phase 1 (offline, no API calls): replay the saved synthetic test
    PYTHONPATH=. python scripts/voting_compare.py --from-round1 [--md OUT.md]

Phase 2 (real photos, ONE call per photo per vision model, paced; answers are saved first, then replayed)
    PYTHONPATH=. python scripts/voting_compare.py --photos ../data/real_photos
    PYTHONPATH=. python scripts/voting_compare.py --replay-only            # re-score the saved answers, no API call

The photo folder needs labels.csv (filename,label). Photos and the saved answers file are git-ignored.
The API key is read from backend/.env and never printed.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import dual_detector as dd  # noqa: E402  (the app's voting rules)

REPO = Path(__file__).resolve().parents[2]
LABELS = ("normal", "bearing_wear", "structural_crack", "corrosion")
DETECTORS = ("cnn", "flash", "gemma")
FLASH_MODEL, GEMMA_MODEL = "Qwen3.8-Flash-Next", "Gemma-4-31B-IT"
SCHEMES = [  # key, title, detectors that must be called (parallel), extra API calls per upload
    ("0", "0  CNN alone", ("cnn",), 0),
    ("A", "A  Pair: CNN + Flash-Next", ("cnn", "flash"), 1),
    ("A_gemma", "A' Pair: CNN + Gemma", ("cnn", "gemma"), 1),
    ("B", "B  3-way vote (2-of-3)", ("cnn", "flash", "gemma"), 2),
    ("C", "C  3-way vote, CNN outvoted -> review", ("cnn", "flash", "gemma"), 2),
    ("U", "U  Unanimity (all must agree)", ("cnn", "flash", "gemma"), 2),
]
MODE_OF = {"B": "vote", "C": "cnn_anchored", "U": "unanimous"}  # scheme -> app DUAL_MODE (A / A' are `pair`)
DEFAULT_ANSWERS = REPO / "data" / "voting_answers.json"
_sleep = time.sleep  # tests replace this

DECISION_RULE = """Decision rule (written BEFORE any real-photo result was seen):
 (a) The scheme with the fewest SILENT wrong answers wins.
 (b) A scheme where the vision models outvote a CNN that was right is OUT, unless that case is flagged for a human
     (the column 'CNN right but silently outvoted' must be 0).
 (c) Among ties, the lowest human-review rate wins.
 (d) Extra calls only matter if the average latency exceeds about 12 s or the slowest exceeds about 45 s.
 (e) If the results are too close or too noisy with this few photos, KEEP `pair` (the default).
 With 20-40 photos only LARGE differences can be seen; a gap of 1 to 3 photos is noise."""


# ---------------- the voting rules (the app's own) ----------------

def decide(scheme: str, labels: dict) -> dict:
    """labels = {"cnn": label|None, "flash": label|None, "gemma": label|None}.
    Returns {"status": "auto"|"review", "label": label or None, "reason": str, "ref": label or None}.
    `ref` is the label used to judge a review: the CNN's label if it answered, else the leading label."""
    cnn, flash, gemma = labels.get("cnn"), labels.get("flash"), labels.get("gemma")
    if scheme == "0":
        if cnn:
            return {"status": "auto", "label": cnn, "reason": "cnn alone", "ref": cnn}
        return {"status": "review", "label": None, "reason": "no answer", "ref": None}
    if scheme == "A":
        return dd.decide_labels("pair", cnn, flash)
    if scheme == "A_gemma":
        return dd.decide_labels("pair", cnn, gemma)
    return dd.decide_labels(MODE_OF[scheme], cnn, flash, gemma)


# ---------------- scoring ----------------

def score(images: list, scheme_key: str) -> dict:
    """images: [{"file","truth","variant","answers": {det: {"label","confidence","latency","error"}|None}}]"""
    scheme = next(s for s in SCHEMES if s[0] == scheme_key)
    c = Counter()
    lats = []
    for im in images:
        labels = {d: _label(im["answers"].get(d)) for d in DETECTORS}
        r = decide(scheme_key, labels)
        involved = [im["answers"].get(d) for d in scheme[2] if im["answers"].get(d)]
        lats.append(max([a.get("latency") or 0.0 for a in involved] or [0.0]))
        truth = im["truth"]
        if r["status"] == "auto":
            c["auto_correct" if r["label"] == truth else "silent_wrong"] += 1
            # the CNN was right, both vision models agreed on another label, and the scheme went with them silently
            if (r["label"] != truth and labels["cnn"] == truth and labels["flash"] and labels["flash"] == labels["gemma"]
                    and labels["flash"] != labels["cnn"]):
                c["silent_outvote"] += 1
        else:
            c["review"] += 1
            c["false_alarm" if r["ref"] == truth else "good_flag"] += 1
            if r["reason"] == "3-way split":
                c["review_split"] += 1
            if r["reason"].startswith("CNN outvoted"):
                c["review_outvoted"] += 1
    n = len(images)
    return {"scheme": scheme_key, "title": scheme[1], "n": n, "auto_correct": c["auto_correct"],
            "silent_wrong": c["silent_wrong"], "silent_outvote": c["silent_outvote"], "review": c["review"], "review_rate": c["review"] / n if n else 0.0,
            "false_alarm": c["false_alarm"], "good_flag": c["good_flag"], "extra_calls": scheme[3],
            "lat_avg": sum(lats) / n if n else 0.0, "lat_max": max(lats) if lats else 0.0}


def _label(a):
    return a["label"] if a and not a.get("error") and a.get("label") in LABELS else None


def dataset_facts(images: list) -> dict:
    splits = outvoted = 0
    for im in images:
        l = {d: _label(im["answers"].get(d)) for d in DETECTORS}
        if all(l.values()) and len(set(l.values())) == 3:
            splits += 1
        if l["cnn"] == im["truth"] and l["flash"] and l["flash"] == l["gemma"] and l["flash"] != l["cnn"]:
            outvoted += 1
    return {"n": len(images), "three_way_splits": splits, "cnn_right_outvoted": outvoted,
            "gemma_missing": sum(1 for im in images if not _label(im["answers"].get("gemma"))),
            "flash_missing": sum(1 for im in images if not _label(im["answers"].get("flash"))),
            "cnn_missing": sum(1 for im in images if not _label(im["answers"].get("cnn")))}


def groups(images: list) -> list:
    variants = {im.get("variant") for im in images}
    if variants & {"clean"} and variants - {"clean"}:
        return [("clean images", [i for i in images if i["variant"] == "clean"]),
                ("hard images (blurred, darkened, rotated, noisy copies)", [i for i in images if i["variant"] != "clean"]),
                ("all images", images)]
    return [("all images", images)]


# ---------------- consistency check against numbers already published ----------------

def pair_breakdown(images, vis):
    """The round-2 categories from docs/dual_detector_results.md for CNN + one vision model."""
    c = Counter()
    for im in images:
        t, cnn, v = im["truth"], _label(im["answers"].get("cnn")), _label(im["answers"].get(vis))
        if not v:
            c["single_detector"] += 1
        elif v == cnn:
            c["agree_correct" if cnn == t else "agree_wrong"] += 1
        else:
            c["disagree_cnn_right" if cnn == t else ("disagree_vision_right" if v == t else "disagree_both_wrong")] += 1
    return c


def check_known(images: list) -> list:
    """Numbers that docs/dual_detector_results.md already published. Returns a list of mismatches (empty = OK)."""
    bad = []

    def expect(name, got, want):
        if got != want:
            bad.append(f"{name}: replay gives {got}, published {want}")

    clean = [i for i in images if i["variant"] == "clean"]
    hard = [i for i in images if i["variant"] != "clean"]
    expect("images", len(images), 100)
    expect("CNN clean correct", sum(_label(i["answers"]["cnn"]) == i["truth"] for i in clean), 20)
    expect("CNN hard correct", sum(_label(i["answers"]["cnn"]) == i["truth"] for i in hard), 62)
    wrong = [i for i in images if _label(i["answers"]["cnn"]) != i["truth"]]
    expect("CNN mistakes", len(wrong), 18)
    expect("CNN mistakes with confidence >= 0.70", sum(i["answers"]["cnn"]["confidence"] >= 0.70 for i in wrong), 11)
    s0 = score(images, "0")
    expect("scheme 0 silent wrong", s0["silent_wrong"], 18)
    a = pair_breakdown(images, "flash")
    for k, v in {"agree_correct": 79, "agree_wrong": 0, "disagree_cnn_right": 3, "disagree_vision_right": 17,
                 "disagree_both_wrong": 1, "single_detector": 0}.items():
        expect(f"pair CNN+Flash-Next {k}", a[k], v)
    sa = score(images, "A")
    expect("scheme A: CNN mistakes caught", sa["good_flag"], 18)
    expect("scheme A: false alarms", sa["false_alarm"], 3)
    expect("scheme A: missed (silent wrong)", sa["silent_wrong"], 0)
    g = pair_breakdown(images, "gemma")
    for k, v in {"agree_correct": 80, "agree_wrong": 0, "disagree_cnn_right": 0, "disagree_vision_right": 17,
                 "disagree_both_wrong": 0, "single_detector": 3}.items():
        expect(f"pair CNN+Gemma {k}", g[k], v)
    return bad


# ---------------- loading / saving ----------------

def from_round1() -> list:
    lab = REPO / "data" / "dual_lab"
    manifest = {r["file"]: r for r in csv.DictReader(open(lab / "manifest.csv"))}
    def load(name):
        return json.load(open(lab / "round1" / f"{name}.json"))["results"]
    raw = {"cnn": load("cnn"), "flash": load(FLASH_MODEL), "gemma": load(GEMMA_MODEL)}
    images = []
    for f, m in manifest.items():
        ans = {}
        for d, res in raw.items():
            r = res.get(f)
            ans[d] = None if r is None else {"label": r["pred"], "confidence": r.get("confidence"),
                                              "latency": r.get("latency"), "error": r.get("error")}
        images.append({"file": f, "truth": m["label"], "variant": m["variant"], "answers": ans})
    return images


def save_answers(path: Path, images: list, models: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": models, "images": images}, indent=1))


def load_answers(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"No saved answers at {path}. Run once with --photos first.")
    return json.loads(path.read_text())["images"]


# ---------------- Phase 2: collect answers from real photos ----------------

def read_labels(folder: Path, labels_file: Path) -> dict:
    if not labels_file.is_file():
        raise SystemExit(f"labels file not found: {labels_file}\nExpected a CSV with the header: filename,label")
    out, problems = {}, []
    with open(labels_file, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or {"filename", "label"} - {h.strip() for h in reader.fieldnames}:
            raise SystemExit("labels.csv must have the header: filename,label")
        for n, row in enumerate(reader, start=2):
            name = (row["filename"] or "").strip()
            label = (row["label"] or "").strip().lower().replace(" ", "_")
            if not (folder / name).is_file():
                problems.append(f"line {n}: photo not found: {name}")
            elif label not in LABELS:
                problems.append(f"line {n}: unknown label '{row['label']}' (allowed: {', '.join(LABELS)})")
            else:
                out[name] = label
    if problems:
        raise SystemExit("labels.csv problems:\n  " + "\n  ".join(problems))
    if not out:
        raise SystemExit("labels.csv has no rows")
    return out


def _ask(dd, model: str, path: str) -> dict:
    os.environ["DUAL_VISION_MODEL"] = model
    start = time.time()
    try:
        r = dd.run_vision_model(path)
        return {"label": r["label"], "confidence": r.get("confidence"), "latency": time.time() - start, "error": None}
    except Exception as exc:
        import httpx
        status = getattr(getattr(exc, "response", None), "status_code", None)
        kind = ("rate_limit" if status == 429 else "timeout" if isinstance(exc, (httpx.TimeoutException, TimeoutError))
                else type(exc).__name__)
        return {"label": None, "confidence": None, "latency": time.time() - start, "error": kind}


def collect(folder: Path, labels: dict, answers_path: Path, flash_model: str, gemma_model: str, per_minute: float) -> list:
    import core  # noqa: F401  (loads backend/.env)
    from app import dual_detector as dd

    done = {im["file"]: im for im in (json.loads(answers_path.read_text())["images"] if answers_path.is_file() else [])}
    gap, models = 60.0 / per_minute, {"cnn": "CNN", "flash": flash_model, "gemma": gemma_model}
    images, first = [], True
    for name, truth in sorted(labels.items()):
        im = done.get(name)
        if im and im["truth"] == truth and all(im["answers"].get(d) for d in DETECTORS):
            images.append(im)  # resume: already answered in an earlier run
            continue
        path = str(folder / name)
        start = time.time()
        try:
            c = dd.run_cnn(path)
            cnn = {"label": c["label"], "confidence": c["confidence"], "latency": time.time() - start, "error": None}
        except Exception as exc:
            cnn = {"label": None, "confidence": None, "latency": time.time() - start, "error": type(exc).__name__}
        ans = {"cnn": cnn}
        for det in ("flash", "gemma"):
            if not first:
                _sleep(gap)
            first = False
            ans[det] = _ask(dd, models[det], path)
        images.append({"file": name, "truth": truth, "variant": "real", "answers": ans})
        save_answers(answers_path, images, models)  # keep every raw answer, even if a later photo fails
        print(f"  {name}: truth={truth} cnn={cnn['label']} flash={ans['flash']['label'] or ans['flash']['error']} "
              f"gemma={ans['gemma']['label'] or ans['gemma']['error']}", flush=True)
    # ONE retry for answers lost to a rate limit, slower
    redo = [(im, d) for im in images for d in ("flash", "gemma") if im["answers"][d].get("error") == "rate_limit"]
    for im, det in redo:
        _sleep(60.0 / min(per_minute, 12))
        im["answers"][det] = _ask(dd, models[det], str(folder / im["file"]))
    if redo:
        print(f"  retried {len(redo)} rate-limited answers once", flush=True)
    save_answers(answers_path, images, models)
    return images


# ---------------- report ----------------

def render(images: list, title_note: str = "") -> str:
    out = []
    for gname, subset in groups(images):
        if not subset:
            continue
        facts = dataset_facts(subset)
        out += [f"### {gname.capitalize()} ({len(subset)} images)", "",
                "| Scheme | Auto answers correct | SILENT wrong | CNN right but silently outvoted | Human review (rate) | False alarms "
                "| Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |", "|---|---|---|---|---|---|---|---|---|"]
        for key, *_ in SCHEMES:
            s = score(subset, key)
            out.append(f"| {s['title']} | {s['auto_correct']} | **{s['silent_wrong']}** | {s['silent_outvote']} | {s['review']} "
                       f"({100 * s['review_rate']:.0f}%) | {s['false_alarm']} | {s['good_flag']} | {s['extra_calls']} | "
                       f"{s['lat_avg']:.1f}s / {s['lat_max']:.1f}s |")
        out += ["", f"Facts about these images: 3-way splits (all three detectors answered, three different labels): "
                f"{facts['three_way_splits']}; CNN right but outvoted by both vision models: {facts['cnn_right_outvoted']}; "
                f"images with a missing detector answer: CNN {facts['cnn_missing']}, Flash-Next {facts['flash_missing']}, "
                f"Gemma {facts['gemma_missing']}.", ""]
    out += ["Definitions: 'silent wrong' = the scheme answered automatically and the answer was wrong (no human asked). "
            "'False alarm' = sent to a human although the CNN's own answer was right; 'good flag' = sent to a human and the "
            "CNN's answer was wrong. Latency is the slowest detector the scheme waits for (they run in parallel).", "",
            DECISION_RULE]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-round1", action="store_true", help="Phase 1: replay data/dual_lab/round1 (no API calls)")
    ap.add_argument("--photos", help="Phase 2: folder with real photos and labels.csv (calls the API)")
    ap.add_argument("--labels")
    ap.add_argument("--replay-only", action="store_true", help="re-score the saved answers, no API call")
    ap.add_argument("--answers", default=str(DEFAULT_ANSWERS), help="where Phase 2 saves the raw answers (git-ignored)")
    ap.add_argument("--per-minute", type=float, default=15.0)
    ap.add_argument("--flash-model", default=FLASH_MODEL)
    ap.add_argument("--gemma-model", default=GEMMA_MODEL)
    ap.add_argument("--skip-check", action="store_true", help="skip the check against the published synthetic numbers")
    ap.add_argument("--md", help="also write the report to this file")
    args = ap.parse_args(argv)
    answers = Path(args.answers)

    if args.from_round1:
        images = from_round1()
        if not args.skip_check:
            bad = check_known(images)
            if bad:
                print("STOP: the replay does not reproduce the published numbers:\n  " + "\n  ".join(bad))
                return 3
            print("Consistency check passed: scheme 0, scheme A and the CNN + Gemma pair reproduce the published numbers exactly.\n")
    elif args.replay_only:
        images = load_answers(answers)
    elif args.photos:
        folder = Path(args.photos)
        if not folder.is_dir():
            raise SystemExit(f"photo folder not found: {folder}")
        labels = read_labels(folder, Path(args.labels) if args.labels else folder / "labels.csv")
        print(f"{len(labels)} photos, {2 * len(labels)} vision requests at <= {args.per_minute:g}/min "
              f"(about {2 * len(labels) / args.per_minute:.1f} min). Raw answers are saved to {answers}\n", flush=True)
        images = collect(folder, labels, answers, args.flash_model, args.gemma_model, args.per_minute)
    else:
        ap.error("choose one of --from-round1, --photos DIR, --replay-only")
    report = render(images)
    print(report)
    if args.md:
        Path(args.md).write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
