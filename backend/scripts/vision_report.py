"""Rounds 1 and 2 scoring from the saved answers in data/dual_lab/round1/ (no API calls).

    PYTHONPATH=. python scripts/vision_report.py [--md OUT.md]

Round 1: accuracy of each vision model and of the CNN, clean vs hard images, latency, errors.
Round 2: each model paired with the CNN. 'single_detector' = the vision model gave no usable answer.
Key number: how many CNN mistakes the disagreement flag caught vs. how many false alarms it raised.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import tf_lab as lab

LAB = lab.REPO / "data" / "dual_lab"
HARD = {"blur", "dark", "rotate", "noise"}


def load():
    manifest = {r["file"]: r for r in csv.DictReader(open(LAB / "manifest.csv"))}
    cnn = json.load(open(LAB / "round1" / "cnn.json"))["results"]
    models = {}
    for f in sorted((LAB / "round1").glob("*.json")):
        if f.stem != "cnn":
            models[f.stem] = json.load(open(f))
    return manifest, cnn, models


def acc(pairs):
    pairs = list(pairs)
    return (f"{100 * sum(ok for ok in pairs) / len(pairs):.0f}% ({sum(pairs)}/{len(pairs)})" if pairs else "n/a")


def round1(manifest, cnn, models):
    rows = []
    entries = [("CNN (Titus)", {"results": cnn, "stopped": None})] + list(models.items())
    for name, m in entries:
        res = m["results"]
        clean = [res[f]["pred"] == manifest[f]["label"] for f in res if manifest[f]["variant"] == "clean"]
        hard = [res[f]["pred"] == manifest[f]["label"] for f in res if manifest[f]["variant"] in HARD]
        lat = [v["latency"] for v in res.values()]
        errs = Counter(v["error"] for v in res.values() if v.get("error"))
        rows.append({"model": name, "n": len(res), "clean": acc(clean), "hard": acc(hard),
                     "avg": sum(lat) / len(lat) if lat else 0, "max": max(lat) if lat else 0,
                     "errors": ", ".join(f"{k} x{v}" for k, v in errs.items()) or "none",
                     "stopped": m.get("stopped")})
    return rows


def round2(manifest, cnn, models):
    rows = []
    cnn_wrong_total = sum(cnn[f]["pred"] != manifest[f]["label"] for f in cnn)
    for name, m in models.items():
        c = Counter()
        for f, v in m["results"].items():
            truth = manifest[f]["label"]
            cnn_ok = cnn[f]["pred"] == truth
            if v.get("error") or v["pred"] not in lab.LABELS:
                c["single_detector"] += 1
                continue
            vlm_ok = v["pred"] == truth
            if v["pred"] == cnn[f]["pred"]:
                c["agree_correct" if cnn_ok else "agree_wrong"] += 1
            else:
                c["disagree_cnn_right" if cnn_ok else ("disagree_vlm_right" if vlm_ok else "disagree_both_wrong")] += 1
        seen = [f for f in m["results"] if not (m["results"][f].get("error") or m["results"][f]["pred"] not in lab.LABELS)]
        cnn_wrong_seen = sum(cnn[f]["pred"] != manifest[f]["label"] for f in seen)
        caught = c["disagree_vlm_right"] + c["disagree_both_wrong"]
        # CNN mistakes it was confident about (>= 0.70): the existing low-confidence rule cannot flag these
        conf_wrong = [f for f in seen if cnn[f]["pred"] != manifest[f]["label"] and cnn[f]["confidence"] >= 0.70]
        conf_caught = sum(1 for f in conf_wrong if m["results"][f]["pred"] != cnn[f]["pred"])
        rows.append({"model": name, **{k: c[k] for k in ("agree_correct", "agree_wrong", "disagree_cnn_right",
                                                         "disagree_vlm_right", "disagree_both_wrong", "single_detector")},
                     "cnn_mistakes_in_scope": cnn_wrong_seen, "caught": caught,
                     "conf_wrong": len(conf_wrong), "conf_caught": conf_caught, "false_alarms": c["disagree_cnn_right"], "missed": c["agree_wrong"], "images": len(m["results"])})
    return rows, cnn_wrong_total


def per_class_cnn(manifest, cnn):
    out = {}
    for f, v in cnn.items():
        out.setdefault((manifest[f]["variant"] in HARD and "hard" or "clean"), []).append(v["pred"] == manifest[f]["label"])
    return {k: acc(v) for k, v in out.items()}


def render(manifest, cnn, models) -> str:
    r1 = round1(manifest, cnn, models)
    r2, cnn_wrong = round2(manifest, cnn, models)
    md = ["### Round 1: each detector alone (same prompt, same images)", "",
          "| Detector | Images answered | Accuracy clean | Accuracy hard | Avg latency | Max latency | Errors |",
          "|---|---|---|---|---|---|---|"]
    for r in r1:
        md.append(f"| {r['model']} | {r['n']} | {r['clean']} | {r['hard']} | {r['avg']:.1f}s | {r['max']:.1f}s | "
                  f"{r['errors']}{' (STOPPED: ' + r['stopped'] + ')' if r['stopped'] else ''} |")
    md += ["", "### Round 2: each vision model paired with the CNN", "",
           "| Vision model | Agree + correct | Agree + WRONG | Disagree, CNN right | Disagree, vision right | "
           "Disagree, both wrong | single_detector | CNN mistakes caught | of them CNN-confident (>=0.70) | "
           "False alarms | Missed (both agree, wrong) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in r2:
        md.append(f"| {r['model']} | {r['agree_correct']} | {r['agree_wrong']} | {r['disagree_cnn_right']} | "
                  f"{r['disagree_vlm_right']} | {r['disagree_both_wrong']} | {r['single_detector']} | "
                  f"{r['caught']} of {r['cnn_mistakes_in_scope']} | {r['conf_caught']} of {r['conf_wrong']} | {r['false_alarms']} | {r['missed']} |")
    md += ["", f"CNN made {cnn_wrong} mistakes on all {len(cnn)} images. 'Caught' = the pair disagreed on an image the CNN "
           "got wrong (a human would have been asked); 'false alarm' = the pair disagreed although the CNN was right; "
           "'missed' = both agreed on the same wrong label."]
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    a = ap.parse_args()
    out = render(*load())
    print(out)
    if a.md:
        Path(a.md).write_text(out)
