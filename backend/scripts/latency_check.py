"""How slow will the demo be? N SEQUENTIAL single calls per vision model on resized real photos (one at a time, no parallelism).

    PYTHONPATH=. python scripts/latency_check.py --photos ../data/real_photos --calls 10

Uses the same function the app uses (app/dual_detector.py: resize to 1024 px, final prompt, parse + one retry). Photos are
spread evenly over the folder. Each call is timed on its own; nothing else should be running against the endpoint while this
measures. Never prints the key.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import dual_detector as dd  # noqa: E402

MODELS = [("Qwen3.8-Flash-Next", "run_vision_model"), ("Gemma-4-31B-IT", "run_vision_model_2")]


def pick(folder: Path, n: int) -> list:
    names = sorted(r["filename"].strip() for r in csv.DictReader(open(folder / "labels.csv")) if (folder / r["filename"].strip()).is_file())
    step = max(1, len(names) // n)
    return [names[i * step] for i in range(min(n, len(names)))]


def measure(calls: int, folder: Path, ask) -> dict:
    """Run `calls` sequential calls; `ask(model_fn_name, path)` returns the result dict or raises. Returns timings."""
    out = {}
    for model, fn in MODELS:
        rows = []
        for name in pick(folder, calls):
            start = time.time()
            try:
                label = ask(fn, str(folder / name))["label"]
                err = None
            except Exception as exc:
                label, err = None, getattr(exc, "kind", None) or type(exc).__name__
            rows.append({"photo": name, "seconds": time.time() - start, "label": label, "error": err})
            print(f"  {model:20} {name}: {rows[-1]['seconds']:6.1f}s {label or 'ERROR ' + str(err)}", flush=True)
        out[model] = rows
    return out


def summary(result: dict) -> str:
    lines = ["| Model | Calls | Average | Slowest | Fastest | Errors |", "|---|---|---|---|---|---|"]
    for model, rows in result.items():
        s = [r["seconds"] for r in rows]
        lines.append(f"| {model} | {len(rows)} | {sum(s) / len(s):.1f}s | {max(s):.1f}s | {min(s):.1f}s | "
                     f"{sum(1 for r in rows if r['error'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default=str(Path(__file__).resolve().parents[2] / "data" / "real_photos"))
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args(argv)
    import core  # noqa: F401  (loads backend/.env)
    result = measure(args.calls, Path(args.photos), lambda fn, path: getattr(dd, fn)(path))
    print("\n" + summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
