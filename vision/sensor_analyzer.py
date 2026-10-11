"""Rule-based analysis of sensor reports (CSV, JSON or "key: value" text).

Thresholds follow common practice (ISO 10816-style vibration limits, typical bearing temperature limits)
and are plain numbers in THRESHOLDS so a maintenance engineer can tune them. A reading is compared with
its alarm level; the strongest family decides the defect.
"""
import csv
import io
import json
import os
import re
from typing import Dict, List

from .labels import to_contract_label

# metric -> (defect class, warn level, alarm level)
THRESHOLDS = {
    "vibration_mm_s": ("bearing_wear", 4.5, 7.1),
    "bearing_temp_c": ("bearing_wear", 80.0, 95.0),
    "temperature_c": ("bearing_wear", 80.0, 95.0),
    "wall_loss_pct": ("corrosion", 10.0, 20.0),
    "crack_width_mm": ("structural_crack", 0.3, 1.0),
}
_ALIASES = {"vibration": "vibration_mm_s", "vib_mm_s": "vibration_mm_s", "temp_c": "temperature_c",
            "temperature": "temperature_c", "bearing_temperature_c": "bearing_temp_c"}


def _num(v):
    try:
        return float(str(v).strip().rstrip("%"))
    except ValueError:
        return None


def parse_readings(path: str) -> Dict[str, List[float]]:
    text = open(path, encoding="utf-8", errors="replace").read()
    ext = os.path.splitext(path)[1].lower()
    rows: List[dict] = []
    if ext == ".json":
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    elif ext == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        one = {}
        for line in text.splitlines():
            m = re.match(r"\s*([A-Za-z_ ]+?)\s*[:=]\s*([-\d.]+)", line)
            if m:
                one[m.group(1).strip().lower().replace(" ", "_")] = m.group(2)
        rows = [one]
    out: Dict[str, List[float]] = {}
    for row in rows:
        norm = {_ALIASES.get(k.strip().lower(), k.strip().lower()): v for k, v in row.items() if k}
        if "wall_loss_pct" not in norm and "wall_thickness_mm" in norm and "nominal_wall_mm" in norm:
            w, n = _num(norm["wall_thickness_mm"]), _num(norm["nominal_wall_mm"])
            if w is not None and n:
                norm["wall_loss_pct"] = str(max(0.0, (n - w) / n * 100))
        for k, v in norm.items():
            n = _num(v)
            if n is not None:
                out.setdefault(k, []).append(n)
    return out


def analyze(path: str) -> dict:
    readings = parse_readings(path)
    findings = []  # (ratio, class, text)
    for metric, (cls, warn, alarm) in THRESHOLDS.items():
        if metric in readings:
            peak = max(readings[metric])
            findings.append((peak / alarm, cls, metric, peak, warn, alarm))
    if not findings:
        return {"input_type": "sensor", "defect": {
            "type": "normal", "confidence": 0.5,
            "explanation": "No recognised sensor channels (expected e.g. vibration_mm_s, bearing_temp_c, wall_loss_pct, crack_width_mm)."}}
    findings.sort(reverse=True)
    top = findings[0]
    ratio, cls, metric, peak, warn, alarm = top
    if peak < warn:
        worst = max(f[0] for f in findings)
        return {"input_type": "sensor", "defect": {
            "type": "normal", "confidence": round(0.95 - 0.25 * worst, 2),
            "explanation": "All sensor readings are below their warning levels: "
                           + ", ".join(f"{m}={p:g} (warn {w:g})" for _, _, m, p, w, _ in findings) + "."}}
    agree = [f for f in findings if f[1] == cls and f[3] >= f[4]]
    conf = min(0.97, 0.6 + 0.25 * min(ratio, 1.4) + 0.04 * (len(agree) - 1))
    parts = [f"{m}={p:g} ({'above alarm' if p >= a else 'above warning'} level {a if p >= a else w:g})" for _, _, m, p, w, a in agree]
    return {"input_type": "sensor", "defect": {
        "type": to_contract_label(cls), "confidence": round(conf, 2),
        "explanation": f"Sensor report indicates {to_contract_label(cls)}: " + "; ".join(parts) + "."}}
