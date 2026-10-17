import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_models", Path(__file__).parents[1] / "scripts" / "compare_models.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def row(**kw):
    base = dict(case="c", latency=2.0, http_retries=0, error=None, json_valid=True, json_retried=False,
                quote_verified=True, manual_ok=True, answered_by="m")
    return {**base, **kw}


def test_summary_and_winner_prefer_grounding_over_speed():
    good_slow = cm.summarize("slow", [row(latency=9), row(latency=9)])
    bad_fast = cm.summarize("fast", [row(latency=1, quote_verified=False), row(latency=1)])
    assert good_slow["quote"] == 100 and bad_fast["quote"] == 50
    assert cm.pick_winner([bad_fast, good_slow])["model"] == "slow"
    assert "| slow | 2 | 100% | 100% | 100% | 9.0s / 9.0s | 0 | 0 |" in cm.table([good_slow])


def test_retries_count_http_and_json():
    s = cm.summarize("m", [row(http_retries=2, json_retried=True), row()])
    assert s["retries"] == 3
