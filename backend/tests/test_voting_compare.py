"""Voting rules, scoring, the synthetic replay check, and the Phase 2 collect / replay-only flow (fake detectors)."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app import dual_detector as dd

spec = importlib.util.spec_from_file_location("voting_compare", Path(__file__).parents[1] / "scripts" / "voting_compare.py")
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)

N, B, S, C = "normal", "bearing_wear", "structural_crack", "corrosion"


def L(cnn=None, flash=None, gemma=None):
    return {"cnn": cnn, "flash": flash, "gemma": gemma}


# ---------------- the rules ----------------

@pytest.mark.parametrize("labels,status,label", [
    (L(C, C), "auto", C),                 # pair agrees
    (L(C, N), "review", None),            # pair disagrees
    (L(None, N), "auto", N),              # CNN missing: the vision model answers alone
    (L(C, None), "auto", C),              # vision model missing: CNN answers alone (single_detector)
    (L(None, None), "review", None),      # nobody answered
])
def test_pair_A(labels, status, label):
    r = vc.decide("A", labels)
    assert (r["status"], r["label"]) == (status, label)


def test_pair_with_gemma_ignores_flash():
    assert vc.decide("A_gemma", L(C, N, C))["label"] == C
    assert vc.decide("A_gemma", L(C, C, N))["status"] == "review"


@pytest.mark.parametrize("scheme", ["B", "C"])
def test_three_voters_agree(scheme):
    r = vc.decide(scheme, L(C, C, C))
    assert (r["status"], r["label"], r["reason"]) == ("auto", C, "unanimous")


@pytest.mark.parametrize("scheme", ["B", "C"])
def test_two_of_three_wins_when_cnn_is_in_the_majority(scheme):
    for labels in (L(C, C, N), L(C, N, C)):  # CNN + one vision model
        r = vc.decide(scheme, labels)
        assert (r["status"], r["label"]) == ("auto", C)


@pytest.mark.parametrize("scheme", ["B", "C"])
def test_three_way_split_goes_to_review(scheme):
    r = vc.decide(scheme, L(C, N, B))
    assert (r["status"], r["reason"], r["label"]) == ("review", "3-way split", None)


def test_cnn_outvoted_B_decides_silently_but_C_asks_a_human():
    labels = L(C, N, N)  # both vision models agree with each other against the CNN
    b, c = vc.decide("B", labels), vc.decide("C", labels)
    assert (b["status"], b["label"]) == ("auto", N)            # B: 2-of-3 wins, silently
    assert (c["status"], c["reason"]) == ("review", "CNN outvoted by both vision models")
    assert c["ref"] == C                                       # a review is judged against the CNN's own label


@pytest.mark.parametrize("scheme", ["B", "C"])
def test_a_missing_model_agreement_wins_disagreement_goes_to_review(scheme):
    assert vc.decide(scheme, L(C, C, None))["status"] == "auto"
    assert vc.decide(scheme, L(C, None, C))["status"] == "auto"
    assert vc.decide(scheme, L(None, N, N))["status"] == "auto"   # CNN missing, vision models agree
    assert vc.decide(scheme, L(C, N, None))["status"] == "review"
    assert vc.decide(scheme, L(None, N, B))["status"] == "review"


@pytest.mark.parametrize("scheme", ["B", "C"])
def test_one_or_zero_voters(scheme):
    assert vc.decide(scheme, L(C, None, None)) ["status"] == "auto"
    assert vc.decide(scheme, L())["status"] == "review"


def test_cnn_alone():
    assert vc.decide("0", L(C, N, N))["label"] == C
    assert vc.decide("0", L())["status"] == "review"


# ---------------- scoring / replay with fake data ----------------

def img(truth, cnn, flash, gemma, variant="clean", lat=(0.0, 4.0, 8.0)):
    def a(label, latency):
        return None if label is None else {"label": label, "confidence": 0.9, "latency": latency, "error": None}
    return {"file": f"{truth}-{cnn}-{flash}-{gemma}.png", "truth": truth, "variant": variant,
            "answers": {"cnn": a(cnn, lat[0]), "flash": a(flash, lat[1]), "gemma": a(gemma, lat[2])}}


FAKE = [
    img(C, C, C, C),          # all right
    img(C, N, C, C),          # CNN wrong, both vision right -> CNN outvoted by the right answer
    img(C, C, N, N),          # CNN right but outvoted by both vision models (wrong)
    img(C, C, N, B),          # 3-way split
    img(C, N, N, N),          # everybody wrong the same way: silent wrong everywhere that decides automatically
    img(C, C, C, None, variant="blur"),  # Gemma missing
]


def test_scores_for_each_scheme():
    s = {k: vc.score(FAKE, k) for k, *_ in vc.SCHEMES}
    assert (s["0"]["silent_wrong"], s["0"]["auto_correct"], s["0"]["review"]) == (2, 4, 0)
    # A: image2 (cnn N vs flash C) review good flag, image3 (C vs N) review false alarm, image4 (C vs N) review false alarm
    assert (s["A"]["review"], s["A"]["false_alarm"], s["A"]["good_flag"], s["A"]["silent_wrong"]) == (3, 2, 1, 1)
    # B: img2 -> C correct; img3 -> N wrong (silent); img4 split review; img5 N wrong (silent); img6 agree
    assert (s["B"]["auto_correct"], s["B"]["silent_wrong"], s["B"]["review"]) == (3, 2, 1)
    # C: img1 auto ok; img2 both vision models (right) outvote the wrong CNN -> review (good flag); img3 outvoted CNN
    # was right -> review (false alarm); img4 split -> review; img5 everybody N -> silent wrong; img6 agree -> ok
    assert (s["C"]["auto_correct"], s["C"]["silent_wrong"], s["C"]["review"]) == (2, 1, 3)
    assert (s["C"]["false_alarm"], s["C"]["good_flag"]) == (2, 1)


def test_scheme_C_review_reasons_and_false_alarms():
    c = vc.score([img(C, C, N, N)], "C")   # CNN right, outvoted
    assert (c["review"], c["false_alarm"], c["good_flag"], c["silent_wrong"]) == (1, 1, 0, 0)
    c = vc.score([img(C, N, C, C)], "C")   # CNN wrong, both vision right: still a review, and a good flag
    assert (c["review"], c["false_alarm"], c["good_flag"], c["silent_wrong"]) == (1, 0, 1, 0)


def test_extra_calls_and_latency_wait_for_the_slowest_detector_used():
    one = [img(C, C, C, C, lat=(0.0, 4.0, 8.0))]
    assert vc.score(one, "0")["lat_max"] == 0.0 and vc.score(one, "0")["extra_calls"] == 0
    assert vc.score(one, "A")["lat_avg"] == 4.0 and vc.score(one, "A")["extra_calls"] == 1
    assert vc.score(one, "A_gemma")["lat_avg"] == 8.0
    assert vc.score(one, "B")["lat_max"] == 8.0 and vc.score(one, "C")["extra_calls"] == 2


def test_dataset_facts():
    f = vc.dataset_facts(FAKE)
    assert (f["three_way_splits"], f["cnn_right_outvoted"], f["gemma_missing"]) == (1, 1, 1)


def test_groups_split_clean_and_hard_only_when_both_exist():
    names = [g for g, _ in vc.groups(FAKE)]
    assert names[0] == "clean images" and names[-1] == "all images"
    assert [g for g, _ in vc.groups([img(C, C, C, C, variant="real")])] == ["all images"]


def test_consistency_check_reports_mismatches_on_other_data():
    bad = vc.check_known(FAKE)
    assert bad and any("images" in b for b in bad)


def test_render_contains_the_decision_rule_and_a_silent_wrong_column():
    out = vc.render(FAKE)
    assert "SILENT wrong" in out and "Decision rule (fixed BEFORE the results were looked at)" in out
    assert "If it is unclear, keep A" in out


# ---------------- Phase 2: collect, save, replay ----------------

@pytest.fixture
def photos(tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    folder.mkdir()
    truth = {"p1.png": C, "p2.png": N, "p3.png": B}
    for n in truth:
        Image.new("RGB", (8, 8), (9, 9, 9)).save(folder / n)
    (folder / "labels.csv").write_text("filename,label\n" + "".join(f"{n},{t}\n" for n, t in truth.items()))
    sleeps = []
    monkeypatch.setattr(vc, "_sleep", sleeps.append)
    state = {"calls": [], "rate_limit_once": set()}
    monkeypatch.setattr(dd, "run_cnn", lambda p: {"label": truth[Path(p).name], "confidence": 0.9})

    def vlm(p):
        model, name = __import__("os").environ["DUAL_VISION_MODEL"], Path(p).name
        state["calls"].append((model, name))
        if (model, name) in state["rate_limit_once"]:
            state["rate_limit_once"].discard((model, name))
            raise httpx.HTTPStatusError("429", request=httpx.Request("POST", "http://x"), response=httpx.Response(429))
        return {"label": N if model == "Gemma-4-31B-IT" and name == "p1.png" else truth[name], "confidence": 0.8, "model": model}

    monkeypatch.setattr(dd, "run_vision_model", vlm)
    return folder, state, sleeps, tmp_path / "answers.json"


def test_collect_calls_each_model_once_per_photo_saves_and_replays(photos, capsys):
    folder, state, sleeps, answers = photos
    assert vc.main(["--photos", str(folder), "--answers", str(answers), "--per-minute", "30"]) == 0
    assert sorted(state["calls"]) == sorted((m, n) for m in ("Qwen3.8-Flash-Next", "Gemma-4-31B-IT")
                                            for n in ("p1.png", "p2.png", "p3.png"))
    saved = json.loads(answers.read_text())
    assert {i["file"] for i in saved["images"]} == {"p1.png", "p2.png", "p3.png"}
    assert saved["images"][0]["answers"]["gemma"]["label"] in vc.LABELS     # every raw answer is stored
    assert set(sleeps) == {2.0}                                              # paced: 60 / 30 per minute
    out = capsys.readouterr().out
    assert "All images (3 images)" in out and "Decision rule" in out


def test_replay_only_makes_no_api_call(photos, monkeypatch, capsys):
    folder, state, _, answers = photos
    vc.main(["--photos", str(folder), "--answers", str(answers)])
    capsys.readouterr()
    monkeypatch.setattr(dd, "run_vision_model", lambda p: pytest.fail("replay must not call the API"))
    monkeypatch.setattr(dd, "run_cnn", lambda p: pytest.fail("replay must not run the CNN"))
    assert vc.main(["--replay-only", "--answers", str(answers)]) == 0
    assert "SILENT wrong" in capsys.readouterr().out


def test_replay_only_without_saved_answers_explains(tmp_path):
    with pytest.raises(SystemExit, match="Run once with --photos first"):
        vc.main(["--replay-only", "--answers", str(tmp_path / "none.json")])


def test_a_rate_limited_answer_is_retried_once(photos):
    folder, state, sleeps, answers = photos
    state["rate_limit_once"].add(("Qwen3.8-Flash-Next", "p2.png"))
    vc.main(["--photos", str(folder), "--answers", str(answers)])
    assert state["calls"].count(("Qwen3.8-Flash-Next", "p2.png")) == 2
    saved = {i["file"]: i for i in json.loads(answers.read_text())["images"]}
    assert saved["p2.png"]["answers"]["flash"]["error"] is None


def test_a_second_rate_limit_is_kept_as_an_error_not_looped(photos, monkeypatch):
    folder, state, _, answers = photos

    def always_429(p):
        state["calls"].append(Path(p).name)
        raise httpx.HTTPStatusError("429", request=httpx.Request("POST", "http://x"), response=httpx.Response(429))

    monkeypatch.setattr(dd, "run_vision_model", always_429)
    vc.main(["--photos", str(folder), "--answers", str(answers)])
    assert len(state["calls"]) == 3 * 2 * 2          # 3 photos x 2 models x (first try + ONE retry)
    saved = json.loads(answers.read_text())["images"]
    assert all(i["answers"]["flash"]["error"] == "rate_limit" for i in saved)


def test_resume_skips_photos_that_are_already_answered(photos):
    folder, state, _, answers = photos
    vc.main(["--photos", str(folder), "--answers", str(answers)])
    n = len(state["calls"])
    vc.main(["--photos", str(folder), "--answers", str(answers)])
    assert len(state["calls"]) == n


def test_bad_labels_file_is_explained(photos):
    folder, *_ = photos
    (folder / "labels.csv").write_text("filename,label\np1.png,rusty\nmissing.png,normal\n")
    with pytest.raises(SystemExit) as e:
        vc.main(["--photos", str(folder)])
    assert "unknown label 'rusty'" in str(e.value) and "photo not found: missing.png" in str(e.value)


def test_photos_and_saved_answers_are_git_ignored():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".gitignore").read_text()
    assert "data/real_photos/" in text and "data/voting_answers" in text
    assert vc.DEFAULT_ANSWERS == root / "data" / "voting_answers.json"
