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


def test_unanimity_U():
    assert vc.decide("U", L(C, C, C))["status"] == "auto"
    for labels in (L(C, C, N), L(C, N, N), L(C, N, B)):       # any disagreement among three -> a human
        assert vc.decide("U", labels)["status"] == "review", labels
    assert vc.decide("U", L(C, C, None))["status"] == "auto"   # two left and they agree
    assert vc.decide("U", L(C, None, N))["status"] == "review"  # two left and they disagree
    assert vc.decide("U", L(C, None, None))["label"] == C       # one left answers alone
    assert vc.decide("U", L())["status"] == "review"


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


def test_scheme_U_and_the_silent_outvote_count():
    u = vc.score(FAKE, "U")
    # img1 ok; img2,3,4 disagree -> review; img5 everybody N (wrong, silent); img6 two left agree -> ok
    assert (u["auto_correct"], u["silent_wrong"], u["review"], u["extra_calls"]) == (2, 1, 3, 2)
    # B silently follows the two vision models against a CNN that was right (img3): the number rule (b) looks at
    assert vc.score(FAKE, "B")["silent_outvote"] == 1
    assert vc.score(FAKE, "C")["silent_outvote"] == 0 and vc.score(FAKE, "U")["silent_outvote"] == 0


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
    assert "SILENT wrong" in out and "Decision rule" in out
    assert "written BEFORE any real-photo result" in out and "AMENDED" in out and "KEEP the current mode" in out
    assert "U  Unanimity" in out and "V  Vision-primary" in out and "A' Pair: CNN + Gemma" in out
    assert "CNN right but outvoted" in out and "CNN dissent noted" in out


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


# ---------------- scheme V (vision-primary) ----------------

def test_scheme_V_rules():
    assert vc.decide("V", L(N, C, C)) ["label"] == C and vc.decide("V", L(N, C, C))["dissent"] is True
    assert vc.decide("V", L(C, C, C))["dissent"] is False
    assert vc.decide("V", L(N, C, N))["label"] == N            # the CNN breaks the tie
    assert vc.decide("V", L(B, C, N))["reason"] == "3-way split"
    assert vc.decide("V", L(N, C, None))["status"] == "review"  # one vision model failed, the other disagrees with the CNN
    assert vc.decide("V", L(C, C, None))["label"] == C
    assert vc.decide("V", L(C, None, None))["status"] == "review"  # both vision models failed: CNN alone is unverified
    assert vc.decide("V", L())["status"] == "review"


def test_scheme_V_scores_and_counts_dissent():
    s = vc.score(FAKE, "V")
    # img1 ok; img2 vision C,C vs CNN N -> auto C, dissent noted (ok); img3 vision N,N vs CNN C -> auto N (wrong, silent,
    # CNN was right and is only noted); img4 3-way split -> review; img5 all N (wrong, silent); img6 Gemma missing, flash+CNN agree
    assert (s["auto_correct"], s["silent_wrong"], s["review"]) == (3, 2, 1)
    assert s["dissent_noted"] == 2 and s["extra_calls"] == 2 and s["silent_outvote"] == 1


def test_table_has_a_V_row_and_the_dissent_column():
    out = vc.render(FAKE)
    row = next(line for line in out.splitlines() if line.startswith("| V  Vision-primary"))
    assert row.split("|")[5].strip() == "2"          # CNN dissent noted
    other = next(line for line in out.splitlines() if line.startswith("| A  Pair"))
    assert other.split("|")[5].strip() == "-"


def test_per_photo_list_marks_wrong_answers_and_the_bearing_corrosion_confusion():
    ims = [img(C, C, C, C), img(B, C, C, C), img(C, N, N, N), img(N, B, B, C, variant="real"), img(C, B, None, None)]
    ims[1]["truth"] = B   # truth bearing_wear, everybody says corrosion: the bearing/corrosion confusion
    out = vc.per_photo(ims, "V")
    assert out.count("WRONG: bearing_wear/corrosion CONFUSION") == 1
    assert "WRONG: other error" in out and "ok" in out and "HUMAN REVIEW" in out
    assert "Wrong automatic answers: 3 (of which bearing_wear/corrosion confusions: 1)." in out
    assert "| Photo | Truth | CNN | Flash-Next | Gemma | Final answer | Verdict |" in out


def test_latency_table_compares_before_and_after():
    def im(fl, gl, ferr=None):
        d = img(C, C, C, C)
        d["answers"]["flash"] = {"label": None if ferr else C, "confidence": 0.9, "latency": fl, "error": ferr}
        d["answers"]["gemma"] = {"label": C, "confidence": 0.9, "latency": gl, "error": None}
        return d
    out = vc.latency_table([im(40.0, 10.0), im(60.0, 20.0, "ValueError")], [im(8.0, 6.0), im(12.0, 10.0)])
    flash = next(line for line in out.splitlines() if line.startswith("| Qwen3.8-Flash-Next"))
    assert "50.0s / 60.0s" in flash and "10.0s / 12.0s" in flash and "40.0s -> 10.0s" in flash and "ValueError x1" in flash
    gemma = next(line for line in out.splitlines() if line.startswith("| Gemma"))
    assert "15.0s / 20.0s" in gemma and "8.0s / 10.0s" in gemma


def test_replay_only_with_per_photo_and_baseline_makes_no_api_call(tmp_path, monkeypatch, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    vc.save_answers(old, [img(C, C, C, C, variant="real")], {})
    vc.save_answers(new, [img(C, N, C, C, variant="real")], {})
    monkeypatch.setattr(dd, "run_vision_model", lambda p: pytest.fail("replay must not call the API"))
    assert vc.main(["--replay-only", "--answers", str(new), "--per-photo", "V", "--baseline", str(old)]) == 0
    out = capsys.readouterr().out
    assert "Per photo, scheme V" in out and "Latency of the vision models, before and after" in out
    assert json.loads(old.read_text())["images"][0]["answers"]["cnn"]["label"] == C   # the baseline file is only read
