# Voting experiment: real photos, final decision, and the earlier synthetic replay

This page records (1) the run on 40 real photos and the final decision, then (2) the earlier replay on synthetic images,
which was only a sanity check.

## Final decision

**The demo uses `DUAL_MODE=vision_primary`: the two vision models (Qwen3.8-Flash-Next and Gemma-4-31B-IT) decide, and the
CNN is only advisory.** `DUAL_DETECTOR` itself defaults to off, but whenever it is on the mode defaults to `vision_primary` (the other modes stay selectable with `DUAL_MODE`); the demo values are in `backend/.env.example`
(`DUAL_DETECTOR=on`, reasoning effort `none`). Rules of `vision_primary`:

- The two vision models agree: that label is the answer, whatever the CNN says. If the CNN disagrees, it is shown as a low-trust
  **CNN dissent note** (label and confidence) in the result and the explanation. It does not trigger a review.
- They disagree: if the CNN matches one of them **and its own confidence is at least 0.70** (the CNN's "a person should verify visually"
  cut-off, `LOW_CONFIDENCE`, defined once in `vision/thresholds.py`), that label wins. If the CNN matches one of them but is below 0.70, it is
  not trusted to break the tie: the diagnosis is UNVERIFIED and a person decides, and the explanation shows the CNN's label and
  confidence and says it was not confident enough. If it matches neither (a three-way split) a person decides.
- One vision model fails (timeout, 5xx, unusable answer): the other vision model plus the CNN decide, with a warning; if
  they disagree a person decides. Both fail: the CNN answers alone and a person verifies.
- Every detector's label and confidence is shown separately, with vote counts such as "2 of 3 say corrosion". Confidences are never averaged.

**Why:** the CNN was trained on generated drawings and was right on only 14 of 40 real photos, so letting it veto the vision models
(as `pair`, `unanimous` and `cnn_anchored` do) sends more than half of all photos to a person (57% to 62%). `vote` and
`vision_primary` answer 36 of 40 automatically and ask a person once. In the new run `vision_primary` produced 3 silently wrong
answers, none of them a bearing_wear/corrosion mix-up; with the low-confidence tie rule described below it produces 1 silently wrong answer
and sends 3 of 40 photos to a person.

**Said plainly:** before the tie rule, rule (a) (fewest silent wrong answers) put the unanimity scheme U first with 1 silent wrong
answer, ahead of `vision_primary` with 3, but U asks a person about 25 of 40 photos (62%). With the tie rule `vision_primary` also has 1
silent wrong answer and asks a person for 3 of 40 (8%). A gap of 2 photos is inside the noise this page warns about, while the gap in
human workload is large, so the choice rests on the practical difference and on the decision that the vision models decide. It is a
judgment call, not something rule (a) alone produces, and the tie rule was added after looking at these same photos (see below).

## Real photos: 40 photos, old run against new run

The 40 photos are **Google images with labels made by hand**, so a few of the labels could be arguable, the models may have seen
some of these pictures before, and **small differences are noise** (one photo is 2.5 percentage points).

- **Old run** = `data/voting_answers.json`: the first prompt, full-size photos. **Replayed offline; no API calls were made for it.**
- **New run** = `data/voting_answers_v2.json`: the new prompt with explicit class definitions, photos shrunk to 1024 px (JPEG q85),
  parse retry. Run on the same 40 photos at 15 requests/min, 4 photos in parallel.

**Caution about the improvement.** The new prompt was written after seeing which photos failed (rusty bearings called corrosion) and then
tested on those same 40 photos. Part of the improvement is therefore tuning to this set. Treat it as encouraging, and confirm it on
photos that were not used to write the prompt. The temperature is 0.2, so a repeat run will differ a little. Also, the first replay
used the OLD prompt and OLD full-size photos, so old and new differ in the prompt, the image size and chance, not in one thing.

### Old run (old prompt, full-size photos)

| Scheme | Auto answers correct | SILENT wrong | CNN right but outvoted | CNN dissent noted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 14 | **26** | 0 | - | 0 (0%) | 0 | 0 | 0 | 0.3s / 10.7s |
| A  Pair: CNN + Flash-Next | 12 | **8** | 0 | - | 20 (50%) | 2 | 18 | 1 | 36.7s / 87.8s |
| A' Pair: CNN + Gemma | 12 | **7** | 0 | - | 21 (52%) | 2 | 19 | 1 | 11.3s / 122.2s |
| B  3-way vote (2-of-3) | 27 | **9** | 1 | - | 4 (10%) | 2 | 2 | 2 | 39.4s / 122.2s |
| C  3-way vote, CNN outvoted -> review | 11 | **7** | 0 | - | 22 (55%) | 3 | 19 | 2 | 39.4s / 122.2s |
| U  Unanimity (all must agree) | 11 | **7** | 0 | - | 22 (55%) | 3 | 19 | 2 | 39.4s / 122.2s |
| V  Vision-primary (CNN advisory) | 27 | **9** | 1 | 18 | 4 (10%) | 2 | 2 | 2 | 39.4s / 122.2s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 1; CNN right but outvoted by both vision models: 1; images with a missing detector answer: CNN 0, Flash-Next 2, Gemma 1.

### New run (new prompt, resized photos)

| Scheme | Auto answers correct | SILENT wrong | CNN right but outvoted | CNN dissent noted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 14 | **26** | 0 | - | 0 (0%) | 0 | 0 | 0 | 0.1s / 0.8s |
| A  Pair: CNN + Flash-Next | 14 | **3** | 0 | - | 23 (57%) | 0 | 23 | 1 | 36.9s / 156.5s |
| A' Pair: CNN + Gemma | 14 | **2** | 0 | - | 24 (60%) | 0 | 24 | 1 | 13.1s / 72.4s |
| B  3-way vote (2-of-3) | 36 | **3** | 0 | - | 1 (2%) | 0 | 1 | 2 | 39.6s / 156.5s |
| C  3-way vote, CNN outvoted -> review | 14 | **3** | 0 | - | 23 (57%) | 0 | 23 | 2 | 39.6s / 156.5s |
| U  Unanimity (all must agree) | 14 | **1** | 0 | - | 25 (62%) | 0 | 25 | 2 | 39.6s / 156.5s |
| V  Vision-primary (CNN advisory) | 36 | **3** | 0 | 22 | 1 (2%) | 0 | 1 | 2 | 39.6s / 156.5s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 0; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 1, Gemma 0.

How to read: **silent wrong** = answered automatically and wrong, nobody asked. **CNN dissent noted** = the CNN disagreed with the
final answer and was only noted (scheme V). **Human review** = sent to a person. Latency is the slowest detector the scheme waits for.

### Low-confidence tie rule (added after the new run)

**What changed.** In `vision_primary`, when Flash-Next and Gemma disagree, the CNN breaks the tie only if its confidence is at least the
CNN's own cut-off (0.70, `LOW_CONFIDENCE`, read from `vision/thresholds.py`; the number is not repeated in the rule). Below it, the case is
UNVERIFIED and goes to a person. When the two vision models agree nothing changes.

Same 40 saved answers (`data/voting_answers_v2.json`), replayed offline with no API calls:

| Scheme V | Auto answers correct | Silent wrong | Human review | CNN dissent noted |
|---|---|---|---|---|
| Before the tie rule | 36 | 3 | 1 (2%) | 22 |
| After the tie rule | 36 | **1** | 3 (8%) | 22 |

The two photos that changed are `img_11` and `img_34` (both truth: normal). The vision models disagreed, and the CNN, at confidence 0.43 and
0.37, picked the wrong side; they now go to a person. The one silent mistake left is `img_39`, where all three detectors agree on corrosion.
No other scheme changes.

**Said plainly: this rule rests on 2 photos in the set that was used to tune the prompt.** What justifies it is not those 2 photos but the
CNN's own 0.70 cut-off: the CNN already tells people to double-check anything below it, and the CNN was right on only 14 of 40
real photos. The idea for the rule did come from looking at these failures, so the improvement here is optimistic. The held-out test on
fresh photos (below) is the fair check, and no rule or threshold was changed after seeing it.

### Scheme V, every wrong or flagged photo

(Listed as measured by the new run, before the tie rule.)

Old run (9 wrong, **6 of them bearing_wear/corrosion mix-ups**, 4 sent to a person):

| Photo | Truth | CNN | Flash-Next | Gemma | Final answer | Verdict |
|---|---|---|---|---|---|---|
| img_02.jpg | bearing_wear | corrosion (0.83) | corrosion (0.95) | corrosion (0.92) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_11.jpg | normal | corrosion (0.43) | corrosion (0.95) | corrosion (0.92) | corrosion | WRONG: other error |
| img_12.jpg | normal | bearing_wear (0.46) | -- (ValueError) | normal (0.95) | HUMAN REVIEW | review: vision model and CNN disagree (the other vision model failed) |
| img_17.jpg | structural_crack | bearing_wear (0.62) | corrosion (0.95) | corrosion (0.95) | corrosion | WRONG: other error |
| img_21.jpg | bearing_wear | bearing_wear (0.63) | corrosion (0.95) | corrosion (0.95) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_22.jpg | bearing_wear | corrosion (1.00) | corrosion (0.99) | corrosion (0.98) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_24.jpg | bearing_wear | bearing_wear (0.55) | -- (ValueError) | corrosion (0.85) | HUMAN REVIEW | review: vision model and CNN disagree (the other vision model failed) |
| img_25.jpg | bearing_wear | structural_crack (0.75) | bearing_wear (0.95) | corrosion (0.92) | HUMAN REVIEW | review: 3-way split |
| img_26.jpg | bearing_wear | corrosion (0.74) | corrosion (0.98) | corrosion (0.98) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_27.jpg | bearing_wear | corrosion (0.94) | corrosion (0.98) | corrosion (0.98) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_28.jpg | bearing_wear | bearing_wear (0.70) | corrosion (0.98) | -- (timeout) | HUMAN REVIEW | review: vision model and CNN disagree (the other vision model failed) |
| img_30.jpg | bearing_wear | corrosion (0.89) | corrosion (0.95) | corrosion (0.92) | corrosion | WRONG: bearing_wear/corrosion CONFUSION |
| img_39.jpg | structural_crack | corrosion (0.98) | corrosion (0.99) | corrosion (0.98) | corrosion | WRONG: other error |

New run (3 wrong, **0 bearing_wear/corrosion mix-ups**, 1 sent to a person):

| Photo | Truth | CNN | Flash-Next | Gemma | Final answer | Verdict |
|---|---|---|---|---|---|---|
| img_11.jpg | normal | corrosion (0.43) | bearing_wear (0.78) | corrosion (0.95) | corrosion | WRONG: other error |
| img_12.jpg | normal | bearing_wear (0.46) | -- (parse_error) | normal (0.92) | HUMAN REVIEW | review: vision model and CNN disagree (the other vision model failed) |
| img_34.jpg | normal | bearing_wear (0.37) | bearing_wear (0.95) | normal (0.95) | bearing_wear | WRONG: other error |
| img_39.jpg | structural_crack | corrosion (0.98) | corrosion (0.99) | corrosion (0.98) | corrosion | WRONG: other error |

- `img_11` and `img_34`: the vision models disagreed and the CNN broke the tie with a very low confidence (0.43 and 0.37) in the
  wrong direction (2 of the 3 silent mistakes). The low-confidence tie rule above now sends these to a person.
- `img_39`: all three detectors agree on corrosion for a photo labelled structural_crack; the label itself may be arguable.
- `img_12`: Flash-Next returned an unusable answer twice (in both runs), so Gemma plus the CNN were compared and disagreed: a person decides.
- The complete 40-photo list for any scheme: `python scripts/voting_compare.py --replay-only --answers ../data/voting_answers_v2.json --per-photo V`.

### Vision-model latency, before and after resizing

| Vision model | Before: avg / slowest | After: avg / slowest | Avg of successful calls before -> after | Errors before | Errors after |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 36.7s / 87.8s | 36.9s / 156.5s | 34.6s -> 33.9s | ValueError x2 | parse_error x1 |
| Gemma-4-31B-IT | 11.3s / 122.2s | 13.1s / 72.4s | 8.5s -> 13.1s | timeout x1 | none |

**Resizing did not make the models faster.** Flash-Next stays around 35 s per photo, so the slowness is on the endpoint, not the image
size (its slowest call, 156 s, is `img_12`, which was asked twice). Gemma's slowest call and its timeout improved, but its average got
a little longer. The new run sent 4 photos at once, which can itself add queueing on the shared endpoint, so the two sets of
latencies are not a clean before/after.

### Held-out validation: 15 fresh photos

**Setup.** `data/fresh_photos` (15 photos: 4 bearing_wear, 4 corrosion, 2 structural_crack, 5 normal; hand labels, same kind of Google images). The FINAL
pipeline was run on them once, **and no prompt, rule or threshold was changed after seeing the result**: new prompt, 1024 px resize, `vision_primary` with the
low-confidence tie rule. Answers are in `data/voting_answers_fresh.json` (git-ignored).

| Scheme | Auto answers correct | SILENT wrong | CNN right but outvoted | CNN dissent noted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 7 | **8** | 0 | - | 0 (0%) | 0 | 0 | 0 | 0.1s / 0.6s |
| A  Pair: CNN + Flash-Next | 7 | **0** | 0 | - | 8 (53%) | 0 | 8 | 1 | 32.1s / 74.2s |
| A' Pair: CNN + Gemma | 7 | **0** | 0 | - | 8 (53%) | 0 | 8 | 1 | 8.2s / 44.1s |
| B  3-way vote (2-of-3) | 13 | **1** | 0 | - | 1 (7%) | 0 | 1 | 2 | 32.2s / 74.2s |
| C  3-way vote, CNN outvoted -> review | 7 | **0** | 0 | - | 8 (53%) | 0 | 8 | 2 | 32.2s / 74.2s |
| U  Unanimity (all must agree) | 7 | **0** | 0 | - | 8 (53%) | 0 | 8 | 2 | 32.2s / 74.2s |
| V  Vision-primary (CNN advisory) | 13 | **1** | 0 | 7 | 1 (7%) | 0 | 1 | 2 | 32.2s / 74.2s |

**Scheme V, every photo** (wrong answers marked):

| Photo | Truth | CNN | Flash-Next | Gemma | Final answer | Verdict |
|---|---|---|---|---|---|---|
| fresh_01.jpg | bearing_wear | corrosion (0.47) | bearing_wear (0.85) | bearing_wear (0.92) | bearing_wear | ok (CNN dissent noted) |
| fresh_02.jpg | bearing_wear | bearing_wear (0.74) | bearing_wear (0.95) | bearing_wear (0.98) | bearing_wear | ok |
| fresh_03.jpg | bearing_wear | structural_crack (0.93) | bearing_wear (0.95) | bearing_wear (0.98) | bearing_wear | ok (CNN dissent noted) |
| fresh_04.jpg | bearing_wear | bearing_wear (0.59) | bearing_wear (0.85) | bearing_wear (0.92) | bearing_wear | ok |
| fresh_05.jpg | corrosion | corrosion (0.64) | corrosion (0.96) | corrosion (1.00) | corrosion | ok |
| fresh_06.jpg | corrosion | corrosion (1.00) | corrosion (0.98) | corrosion (0.99) | corrosion | ok |
| fresh_07.jpg | corrosion | corrosion (0.86) | corrosion (0.95) | corrosion (0.98) | corrosion | ok |
| fresh_08.jpg | corrosion | corrosion (1.00) | corrosion (0.99) | corrosion (1.00) | corrosion | ok |
| fresh_09.jpg | structural_crack | structural_crack (0.71) | structural_crack (0.95) | structural_crack (0.95) | structural_crack | ok |
| fresh_10.jpg | structural_crack | bearing_wear (0.62) | structural_crack (0.95) | structural_crack (0.92) | structural_crack | ok (CNN dissent noted) |
| fresh_13.jpg | normal | corrosion (0.66) | normal (0.95) | normal (0.95) | normal | ok (CNN dissent noted) |
| fresh_14.jpg | normal | corrosion (0.90) | bearing_wear (0.85) | normal (0.95) | HUMAN REVIEW | review: 3-way split |
| fresh_15.jpg | normal | bearing_wear (0.78) | normal (0.99) | normal (0.98) | normal | ok (CNN dissent noted) |
| fresh_16.jpg | normal | bearing_wear (0.46) | normal (0.98) | normal (0.98) | normal | ok (CNN dissent noted) |
| fresh_17.jpg | normal | corrosion (0.94) | bearing_wear (0.90) | bearing_wear (0.95) | bearing_wear | WRONG: other error |

Wrong automatic answers: 1 (of which bearing_wear/corrosion confusions: 0).

**Did the earlier 36/40 hold up? Roughly, yes, but 15 photos cannot say more.**

| Scheme V | Photos | Answered automatically and correctly | Silently wrong | Sent to a person |
|---|---|---|---|---|
| Tuning set (40 photos, after the tie rule) | 40 | 36 (90%) | 1 (2.5%) | 3 (7.5%) |
| Fresh photos (15 photos) | 15 | 13 (87%) | 1 (6.7%) | 1 (6.7%) |

- One fresh photo is 6.7 percentage points. A 95% range for 13 out of 15 is roughly 62% to 96%, which includes the tuning-set result, so this neither
  confirms nor contradicts it. The silent-wrong rate looks higher (6.7% against 2.5%), but that is one photo in each set.
- **The CNN alone got 7 of 15 right (47%)** on the fresh photos (14 of 40, 35%, on the tuning set): it is still weak on real photos. Every scheme that uses the vision
  models beats it; V and B answered 13 of 15 automatically.
- **The one silent mistake, `fresh_17`, is a normal part that both vision models called bearing_wear.** The one photo sent to a person, `fresh_14`, is a normal part with
  a three-way split. Together with `img_11`, `img_12` and `img_34` earlier, the hard cases are mostly photos labelled **normal** that look like they have wear:
  the vision models lean towards finding a defect.
- **The low-confidence tie rule was not exercised** on the fresh photos: no photo had the two vision models disagree while the CNN matched one of them. So this
  validation says nothing about that rule; it rests on the 2 tuning-set photos described above.
- Pair / unanimity / `cnn_anchored` again had 0 silent mistakes and again sent about half (53%) to a person; the CNN dissent note appeared on 7 of 15 photos.

### How slow will the demo be? (10 single calls per model, one at a time)

Measured with `backend/scripts/latency_check.py`: resized photos spread over `data/real_photos`, same function as the app, nothing else running, one call after another.

| Model | Calls | Average | Slowest | Fastest | Errors |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 10 | 32.1s | 57.8s | 10.4s | 0 |
| Gemma-4-31B-IT | 10 | 8.2s | 20.7s | 3.7s | 0 |

Each call: Flash-Next 40.5s, 10.4s, 17.6s, 57.8s, 25.5s, 22.3s, 23.6s, 54.0s, 43.9s, 25.1s; Gemma 11.1s, 6.2s, 3.7s, 20.7s, 4.5s, 4.6s, 7.2s, 6.3s, 11.2s, 6.9s.

The two vision models run side by side, so an upload waits for the slower one: **expect about 30 s on average and up to about a minute for the detectors, plus
the text writer (MiMo-V2.6-Flash) afterwards.** Flash-Next is the bottleneck: Gemma alone would be about 8 s. Earlier runs with several photos at once saw Flash-Next
at 37 to 46 s on average with slowest calls over 150 s (including one unusable answer that was asked twice), so a busy shared endpoint makes it worse. For the
demo: upload one photo at a time, show the "analysing" state, and keep the 60 s per-detector timeout (a slow detector is left out and the other answers).

### Decision rule (amended once)

The rule was written before any real-photo result:

 (a) The scheme with the fewest SILENT wrong answers wins.
 (b) A scheme where the vision models outvote a CNN that was right is OUT, unless that case is flagged for a human.
 (c) Among ties, the lowest human-review rate wins.
 (d) Extra calls only matter if the average latency exceeds about 12 s or the slowest exceeds about 45 s.
 (e) If the results are too close or too noisy with this few photos, keep the current mode.

**Rule (b) was amended after the first real-photo run.** A CNN dissent that is shown as a low-trust note (scheme V) no longer counts as a silent
override. The reason is the measurement: the CNN, trained on drawings, was right on only 14 of 40 real photos, so treating its
disagreement as a veto would mostly cancel correct vision answers. For the other schemes (b) is unchanged. This amendment was made
after seeing real-photo numbers, which is a reason to read the final decision with some caution. Rule (d): the schemes that wait for two
vision models are over the limits (about 37 to 40 s average, over 150 s slowest) because Flash-Next is slow, and so is the pair A; this
is a latency problem of the model, not of the voting rule.

**20 to 40 photos can only reveal large differences.** The large differences here are: the CNN alone is far worse than any scheme that
uses the vision models (26 silent mistakes against 3), and `pair` / `unanimous` / `cnn_anchored` hand over 57% to 62% of photos to a person while
`vote` / `vision_primary` hand over 2%. The differences among silent-wrong counts of 1, 2 and 3 are not large.

---

# Earlier replay on synthetic images (Phase 1, a sanity check only)

## The schemes (and the app setting each one matches)

| | How it decides | `DUAL_MODE` |
|---|---|---|
| 0 | CNN alone. | (no dual detector) |
| A | Pair: CNN + Qwen3.8-Flash-Next. If they disagree, a human reviews it. | `pair` |
| A' | The same pair idea with Gemma-4-31B-IT instead of Flash-Next (reported for completeness). | `pair` with `DUAL_VISION_MODEL=Gemma-4-31B-IT` |
| B | Three equal voters (CNN, Flash-Next, Gemma). Two of three win. A three-way split goes to a human. | `vote` |
| C | Same voters, but if the two vision models agree with each other and outvote the CNN, a human reviews it instead of the vote deciding silently. Otherwise two of three win; a three-way split goes to a human. | `cnn_anchored` |
| U | Unanimity: the answer is automatic ONLY if all available detectors agree; any disagreement goes to a human. | `unanimous` |

A detector with no answer for an image counts as missing. With two voters left, they must agree (otherwise a human reviews
it). With one left, it answers alone (like `single_detector` in the app). Gemma has no answer for 3 images (server errors that
survived the retry); for those images it is treated as missing. The replay uses the app's own rule function
(`decide_labels` in `backend/app/dual_detector.py`), so what is scored here is exactly what the modes do.

## Consistency check (passed again)

The replay must reproduce the numbers already published in `docs/dual_detector_results.md` before anything else is trusted.
It does, exactly, for schemes 0, A and A': the CNN alone gets 20/20 on clean and 62/80 on hard images (18 mistakes, 11 of
them with confidence at least 0.70); the pair CNN + Flash-Next catches 18 of 18 CNN mistakes with 3 false alarms and 0
missed; the pair CNN + Gemma has 80 agree-and-correct, 17 disagreements where the vision model was right and 3
single-detector images. The tool re-runs this check every time and stops if it ever fails.

## Results

How to read the columns: **silent wrong** = the scheme answered by itself and was wrong, so no human was asked (the most
important number). **CNN right but silently outvoted** = the CNN had the right answer, both vision models agreed on
another one, and the scheme went with them without asking anyone. **False alarm** = sent to a human although the CNN's own
answer was right. **Good flag** = sent to a human and the CNN's answer was wrong. **Extra calls** = additional Token Factory
requests per upload. **Latency** = the slowest detector the scheme waits for (they run in parallel).

### Clean images (20 images)

| Scheme | Auto answers correct | SILENT wrong | CNN right but silently outvoted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 20 | **0** | 0 | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 19 | **0** | 0 | 1 (5%) | 1 | 0 | 1 | 3.6s / 6.7s |
| A' Pair: CNN + Gemma | 20 | **0** | 0 | 0 (0%) | 0 | 0 | 1 | 7.9s / 39.6s |
| B  3-way vote (2-of-3) | 20 | **0** | 0 | 0 (0%) | 0 | 0 | 2 | 8.2s / 39.6s |
| C  3-way vote, CNN outvoted -> review | 20 | **0** | 0 | 0 (0%) | 0 | 0 | 2 | 8.2s / 39.6s |
| U  Unanimity (all must agree) | 19 | **0** | 0 | 1 (5%) | 1 | 0 | 2 | 8.2s / 39.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 0; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 1.

### Hard images (blurred, darkened, rotated, noisy copies) (80 images)

| Scheme | Auto answers correct | SILENT wrong | CNN right but silently outvoted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 62 | **18** | 0 | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 60 | **0** | 0 | 20 (25%) | 2 | 18 | 1 | 5.5s / 32.2s |
| A' Pair: CNN + Gemma | 62 | **1** | 0 | 17 (21%) | 0 | 17 | 1 | 7.3s / 42.6s |
| B  3-way vote (2-of-3) | 78 | **0** | 0 | 2 (2%) | 0 | 2 | 2 | 8.7s / 42.6s |
| C  3-way vote, CNN outvoted -> review | 62 | **0** | 0 | 18 (22%) | 0 | 18 | 2 | 8.7s / 42.6s |
| U  Unanimity (all must agree) | 60 | **0** | 0 | 20 (25%) | 2 | 18 | 2 | 8.7s / 42.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 1; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 2.

### All images (100 images)

| Scheme | Auto answers correct | SILENT wrong | CNN right but silently outvoted | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|---|
| 0  CNN alone | 82 | **18** | 0 | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 79 | **0** | 0 | 21 (21%) | 3 | 18 | 1 | 5.1s / 32.2s |
| A' Pair: CNN + Gemma | 82 | **1** | 0 | 17 (17%) | 0 | 17 | 1 | 7.4s / 42.6s |
| B  3-way vote (2-of-3) | 98 | **0** | 0 | 2 (2%) | 0 | 2 | 2 | 8.6s / 42.6s |
| C  3-way vote, CNN outvoted -> review | 82 | **0** | 0 | 18 (18%) | 0 | 18 | 2 | 8.6s / 42.6s |
| U  Unanimity (all must agree) | 79 | **0** | 0 | 21 (21%) | 3 | 18 | 2 | 8.6s / 42.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 1; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 3.

## What this means

- **Any vision model removes almost all silent mistakes.** The CNN alone answers 18 images wrongly without anyone noticing
  (all in the edited copies). A, B, C and U all have 0; the Gemma pair has 1, a CNN mistake on an image where Gemma gave no
  answer, so nobody double-checked it.
- **The schemes differ in how many cases they hand to a person.** B asks for 2 of 100 images, the Gemma pair 17, C 18, and
  A and U 21 each. B is quiet because the two vision models are accurate on these synthetic images and agree with each
  other, so their vote overrules the CNN's mistakes (98 correct automatic answers).
- **U adds nothing here.** It flags exactly the same 21 images as the pair A but costs one more call and waits for the
  slowest of three detectors.
- **The risk this test cannot show:** B lets two vision models overrule a correct CNN without asking anyone. On these images
  that never happened (the column "CNN right but silently outvoted" is 0 everywhere), but both vision models can make the same
  mistake on real photos. C and U exist to guard against that, at the cost of asking a human more often.
- **Latency:** schemes with three detectors average 8.6 s and the slowest took 42.6 s (the pair A: 5.1 s and 32.2 s).

## Decision rule at the time of this replay

The same rule as above, before rule (b) was amended; at that time (e) said to keep `pair`. Applying it mechanically to the synthetic
numbers (a sanity check, not a decision): A, B, C and U tie at 0 silent wrong answers; B has the lowest review rate (2%); but the
images are about 20 independent scenes, so by (e) this was too close to move away from `pair`.

## How much to trust this

- **About 20 independent images, not 100.** The 80 hard images are blurred, darkened, rotated or noisy copies of the same 20
  originals (5 per class). One original appears 5 times, so a single original that behaves differently moves a result by 5
  images; on the clean set one image is 5 percentage points. A difference of 1 to 3 images is noise.
- **Synthetic images from the CNN's own generator,** with a prompt that describes how that generator draws each defect. That
  helps the vision models, so the three-detector schemes look better here than they may on real photos.
- One run, one setting (`reasoning_effort none`), and the shared endpoint's load changes latencies a lot.
- **This replay is only a sanity check. The real-photo run is the deciding one.**

## Phase 2: how the real-photo run is done

`backend/scripts/voting_compare.py` calls the CNN, Qwen3.8-Flash-Next and Gemma-4-31B-IT once per photo (`reasoning_effort none`,
15 requests/min, a rate-limited answer is retried once, `--workers N` for several photos at once), saves every raw answer to a
git-ignored file (`--answers PATH`; the first run's `data/voting_answers.json` was kept as the baseline and never overwritten), and then
prints the table for schemes 0, A, A', B, C, U and V. `--replay-only` re-scores a saved file with no API call; `--per-photo V` lists every
photo; `--baseline OLD.json` prints the latency before/after.

```bash
cd backend
PYTHONPATH=. python scripts/voting_compare.py --photos ../data/real_photos --answers ../data/voting_answers_v3.json --workers 4
PYTHONPATH=. python scripts/voting_compare.py --replay-only --answers ../data/voting_answers_v2.json --per-photo V --baseline ../data/voting_answers.json
```
