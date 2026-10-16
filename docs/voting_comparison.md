# Voting experiment, Phase 1 (synthetic replay)

Six ways of combining the detectors, replayed offline from the answers already saved in the synthetic test. **No API
calls were made in this replay.** This is only a sanity check: the real-photo run (Phase 2) is the deciding one.
Nothing here declares a winner, and the app's default (`DUAL_MODE=pair`) is unchanged.

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

## Decision rule (written now, before any real-photo result)

 (a) The scheme with the fewest SILENT wrong answers wins.
 (b) A scheme where the vision models outvote a CNN that was right is OUT, unless that case is flagged for a human.
 (c) Among ties, the lowest human-review rate wins.
 (d) Extra calls only matter if the average latency exceeds about 12 s or the slowest exceeds about 45 s.
 (e) If the results are too close or too noisy with this few photos, KEEP `pair` (the default).

**20 to 40 real photos can only reveal large differences.** A gap of one to three photos is noise, and in that case the rule
says to keep `pair`.

Applying the rule mechanically to the synthetic numbers (a sanity check, not a decision): A, B, C and U tie at 0 silent
wrong answers; nobody is out under (b) because no correct CNN answer was silently outvoted; B has the lowest review rate
(2%) and its latency stays under the limits in (d) (8.6 s average, 42.6 s slowest, which is close to the 45 s limit); but
the images are about 20 independent scenes, so by (e) this is too close to move away from `pair`.

## How much to trust this

- **About 20 independent images, not 100.** The 80 hard images are blurred, darkened, rotated or noisy copies of the same 20
  originals (5 per class). One original appears 5 times, so a single original that behaves differently moves a result by 5
  images; on the clean set one image is 5 percentage points. A difference of 1 to 3 images is noise.
- **Synthetic images from the CNN's own generator,** with a prompt that describes how that generator draws each defect. That
  helps the vision models, so the three-detector schemes look better here than they may on real photos.
- One run, one setting (`reasoning_effort none`), and the shared endpoint's load changes latencies a lot.
- **This replay is only a sanity check. The real-photo run is the deciding one.**

## Phase 2: the same comparison on real photos

`backend/scripts/voting_compare.py` calls the CNN, Qwen3.8-Flash-Next and Gemma-4-31B-IT once per photo
(`reasoning_effort none`, paced to 15 requests/min, a rate-limited answer is retried once), saves every raw answer to
`data/voting_answers.json` (git-ignored, like `data/real_photos/`), then prints the same table for all six schemes. 
`--replay-only` re-scores the saved file with no API call.

```bash
cd backend
PYTHONPATH=. python scripts/voting_compare.py --photos ../data/real_photos      # photos + labels.csv in that folder
PYTHONPATH=. python scripts/voting_compare.py --replay-only                     # re-score, no API call
```
