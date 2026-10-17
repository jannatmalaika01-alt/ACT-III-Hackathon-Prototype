# Voting experiment, Phase 1 (synthetic replay)

Four ways of combining the detectors, replayed offline from the answers already saved in the synthetic test. **No API
calls were made and no app code was changed.** This is only a sanity check: the real-photo run (Phase 2) is the deciding one.
Nothing here declares a winner.

## The schemes

| | How it decides |
|---|---|
| 0 | CNN alone. |
| A | Pair: CNN + Qwen3.8-Flash-Next. If they disagree, a human reviews it. |
| A' | Same pair idea with Gemma-4-31B-IT instead of Flash-Next (reported for completeness). |
| B | Three equal voters (CNN, Flash-Next, Gemma). Two of three win. A three-way split goes to a human. |
| C | Same voters, but if the two vision models agree with each other and outvote the CNN, a human reviews it instead of the vote deciding silently. Otherwise two of three win; a three-way split goes to a human. |

A detector with no answer for an image counts as missing. With two voters left, agreement wins and disagreement goes to
a human. With one left, it answers alone (like `single_detector` in the app). Gemma has no answer for 3 images (server
errors that survived the retry); for those images it is treated as missing.

## Consistency check (passed)

Before scoring B and C, the replay had to reproduce the numbers already published in `docs/dual_detector_results.md`.
It does, exactly: the CNN alone gets 20/20 on clean and 62/80 on hard images (18 mistakes, 11 of them with confidence
at least 0.70); the pair CNN + Flash-Next catches 18 of 18 CNN mistakes with 3 false alarms and 0 missed; the pair
CNN + Gemma has 80 agree-and-correct, 17 disagreements where the vision model was right and 3 single-detector images.
(The tool re-runs this check every time and stops if it ever fails.)

## Results

How to read the columns: **silent wrong** = the scheme answered by itself and was wrong, so no human was asked (the most
important number). **False alarm** = sent to a human although the CNN's own answer was right. **Good flag** = sent to a
human and the CNN's answer was wrong. **Extra calls** = additional Token Factory requests per upload. **Latency** = the
slowest detector the scheme waits for (they run in parallel).

### Clean images (20 images)

| Scheme | Auto answers correct | SILENT wrong | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|
| 0  CNN alone | 20 | **0** | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 19 | **0** | 1 (5%) | 1 | 0 | 1 | 3.6s / 6.7s |
| A' Pair: CNN + Gemma | 20 | **0** | 0 (0%) | 0 | 0 | 1 | 7.9s / 39.6s |
| B  3-way vote (2-of-3) | 20 | **0** | 0 (0%) | 0 | 0 | 2 | 8.2s / 39.6s |
| C  3-way vote, CNN outvoted -> review | 20 | **0** | 0 (0%) | 0 | 0 | 2 | 8.2s / 39.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 0; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 1.

### Hard images (blurred, darkened, rotated, noisy copies) (80 images)

| Scheme | Auto answers correct | SILENT wrong | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|
| 0  CNN alone | 62 | **18** | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 60 | **0** | 20 (25%) | 2 | 18 | 1 | 5.5s / 32.2s |
| A' Pair: CNN + Gemma | 62 | **1** | 17 (21%) | 0 | 17 | 1 | 7.3s / 42.6s |
| B  3-way vote (2-of-3) | 78 | **0** | 2 (2%) | 0 | 2 | 2 | 8.7s / 42.6s |
| C  3-way vote, CNN outvoted -> review | 62 | **0** | 18 (22%) | 0 | 18 | 2 | 8.7s / 42.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 1; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 2.

### All images (100 images)

| Scheme | Auto answers correct | SILENT wrong | Human review (rate) | False alarms | Good flags (CNN was wrong) | Extra calls per upload | Latency avg / max |
|---|---|---|---|---|---|---|---|
| 0  CNN alone | 82 | **18** | 0 (0%) | 0 | 0 | 0 | 0.0s / 0.0s |
| A  Pair: CNN + Flash-Next | 79 | **0** | 21 (21%) | 3 | 18 | 1 | 5.1s / 32.2s |
| A' Pair: CNN + Gemma | 82 | **1** | 17 (17%) | 0 | 17 | 1 | 7.4s / 42.6s |
| B  3-way vote (2-of-3) | 98 | **0** | 2 (2%) | 0 | 2 | 2 | 8.6s / 42.6s |
| C  3-way vote, CNN outvoted -> review | 82 | **0** | 18 (18%) | 0 | 18 | 2 | 8.6s / 42.6s |

Facts about these images: 3-way splits (all three detectors answered, three different labels): 1; CNN right but outvoted by both vision models: 0; images with a missing detector answer: CNN 0, Flash-Next 0, Gemma 3.

## What this means

- **Adding any vision model removes almost all silent mistakes.** The CNN alone answers 18 images wrongly without anyone
  noticing (all in the edited copies). A, B and C have 0 silent wrong answers; the Gemma pair has 1, and that one is a CNN
  mistake on an image where Gemma gave no answer, so nobody double-checked it.
- **The schemes differ in how many cases they hand to a person, not in silent mistakes.** The pair A asks a human for 21 of
  100 images, C for 18, the Gemma pair for 17, and B for only 2. B is so quiet because the two vision models are accurate
  on these synthetic images and agree with each other, so their vote overrules the CNN's mistakes (98 correct automatic answers).
- **The price of B and C** is one more call from the shared quota on every upload (2 extra instead of 1) and a longer wait:
  the slowest of three detectors, about 8.6 s on average and up to 42.6 s, against 5.1 s and 32.2 s for the pair A.
- **The risk that this test cannot show:** B lets two vision models overrule a correct CNN without asking anyone. On these
  images that never happened (0 CNN-correct images were outvoted), but the two vision models can make the same mistake on
  real photos. C exists to guard against that, at the cost of asking a human more often.
- **Applying the decision rule mechanically to these numbers** (not a decision): A, B and C tie at 0 silent wrong answers;
  B has the lowest review rate, but B and C do not beat A by 2-3 fewer silent wrong answers, so the rule's last step
  ("if unclear, keep A") applies. The synthetic data cannot separate them.

Decision rule (fixed BEFORE the results were looked at):
 1. The scheme with the fewest SILENT wrong answers wins.
 2. If tied, the lower human-review rate wins.
 3. B or C must beat A by at least 2-3 fewer silent wrong answers to justify 2 extra calls per upload and the extra complexity.
 4. If it is unclear, keep A (the pair): it is simpler and already built.

## How much to trust this

- **About 20 independent images, not 100.** The 80 hard images are blurred, darkened, rotated or noisy copies of the
  same 20 originals (5 per class). One original appears 5 times, so a single original that behaves differently moves a
  result by 5 images. On the clean set one image is 5 percentage points.
- **A difference of 1 to 3 images is noise.** Only large gaps (such as 18 silent mistakes versus 0) mean something.
- **Synthetic images from the CNN's own generator,** with a prompt that describes how that generator draws each defect.
  That helps the vision models, so B and C look better here than they may on real factory photos.
- One run, one model setting (`reasoning_effort none`), and the shared endpoint's load changes latencies a lot.
- **This replay is only a sanity check. The real-photo run is the deciding one.**

## Phase 2: the same comparison on real photos

`backend/scripts/voting_compare.py` calls the CNN, Qwen3.8-Flash-Next and Gemma-4-31B-IT once per photo
(`reasoning_effort none`, paced to 15 requests/min, a rate-limited answer is retried once), saves every raw answer to
`data/voting_answers.json` (git-ignored, like `data/real_photos/`), and then replays schemes 0, A, A', B and C from that
file. `--replay-only` re-scores the saved file with no API call. Command and photo format: see the README "Dual detector" section.
