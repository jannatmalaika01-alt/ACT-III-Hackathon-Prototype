# Dual detector and Token Factory model results

Branch `dual-detector`, run on 2026-10-15 against the shared AMD Token Factory endpoint. Everything here is
reproducible with the scripts in `backend/scripts/` (listed at the end). The test images are git-ignored and are
rebuilt by `build_dual_testset.py`.

## Plain-English summary

- **Which models can see images?** Seven can: Qwen3.8-Flash-Next, Gemma-4-31B-IT, Qwen3.8-27B, MiniCPM-V-4.7-35B-A3B,
  MiniCPM-V-4.7-1B, DeepSeek-V4-Flash-Vision-Exp and DeepSeek-V4.1-Flash (too slow to test fully). **MiMo-V2.6-Flash accepts
  images but cannot see them**: it answered "White" or "Black" for a red square and a blue square, so it is not usable for
  vision even though the catalog lists image input.
- **Best vision model: Qwen3.8-Flash-Next.** Alone it got 95% on clean and 96% on hard images, answered all 100 images
  with no errors, and averaged 5.1 s. Gemma-4-31B-IT was the most accurate (95% / 98%) but the endpoint had no
  available workers for it during part of the day (HTTP 500 / 503), so it is not dependable right now.
- **Did the dual detector help? On this test, yes.** The CNN is perfect on clean images (20/20) but drops to 78% (62/80) on
  the harder copies (blurred, darkened, rotated). Paired with Qwen3.8-Flash-Next, the disagreement flag caught **18 of the
  CNN's 18 mistakes** (11 of 11 mistakes the CNN was confident about, which the existing "confidence below 0.70" rule can
  never flag) and raised **3 false alarms** on the 82 images the CNN got right. For comparison, the existing confidence rule
  would have sent only 7 of the 18 CNN mistakes to a human.
- **Cost of turning it on:** one extra shared-quota LLM call per image upload and about 5 s extra latency (up to 30+ s
  when the endpoint is busy). It is off by default (`DUAL_DETECTOR=off`).
- **Best text model with `reasoning_effort: none`:** four models tie at 18/18 on every measure (MiMo-V2.6-Flash,
  DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, MiniCPM5-2B). The script's tie-break (speed) picks MiniCPM5-2B, but with
  18 cases the four are indistinguishable. Keeping MiMo-V2.6-Flash is reasonable: it is the current default, it scored
  18/18 here and in the earlier run, and it averaged 8.9 s.

## Step 1: which models accept images

The Token Factory catalog (`/models`) lists input modalities; each model was also tested with a real image (a solid red
160x160 square, "what colour is this?") so "yes" means the model really saw it. MinerU2.5-Pro was skipped (OCR only).
`reasoning_effort` shows the value each model accepts (the script prefers `none`).

| Model | Accepts images | Really sees them | reasoning_effort accepted | Note |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | yes | yes | none | one probe hit another user's concurrency limit and was repeated |
| Qwen3.8-27B | yes | yes | none | |
| Gemma-4-31B-IT | yes | yes | none | worker outage later in the day (see limits) |
| MiniCPM-V-4.7-35B-A3B | yes | yes | none | |
| MiniCPM-V-4.7-1B | yes | yes | low (rejects `none` with HTTP 422) | |
| DeepSeek-V4-Flash-Vision-Exp | yes | yes | none | |
| DeepSeek-V4.1-Flash | yes | yes | low tested | very slow: 61 s, then 120 s timeouts |
| MiMo-V2.6-Flash | accepted | **no**: answers White/Black for red and blue | none | not usable for vision |
| DeepSeek-V4-Flash | no (HTTP 400 "does not support image input") | no | none | |
| GLM-5.3-Flash | no (HTTP 400) | no | none | |
| MiniCPM5-2B | no (HTTP 400) | no | none | |

## Step 2: test set

100 images in `data/dual_lab/` (git-ignored): 5 new generated images per class (normal, bearing_wear, structural_crack,
corrosion) from seeds the CNN never trained on, each with 4 harder copies: blurred (Gaussian 3 px), darkened (30% brightness),
rotated 35 degrees, and noisy (sigma 28). The true label is kept in `manifest.csv`. "Clean" = the 20 originals, "hard" = the 80 copies.

## Steps 3 and 5: Round 1 and Round 2

All vision models got the same prompt, `json_object` output, and the `reasoning_effort` from step 1. Labels outside the
allowed list count as wrong. Answers lost to a rate limit were retried **once** (a single mop-up pass at 12 requests/min):
25 of 28 recovered; the 3 left (Gemma, HTTP 500 worker outage) are counted as `single_detector` in round 2. DeepSeek-V4.1-Flash was
stopped after 3 consecutive 120 s timeouts (15 images answered), so its row is not comparable.

### Round 1: each detector alone (same prompt, same images)

| Detector | Images answered | Accuracy clean | Accuracy hard | Avg latency | Max latency | Errors |
|---|---|---|---|---|---|---|
| CNN (Titus) | 100 | 100% (20/20) | 78% (62/80) | 0.0s | 0.0s | none |
| DeepSeek-V4-Flash-Vision-Exp | 100 | 80% (16/20) | 90% (72/80) | 4.3s | 12.5s | none |
| DeepSeek-V4.1-Flash | 15 | 0% (0/2) | 31% (4/13) | 30.8s | 120.9s | rate_limit x7, http_500 x1, timeout x3 (STOPPED: stopped after 3 consecutive timeouts: no answer within 120s) |
| Gemma-4-31B-IT | 100 | 95% (19/20) | 98% (78/80) | 7.4s | 42.6s | http_500 x3 |
| MiniCPM-V-4.7-1B | 100 | 90% (18/20) | 90% (72/80) | 3.8s | 7.5s | none |
| MiniCPM-V-4.7-35B-A3B | 100 | 80% (16/20) | 94% (75/80) | 5.0s | 10.6s | none |
| Qwen3.8-27B | 100 | 80% (16/20) | 91% (73/80) | 7.5s | 47.6s | none |
| Qwen3.8-Flash-Next | 100 | 95% (19/20) | 96% (77/80) | 5.1s | 32.2s | none |

### Round 2: each vision model paired with the CNN

| Vision model | Agree + correct | Agree + WRONG | Disagree, CNN right | Disagree, vision right | Disagree, both wrong | single_detector | CNN mistakes caught | of them CNN-confident (>=0.70) | False alarms | Missed (both agree, wrong) |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp | 72 | 1 | 10 | 16 | 1 | 0 | 17 of 18 | 10 of 11 | 10 | 1 |
| DeepSeek-V4.1-Flash | 2 | 0 | 0 | 2 | 0 | 11 | 2 of 2 | 1 of 1 | 0 | 0 |
| Gemma-4-31B-IT | 80 | 0 | 0 | 17 | 0 | 3 | 17 of 17 | 10 of 10 | 0 | 0 |
| MiniCPM-V-4.7-1B | 73 | 1 | 9 | 17 | 0 | 0 | 17 of 18 | 10 of 11 | 9 | 1 |
| MiniCPM-V-4.7-35B-A3B | 74 | 0 | 8 | 17 | 1 | 0 | 18 of 18 | 11 of 11 | 8 | 0 |
| Qwen3.8-27B | 73 | 1 | 9 | 16 | 1 | 0 | 17 of 18 | 10 of 11 | 9 | 1 |
| Qwen3.8-Flash-Next | 79 | 0 | 3 | 17 | 1 | 0 | 18 of 18 | 11 of 11 | 3 | 0 |

CNN made 18 mistakes on all 100 images. 'Caught' = the pair disagreed on an image the CNN got wrong (a human would have been asked); 'false alarm' = the pair disagreed although the CNN was right; 'missed' = both agreed on the same wrong label.

Reading round 2: "Disagree, vision right" are the CNN mistakes the second detector fixed or exposed. "Of them CNN-confident"
counts mistakes where the CNN reported at least 0.70 confidence: the old low-confidence rule would have sent only 7 of the 18
CNN mistakes to a human, the other 11 needed the second detector. Confidences are never averaged; both are shown.

## Step 4: the dual detector in the backend

`backend/app/dual_detector.py`. With `DUAL_DETECTOR=on`, `POST /cases/{id}/analyze` runs the CNN and the Token Factory
vision model (`DUAL_VISION_MODEL`, default Qwen3.8-Flash-Next; `DUAL_VISION_EFFORT`, default `none`; `DUAL_VISION_TIMEOUT`, default 60 s)
in parallel and returns `dual_detector` in the response (also stored in `diagnosis_raw.dual_detector`) with:

- `agreement`: `agree`, `disagree`, or `single_detector` (one detector failed or timed out)
- `cnn` and `vision_model`: each with its own label and confidence (never averaged or merged)
- `needs_human_review`: true on `disagree`. The agent then skips the LLM, marks the diagnosis UNVERIFIED, states both results, and still cites the manual.

`/health` shows `dual_detector` and `vision_model`. 16 new tests cover agree, disagree, single_detector (either detector failing),
parallel execution, timeouts, the off-by-default path, and the agent routing. All 119 backend tests pass.

## Step 6: text diagnosis, all chat models, reasoning_effort "none"

Same 18 cases and real retriever as the earlier comparison, paced at 15 requests/min. A model that rejects `none` falls back to `low`
(MiniCPM-V-4.7-1B did). No model hit a spending cap or was stopped for 429s.

| Model | Cases | Valid JSON | Evidence quote verified | Correct manual cited | Avg / max latency | Retries (HTTP + JSON) | Errors |
|---|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash | 18 | 100% | 100% | 100% | 8.9s / 20.4s | 0 | 0 |
| Qwen3.8-27B | 18 | 100% | 89% | 100% | 8.6s / 16.6s | 0 | 0 |
| Qwen3.8-Flash-Next | 18 | 100% | 89% | 100% | 9.1s / 32.9s | 0 | 0 |
| Gemma-4-31B-IT | 18 | 39% | 39% | 39% | 22.1s / 121.0s | 1 | 11 |
| DeepSeek-V4-Flash | 18 | 100% | 100% | 100% | 9.0s / 16.5s | 0 | 0 |
| DeepSeek-V4-Flash-Vision-Exp | 18 | 100% | 100% | 100% | 8.4s / 18.5s | 0 | 0 |
| MiniCPM-V-4.7-35B-A3B | 18 | 100% | 89% | 100% | 10.7s / 13.5s | 0 | 0 |
| MiniCPM-V-4.7-1B | 18 | 100% | 50% | 100% | 16.6s / 53.4s | 2 | 0 |
| GLM-5.3-Flash | 18 | 94% | 94% | 94% | 40.7s / 125.0s | 0 | 1 |
| MiniCPM5-2B | 18 | 100% | 100% | 100% | 8.0s / 10.1s | 0 | 0 |
| DeepSeek-V4.1-Flash | 18 | 94% | 94% | 94% | 14.8s / 120.8s | 1 | 1 |

Notes: Gemma-4-31B-IT's 11 errors are one timeout and ten HTTP 500s from the provider
(`No available workers (all circuits open or unhealthy)`): an endpoint outage, not a model-quality result.
MiniCPM-V-4.7-1B returns valid JSON but its evidence quotes are usually not exact copies of the manual (50% verified).
GLM-5.3-Flash and DeepSeek-V4.1-Flash each lost one case to a 120 s timeout and are the slowest.

## Limits (read before trusting any number)

- **Synthetic images.** Every image comes from the same generator that trained the CNN. Real factory photos may behave very differently, in either direction.
- **Same-source training data.** The CNN has only ever seen this generator's style; the vision LLMs never trained on it. The CNN's drop on the hard set
  reflects that blur, darkness and a 35 degree rotation are harsher than anything in its training, so "CNN made 18 mistakes" is partly a property of the test I chose.
- **The prompt describes each defect.** The class descriptions in the prompt ("concentric scoring", "thin dark line", "rust speckles") match how the generator draws them, which helps the LLMs on synthetic data.
- **Small sample.** 20 clean images (one image = 5 points) and 80 hard ones, one run per model, temperature 0.2. Differences of a few points are noise.
- **Vision models' clean-image errors are systematic**: they call faint bearing wear and cracks "normal" (that is why several score 80% on clean but 90%+ on hard).
- **Rate limits.** The server allows 20 requests/min for this key and the quota is shared; some answers were lost to 429s (mop-up pass described above), and Gemma had an outage.
- **Text comparison:** 18 cases per model and only `reasoning_effort: none` tested here (MiniCPM-V-4.7-1B ran at `low` after `none` was rejected). The four-way tie at 18/18 cannot be broken with this sample.
- **Latency** is dominated by the shared endpoint's load, not just the model, and varied a lot between runs.
- DeepSeek-V4.1-Flash was tested on only 15 images.

## How to reproduce

```bash
cd backend && export PYTHONPATH=.        # key and URL come from backend/.env
python scripts/vision_probe.py           # step 1
python scripts/build_dual_testset.py     # step 2
python scripts/vision_round1.py          # step 3 (about 40 min); add --retry-rate-limited once afterwards
python scripts/vision_report.py          # rounds 1 and 2 tables
python scripts/compare_models.py --full --reasoning-effort none --models <names>   # step 6
```

> Note: the CNN figures in this file (for example "18 mistakes") were measured with the CNN weights from before Titus's retrain. Real-photo figures with the current weights are in the last section of `docs/voting_comparison.md`.
