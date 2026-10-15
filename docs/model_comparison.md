# Token Factory model comparison

Run on 2026-10-15: 18 cases per model (56 requests). The agent's real prompt and the real CPU retriever were used; confidence was set to 0.90 so every case reaches the LLM. reasoning_effort=low. 'Evidence quote verified' means the quote the model gave really appears in the cited manual chunk; 'correct manual' means the cited manual is the one that covers that defect.

| Model | Cases | Valid JSON | Evidence quote verified | Correct manual cited | Avg / max latency | Retries (HTTP + JSON) | Errors |
|---|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash | 18 | 94% | 94% | 94% | 27.5s / 133.6s | 0 | 1 |
| Qwen3.8-27B | 18 | 94% | 89% | 94% | 61.5s / 121.0s | 0 | 1 |
| DeepSeek-V4-Flash | 18 | 72% | 72% | 72% | 84.1s / 179.5s | 0 | 5 |

**Recommended: MiMo-V2.6-Flash** (ranked by verified evidence, correct citation, valid JSON, then errors/retries and speed).

## Failures

**MiMo-V2.6-Flash** (reasoning_effort=low): 
- crack / weld joint A3: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)

**Qwen3.8-27B** (reasoning_effort=low): 
- structural_crack / inspection test_04.png: json=True quote=False manual=True
- corrosion / inspection test_14.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)

**DeepSeek-V4-Flash** (reasoning_effort=low): 
- bearing_wear / inspection test_02.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)
- corrosion / inspection test_06.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)
- bearing_wear / inspection test_07.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)
- structural_crack / inspection test_13.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)
- corrosion / inspection test_20.png: LLMUnavailableError: All LLM providers failed (tokenfactory: ReadTimeout)

## MiMo-V2.6-Flash with reasoning_effort "none"

Same 18 cases, same prompt and retriever (`compare_models.py --models MiMo-V2.6-Flash --reasoning-effort none --full`).

| Model | Cases | Valid JSON | Evidence quote verified | Correct manual cited | Avg / max latency | Retries (HTTP + JSON) | Errors |
|---|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash (low) | 18 | 94% | 94% | 94% | 27.5s / 133.6s | 0 | 1 |
| MiMo-V2.6-Flash (none) | 18 | 100% | 100% | 100% | 19.8s / 47.9s | 0 | 0 |

## How to read these numbers

- Every error in the runs above is a 120 s read timeout, not a bad answer. None came from rate limits (0 HTTP retries). Pacing was under 20 requests/min, but the endpoint is shared, so latency also depends on what teammates are running.
- 18 cases per model is a small sample: one failure moves a rate by about 5 points. Treat 94% vs 100% as "no clear difference in quality"; the difference that is clear is DeepSeek-V4-Flash being slowest (5 timeouts) and the timeouts overall.
- The test inputs are generated images/defect names, not real factory photos, and the retriever is the same for every model.
- `none` was tried on MiMo only. Thinking is the likely reason for the slow answers (all models are slower than a plain completion), but the runs do not prove it; `none` gave faster answers here with no loss of quality.
