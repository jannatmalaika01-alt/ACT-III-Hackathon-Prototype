# Vision and Knowledge (Titus)

Steps 2 and 3 of the flow: **detect and read** (image / sensor report / document) and **find the fix** (RAG over the
manuals and SOPs). Everything lives in new folders (`vision/`, `knowledge/`, `data/`, `vk_tests/`); nothing in `backend/`,
`frontend/` or `Last/` was changed.

## Run it
```bash
pip install -r requirements-vision.txt
python run_pipeline.py                              # contract JSON for data/test/test_02.png
python run_pipeline.py data/test_reports/s01_motor_vibration.csv
python evaluate.py                                  # the Oct 9 test run (results below)
python -m pytest vk_tests -q                        # 49 tests, includes backend compatibility, citation and secret checks
python -m vision.train                              # retrain the detector (about 3 minutes on CPU)
```
Optional: install the Tesseract program for OCR of scanned PDFs, and `sentence-transformers` for semantic search.
Without them the code still runs (TF-IDF retrieval, scanned pages reported as unreadable instead of invented).

## What each input type does
| Input | How the defect is found | Output `input_type` |
|---|---|---|
| Image (png/jpg/webp) | `vision/defect_detector.py`: small CNN, real softmax confidence | `image` |
| Sensor report (csv / json / `key: value` txt) | `vision/sensor_analyzer.py`: vibration, bearing temperature, wall loss, crack width against warn/alarm levels (editable in `THRESHOLDS`) | `sensor` |
| Document (pdf / txt, scans via OCR) | `vision/document_analyzer.py`: defect keywords in the text, ignoring negated sentences ("no visible cracks") | `document` |

Then `knowledge/pipeline.py` retrieves manual chunks, takes the action and priority from the best chunk that contains an
"Action Required" procedure, and returns exactly the shared contract (`case_id, input_type, defect, evidence[source,page,chunk],
recommendation[action,priority], status, task_id`). If no chunk supports the defect, the action says "manual inspection
required" instead of guessing.

## Plugging into the backend (nothing of theirs edited)
* **Defect report to the agent:** `DefectDetector.defect_report(result, location, image_ref)` returns the body of
  `POST /cases/{id}/analyze` (or `/diagnose`): `defect_type, location, confidence, image_ref`.
  `python push_to_backend.py <file> --api http://localhost:8000` runs upload -> detect -> analyze in one go.
* **Real manuals for the agent's citations:** `backend/core.py::retrieve()` searches 3 hard-coded SOP strings.
  `knowledge/retrieval/adapter.py::retrieve(query, n_results)` returns the same shape (`manual, page, text`) from the real
  PDFs. To switch, Jannat changes one import in `core.py`/`agent.py`. `vk_tests/test_backend_compat.py` already proves it
  works inside the real `agent.py` and the real API, including the "no source -> refuse" path.
* Defect names: the contract uses readable names (`bearing wear`, `structural crack`, `corrosion`, `normal`). The retriever also
  understands the agent's names (`crack`, `loose_bolt`, `misalignment`).

## Test results (repeatable)
`python evaluate.py --write-report` runs everything with fixed seeds and rewrites `docs/EVALUATION.md` (full per-file table) and
`docs/eval_results.json`. Two runs give identical numbers. Summary of the last run (keyword-only retrieval, see limits):

| Check | Result |
|---|---|
| Detection, 20 test images (10 original + 10 held-out harder) | 20 of 20 |
| Detection, sensor reports / documents (one scanned PDF via OCR) | 4 of 4 / 5 of 5 |
| Stress set: blurred, darkened, rotated, noisy copies of 100 unseen images (400 total) | 394 of 400 (98%); old detector 306 of 400 (77%) |
| Retrieval, right page in top-1 / top-3 of 30 questions | 28 of 30 / 29 of 30 |
| Off-topic questions (bread, tax, cat...) correctly given no evidence | 10 of 10 |
| Citations: every evidence chunk found verbatim on its cited PDF page, action copied from evidence | 29 of 29 |
| End to end: right defect and the top cited page is its procedure | 21 of 21 |

Citation checking lives in `knowledge/verify.py` (`verify_citation(manual, page)` for the agent's final answer,
`verify_contract(contract)` for pipeline output); `vk_tests/test_citations_and_relevance.py` proves it catches a made-up manual,
a wrong page, edited text and an invented action.

## Real photos
Put photos in a folder with a `labels.csv` (`filename,label`, labels `normal|bearing_wear|structural_crack|corrosion`; the same
layout as `data/real_photos` used by `backend/scripts/real_photo_check.py`) and run
`python evaluate.py --real-dir data/real_photos`. To teach the CNN from them, copy them into `data/train/<class>/` and run
`python -m vision.train`.

## Secrets
Nothing in this folder contains credentials (checked by `vk_tests/test_no_secrets.py`). `knowledge/redact.py::redact()` masks keys,
bearer tokens and `*_KEY=` / `*_SECRET=` / `*_TOKEN=` values; `push_to_backend.py` prints everything through it. Never commit `.env`.

## Honest limits
* **No real factory photos were available**, so every image here, including the stress set, is generated. "98%" means the model
  survives blur, darkness, rotation and noise on generated images; it is NOT accuracy on real photos, which is expected to be
  lower. Run `--real-dir` with real photos before quoting a real number. The dual detector (vision-language model) exists for
  exactly this gap.
* Retrieval ran keyword-only (TF-IDF) because MiniLM could not be downloaded where this was tested. Paraphrased questions can miss
  (2 of 30 here); run `python evaluate.py` where `sentence-transformers` is installed and check the `embedder:` line.
* A hit must share at least two meaningful words with the query (or MiniLM must score it >= 0.40), so unrelated questions return no
  evidence and the backend answers 422 "no relevant SOP" instead of citing a random page. A detection of "normal" retrieves routine
  inspection guidance.
* The manuals are short synthetic samples written for the prototype (`setup_manuals.py`). Real PDFs dropped into `data/raw/` are
  indexed automatically, and citations are verified against whatever PDFs are there.
* Sensor thresholds are typical values, not site-specific limits.
* Detector confidence below 0.70 means a person should verify; the backend agent already routes those to human verification.
