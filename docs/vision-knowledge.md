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
python -m pytest vk_tests -q                        # 20 tests, includes the backend compatibility checks
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

## Test run (Oct 9 note) - `python evaluate.py`
| Check | Result |
|---|---|
| Image detection, 10 original images | 10 of 10 |
| Image detection, 10 held-out harder images (textured, noisy, blurred) | 10 of 10 |
| Sensor reports | 4 of 4 |
| Documents incl. one scanned PDF read by OCR | 5 of 5 |
| Retrieval, correct manual page in top-1 / top-3 of 15 questions | 14 of 15 / 15 of 15 |
| End to end: right defect AND top cited page is its procedure | 21 of 21 |

Retrieval ran in TF-IDF mode in this run (no internet for the MiniLM download). Run `python evaluate.py` on a machine with
`sentence-transformers` to get the semantic-mode numbers; the first line prints which embedder was used.

## Honest limits
* There is **no real photo set yet**. The detector is trained on generated images (`vision/synthetic.py`), and the test images
  are generated too, so "20 of 20" shows the model and pipeline work, not accuracy on real factory photos. To fix that, put
  real photos in `data/train/<normal|bearing_wear|structural_crack|corrosion>/` and run `python -m vision.train` (they are
  mixed in automatically). Add real photos to `data/test/` and `data/test_cases.csv` to measure honestly.
* The manuals are short synthetic samples written for the prototype (`setup_manuals.py`). Real PDFs dropped into `data/raw/`
  are indexed automatically.
* Sensor thresholds are typical values, not site-specific limits.
* Detector confidence is below 0.70 for unclear images; the backend agent already routes those to human verification.
