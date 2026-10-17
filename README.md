# Factory Agent: AI quality and maintenance assistant

> AMD Developer Hackathon, ACT III. Track: Intelligent Industry.

Upload a **photo of a machine part, a sensor report, or a maintenance document (PDF/text)**. The app finds the defect, looks up the right procedure in
the maintenance manuals and **shows the exact manual page it relied on**, writes a recommended action, asks a **person to approve or reject** it, and
then hands the approved case to **Evolus**, which prepares the maintenance task.

```
photo / sensor report / PDF  ->  defect detection  ->  manuals search with a verified citation  ->  recommendation
        ->  human approval  ->  Evolus task
```

What makes it trustworthy: every recommendation quotes the manual and the quote is checked against the page; when the detectors disagree or are unsure, the
case is marked **UNVERIFIED** and a person decides; each model's answer is shown separately (confidences are never averaged); and nothing crashes when an AI
service is slow or down.

---

## Quick start: one command, no keys

You need [Docker](https://docs.docker.com/get-docker/) (Docker Desktop, or Docker Engine with Compose v2.24 or newer). Nothing else, and **no API keys**.

```bash
git clone <this repository>
cd ACT-III-Hackathon-Prototype
docker compose up --build
```

The first build downloads Python and Node packages and a small search model (the backend image is about 2.6 GB) and takes a few minutes; later starts take seconds.
When the log says both services are healthy, open:

| What | Where |
|---|---|
| The web app | http://localhost:3000 (demo login: **admin** / **factory123**, a demo-only login, not real authentication) |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

Try it: click **New case**, upload a file from [`data/test/`](data/test) (photos) or [`data/test_reports/`](data/test_reports) (sensor reports and documents), open the
case, read the diagnosis and its manual citation, then **Approve**. Stop with `Ctrl+C` and `docker compose down`.
Ports already in use? `FRONTEND_PORT=3010 BACKEND_PORT=8010 docker compose up --build`.

**What "mock mode" means (the default, no keys):**

| Part | Default (no keys) | Real mode (opt-in, see below) |
|---|---|---|
| Image detector | our small CNN, running locally | + two AMD Token Factory vision models vote on every photo |
| Recommendation text | an **offline template**: the manual's own "Action Required" step, labelled as a template (not an AI answer) | written by an AI model on AMD Token Factory (with a backup model) |
| Database | in memory (data is lost on restart) | Supabase |
| Evolus task | a fake id such as `MOCK-TASK-0001` | a real Evolus workflow run with the Evolus agent |

---

## How it works

```
 Browser  --->  Next.js web app (:3000)  --/api/backend/*-->  FastAPI backend (:8000)
                                                                 |        |         |
                                                  (memory or Supabase)    |         +--> Evolus (REST: start workflow, poll result)
                                                                          +--> AMD Token Factory (vision models, text models)

 Inside the backend, for one uploaded case:

   upload (photo | sensor report | PDF/text)
      |
      v
 1  DETECT      photo        -> CNN (ours, advisory)  +  Qwen3.8-Flash-Next and Gemma-4-31B-IT look at the same photo
                sensor report-> threshold analyser (vibration, temperature, wall loss, crack width)
                document     -> keyword reader (ignores "no visible cracks"); scanned PDFs need optional OCR
      |          mode "vision_primary": the vision models decide, the CNN only breaks a tie (if it is >= 70% sure) or leaves a note;
      |          disagreement or low confidence -> the case is marked UNVERIFIED for a person
      v
 2  FIND        search the 5 manuals in data/raw (on the CPU) -> best passage, with manual name and page
      |
      v
 3  WRITE       MiMo-V2.6-Flash writes the explanation and action. The quote it gives must appear in the cited page, otherwise a backup
      |          model (DeepSeek-V4-Flash) gets ONE try, otherwise the case says "recommendation unavailable" with the citation still shown.
      |          No key at all -> the offline template described above.
      v
 4  DECIDE      a person approves or rejects in the web app (approved photos are saved for a later retrain of the CNN)
      |
      v
 5  EVOLUS      the approved case starts an Evolus workflow; the agent's answer is parsed and stored on the case
```

Design rules worth knowing: confidences from different models are **never averaged**; a detector or writer that fails or times out is simply left out
(an upload never breaks); a manual citation is **checked**, never invented; at most one Evolus run per case at a time, and a timed-out run is **resumed** by Retry.

### AMD Token Factory: which model does what

| Role | Model | Notes |
|---|---|---|
| Writes the diagnosis (primary) | **MiMo-V2.6-Flash** | scored 18/18 on JSON, verified quote and correct manual in our 18-case test |
| Writes the diagnosis (backup, tried once) | **DeepSeek-V4-Flash** (reasoning effort `none`) | also 18/18; used when the primary fails or its citation does not check out |
| Vision model 1 | **Qwen3.8-Flash-Next** | reliable, about 32 s per photo when called one at a time |
| Vision model 2 | **Gemma-4-31B-IT** | most accurate alone in our tests, about 8 s per photo; the endpoint had worker outages for it on one day |
| **Never use for vision** | MiMo-V2.6-Flash | accepts images but cannot see them (it answered "white" for a red square) |

We probed every chat model in the Token Factory catalog; the comparison tables, the voting experiments and the real-photo results are in
[`docs/model_comparison.md`](docs/model_comparison.md), [`docs/dual_detector_results.md`](docs/dual_detector_results.md) and
[`docs/voting_comparison.md`](docs/voting_comparison.md). Photos sent to the vision models are first shrunk to 1024 px. The retrieval step stays on the CPU
because the shared endpoints have no embedding model.

### Evolus: the three building blocks

1. **An agent with a manuals library**: the "Factory Maintenance Agent" answers from the manuals uploaded to its Evolus library and says when something is general practice.
2. **An automation workflow**: `Start -> Ask an agent -> Automation output`.
3. **An API call**: the backend starts the workflow with `POST .../workflows/{id}/execute`, then polls `GET .../executions/{id}` until the answer is ready, parses it
   (defect, severity, probable cause, safety precautions, recommended steps, citation, confidence, needs human review, summary) and stores it on the case.

The verified request and response shapes, measured timings, the retry-resumes-the-run behaviour and the execution id of a live end-to-end run are in
**[`docs/evolus_integration.md`](docs/evolus_integration.md)**.

---

## Switch to real mode (opt-in)

Real mode is just keys in one git-ignored file. Mock mode keeps working for everything you do not configure.

1. `cp backend/.env.example backend/.env` and open it. **Part 1** holds the safe defaults; **Part 2** lists every other setting, commented out, with a note on what it does.
2. Uncomment and fill in the block you have keys for:

| You want | Uncomment | Then |
|---|---|---|
| AI-written recommendations | `TOKENFACTORY_BASE_URL`, `TOKENFACTORY_API_KEY` (model names have sensible defaults) | restart |
| AI vision vote on photos | the key above, plus `DUAL_DETECTOR=on` and `DUAL_MODE=vision_primary` | restart |
| Real Evolus tasks | `EVOLUS_MODE=rest`, `EVOLUS_API_KEY`, `EVOLUS_APPLICATION_ID`, `EVOLUS_WORKFLOW_ID` (switch the workflow on in Evolus) | restart; approving now takes minutes, the page shows progress |
| A real database | `USE_IN_MEMORY_DB=false`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`; run `backend/supabase/migrations/001_init.sql` once | restart |

3. With Docker: `docker compose up --build` again (compose reads `backend/.env` automatically if it exists). Without Docker: see the next section.
4. Check `http://localhost:8000/health`: it shows the Evolus mode, whether the database is in memory, which model writes the text (`"offline"` means no key was found),
   and whether the dual detector is on. Live checks: `cd backend && PYTHONPATH=. python scripts/tokenfactory_check.py`.

Keys are only ever read on the server. They are never put in an image, a log or the browser. **Never commit `backend/.env`.**

### Run without Docker

```bash
# backend (Python 3.11)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU-only torch keeps it small
pip install -r requirements.txt
USE_IN_MEMORY_DB=true uvicorn app.main:create_app --factory --port 8000

# frontend (Node 20+), in another terminal
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev                           # http://localhost:3000
```

---

## Sample inputs in `data/`

Everything below ships in the repository, works with no keys, and is what the tests and the demo use.

| Folder | What it is | What you should see |
|---|---|---|
| [`data/test/`](data/test) + [`data/test_cases.csv`](data/test_cases.csv) | 20 generated machine-part photos with their true labels (6 normal, 5 bearing wear, 5 corrosion, 4 structural crack) | `test_02.png` bearing wear, `test_04.png` structural crack, `test_05.png` corrosion, `test_01.png` normal, each with a manual citation |
| [`data/test_reports/`](data/test_reports) | 4 sensor reports (`s01` to `s04`: CSV, JSON, TXT) and 5 documents (`d01` to `d05`: TXT and PDF), expected results in `manifest.csv` | `s01` bearing wear, `s02` corrosion, `s03` structural crack, `s04` normal; `d01`, `d02`, `d04` the matching defect |
| [`data/raw/`](data/raw) | the 5 maintenance manuals (PDF) the search runs on | every citation points into one of these (manual name and page) |
| [`data/retrieval_eval.csv`](data/retrieval_eval.csv) | 15 questions with the manual page that should answer each | used by `evaluate.py` |

Two samples are **meant** to end up flagged for a person: `d03_clean_audit.txt` (a clean audit: low detection confidence) and `d05_scanned_crack_report.pdf`
(a scanned image of a report: reading it needs the optional Tesseract OCR program, which is not installed in the Docker image, so the app says it cannot read the page
instead of guessing). Real photos used for our validation are not in the repository.

---

## Tests

```bash
cd backend && python -m pytest -q                 # about 420 tests; no network, no keys (models and Evolus are mocked)
cd frontend && npx tsc --noEmit                   # type check
python -m pytest vk_tests -q                      # vision and knowledge package tests (pip install -r requirements-vision.txt first)
```

Live checks that spend the shared quota (you need keys), each with a description at its top: `backend/scripts/tokenfactory_check.py`, `backend/scripts/latency_check.py`,
`backend/scripts/voting_compare.py`, `backend/scripts/real_photo_check.py`. A smoke test of a running server: `cd backend && python smoke_test.py`.

---

## Project structure

```
backend/                FastAPI service
  app/                  main.py (API), service.py (case lifecycle), evolus.py (mock + real client, reply parser), dual_detector.py (CNN + vision models),
                        feedback.py (approved-photo store), db.py (memory / Supabase), models.py, config.py, security.py
  agent.py, core.py     the AI agent (retrieve, write, verify citation, fallbacks) and the model clients
  scripts/              live checks and experiments (see Tests)    supabase/  database migration    tests/  the test suite
frontend/               Next.js web app (login, dashboard, upload, case page with progress and approval)
vision/                 the CNN, sensor and document analysers, training code, thresholds, weights
knowledge/              reading and chunking the manuals, search, citation helpers
data/                   sample inputs (above); photos, answers and experiments you generate stay git-ignored
docs/                   evolus_integration.md, model_comparison.md, dual_detector_results.md, voting_comparison.md, vision-knowledge.md, screenshots
docker/, docker-compose.yml, backend/Dockerfile, frontend/Dockerfile      the one-command setup
scripts/                retrain_from_feedback.py (trains a candidate CNN from approved photos; never overwrites the live one)
Last/                   an old copy of early backend files, not used
```

---

## Known limits (please read)

- **The AI service is shared and slow.** One Token Factory key is shared by the whole team: **30 requests per minute**, 8 at a time. Qwen3.8-Flash-Next takes about **32 s per photo** (10 to 58 s),
  Gemma about 8 s, and a photo with the vote on takes about 30 s up to a minute. The Evolus agent took **245 s** on one run and about **19 minutes** on another, so Evolus is waited for in the background
  (up to 10 minutes), the page shows progress, and a timed-out run is resumed by Retry.
- **Small tests.** On 40 hand-labelled Google photos the recommended setting answered 36 automatically and correctly, 1 silently wrong and sent 3 to a person, **but we tuned the prompt and one rule on those
  same photos**. On 15 fresh photos it got 13 correct, 1 silently wrong, 1 to a person. Differences of a photo or two are noise. Details and caveats are in `docs/voting_comparison.md`.
- **The CNN is weak on real photos** (right on 14 of 40) because it was trained on generated drawings; that is why the vision models decide and the CNN is advisory.
- Mock-mode recommendations are a template copied from the manual, not an AI judgement. For a part that looks fine, the citation is a general page and not meaningful.
- The in-memory database loses everything on restart. The demo login is not authentication: anyone who reaches the API can approve or reject. There is one backend process; the "one Evolus run per case" guard lives in it.
- Scanned PDFs need the optional OCR program. `docker compose up --build` was tested from a clean clone (no `backend/.env`, no keys) on an Apple-silicon Mac with Docker Desktop 29.5: both containers became healthy, and a photo, a sensor report and a PDF each went through detection, citation, approval and a mock Evolus task. Intel, Linux and Windows machines were not tried.

## Security notes

Never commit `backend/.env` or any key; keys are server-side only and are never baked into an image (`.dockerignore` excludes every `.env`). Uploads are limited by type and size. The Evolus step is
claimed once per case and its run id is kept, so an approval is never lost or duplicated by a retry.

## Screenshots (from an earlier build; the layout is the same)

[Login](docs/screenshots/01-login.png) · [Dashboard](docs/screenshots/02-dashboard.png) · [Upload](docs/screenshots/03-upload-page.png) · [Case details](docs/screenshots/04-case-details.png) ·
[Detection results](docs/screenshots/05-detection-results.png) · [Approval](docs/screenshots/06-approval-workflow.png) · [Evolus task](docs/screenshots/07-evolus-task.png)
