# Evolus integration (EVOLUS_MODE=rest)

When a person approves a case, the backend hands the defect to an Evolus workflow, waits for the Evolus agent's answer, parses it, and stores it
on the case. `EVOLUS_MODE=mock` (the default) is unchanged and needs no Evolus account. No keys appear on this page.

## The three Evolus building blocks used

1. **An agent with a manuals library.** The workflow step "Ask an agent" runs the **Factory Maintenance Agent**. It answers from the manuals
   uploaded to its Evolus library (in the live run it quoted `corrosion_control_guide.txt`, pages 2 to 4) and says when something is general practice and not from the library.
2. **An automation workflow.** `Start -> Ask an agent "Factory Maintenance Agent" -> Automation output`. The workflow must be switched on, otherwise Evolus answers 409.
3. **The API call.** The backend starts the workflow with one REST call and then checks its result with another.

## Verified request and response shapes

Base URL `https://api.evolus.ai/api/v1`; every call sends the headers `X-ApiKey` and `X-Application-Id`.

**Start a run** (verified live): `POST /workflows/{workflowId}/execute`, `multipart/form-data` with one text field `input`.
Answer: **HTTP 202**, JSON `{jobId, executionId, enqueuedAt, traceId}` and the header `X-Workflow-Execution-Id` (same id as `executionId`).

**Get the result** (verified live; the public docs at evolus.ai/developers name only `status`, the full reference sits inside a logged-in workspace, and there is no public OpenAPI file):
`GET /workflows/{workflowId}/executions/{executionId}` -> **HTTP 200**, JSON:

| Field | Seen in the live runs |
|---|---|
| `status` | `Running` while the agent works, `Completed` at the end (the docs also list `Pending` and `Failed`; `Failed` was not seen in a live run) |
| `output` | `null` while running; the agent's **text** (a string) when completed |
| `error` | `null` (a reason string is expected when the run fails) |
| `input` | exactly the text we sent |
| `stepOutputs`, `stepStates` | per-step results and status (step 1 = the agent: `done`, with its `durationMs`) |
| `totalStepsExecuted` | 2 when completed, 0 while the agent is still running |
| `requestedAt`, `completedAt` | UTC timestamps, so the real duration can be checked |
| `id`, `workflowId`, `applicationId`, `traceId`, `hangfireJobId`, `fileUrls`, `fileNames`, `parameters`, `consumption`, ... | present, not used |

**What we send as `input`** (built by `build_evolus_input`, real example from the live run):

```
Defect: corrosion
Location: location not specified in upload
Case ID: ed3870f0-d30d-4865-9ec0-f801a4a2ab9e
Severity proposed by our pipeline: medium
Detector results (each model separately, confidences are never averaged):
- CNN (small image model trained on drawings, low trust on real photos): structural_crack (confidence 0.82)
- Qwen3.8-Flash-Next: corrosion (confidence 0.95)
- Gemma-4-31B-IT: corrosion (confidence 0.98)
Votes: 2 of 3 say corrosion
CNN note (low trust): the CNN said structural_crack with confidence 0.82, but the vision models agree on corrosion.
Our pipeline's explanation: ...
Our pipeline's recommended action: ...
Manual cited by our pipeline: corrosion_control_guide.pdf, page 2
```

**What comes back**: free text with these fields (markdown such as `**defect:** ...` plus bullet and numbered lists):
`defect, severity, probable_cause, safety_precautions, recommended_steps, citation, confidence, needs_human_review, summary`.

## How the backend uses it

- `RestEvolusClient.create_task` (in `backend/app/evolus.py`): POST the `input`, accept 202, then GET the execution **every 5 seconds for at most 600 seconds**
  (`EVOLUS_POLL_INTERVAL`, `EVOLUS_POLL_TIMEOUT`; the default was 240 s until the 245 s run below showed it was too tight). A few transient errors while polling (network, 429, 5xx) are tolerated.
- **Approving does not block.** In rest mode (`EVOLUS_BACKGROUND=true`, the default) `POST /cases/{id}/approve` saves the approval and returns at once with status `approved`; the Evolus step runs
  in a background thread. The page keeps working and shows the run as in progress. With `EVOLUS_BACKGROUND=false` (or in mock mode) the step runs inside the request, as before.
- **The run is remembered as soon as it exists.** The moment Evolus accepts the run, its execution id and start time are saved in `diagnosis_raw.evolus` (`state: running`), so a retry can find it even after a server restart.
- **Retry resumes instead of restarting.** `POST /cases/{id}/retry-task` looks once at the earlier execution:
  - `Pending` or `Running`: keep waiting for **that** run (no second run is started);
  - `Completed`: use its answer (nothing is run again);
  - `Failed`, or unknown (404, error, unreadable status): start a new run (the old id is kept as `previous_execution_id`).
  A case that was left `approved` by a restart mid-run (nothing running) can be retried the same way.
- **No two runs for one case at the same time.** Starting a run claims the case in memory; a second approve or retry while the first is running is refused with HTTP 409
  ("An Evolus run for this case is already in progress"). The claim is always released when the run ends, even if it crashes. This guard lives in one backend process; with several backend processes
  the resume rule above still prevents a second run for the same unfinished execution, but two processes could in theory both start a first run.
- **Failures never crash the request.** A timeout, a `Failed` run, a 401/403/404, a refused start (409) or too many network errors raise `EvolusError`; the case becomes `task_failed` with the
  reason (and the id of the run that did not finish), and Retry resumes or restarts as described above.
- **The answer is stored on the case** in `diagnosis_raw.evolus` (an existing JSON column, no database change): `raw_text` (always kept), `parsed` (the nine fields), `notes`, `execution_id`, `status`,
  `resumed`, `previous_execution_id`, `duration_seconds` (this call) and `evolus_seconds` (Evolus's own clock, completedAt minus requestedAt). `citations` (the checked citations, see below), `citation_check` (counts) and `agent_needs_human_review` (what the agent itself said). `safety_precautions`, `recommended_steps` and `citation` are lists;
  `confidence` is a number when the agent gave one and otherwise `None` with its words in `confidence_text` (the live agent said "high"); `needs_human_review` is a boolean (if the reply does not say, it defaults to `true`).
  The case's `evolus_task_id` is the execution id. `GET /cases/{id}` also returns `evolus: {running, started_at, execution_id}`.

## What the page shows

- **While the agent works** (the case is `approved`, or the server says a run is in progress, including a retry): the "Evolus task" step spins and the right-hand card shows
  *"Evolus agent is preparing the maintenance task, this can take a few minutes"* with the elapsed time (counted from the server's start time, so a reload shows the right value). The rest of the page stays usable and the case
  is re-checked every 3 seconds. After 5 minutes it adds that the shared AI service is busy and that we keep waiting for up to 10 minutes.
- **If the wait runs out** the card turns into a calm message: *"Evolus is taking longer than expected. Your approval is saved and the run may still finish; Retry picks it up where it left off."* with a **Retry** button (other
  failures say the task could not be created this time). Retry is hidden while a run is in progress.
- Approving or retrying shows a short notice instead of waiting: *"Approved. The Evolus agent is preparing the maintenance task; this can take a few minutes."*

## Citation check: every quote is looked up in our manuals

The agent cites a document, a page and a quoted passage. It names the document `corrosion_control_guide.txt` where our file is `corrosion_control_guide.pdf`, and it sometimes changes a spelling inside a quote (for example
"pressurized" for the manual's "pressurised"). So after parsing the reply, `backend/app/citations.py` checks **every quote** against the text of the manuals in the backend's retrieval index (`data/raw`, the same index the diagnosis uses).

- **Where it looks:** the document with the same stem (`.txt` and `.pdf` count as the same) and the cited page first; then every other page of that document; then every page of every manual. If the text is somewhere other than
  where the agent said, the real document and page are stored and the difference is noted.
- **How it compares:** whitespace, case and punctuation are ignored. A quote can be an exact stretch of a page (after that), or a stretch of about the same length that is at least **0.9 similar** (character similarity, so a changed spelling or a missing or extra word still matches).
- **Three results per citation:**

| Status | Meaning | What is stored |
|---|---|---|
| `verified` | the quote is in the manual | the agent's quote, the real `.pdf` name and page |
| `corrected` ("corrected to source") | no exact match but a passage is at least 0.9 similar | `quote` is REPLACED by the manual's exact text (punctuation included); the agent's wording is kept in `agent_quote` |
| `unverified` | nothing is that close, or the citation has no quote at all | the agent's text is kept, `note` says why, and `similarity` is the closest match found (0 if nothing came close) |

- **Human review:** if any quote is `unverified`, `parsed.needs_human_review` becomes `true` even when the agent said `false` (its own answer is kept in `agent_needs_human_review`), a note is added to the result, and the case's activity line says
  "N citation(s) NOT verified against the manuals". `verified` and `corrected` citations do not change the agent's own flag. If the manuals cannot be loaded, every citation is left `unverified` (the check fails closed) and the task is still created.
- **Stored on the case** in `diagnosis_raw.evolus.citations` (list of `{document, page, quote, status, similarity, agent_quote?, claimed?, note?, raw}`) with the counts in `citation_check`.
- **On the case page** the Evolus result card lists each citation with its status label (verified / corrected to source / unverified), shows what the agent had written for a corrected one, and, if any quote is unverified, a red warning in the card, a banner at the top of the page and
  "needs human review (citation check)" on the review label. Results stored before the check existed show their citations marked "not checked".
- **Real data:** the three quotes of the live run above (`dbdee6ac-33ca-f111-bf5b-6045bd9ce552`) all come out `verified` against the real index (they are in the tests, together with the "pressurized" spelling, which comes out `corrected`, and an invented quote, which comes out `unverified`).

What this does **not** do: it checks that the words exist in our manuals, not that the passage supports the agent's advice. A very short or generic quote can match somewhere by chance. The agent's library holds its own `.txt` versions of the manuals, so a re-flowed or edited
version there could show up as `unverified` even though the advice is fine; the review flag is deliberately cautious. A quote that joins two separate places on a page will not match.

## Timings (measured, shared endpoint)

| Run | Requested | Completed | Duration |
|---|---|---|---|
| `dbdee6ac-33ca-f111-bf5b-6045bd9ce552` (our end-to-end test) | 14:04:36 UTC | 14:08:41 UTC | **245 s** (Evolus's own timestamps) |
| `f0a26733-32ca-f111-bf5b-6045bd9ce552` (the earlier example) | 13:53:59 UTC | 14:12:38 UTC | **about 18.6 minutes** (it stayed `Running` with 0 steps for a long time) |

So "about 2 minutes" is the best case, not the rule. The first default of 240 s was too tight: the end-to-end run above needed 245 s by Evolus's clock (it only passed because the test machine went to sleep
and the client's clock does not count sleep, so it read 155.5 s). **The default is now 600 s** (`EVOLUS_POLL_TIMEOUT`, in code and `.env.example`). Even that does not cover a run like the 19-minute one, which is why a timed-out
case keeps its execution id and Retry resumes it: the run is not lost, and no second run is paid for.

## Proof of the end-to-end test

A real case (`img_18`, a corrosion photo) went through the live pipeline: upload, `/analyze` (CNN, Qwen3.8-Flash-Next and Gemma-4-31B-IT, mode `vision_primary`; the diagnosis text was
written by the backup writer DeepSeek-V4-Flash, so the primary writer did not give a usable answer), then **approve with `EVOLUS_MODE=rest`**. Result: case status `task_created`, `evolus_task_id` =
`dbdee6ac-33ca-f111-bf5b-6045bd9ce552`, Evolus status `Completed`, 2 steps done, parsed answer: severity `medium`, 2 safety precautions, 5 recommended steps, 3 citations from the manuals library,
confidence "high", `needs_human_review` false, and a summary that treats the CNN's `structural_crack` as low-trust. The agent's real reply is a test fixture
(`REAL_REPLY` in `backend/tests/test_evolus_parser.py`).

## How the resume and the guard were checked

Unit tests with mocked HTTP (`backend/tests/test_evolus_resume.py`) cover: a Pending or Running run is resumed without a second POST, a Completed run is only read, a Failed or unknown run is replaced by a new one,
the id is saved mid-run, a stalled `approved` case can be retried, a second approve or retry during a run gets 409 (also with two real threads), and the claim is released after a failure or a crash. I also made sure these tests fail
when the guard or the resume logic is switched off. In addition, the real backend was run over HTTP against a small fake Evolus server whose run lasts 7 s with a 4 s wait limit: approve returned in under 0.01 s, a retry during the
run got 409, the wait timed out at 4 s and kept the id, the retry returned at once and resumed the **same** run, which then completed (`task_created`, `resumed: true`), and the fake server saw exactly **one** POST. The real Evolus
service was not used for this check, so the resume against the live API is verified only through the same GET endpoint used in the live end-to-end run above.

## What is not verified

- A `Failed` run on the live service: the shape of its `error` field is assumed from the docs and tolerated, not observed.
- The webhook path (`callbackUrl`) is not used.
- The new progress and error states on the case page were checked with the type checker and the linter and through the API states they read, but not by looking at the page in a browser.
- Timing under load: the two live runs above are all the data there is.
