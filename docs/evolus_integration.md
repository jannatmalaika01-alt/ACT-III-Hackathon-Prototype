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

- `RestEvolusClient.create_task` (in `backend/app/evolus.py`): POST the `input`, accept 202, then GET the execution **every 5 seconds for at most 240 seconds**
  (`EVOLUS_POLL_INTERVAL`, `EVOLUS_POLL_TIMEOUT`). A few transient errors while polling (network, 429, 5xx) are tolerated.
- **Failures never crash the request.** A timeout, a `Failed` run, a 401/403/404, a refused start (409) or too many network errors raise `EvolusError`; the case becomes
  `task_failed` with the reason, and `POST /cases/{id}/retry-task` tries again. A retry starts a **new** run (it does not resume the old one); the timeout message names
  the old execution id so it can be looked at in Evolus.
- **The answer is stored on the case** in `diagnosis_raw.evolus` (an existing JSON column, no database change): `raw_text` (always kept), `parsed` (the nine fields),
  `notes`, `execution_id`, `status`, `duration_seconds`. `safety_precautions`, `recommended_steps` and `citation` are lists; `confidence` is a number when the agent gave one
  and otherwise `None` with its words in `confidence_text` (the live agent said "high"); `needs_human_review` is a boolean (if the reply does not say, it defaults to `true`).
  The case's `evolus_task_id` is the execution id.
- Approving waits for all of this, so the HTTP request can take minutes. The frontend dev proxy timeout was raised to 300 s (`experimental.proxyTimeout` in `frontend/next.config.ts`).

## Timings (measured, shared endpoint)

| Run | Requested | Completed | Duration |
|---|---|---|---|
| `dbdee6ac-33ca-f111-bf5b-6045bd9ce552` (our end-to-end test) | 14:04:36 UTC | 14:08:41 UTC | **245 s** (Evolus's own timestamps) |
| `f0a26733-32ca-f111-bf5b-6045bd9ce552` (the earlier example) | 13:53:59 UTC | 14:12:38 UTC | **about 18.6 minutes** (it stayed `Running` with 0 steps for a long time) |

So "about 2 minutes" is the best case, not the rule. **The default 240 s timeout is too tight for this endpoint**: the end-to-end run above needed 245 s by Evolus's clock.
It only passed because the test machine went to sleep while waiting and the client's clock (which does not count sleep) read 155.5 s. For a demo set `EVOLUS_POLL_TIMEOUT` higher (for example 600)
and start the approval early.

## Proof of the end-to-end test

A real case (`img_18`, a corrosion photo) went through the live pipeline: upload, `/analyze` (CNN, Qwen3.8-Flash-Next and Gemma-4-31B-IT, mode `vision_primary`; the diagnosis text was
written by the backup writer DeepSeek-V4-Flash, so the primary writer did not give a usable answer), then **approve with `EVOLUS_MODE=rest`**. Result: case status `task_created`, `evolus_task_id` =
`dbdee6ac-33ca-f111-bf5b-6045bd9ce552`, Evolus status `Completed`, 2 steps done, parsed answer: severity `medium`, 2 safety precautions, 5 recommended steps, 3 citations from the manuals library,
confidence "high", `needs_human_review` false, and a summary that treats the CNN's `structural_crack` as low-trust. The agent's real reply is a test fixture
(`REAL_REPLY` in `backend/tests/test_evolus_parser.py`).

## What is not verified

- A `Failed` run on the live service: the shape of its `error` field is assumed from the docs and tolerated, not observed.
- The webhook path (`callbackUrl`) is not used.
- Duplicate runs: a retry after a timeout starts another run while the first may still finish.
- Timing under load: the two runs above are all the data there is.
