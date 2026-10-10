# AI Factory Maintenance Agent - Backend (FastAPI + Supabase/PostgreSQL)

Handles: case upload -> diagnosis storage -> human approve/reject -> Evolus task creation,
plus status webhooks, error handling and tests.

```
pending --approve--> approved --ok--> task_created
   |                     \--error--> task_failed --retry--> task_created
   \--reject--> rejected
```
A human decision is saved first and never undone. If Evolus fails, the case becomes
`task_failed` (error saved) and can be retried with `POST /cases/{id}/retry-task`.

## 1. Set up Supabase (once)
1. Create a project at supabase.com.
2. Dashboard -> SQL Editor -> paste `supabase/migrations/001_init.sql` -> Run.
   (Creates tables `cases`, `approvals`, `case_events`, the private storage bucket `case-uploads`.)
3. Project Settings -> API: copy the project URL and the **service_role** key.

## 2. Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, WEBHOOK_SECRET
uvicorn app.main:create_app --factory --reload --port 8000
```
Docs UI: http://localhost:8000/docs
No Supabase yet? Set `USE_IN_MEMORY_DB=true` in `.env` (data is lost on restart).

### Check the database is set up right
```bash
python scripts/check_db.py     # tables + columns, private bucket 'case-uploads', service_role key
```
`001_init.sql` is safe to run more than once (it never drops or renames anything).

## 3. Tests
```bash
python -m pytest -q                      # 44 tests, no Supabase needed
PYTHONPATH=. python scripts/e2e_smoke.py # full path against the running server
```
Run the server with `EVOLUS_MOCK_FAIL_FIRST_N=1` to make the smoke test exercise failed-task -> retry.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/cases` | upload file (multipart: `file`, `uploaded_by`) -> new `pending` case |
| POST | `/cases/{id}/diagnosis` | AI pipeline stores diagnosis (+ manual & page citation required) |
| POST | `/cases/{id}/approve` | human approves -> creates Evolus task (check returned `status`) |
| POST | `/cases/{id}/reject` | human rejects (reason required) |
| POST | `/cases/{id}/retry-task` | retry after `task_failed` |
| GET | `/cases`, `/cases/{id}` | list (filter `?status=`) / detail with approvals + events |
| POST | `/webhooks/case-status` | status update from Evolus (HMAC-signed) |

## Webhook
Header `X-Webhook-Signature: sha256=<HMAC-SHA256 of the raw body using WEBHOOK_SECRET>`.
Body: `{"event_id": "...", "case_id": "<uuid>", "status": "task_created"|"task_failed",
"evolus_task_id": "...", "error": "..."}`
`event_id` must be unique per delivery - repeats are ignored (`result: "duplicate"`).
Sign from a shell: `printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$WEBHOOK_SECRET"`

## For Karim - integration points
- `app/models.py::DiagnosisIn` - align field names with the team JSON contract.
- `app/evolus.py::McpEvolusClient` - implement the real Evolus MCP call (set `EVOLUS_MODE=mcp`).
- No login yet: add auth (e.g. Supabase JWT check) before anyone outside the team can reach approve/reject.
- Approve calls Evolus inside the request; fine for a demo, move to a background job if it gets slow.

## Hooking up the AI agent (`/diagnose`)
`app/diagnosis_adapter.py::normalize_agent_output()` cleans the agent's answer ("Severe" -> `high`,
`87` -> `0.87`, `"p. 14"` -> `14`) and reports a clear 422 if it can't (unknown severity, missing
manual/page) instead of letting PostgreSQL throw a 500. Then store it with `service.attach_diagnosis()`
so the "only while pending" and "must cite a manual page" rules still apply.

```python
@app.post("/cases/{case_id}/diagnose")
def diagnose_case(case_id: UUID, body: DiagnoseIn):          # plain `def`, not `async def`
    case = service.get_case_detail(str(case_id))              # 404 if the case doesn't exist
    if case["status"] != "pending":                           # check BEFORE spending a Groq call
        raise HTTPException(409, "Only pending cases can be diagnosed")
    raw = run_agent(body)                                     # agent.py: Groq + LangGraph + Chroma
    try:
        diag = normalize_agent_output(raw, location=body.location, image_ref=body.image_ref)
    except DiagnosisFormatError as exc:
        raise HTTPException(422, {"agent_output_problems": exc.problems})
    return service.attach_diagnosis(str(case_id), diag)
```

## Secrets
`.env` is never committed and never sent in chat. Everyone copies `.env.example` to `.env` and
fills in their own values. The Supabase `service_role` key gives full database access: share it
privately (or invite the person to the Supabase project so they can read it themselves).
