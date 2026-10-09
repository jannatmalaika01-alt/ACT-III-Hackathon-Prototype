"""FastAPI app.   Run:  uvicorn app.main:create_app --factory --reload"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .db import InMemoryRepo, SupabaseRepo
from .evolus import build_evolus_client
from .models import ApproveIn, CaseStatus, DiagnosisIn, RejectIn, WebhookIn
from .security import verify_signature
from .service import CaseNotFound, CaseNotReady, CaseService, InvalidTransition

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "application/pdf", "text/plain", "text/csv", "application/json",
}


def build_repo(settings: Settings):
    if settings.use_memory_db:
        return InMemoryRepo()
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set. Fill in .env, "
            "or set USE_IN_MEMORY_DB=true to run without Supabase."
        )
    return SupabaseRepo(settings.supabase_url, settings.supabase_key, settings.storage_bucket)


def create_app(settings: Optional[Settings] = None, repo=None, evolus=None) -> FastAPI:
    settings = settings or get_settings()
    repo = repo if repo is not None else build_repo(settings)
    evolus = evolus if evolus is not None else build_evolus_client(settings)
    service = CaseService(repo, evolus)

    app = FastAPI(title="AI Factory Maintenance Agent - Backend", version="0.1.0")
    app.state.settings, app.state.repo, app.state.evolus, app.state.service = settings, repo, evolus, service

    # ---- errors from the service layer -> HTTP ----
    @app.exception_handler(CaseNotFound)
    async def _not_found(_: Request, exc: CaseNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidTransition)
    async def _conflict(_: Request, exc: InvalidTransition):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(CaseNotReady)
    async def _not_ready(_: Request, exc: CaseNotReady):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # ---- routes ----
    @app.get("/health")
    def health():
        return {"status": "ok", "evolus_mode": settings.evolus_mode, "in_memory_db": settings.use_memory_db}

    @app.post("/cases", status_code=201)
    async def upload_case(file: UploadFile = File(...), uploaded_by: str = Form("unknown")):
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(415, f"Unsupported file type: {file.content_type}")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(400, "Empty file")
        return await run_in_threadpool(
            service.create_case, data, file.filename or "upload", file.content_type, uploaded_by
        )

    @app.get("/cases")
    def list_cases(status: Optional[CaseStatus] = None,
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        items = repo.list_cases(status.value if status else None, limit, offset)
        return {"items": items, "limit": limit, "offset": offset}

    @app.get("/cases/{case_id}")
    def get_case(case_id: UUID):
        return service.get_case_detail(str(case_id))

    @app.post("/cases/{case_id}/diagnosis")
    def attach_diagnosis(case_id: UUID, body: DiagnosisIn):
        return service.attach_diagnosis(str(case_id), body)

    @app.post("/cases/{case_id}/approve")
    def approve_case(case_id: UUID, body: ApproveIn):
        # 200 even if the Evolus task failed: the approval itself succeeded.
        # Check the returned `status` ('task_created' or 'task_failed').
        return service.approve(str(case_id), body)

    @app.post("/cases/{case_id}/reject")
    def reject_case(case_id: UUID, body: RejectIn):
        return service.reject(str(case_id), body)

    @app.post("/cases/{case_id}/retry-task")
    def retry_task(case_id: UUID):
        return service.retry_task(str(case_id))

    @app.post("/webhooks/case-status")
    async def case_status_webhook(request: Request,
                                  x_webhook_signature: Optional[str] = Header(default=None)):
        raw = await request.body()
        if not verify_signature(settings.webhook_secret, raw, x_webhook_signature):
            raise HTTPException(401, "Invalid or missing webhook signature")
        try:
            body = WebhookIn.model_validate_json(raw)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_context=False, include_input=False))
        return await run_in_threadpool(service.apply_webhook, body)

    return app
