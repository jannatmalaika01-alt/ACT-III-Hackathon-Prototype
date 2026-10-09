"""Business rules. The API layer stays thin; everything that matters lives here.

Key decisions:
  * A human's approval/rejection is recorded FIRST and is never undone, even if
    Evolus then fails. A failed task becomes status 'task_failed' and can be retried.
  * Approval requires a complete diagnosis WITH a manual + page citation.
  * Every status change goes through _move(): validated against ALLOWED_TRANSITIONS,
    applied as a compare-and-set, and logged in case_events.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .evolus import EvolusClient, EvolusError
from .models import ALLOWED_TRANSITIONS, ApproveIn, CaseStatus, DiagnosisIn, RejectIn, WebhookIn

log = logging.getLogger("service")

REQUIRED_FOR_APPROVAL = ("defect", "recommended_action", "source_manual", "source_page")


class CaseNotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


class CaseNotReady(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: Optional[str]) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base) or "upload"
    return base[:120]


class CaseService:
    def __init__(self, repo, evolus: EvolusClient) -> None:
        self.repo = repo
        self.evolus = evolus

    # ---------- helpers ----------
    def _get(self, case_id: str) -> Dict[str, Any]:
        case = self.repo.get_case(case_id)
        if case is None:
            raise CaseNotFound(f"Case {case_id} not found")
        return case

    def _move(self, case_id: str, from_status: CaseStatus, to_status: CaseStatus,
              fields: Optional[Dict[str, Any]] = None, note: Optional[str] = None,
              source: str = "api", event_id: Optional[str] = None) -> Dict[str, Any]:
        if to_status not in ALLOWED_TRANSITIONS[from_status]:
            raise InvalidTransition(f"Cannot go from '{from_status.value}' to '{to_status.value}'")
        row = self.repo.transition_case(case_id, from_status.value, to_status.value, fields or {})
        if row is None:
            raise InvalidTransition(f"Case {case_id} is no longer '{from_status.value}' (changed by someone else?)")
        self.repo.add_event(case_id, from_status.value, to_status.value, note, source, event_id)
        return row

    # ---------- upload ----------
    def create_case(self, data: bytes, filename: str, content_type: str, uploaded_by: str) -> Dict[str, Any]:
        case_id = str(uuid.uuid4())
        safe = _safe_filename(filename)
        path = f"{case_id}/{safe}"
        self.repo.upload_file(path, data, content_type)  # if this fails, no row is created
        case = self.repo.create_case(case_id, path, safe, content_type, uploaded_by)
        self.repo.add_event(case_id, None, "pending", "case created (file uploaded)", "api", None)
        return case

    # ---------- diagnosis from the AI pipeline ----------
    def attach_diagnosis(self, case_id: str, diag: DiagnosisIn) -> Dict[str, Any]:
        self._get(case_id)
        fields = {**diag.model_dump(exclude={"raw"}), "diagnosis_raw": diag.raw, "diagnosed_at": _now_iso()}
        row = self.repo.set_diagnosis(case_id, fields)
        if row is None:
            raise InvalidTransition("A diagnosis can only be attached while the case is 'pending'")
        self.repo.add_event(case_id, "pending", "pending", "diagnosis attached", "api", None)
        return row

    # ---------- human decisions ----------
    def approve(self, case_id: str, body: ApproveIn) -> Dict[str, Any]:
        case = self._get(case_id)
        if case["status"] != CaseStatus.PENDING.value:
            raise InvalidTransition(f"Only pending cases can be approved (this one is '{case['status']}')")
        missing = [f for f in REQUIRED_FOR_APPROVAL if not case.get(f)]
        if missing:
            raise CaseNotReady(f"Case has no complete diagnosis/citation yet (missing: {', '.join(missing)})")

        self._move(case_id, CaseStatus.PENDING, CaseStatus.APPROVED, note=f"approved by {body.decided_by}")
        self.repo.add_approval(case_id, "approved", body.decided_by, body.reason)
        return self._create_task(case_id)

    def reject(self, case_id: str, body: RejectIn) -> Dict[str, Any]:
        case = self._get(case_id)
        if case["status"] != CaseStatus.PENDING.value:
            raise InvalidTransition(f"Only pending cases can be rejected (this one is '{case['status']}')")
        row = self._move(case_id, CaseStatus.PENDING, CaseStatus.REJECTED,
                         note=f"rejected by {body.decided_by}: {body.reason}")
        self.repo.add_approval(case_id, "rejected", body.decided_by, body.reason)
        return row

    # ---------- Evolus task ----------
    def _create_task(self, case_id: str) -> Dict[str, Any]:
        """Case is 'approved' or 'task_failed'. Never raises for Evolus problems."""
        case = self._get(case_id)
        current = CaseStatus(case["status"])
        attempts = (case.get("task_attempts") or 0) + 1
        try:
            task_id = self.evolus.create_task(case)
        except EvolusError as exc:
            return self._task_failed(case_id, current, attempts, str(exc))
        except Exception as exc:  # anything unexpected must not lose the approval
            log.exception("unexpected error creating Evolus task for %s", case_id)
            return self._task_failed(case_id, current, attempts, f"unexpected error: {exc}")
        return self._move(case_id, current, CaseStatus.TASK_CREATED,
                          {"evolus_task_id": task_id, "task_error": None, "task_attempts": attempts},
                          note=f"Evolus task {task_id} created")

    def _task_failed(self, case_id: str, current: CaseStatus, attempts: int, error: str) -> Dict[str, Any]:
        return self._move(case_id, current, CaseStatus.TASK_FAILED,
                          {"task_error": error, "task_attempts": attempts},
                          note=f"task creation failed: {error}")

    def retry_task(self, case_id: str) -> Dict[str, Any]:
        case = self._get(case_id)
        if case["status"] != CaseStatus.TASK_FAILED.value:
            raise InvalidTransition(f"Only 'task_failed' cases can be retried (this one is '{case['status']}')")
        return self._create_task(case_id)

    # ---------- webhook ----------
    def apply_webhook(self, body: WebhookIn) -> Dict[str, Any]:
        case_id = str(body.case_id)
        case = self._get(case_id)

        if self.repo.event_exists(body.event_id):
            return {"result": "duplicate", "case": case}

        current = CaseStatus(case["status"])
        target = CaseStatus(body.status)
        if current == target == CaseStatus.TASK_CREATED:
            return {"result": "noop", "case": case}  # task already exists; never overwrite its id
        # (task_failed -> task_failed is allowed: Evolus is reporting a newer failure reason)

        if target == CaseStatus.TASK_CREATED:
            fields = {"evolus_task_id": body.evolus_task_id, "task_error": None}
        else:
            fields = {"task_error": body.error}

        row = self._move(case_id, current, target, fields,
                         note=body.note or f"webhook: {target.value}", source="webhook", event_id=body.event_id)
        return {"result": "applied", "case": row}

    # ---------- reads ----------
    def get_case_detail(self, case_id: str) -> Dict[str, Any]:
        case = self._get(case_id)
        return {**case, "approvals": self.repo.list_approvals(case_id), "events": self.repo.list_events(case_id)}
