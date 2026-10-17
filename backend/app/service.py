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
import threading
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
    def __init__(self, repo, evolus: EvolusClient, background: bool = False) -> None:
        self.repo = repo
        self.evolus = evolus
        # background=True (EVOLUS_MODE=rest): approving returns at once and the Evolus step runs in a thread, because the
        # agent can take minutes. At most ONE Evolus run per case at a time (see _begin).
        self.background = background
        self._running: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._threads: list = []

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
        return self._launch(case_id)

    def reject(self, case_id: str, body: RejectIn) -> Dict[str, Any]:
        case = self._get(case_id)
        if case["status"] != CaseStatus.PENDING.value:
            raise InvalidTransition(f"Only pending cases can be rejected (this one is '{case['status']}')")
        row = self._move(case_id, CaseStatus.PENDING, CaseStatus.REJECTED,
                         note=f"rejected by {body.decided_by}: {body.reason}")
        self.repo.add_approval(case_id, "rejected", body.decided_by, body.reason)
        return row

    # ---------- Evolus task ----------
    def _begin(self, case_id: str) -> None:
        """Claim the case for ONE Evolus run. A second claim while the first is running is refused (no duplicate runs)."""
        with self._lock:
            if case_id in self._running:
                raise InvalidTransition("An Evolus run for this case is already in progress")
            self._running[case_id] = {"started_at": _now_iso(), "execution_id": None}

    def _end(self, case_id: str) -> None:
        with self._lock:
            self._running.pop(case_id, None)

    def _launch(self, case_id: str) -> Dict[str, Any]:
        """Run the Evolus step for an approved / failed case: in a background thread (rest mode) or right here."""
        self._begin(case_id)
        if self.background:
            thread = threading.Thread(target=self._run_guarded, args=(case_id,), name=f"evolus-{case_id[:8]}", daemon=True)
            self._threads.append(thread)
            thread.start()
            return self._get(case_id)  # still 'approved' (or 'task_failed' on a retry); the page shows the run as in progress
        try:
            return self._create_task(case_id)
        finally:
            self._end(case_id)

    def _run_guarded(self, case_id: str) -> None:
        try:
            self._create_task(case_id)
        except Exception:  # a background thread must never die silently or leave the case claimed
            log.exception("Evolus step crashed for case %s", case_id)
        finally:
            self._end(case_id)

    def wait_for_background(self, timeout: float = 10.0) -> None:
        """Wait for background Evolus steps (used by tests and on shutdown)."""
        for t in list(self._threads):
            t.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    def _remember_run(self, case_id: str, execution_id: str) -> None:
        """Save the execution id as soon as Evolus has accepted the run, so a retry (even after a restart) can resume it."""
        with self._lock:
            if case_id in self._running:
                self._running[case_id]["execution_id"] = execution_id
        case = self.repo.get_case(case_id) or {}
        started = (self._running.get(case_id) or {}).get("started_at") or _now_iso()
        raw = {**(case.get("diagnosis_raw") or {}), "evolus": {"execution_id": execution_id, "state": "running", "started_at": started}}
        self.repo.update_fields(case_id, {"diagnosis_raw": raw})

    def _create_task(self, case_id: str) -> Dict[str, Any]:
        """Case is 'approved' or 'task_failed'. Never raises for Evolus problems."""
        case = self._get(case_id)
        current = CaseStatus(case["status"])
        attempts = (case.get("task_attempts") or 0) + 1
        kwargs: Dict[str, Any] = {}
        if getattr(self.evolus, "supports_resume", False):
            previous = (case.get("diagnosis_raw") or {}).get("evolus") or {}
            kwargs = {"resume_execution_id": previous.get("execution_id") if previous.get("state") != "completed" else None,
                      "on_started": lambda eid: self._remember_run(case_id, eid)}
        try:
            task_id = self.evolus.create_task(case, **kwargs)
        except EvolusError as exc:
            return self._task_failed(case_id, current, attempts, str(exc), getattr(exc, "execution_id", None))
        except Exception as exc:  # anything unexpected must not lose the approval
            log.exception("unexpected error creating Evolus task for %s", case_id)
            return self._task_failed(case_id, current, attempts, f"unexpected error: {exc}")
        case = self._get(case_id)  # re-read: the run's id was saved into diagnosis_raw while we waited
        fields = {"evolus_task_id": str(task_id), "task_error": None, "task_attempts": attempts}
        note = f"Evolus task {task_id} created"
        result = getattr(task_id, "result", None)  # REST mode: the agent's raw text and parsed fields
        if result:
            fields["diagnosis_raw"] = {**(case.get("diagnosis_raw") or {}), "evolus": result}
            note += (f" ({'resumed an earlier run; ' if result.get('resumed') else ''}agent answered in "
                     f"{result.get('evolus_seconds') or result.get('duration_seconds')} s; "
                     f"needs_human_review={result['parsed']['needs_human_review']})")
        return self._move(case_id, current, CaseStatus.TASK_CREATED, fields, note=note)

    def _task_failed(self, case_id: str, current: CaseStatus, attempts: int, error: str,
                     execution_id: Optional[str] = None) -> Dict[str, Any]:
        fields: Dict[str, Any] = {"task_error": error, "task_attempts": attempts}
        if execution_id:  # keep the id of the run that did not finish, so Retry can resume it
            case = self._get(case_id)
            fields["diagnosis_raw"] = {**(case.get("diagnosis_raw") or {}),
                                       "evolus": {"execution_id": execution_id, "state": "failed", "error": error[:300],
                                                  "started_at": ((case.get("diagnosis_raw") or {}).get("evolus") or {}).get("started_at")}}
        return self._move(case_id, current, CaseStatus.TASK_FAILED, fields, note=f"task creation failed: {error}")

    def retry_task(self, case_id: str) -> Dict[str, Any]:
        case = self._get(case_id)
        running = case_id in self._running
        stalled = case["status"] == CaseStatus.APPROVED.value and not running  # e.g. the server restarted mid-run
        if case["status"] != CaseStatus.TASK_FAILED.value and not stalled:
            if running:
                raise InvalidTransition("An Evolus run for this case is already in progress")
            raise InvalidTransition(f"Only 'task_failed' cases can be retried (this one is '{case['status']}')")
        return self._launch(case_id)  # refuses a second run while one is in progress; resumes the old run when it can

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
        run = self._running.get(case_id)
        saved = (case.get("diagnosis_raw") or {}).get("evolus") or {}
        evolus = {"running": run is not None, "started_at": (run or {}).get("started_at") or saved.get("started_at"),
                  "execution_id": (run or {}).get("execution_id") or saved.get("execution_id")}
        return {**case, "evolus": evolus, "approvals": self.repo.list_approvals(case_id), "events": self.repo.list_events(case_id)}
