"""Evolus task creation.

Two implementations behind one tiny interface:
  * MockEvolusClient - for development and tests (can simulate failures)
  * RestEvolusClient - the real Evolus REST API (EVOLUS_MODE=rest)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol

import httpx

from .config import Settings


class EvolusError(Exception):
    """Raised when a task could not be created in Evolus."""


class EvolusClient(Protocol):
    def create_task(self, case: Dict[str, Any]) -> str:
        """Create a task for an approved case and return Evolus's task id."""


SEVERITY_TO_PRIORITY = {"low": "low", "medium": "medium", "high": "high", "critical": "urgent"}


def build_task_payload(case: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an approved case into the fields a maintenance task needs."""
    return {
        "title": f"[{case.get('severity') or 'n/a'}] {case.get('defect')}",
        "description": (
            f"{case.get('explanation')}\n\n"
            f"Recommended action: {case.get('recommended_action')}\n"
            f"Source: {case.get('source_manual')}, page {case.get('source_page')}\n"
            f"Case ID: {case.get('id')}"
        ),
        "priority": SEVERITY_TO_PRIORITY.get(case.get("severity") or "", "medium"),
        "external_ref": case.get("id"),
    }


class MockEvolusClient:
    def __init__(self, fail_first_n: int = 0) -> None:
        self._fail_remaining = fail_first_n
        self._counter = 0
        self.created: List[Dict[str, Any]] = []

    def fail_next(self, n: int = 1) -> None:
        """Make the next n create_task() calls fail (used by tests)."""
        self._fail_remaining = n

    def create_task(self, case: Dict[str, Any]) -> str:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise EvolusError("Evolus unavailable (simulated failure)")
        self._counter += 1
        self.created.append(build_task_payload(case))
        return f"MOCK-TASK-{self._counter:04d}"


class RestEvolusClient:
    """Runs an Evolus workflow for an approved case (EVOLUS_MODE=rest).

    POST {base}/workflows/{workflow_id}/execute, headers X-ApiKey + X-Application-Id, multipart/form-data
    with `input` (text) and `parameters` (a flat JSON object). Evolus answers 202 Accepted with
    {jobId, executionId, enqueuedAt, traceId} and repeats the id in the X-Workflow-Execution-Id header
    (per https://evolus.ai/developers). The workflow must be switched on, otherwise Evolus answers 409.

    Any failure (config, network, auth, 4xx/5xx, no id in the answer) raises EvolusError, so the case is
    marked task_failed and can be retried. The API key is never put in an error message."""

    ID_KEYS = ("executionId", "execution_id", "jobId", "job_id", "taskId", "task_id", "id")

    def __init__(self, base_url: str, api_key: str, application_id: str, workflow_id: str,
                 timeout: float = 30.0) -> None:
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.application_id, self.workflow_id, self.timeout = application_id, workflow_id, timeout

    def create_task(self, case: Dict[str, Any]) -> str:
        missing = [n for n, v in (("EVOLUS_API_KEY", self.api_key), ("EVOLUS_APPLICATION_ID", self.application_id),
                                  ("EVOLUS_WORKFLOW_ID", self.workflow_id)) if not v]
        if missing:
            raise EvolusError("Evolus is not configured: set " + ", ".join(missing))
        payload = build_task_payload(case)
        # files={...: (None, value)} makes httpx send multipart/form-data text fields
        fields = {"input": (None, payload["description"]), "parameters": (None, json.dumps(payload))}
        try:
            resp = httpx.post(f"{self.base_url}/workflows/{self.workflow_id}/execute",
                              headers={"X-ApiKey": self.api_key, "X-Application-Id": self.application_id},
                              files=fields, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise EvolusError(f"Evolus request failed: {type(exc).__name__}") from None
        if resp.status_code == 409:
            raise EvolusError("Evolus refused the run (409): the workflow is probably switched off")
        if resp.status_code in (401, 403):
            raise EvolusError(f"Evolus rejected the credentials (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            raise EvolusError(f"Evolus returned HTTP {resp.status_code}")
        task_id = self._find_id(resp)
        if not task_id:
            raise EvolusError("Evolus answered but no execution id was found in the response")
        return task_id

    def _find_id(self, resp) -> Optional[str]:
        header = resp.headers.get("X-Workflow-Execution-Id")
        if header:
            return header.strip()
        try:
            body = resp.json()
        except ValueError:
            return None
        for obj in (body, body.get("data") if isinstance(body, dict) else None):
            if isinstance(obj, dict):
                for key in self.ID_KEYS:
                    if obj.get(key) not in (None, ""):
                        return str(obj[key])
        return None


class McpEvolusClient:
    """Not used: the team went with the REST API (EVOLUS_MODE=rest)."""

    def create_task(self, case: Dict[str, Any]) -> str:
        raise EvolusError("EVOLUS_MODE=mcp is not implemented; use EVOLUS_MODE=rest")


def build_evolus_client(settings: Settings) -> EvolusClient:
    if settings.evolus_mode == "rest":
        return RestEvolusClient(settings.evolus_base_url, settings.evolus_api_key,
                                settings.evolus_application_id, settings.evolus_workflow_id)
    if settings.evolus_mode == "mcp":
        return McpEvolusClient()
    return MockEvolusClient(fail_first_n=settings.evolus_mock_fail_first_n)
