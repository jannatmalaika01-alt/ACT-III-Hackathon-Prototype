"""Evolus task creation.

Two implementations behind one tiny interface:
  * MockEvolusClient - for development and tests (can simulate failures)
  * McpEvolusClient  - placeholder: Karim / whoever owns the MCP side fills it in
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol

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


class McpEvolusClient:
    """TODO(Karim): call Evolus' 'create task' tool through the MCP connection.

    Use build_task_payload(case) for the fields, return the task id Evolus gives
    back, and raise EvolusError on ANY failure (timeout, auth, tool error) so the
    case is marked task_failed instead of crashing the request.
    """

    def create_task(self, case: Dict[str, Any]) -> str:
        raise EvolusError("McpEvolusClient is not implemented yet (see app/evolus.py)")


def build_evolus_client(settings: Settings) -> EvolusClient:
    if settings.evolus_mode == "mcp":
        return McpEvolusClient()
    return MockEvolusClient(fail_first_n=settings.evolus_mock_fail_first_n)
