"""Status machine + request schemas (the 'contract' for the API)."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class CaseStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TASK_CREATED = "task_created"
    TASK_FAILED = "task_failed"


# Which status may follow which. Anything not listed here is rejected with 409.
#
#   pending ──approve──> approved ──ok──> task_created
#      │                     └──error──> task_failed ──retry──> task_created
#      └──reject──> rejected                  └─(retry fails)─> task_failed
ALLOWED_TRANSITIONS: Dict[CaseStatus, Set[CaseStatus]] = {
    CaseStatus.PENDING: {CaseStatus.APPROVED, CaseStatus.REJECTED},
    CaseStatus.APPROVED: {CaseStatus.TASK_CREATED, CaseStatus.TASK_FAILED},
    CaseStatus.TASK_FAILED: {CaseStatus.TASK_CREATED, CaseStatus.TASK_FAILED},
    CaseStatus.REJECTED: set(),
    CaseStatus.TASK_CREATED: set(),
}

Severity = Literal["low", "medium", "high", "critical"]


class DiagnosisIn(BaseModel):
    """What the AI pipeline sends once it has diagnosed a case.

    NOTE: align these field names with the team's agreed JSON contract (Karim).
    A recommendation without a manual + page citation is not allowed.
    """

    defect: str = Field(min_length=1)
    severity: Severity
    explanation: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    source_manual: str = Field(min_length=1)
    source_page: int = Field(gt=0)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    raw: Optional[Dict[str, Any]] = None  # full model output, stored for debugging


class ApproveIn(BaseModel):
    decided_by: str = Field(min_length=1)
    reason: Optional[str] = None


class RejectIn(BaseModel):
    decided_by: str = Field(min_length=1)
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a rejection needs a reason")
        return v.strip()


class WebhookIn(BaseModel):
    """Status update pushed to us (e.g. by Evolus) about a case's task."""

    event_id: str = Field(min_length=1, max_length=200)  # unique per delivery -> de-duplication
    case_id: UUID
    status: Literal["task_created", "task_failed"]
    evolus_task_id: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None

    @model_validator(mode="after")
    def check_fields_for_status(self) -> "WebhookIn":
        if self.status == "task_created" and not self.evolus_task_id:
            raise ValueError("evolus_task_id is required when status is task_created")
        if self.status == "task_failed" and not (self.error and self.error.strip()):
            raise ValueError("error is required when status is task_failed")
        return self
