"""Evolus task creation.

Two implementations behind one tiny interface:
  * MockEvolusClient - for development and tests (can simulate failures)
  * RestEvolusClient - the real Evolus REST API (EVOLUS_MODE=rest): starts the workflow, waits for the
    "Factory Maintenance Agent" to finish, and parses its answer (see docs/evolus_integration.md)
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import httpx

from . import citations
from .config import Settings


class EvolusError(Exception):
    """Raised when a task could not be created in Evolus. `execution_id` is set when a run was already started, so a
    retry can pick that run up again instead of starting a second one."""

    def __init__(self, message: str, execution_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.execution_id = execution_id


_sleep = time.sleep          # tests replace these two so they do not really wait
_monotonic = time.monotonic

POLL_INTERVAL = 5.0          # seconds between result checks
POLL_TIMEOUT = 600.0         # give up after this long (measured: 245 s on one run, about 19 minutes on another)


class TaskId(str):
    """An Evolus execution id (a plain string, so it works wherever a task id is expected) that also carries the
    agent's answer in `.result` (raw text + parsed fields). The mock client returns plain strings without a result."""

    result: Optional[Dict[str, Any]] = None


class EvolusClient(Protocol):
    def create_task(self, case: Dict[str, Any], resume_execution_id: Optional[str] = None, on_started=None) -> str:
        """Create a task for an approved case and return Evolus's task id. Clients with `supports_resume` pick an earlier
        execution up again when `resume_execution_id` is still Pending or Running, and call `on_started(execution_id)`
        as soon as an execution exists."""


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


# ---------------- what we send to the workflow ----------------

def _confidence(x: Optional[dict]) -> str:
    c = (x or {}).get("confidence")
    return "unknown" if c is None else f"{c}"


def build_evolus_input(case: Dict[str, Any]) -> str:
    """A short readable text block for the workflow's `input`: the defect, where it is, every detector's own answer
    (never averaged), the vote counts, the CNN note and the pipeline's diagnosis with its manual citation."""
    raw = case.get("diagnosis_raw") or {}
    report = raw.get("defect_report") or {}
    dual = raw.get("dual_detector") or report.get("detectors") or {}
    lines = [f"Defect: {report.get('defect_type') or case.get('defect') or 'unknown'}",
             f"Location: {report.get('location') or 'not specified'}",
             f"Case ID: {case.get('id')}"]
    if case.get("severity"):
        lines.append(f"Severity proposed by our pipeline: {case['severity']}")
    if dual:
        lines.append("Detector results (each model separately, confidences are never averaged):")
        cnn = dual.get("cnn") or {}
        if cnn:
            lines.append(f"- CNN (small image model trained on drawings, low trust on real photos): "
                         f"{cnn.get('label') or 'no answer'} (confidence {_confidence(cnn)})")
        for key in ("vision_model", "vision_model_2"):
            d = dual.get(key)
            if d:
                lines.append(f"- {(d.get('name') or 'vision model').split('/')[-1]}: {d.get('label') or 'no answer'} "
                             f"(confidence {_confidence(d)})")
        if dual.get("suggestion"):
            lines.append(f"Votes: {dual['suggestion']}")
        dissent = dual.get("cnn_dissent")
        if dissent:
            lines.append(f"CNN note (low trust): the CNN said {dissent.get('label')} with confidence "
                         f"{dissent.get('confidence')}, but the vision models agree on {report.get('defect_type')}.")
        if dual.get("needs_human_review"):
            lines.append("The detectors need a human check: " + str(dual.get("note") or dual.get("decision") or "see above"))
        for w in dual.get("warnings") or []:
            lines.append(f"Warning: {w}")
    elif case.get("confidence") is not None:
        lines.append(f"Detection confidence: {case['confidence']}")
    if case.get("explanation"):
        lines.append("Our pipeline's explanation: " + " ".join(str(case["explanation"]).split())[:500])
    if case.get("recommended_action"):
        lines.append(f"Our pipeline's recommended action: {case['recommended_action']}")
    if case.get("source_manual"):
        lines.append(f"Manual cited by our pipeline: {case['source_manual']}, page {case.get('source_page')}")
    return "\n".join(lines)


# ---------------- parsing the agent's reply ----------------

_FIELD_ALIASES = {
    "defect": ("defect", "defect_type", "finding", "issue"),
    "severity": ("severity", "severity_level"),
    "probable_cause": ("probable_cause", "probable_causes", "cause", "causes", "root_cause"),
    "safety_precautions": ("safety_precautions", "safety_precaution", "safety", "precautions", "safety_notes"),
    "recommended_steps": ("recommended_steps", "recommended_step", "recommended_actions", "recommended_action",
                          "steps", "next_steps", "actions", "recommendation", "recommendations"),
    "citation": ("citation", "citations", "source", "sources", "reference", "references", "manual"),
    "confidence": ("confidence", "confidence_score"),
    "needs_human_review": ("needs_human_review", "needs_review", "human_review", "human_review_needed",
                           "requires_human_review"),
    "summary": ("summary", "tldr", "overview"),
}
_ALIAS_TO_FIELD = {a: f for f, aliases in _FIELD_ALIASES.items() for a in aliases}
LIST_FIELDS = ("safety_precautions", "recommended_steps", "citation")   # the agent may cite several manual passages
FIELDS = tuple(_FIELD_ALIASES)
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25cf]+\s+|\d+[.)]\s+)")
_BOLD_HEAD = re.compile(r"^(\*\*|__)(.+?)(\*\*|__)\s*(.*)$")
_PLAIN_HEAD = re.compile(r"^([A-Za-z][A-Za-z _/-]{1,40}?)\s*[:\uff1a]\s*(.*)$")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _strip_marks(text: str) -> str:
    return re.sub(r"(\*\*|__)", "", text).strip()


def _header(line: str):
    """(field, rest_of_line) if the line starts a known field, else None. Handles `**field:** x`, `**Field**: x`,
    `- field: x`, `field: x`, `## Field` and a bold-only `**Field**`."""
    s = _BULLET.sub("", line.strip(), count=1)
    heading = bool(re.match(r"^#{1,6}\s*", s))
    s = re.sub(r"^#{1,6}\s*", "", s)
    m = _BOLD_HEAD.match(s)
    if m:
        inner, rest = m.group(2).strip(), m.group(4).strip()
        colon = inner.endswith((":", "\uff1a"))
        if rest[:1] in (":", "\uff1a"):
            colon, rest = True, rest[1:].strip()
        field = _ALIAS_TO_FIELD.get(_norm_name(inner))
        if field and (colon or not rest or heading):
            return field, _strip_marks(rest)
        return None
    m = _PLAIN_HEAD.match(s)
    if m:
        field = _ALIAS_TO_FIELD.get(_norm_name(m.group(1)))
        return (field, _strip_marks(m.group(2))) if field else None
    if heading:
        field = _ALIAS_TO_FIELD.get(_norm_name(s))
        return (field, "") if field else None
    return None


def _truthy(text: Optional[str]) -> Optional[bool]:
    t = (text or "").strip().lower().strip("*_`.! ")
    if not t:
        return None
    if re.match(r"^(true|yes|y|1|required|needed|needs|review|human review)\b", t):
        return True
    if re.match(r"^(false|no|n|0|none|not\b|never)", t):
        return False
    return None


def _number(text: Optional[str]) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%)?", text or "")
    if not m:
        return None
    v = float(m.group(1))
    v = v / 100.0 if (m.group(2) or v > 1) else v
    return round(v, 4) if 0 <= v <= 1 else None


def parse_agent_reply(text: str) -> Dict[str, Any]:
    """Parse the Evolus agent's free-text reply into defect, severity, probable_cause, safety_precautions (list),
    recommended_steps (list), citation, confidence (0-1 or None), needs_human_review (bool) and summary. Tolerates
    markdown (`**field:**`, headings, bullets, numbered lists) and a JSON reply. The raw text is always kept.
    `confidence` is a 0-1 number when the agent gave one, otherwise None, and `confidence_text` keeps its words ("high").
    Anything it cannot read is None / [] and listed in `notes`; it never raises."""
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    values: Dict[str, Any] = {}
    stripped = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
    try:
        obj = json.loads(stripped) if stripped.startswith("{") else None
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            f = _ALIAS_TO_FIELD.get(_norm_name(str(k)))
            if f and f not in values:
                values[f] = v
    else:
        current, buf, blank = None, {}, False
        for line in raw.splitlines():
            h = _header(line) if line.strip() else None
            if h:
                current, blank = h[0], False
                buf.setdefault(current, [])
                if h[1]:
                    buf[current].append(h[1])
            elif not line.strip():
                blank = bool(current and buf.get(current))     # a blank line after a value ends a paragraph
            elif current:
                if blank and not (current in LIST_FIELDS and _BULLET.match(line)):
                    current = None                              # text after the answer ("Stay safe!") is not part of it
                else:
                    buf[current].append(line.strip())
                    blank = False
        for f, lines in buf.items():
            if f in LIST_FIELDS:
                values[f] = [i for i in (_strip_marks(_BULLET.sub("", ln, count=1)) for ln in lines) if i]
            else:
                values[f] = _strip_marks(" ".join(lines))
    notes: List[str] = []
    parsed: Dict[str, Any] = {}
    for f in FIELDS:
        v = values.get(f)
        if f in LIST_FIELDS:
            items = [str(i).strip() for i in v] if isinstance(v, list) else ([str(v).strip()] if v else [])
            parsed[f] = [i for i in items if i]
        elif f == "needs_human_review":
            parsed[f] = None
        elif f == "confidence":
            text = str(v).strip() if v not in (None, "") else None
            parsed[f] = _number(text)
            parsed["confidence_text"] = text            # the agent often says "high" instead of a number: keep its words
            if text and parsed[f] is None and text.lower().strip("*_`. ") not in ("high", "medium", "low", "very high", "very low"):
                notes.append(f"confidence {text[:30]!r} is neither a number nor high/medium/low")
        else:
            parsed[f] = str(v).strip() if v not in (None, "") else None
    sev = (parsed["severity"] or "").lower()
    parsed["severity"] = next((w for w in ("critical", "high", "medium", "low") if sev.startswith(w)), parsed["severity"])
    flag = _truthy(str(values["needs_human_review"]) if "needs_human_review" in values else None)
    if flag is None:
        flag = True  # when we cannot tell, a person should look
        notes.append("needs_human_review was missing or unclear: defaulting to true")
    parsed["needs_human_review"] = flag
    parsed.setdefault("confidence_text", None)
    for f in FIELDS:
        if f not in values and f != "needs_human_review":
            notes.append(f"{f} not found in the reply")
    return {"raw_text": raw, "parsed": parsed, "notes": notes}


def _evolus_seconds(execution: Dict[str, Any]) -> Optional[float]:
    """The run's real duration by Evolus's own clock (completedAt - requestedAt), or None."""
    def parse(v):
        try:
            return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", str(v)).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    a, b = parse(execution.get("requestedAt")), parse(execution.get("completedAt"))
    return round((b - a).total_seconds(), 1) if a and b else None


def output_text(execution: Dict[str, Any]) -> str:
    """The agent's answer text from an execution record: `output`, else the last non-empty step output."""
    def flat(o) -> str:
        if o is None:
            return ""
        if isinstance(o, str):
            return o
        if isinstance(o, list):
            return "\n".join(t for t in (flat(i) for i in o) if t)
        if isinstance(o, dict):
            for k in ("text", "output", "content", "answer", "message", "response", "result", "value"):
                if k in o and flat(o[k]).strip():
                    return flat(o[k])
            return json.dumps(o, ensure_ascii=False)
        return str(o)

    text = flat(execution.get("output"))
    if text.strip():
        return text
    steps = execution.get("stepOutputs") or {}
    for k in reversed(list(steps)):
        t = flat(steps[k])
        if t.strip():
            return t
    return ""


class MockEvolusClient:
    def __init__(self, fail_first_n: int = 0) -> None:
        self._fail_remaining = fail_first_n
        self._counter = 0
        self.created: List[Dict[str, Any]] = []

    def fail_next(self, n: int = 1) -> None:
        """Make the next n create_task() calls fail (used by tests)."""
        self._fail_remaining = n

    supports_resume = False

    def create_task(self, case: Dict[str, Any], resume_execution_id: Optional[str] = None, on_started=None) -> str:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise EvolusError("Evolus unavailable (simulated failure)")
        self._counter += 1
        self.created.append(build_task_payload(case))
        return f"MOCK-TASK-{self._counter:04d}"


class RestEvolusClient:
    """Runs the Evolus workflow for an approved case (EVOLUS_MODE=rest) and waits for the agent's answer.

    1. POST {base}/workflows/{workflow_id}/execute, headers X-ApiKey + X-Application-Id, multipart form field `input`
       (text from build_evolus_input). Evolus answers 202 with {jobId, executionId, enqueuedAt, traceId} and the header
       X-Workflow-Execution-Id.
    2. GET {base}/workflows/{workflow_id}/executions/{executionId} every `poll_interval` seconds (5) until `status` is
       Completed or Failed, for at most `poll_timeout` seconds (600). Statuses seen: Pending, Running, Completed, Failed.
    3. The agent's text (`output`) is parsed by parse_agent_reply; create_task returns a TaskId (the execution id) whose
       `.result` holds the raw text and the parsed fields.
    Resume: create_task(case, resume_execution_id=...) first looks ONCE at that earlier run. Pending/Running: keep waiting for
    it (no second run); Completed: use its answer; Failed or unknown: start a new run. `on_started(execution_id)` is called as
    soon as a run exists, so the caller can remember it.

    Any failure (config, network, auth, 4xx/5xx, a Failed run, no answer within the timeout) raises EvolusError, so the case
    becomes task_failed and can be retried. The API key is never put in an error message."""

    ID_KEYS = ("executionId", "execution_id", "jobId", "job_id", "taskId", "task_id", "id")
    FAILED = ("failed", "error", "errored", "cancelled", "canceled", "aborted", "timedout", "timed_out")
    MAX_TRANSIENT = 3

    def __init__(self, base_url: str, api_key: str, application_id: str, workflow_id: str,
                 timeout: float = 30.0, poll_interval: float = POLL_INTERVAL, poll_timeout: float = POLL_TIMEOUT) -> None:
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.application_id, self.workflow_id, self.timeout = application_id, workflow_id, timeout
        self.poll_interval, self.poll_timeout = poll_interval, poll_timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {"X-ApiKey": self.api_key, "X-Application-Id": self.application_id}

    supports_resume = True

    def create_task(self, case: Dict[str, Any], resume_execution_id: Optional[str] = None, on_started=None) -> str:
        missing = [n for n, v in (("EVOLUS_API_KEY", self.api_key), ("EVOLUS_APPLICATION_ID", self.application_id),
                                  ("EVOLUS_WORKFLOW_ID", self.workflow_id)) if not v]
        if missing:
            raise EvolusError("Evolus is not configured: set " + ", ".join(missing))
        started = _monotonic()
        execution_id, resumed, previous, execution = None, False, None, None
        if resume_execution_id:
            # Pending / Running -> keep waiting for that run; Completed -> use its answer; Failed or unknown -> a new run
            state, body = self._peek(resume_execution_id)
            if state in ("running", "completed"):
                execution_id, resumed = resume_execution_id, True
                execution = body if state == "completed" else None
            else:
                previous = resume_execution_id
        if execution_id is None:
            execution_id = self._start(build_evolus_input(case))
        if on_started:
            on_started(execution_id)
        if execution is None:
            execution = self._wait(execution_id)
        text = output_text(execution)
        result = parse_agent_reply(text)
        if not text.strip():
            result["notes"].insert(0, "Evolus finished but returned no answer text")
        self._check_citations(result)
        result.update(execution_id=execution_id, status=execution.get("status"), resumed=resumed,
                      previous_execution_id=previous, duration_seconds=round(_monotonic() - started, 1),
                      evolus_seconds=_evolus_seconds(execution))
        task = TaskId(execution_id)
        task.result = result
        return task

    @staticmethod
    def _check_citations(result: Dict[str, Any]) -> None:
        """Look every quote of the agent up in our manuals (app/citations.py). A quote that is not in them makes the answer need a
        person: the agent's own review flag is kept as `agent_needs_human_review`, and `parsed.needs_human_review` becomes true."""
        parsed = result["parsed"]
        cites = citations.verify_citations(parsed["citation"])
        counts = citations.summarize(cites)
        result.update(citations=cites, citation_check=counts, agent_needs_human_review=parsed["needs_human_review"])
        if counts["unverified"]:
            parsed["needs_human_review"] = True
            result["notes"].append(f"{counts['unverified']} citation(s) could not be verified against our manuals: a person must check them")
        if counts["corrected"]:
            result["notes"].append(f"{counts['corrected']} quote(s) were corrected to the manual's exact wording")
        if not cites:
            result["notes"].append("the agent gave no citation")

    RUNNING = ("pending", "queued", "enqueued", "running", "processing", "inprogress", "waiting", "scheduled")

    def _peek(self, execution_id: str):
        """ONE look at an earlier execution: ("running" | "completed" | "failed" | "unknown", record)."""
        try:
            resp = httpx.get(f"{self.base_url}/workflows/{self.workflow_id}/executions/{execution_id}",
                             headers=self._headers, timeout=self.timeout)
            if resp.status_code != 200:
                return "unknown", {}
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            return "unknown", {}
        body = body if isinstance(body, dict) else {}
        state = re.sub(r"[^a-z]", "", str(body.get("status") or "").lower())
        if state == "completed":
            return "completed", body
        if state in self.FAILED:
            return "failed", body
        return ("running", body) if state in self.RUNNING else ("unknown", body)

    def _start(self, text: str) -> str:
        try:
            # files={...: (None, value)} makes httpx send multipart/form-data text fields
            resp = httpx.post(f"{self.base_url}/workflows/{self.workflow_id}/execute", headers=self._headers,
                              files={"input": (None, text)}, timeout=self.timeout)
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

    def _wait(self, execution_id: str) -> Dict[str, Any]:
        """Poll until Completed (returns the execution record) or raise EvolusError."""
        url = f"{self.base_url}/workflows/{self.workflow_id}/executions/{execution_id}"
        deadline = _monotonic() + self.poll_timeout
        transient, last_status, last_problem = 0, None, None
        while True:
            try:
                resp = httpx.get(url, headers=self._headers, timeout=self.timeout)
            except httpx.HTTPError as exc:
                transient, last_problem = transient + 1, type(exc).__name__
            else:
                code = resp.status_code
                if code in (401, 403):
                    raise EvolusError(f"Evolus rejected the credentials while waiting for the result (HTTP {code})", execution_id)
                if code == 404:
                    raise EvolusError(f"Evolus does not know execution {execution_id} (HTTP 404)", execution_id)
                if code == 429 or code >= 500:
                    transient, last_problem = transient + 1, f"HTTP {code}"
                elif code >= 400:
                    raise EvolusError(f"Evolus returned HTTP {code} while waiting for the result", execution_id)
                else:
                    transient = 0
                    try:
                        body = resp.json()
                    except ValueError:
                        body = {}
                    body = body if isinstance(body, dict) else {}
                    last_status = str(body.get("status") or "")
                    state = re.sub(r"[^a-z]", "", last_status.lower())
                    if state == "completed":
                        return body
                    if state in self.FAILED:
                        why = body.get("error")
                        why = why if isinstance(why, str) else (json.dumps(why) if why else "no reason given")
                        raise EvolusError(f"Evolus execution {execution_id} {last_status.lower()}: {why[:200]}", execution_id)
            if transient > self.MAX_TRANSIENT:
                raise EvolusError(f"Evolus stopped answering while waiting for execution {execution_id} ({last_problem})", execution_id)
            if _monotonic() >= deadline:
                raise EvolusError(f"Evolus did not finish within {self.poll_timeout:g} s: execution {execution_id} "
                                  f"is still {last_status or 'unknown'}. Retry will pick that run up again.", execution_id)
            _sleep(self.poll_interval)

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
                                settings.evolus_application_id, settings.evolus_workflow_id,
                                poll_interval=settings.evolus_poll_interval, poll_timeout=settings.evolus_poll_timeout)
    if settings.evolus_mode == "mcp":
        return McpEvolusClient()
    return MockEvolusClient(fail_first_n=settings.evolus_mock_fail_first_n)
