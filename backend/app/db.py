"""Data access. Same methods, two implementations:

  * SupabaseRepo  - real PostgreSQL (+ Storage) through supabase-py
  * InMemoryRepo  - dicts; used by tests and for running with no Supabase

transition_case() is a compare-and-set: it only updates the row if it is STILL in
`from_status`. That stops two requests (e.g. a double-click on Approve) from both
winning.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupabaseRepo:
    def __init__(self, url: str, key: str, bucket: str) -> None:
        from supabase import create_client  # imported here so tests don't need it

        self.client = create_client(url, key)
        self.bucket = bucket

    def _t(self, name: str):
        return self.client.table(name)

    # ---- storage ----
    def upload_file(self, path: str, data: bytes, content_type: str) -> None:
        self.client.storage.from_(self.bucket).upload(path, data, {"content-type": content_type})

    # ---- cases ----
    def create_case(self, case_id: str, file_path: str, file_name: str,
                    content_type: str, uploaded_by: str) -> Dict[str, Any]:
        res = self._t("cases").insert({
            "id": case_id, "file_path": file_path, "file_name": file_name,
            "content_type": content_type, "uploaded_by": uploaded_by,
        }).execute()
        return res.data[0]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        res = self._t("cases").select("*").eq("id", case_id).limit(1).execute()
        return res.data[0] if res.data else None

    def list_cases(self, status: Optional[str], limit: int, offset: int) -> List[Dict[str, Any]]:
        q = self._t("cases").select("*")
        if status:
            q = q.eq("status", status)
        res = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        return res.data

    def set_diagnosis(self, case_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        res = self._t("cases").update(fields).eq("id", case_id).eq("status", "pending").execute()
        return res.data[0] if res.data else None

    def transition_case(self, case_id: str, from_status: str, to_status: str,
                        fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        res = (self._t("cases").update({**fields, "status": to_status})
               .eq("id", case_id).eq("status", from_status).execute())
        return res.data[0] if res.data else None

    # ---- approvals ----
    def add_approval(self, case_id: str, decision: str, decided_by: str,
                     reason: Optional[str]) -> Dict[str, Any]:
        res = self._t("approvals").insert({
            "case_id": case_id, "decision": decision,
            "decided_by": decided_by, "reason": reason,
        }).execute()
        return res.data[0]

    def list_approvals(self, case_id: str) -> List[Dict[str, Any]]:
        return self._t("approvals").select("*").eq("case_id", case_id).order("decided_at").execute().data

    # ---- events ----
    def add_event(self, case_id: str, from_status: Optional[str], to_status: str,
                  note: Optional[str], source: str, event_id: Optional[str]) -> Dict[str, Any]:
        res = self._t("case_events").insert({
            "case_id": case_id, "event_id": event_id, "source": source,
            "from_status": from_status, "to_status": to_status, "note": note,
        }).execute()
        return res.data[0]

    def event_exists(self, event_id: str) -> bool:
        res = self._t("case_events").select("id").eq("event_id", event_id).limit(1).execute()
        return bool(res.data)

    def list_events(self, case_id: str) -> List[Dict[str, Any]]:
        return self._t("case_events").select("*").eq("case_id", case_id).order("created_at").execute().data


class InMemoryRepo:
    def __init__(self) -> None:
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.approvals: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.files: Dict[str, bytes] = {}

    def upload_file(self, path: str, data: bytes, content_type: str) -> None:
        self.files[path] = data

    def create_case(self, case_id: str, file_path: str, file_name: str,
                    content_type: str, uploaded_by: str) -> Dict[str, Any]:
        now = _now()
        row = {
            "id": case_id, "status": "pending",
            "file_path": file_path, "file_name": file_name,
            "content_type": content_type, "uploaded_by": uploaded_by,
            "defect": None, "severity": None, "explanation": None,
            "recommended_action": None, "source_manual": None, "source_page": None,
            "confidence": None, "diagnosis_raw": None, "diagnosed_at": None,
            "evolus_task_id": None, "task_error": None, "task_attempts": 0,
            "created_at": now, "updated_at": now,
        }
        self.cases[case_id] = row
        return copy.deepcopy(row)

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        row = self.cases.get(case_id)
        return copy.deepcopy(row) if row else None

    def list_cases(self, status: Optional[str], limit: int, offset: int) -> List[Dict[str, Any]]:
        rows = [r for r in self.cases.values() if status is None or r["status"] == status]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return copy.deepcopy(rows[offset: offset + limit])

    def set_diagnosis(self, case_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.cases.get(case_id)
        if row is None or row["status"] != "pending":
            return None
        row.update(fields)
        row["updated_at"] = _now()
        return copy.deepcopy(row)

    def transition_case(self, case_id: str, from_status: str, to_status: str,
                        fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.cases.get(case_id)
        if row is None or row["status"] != from_status:
            return None
        row.update(fields)
        row["status"] = to_status
        row["updated_at"] = _now()
        return copy.deepcopy(row)

    def add_approval(self, case_id: str, decision: str, decided_by: str,
                     reason: Optional[str]) -> Dict[str, Any]:
        if any(a["case_id"] == case_id for a in self.approvals):
            raise ValueError("case already has a decision")  # mirrors the unique index
        row = {"id": f"appr-{len(self.approvals) + 1}", "case_id": case_id, "decision": decision,
               "decided_by": decided_by, "reason": reason, "decided_at": _now()}
        self.approvals.append(row)
        return copy.deepcopy(row)

    def list_approvals(self, case_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy([a for a in self.approvals if a["case_id"] == case_id])

    def add_event(self, case_id: str, from_status: Optional[str], to_status: str,
                  note: Optional[str], source: str, event_id: Optional[str]) -> Dict[str, Any]:
        if event_id and any(e["event_id"] == event_id for e in self.events):
            raise ValueError("duplicate event_id")  # mirrors the unique constraint
        row = {"id": len(self.events) + 1, "case_id": case_id, "event_id": event_id,
               "source": source, "from_status": from_status, "to_status": to_status,
               "note": note, "created_at": _now()}
        self.events.append(row)
        return copy.deepcopy(row)

    def event_exists(self, event_id: str) -> bool:
        return any(e["event_id"] == event_id for e in self.events)

    def list_events(self, case_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy([e for e in self.events if e["case_id"] == case_id])
