export type Status = "pending" | "approved" | "rejected" | "task_created" | "task_failed";
export type Severity = "low" | "medium" | "high" | "critical";

export interface Approval {
  id: string | number;
  decision: "approved" | "rejected";
  decided_by: string;
  reason: string | null;
  decided_at: string;
}

export interface CaseEvent {
  id: string | number;
  from_status: string | null;
  to_status: string;
  note: string | null;
  source: string; // "api", "webhook" or "analyzer"
  created_at: string;
}

// INTEGRATION(KARIM, SUVE): mirrors one row of the "cases" table (backend/app/db.py).
// The AI fields stay null until the AI pipeline posts a diagnosis (POST /cases/{id}/diagnosis).
export interface Case {
  id: string;
  status: Status;
  file_name: string;
  content_type: string;
  uploaded_by: string;
  // INTEGRATION(JANNAT, TITUS, KRISHIV): these come from the AI pipeline (DiagnosisIn in models.py)
  defect: string | null;
  severity: Severity | null;
  explanation: string | null;
  recommended_action: string | null;
  source_manual: string | null;
  source_page: number | null;
  confidence: number | null;
  // INTEGRATION(EVOLUS, KARIM)
  evolus_task_id: string | null;
  task_error: string | null;
  task_attempts: number;
  created_at: string;
  updated_at: string;
  // Only present on GET /cases/{id}
  approvals?: Approval[];
  events?: CaseEvent[];
}

export const LOW_CONFIDENCE = 0.7; // same as LOW_CONFIDENCE_THRESHOLD in agent.py

export const hasDiagnosis = (c: Case) => !!c.defect;

// Same rule as REQUIRED_FOR_APPROVAL in backend/app/service.py
export const missingForApproval = (c: Case) =>
  (["defect", "recommended_action", "source_manual", "source_page"] as const).filter((k) => !c[k]);

export const shortId = (id: string) => id.slice(0, 8);

export function inputKind(contentType: string): string {
  if (contentType.startsWith("image/")) return "Machine photo";
  if (contentType === "text/csv" || contentType === "application/json") return "Sensor report";
  return "Document";
}