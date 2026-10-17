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

// What one detector said. `error` is set (and `label` null) when it failed or timed out.
export interface DetectorAnswer {
  label: string | null;
  confidence: number | null;
  name?: string;
  error?: string | null;
}

// The dual detector's result, stored by the backend in diagnosis_raw.dual_detector (backend/app/dual_detector.py).
export interface DualDetector {
  mode: string;                       // vision_primary (default) | pair | unanimous | vote | cnn_anchored
  agreement: "agree" | "disagree" | "single_detector";
  needs_human_review: boolean;
  decision: string;                   // why: "vision models agree", "3-way split", ...
  suggestion?: string;                // "2 of 3 say corrosion"
  note?: string;
  cnn: DetectorAnswer;
  vision_model?: DetectorAnswer;
  vision_model_2?: DetectorAnswer;
  answer?: { label: string; confidence: number | null; from: string };
  warnings?: string[];
  cnn_dissent?: { label: string; confidence: number | null; trust: string; note?: string } | null;
  cnn_tie?: { label: string; confidence: number | null; threshold: number; used: boolean } | null;
}

// The Evolus agent's reply, parsed by the backend (backend/app/evolus.py parse_agent_reply), in diagnosis_raw.evolus.
export interface EvolusResult {
  execution_id: string;
  status?: string;
  raw_text: string;
  parsed: {
    defect: string | null;
    severity: string | null;
    probable_cause: string | null;
    safety_precautions: string[];
    recommended_steps: string[];
    citation: string[];
    confidence: number | null;
    confidence_text?: string | null;
    needs_human_review: boolean;
    summary: string | null;
  };
  notes: string[];
  duration_seconds?: number;
  evolus_seconds?: number | null;
  resumed?: boolean;
}

export interface DiagnosisRaw {
  needs_manual_verification?: boolean;
  defect_report?: { defect_type?: string; location?: string; confidence?: number };
  dual_detector?: DualDetector;
  evolus?: Partial<EvolusResult> & { state?: string };
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
  // Only on GET /cases/{id}: is an Evolus run in progress for this case right now? (EVOLUS_MODE=rest runs it in the background)
  evolus?: { running: boolean; started_at: string | null; execution_id: string | null };
  created_at: string;
  updated_at: string;
  diagnosis_raw?: DiagnosisRaw | null;
  // Only present on GET /cases/{id}
  approvals?: Approval[];
  events?: CaseEvent[];
}

export const LOW_CONFIDENCE = 0.7; // same as LOW_CONFIDENCE_THRESHOLD in agent.py

export const hasDiagnosis = (c: Case) => !!c.defect;

// Same rule as REQUIRED_FOR_APPROVAL in backend/app/service.py
export const missingForApproval = (c: Case) =>
  (["defect", "recommended_action", "source_manual", "source_page"] as const).filter((k) => !c[k]);

// True while the Evolus agent is preparing the task: the backend says so, and a case that is "approved" but has no task yet is waiting too.
export const evolusInProgress = (c: Case) => !!c.evolus?.running || c.status === "approved";

export const shortId = (id: string) => id.slice(0, 8);

export function inputKind(contentType: string): string {
  if (contentType.startsWith("image/")) return "Machine photo";
  if (contentType === "text/csv" || contentType === "application/json") return "Sensor report";
  return "Document";
}