import type { Case, CaseEvent, Severity, Status } from "./types";

// This file pretends to be the backend, with the same fields and the same error messages.
const KEY = "act3_mock_cases_v3";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const ago = (m: number) => new Date(Date.now() - m * 60000).toISOString();
// crypto.randomUUID only works on https or localhost, so we use our own
const uuid = () =>
  "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 3) | 8).toString(16);
  });

interface Diag {
  defect: string; severity: Severity; confidence: number; explanation: string;
  recommended_action: string; source_manual: string; source_page: number;
}
const DIAGS: Diag[] = [
  { defect: "crack", severity: "high", confidence: 0.91, explanation: "A crack was found in weld joint A3.",
    recommended_action: "Flag for immediate manual inspection by a certified welder. Do not clear equipment until the joint is re-tested.",
    source_manual: "sop_weld_inspection.pdf", source_page: 1 },
  { defect: "corrosion", severity: "medium", confidence: 0.78, explanation: "Surface corrosion on pipe section B7.",
    recommended_action: "Log and schedule inspection within 7 days.", source_manual: "sop_corrosion_handling.pdf", source_page: 1 },
  { defect: "loose_bolt", severity: "high", confidence: 0.95, explanation: "A loose bolt was found on panel D5.",
    recommended_action: "Create an immediate tightening work order.", source_manual: "sop_mechanical_alignment.pdf", source_page: 1 },
  { defect: "misalignment", severity: "low", confidence: 0.65, explanation: "Possible misalignment on conveyor belt C2 (low confidence).",
    recommended_action: "Manually verify the detection before any maintenance action.", source_manual: "sop_mechanical_alignment.pdf", source_page: 1 },
];

const ev = (id: number, from: string | null, to: string, note: string, t: string): CaseEvent =>
  ({ id, from_status: from, to_status: to, note, source: "api", created_at: t });

function mk(status: Status, mins: number, d: Diag | null,
            o: { task?: string; error?: string; reason?: string } = {}): Case {
  const t = ago(mins);
  const decided = status !== "pending";
  const events = [ev(1, null, "pending", "case created (file uploaded)", t)];
  if (d) events.push(ev(2, "pending", "pending", "diagnosis attached", t));
  if (decided) events.push(ev(3, "pending", status, o.reason ?? o.error ?? `Evolus task ${o.task} created`, t));
  return {
    id: uuid(), status, file_name: d ? `${d.defect}.png` : "upload.png", content_type: "image/png", uploaded_by: "admin",
    defect: d?.defect ?? null, severity: d?.severity ?? null, explanation: d?.explanation ?? null,
    recommended_action: d?.recommended_action ?? null, source_manual: d?.source_manual ?? null,
    source_page: d?.source_page ?? null, confidence: d?.confidence ?? null,
    evolus_task_id: o.task ?? null, task_error: o.error ?? null,
    task_attempts: status === "task_created" || status === "task_failed" ? 1 : 0,
    created_at: t, updated_at: t,
    approvals: decided
      ? [{ id: 1, decision: status === "rejected" ? "rejected" : "approved", decided_by: "admin", reason: o.reason ?? null, decided_at: t }]
      : [],
    events,
  };
}

function load(): Case[] {
  if (typeof window === "undefined") return [];
  const raw = localStorage.getItem(KEY);
  if (raw) return JSON.parse(raw);
  const seed = [
    mk("pending", 5, DIAGS[0]),
    mk("pending", 20, DIAGS[3]),
    mk("task_created", 90, DIAGS[1], { task: "MOCK-TASK-0001" }),
    mk("task_failed", 150, DIAGS[2], { error: "Evolus unavailable (simulated failure)" }),
    mk("rejected", 220, DIAGS[0], { reason: "Wrong joint. Inspect A2 first." }),
  ];
  localStorage.setItem(KEY, JSON.stringify(seed));
  return seed;
}
const save = (c: Case[]) => localStorage.setItem(KEY, JSON.stringify(c));

function find(all: Case[], id: string): Case {
  const c = all.find((x) => x.id === id);
  if (!c) throw new Error("404: Case not found");
  return c;
}

function push(c: Case, from: string, to: string, note: string) {
  c.events = [...(c.events ?? []), ev((c.events?.length ?? 0) + 1, from, to, note, new Date().toISOString())];
  c.updated_at = new Date().toISOString();
}

export async function mockList(): Promise<Case[]> { await sleep(400); return load(); }

export async function mockGet(id: string): Promise<Case> { await sleep(250); return find(load(), id); }

export async function mockUpload(by: string): Promise<Case> {
  await sleep(700);
  const all = load();
  const c = mk("pending", 0, null);
  c.uploaded_by = by;
  save([c, ...all]);
  // The AI diagnosis "arrives" a few seconds later, like the real pipeline would do
  setTimeout(() => {
    const cur = load();
    const row = cur.find((x) => x.id === c.id);
    if (!row) return;
    Object.assign(row, DIAGS[Math.floor(Math.random() * DIAGS.length)]);
    row.recommended_action = row.recommended_action ?? "";
    push(row, "pending", "pending", "diagnosis attached");
    save(cur);
  }, 3500);
  return c;
}

export async function mockApprove(id: string, by: string): Promise<Case> {
  await sleep(700);
  const all = load();
  const c = find(all, id);
  if (c.status !== "pending") throw new Error(`409: Only pending cases can be approved (this one is '${c.status}')`);
  const missing = (["defect", "recommended_action", "source_manual", "source_page"] as const).filter((k) => !c[k]);
  if (missing.length) throw new Error(`409: Case has no complete diagnosis/citation yet (missing: ${missing.join(", ")})`);
  c.approvals = [{ id: 1, decision: "approved", decided_by: by, reason: null, decided_at: new Date().toISOString() }];
  c.status = "task_created";
  c.evolus_task_id = `MOCK-TASK-${String(1000 + all.length).slice(1).padStart(4, "0")}`;
  c.task_attempts = 1;
  push(c, "pending", "task_created", `approved by ${by}, Evolus task ${c.evolus_task_id} created`);
  save(all);
  return c;
}

export async function mockReject(id: string, by: string, reason: string): Promise<Case> {
  await sleep(500);
  const all = load();
  const c = find(all, id);
  if (c.status !== "pending") throw new Error(`409: Only pending cases can be rejected (this one is '${c.status}')`);
  if (!reason.trim()) throw new Error("422: a rejection needs a reason");
  c.approvals = [{ id: 1, decision: "rejected", decided_by: by, reason: reason.trim(), decided_at: new Date().toISOString() }];
  c.status = "rejected";
  push(c, "pending", "rejected", `rejected by ${by}: ${reason.trim()}`);
  save(all);
  return c;
}

export async function mockRetry(id: string): Promise<Case> {
  await sleep(700);
  const all = load();
  const c = find(all, id);
  if (c.status !== "task_failed") throw new Error(`409: Only 'task_failed' cases can be retried (this one is '${c.status}')`);
  c.status = "task_created";
  c.evolus_task_id = `MOCK-TASK-R${c.id.slice(0, 4)}`;
  c.task_error = null;
  c.task_attempts += 1;
  push(c, "task_failed", "task_created", `Evolus task ${c.evolus_task_id} created`);
  save(all);
  return c;
}