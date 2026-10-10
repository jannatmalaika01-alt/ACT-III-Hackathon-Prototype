/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Case, Severity, Status } from "./types";

const STATUSES: Status[] = ["pending", "approved", "rejected", "task_created", "task_failed"];
const SEVERITIES: Severity[] = ["low", "medium", "high", "critical"];

// INTEGRATION(KARIM, SUVE): every case from the backend passes through here.
// When a field name or format differs from the backend, fix it HERE and nowhere else.
export function normalizeCase(raw: any): Case {
  const conf = raw.confidence == null ? null : Number(raw.confidence);
  const sev = raw.severity == null ? null : String(raw.severity).toLowerCase();
  const status = String(raw.status ?? "pending").toLowerCase();
  return {
    ...raw, // keeps extra fields so they show in the Raw JSON panel
    id: String(raw.id),
    status: STATUSES.includes(status as Status) ? (status as Status) : "pending",
    confidence: conf == null ? null : conf > 1 ? conf / 100 : conf,
    severity: sev && SEVERITIES.includes(sev as Severity) ? (sev as Severity) : null,
    source_page: raw.source_page == null ? null : Number(raw.source_page),
    task_attempts: raw.task_attempts ?? 0,
    approvals: raw.approvals ?? [],
    events: raw.events ?? [],
  };
}

// GET /cases returns {items: [...], limit, offset}
export const normalizeList = (raw: any): Case[] =>
  (Array.isArray(raw) ? raw : raw.items ?? []).map(normalizeCase);