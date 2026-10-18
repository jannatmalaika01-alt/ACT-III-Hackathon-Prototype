import { normalizeCase, normalizeList } from "./adapter";
import type { Case } from "./types";
import { mockApprove, mockGet, mockList, mockReject, mockRetry, mockUpload } from "./mock";

const MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";

// The browser calls /api/backend/*. Next.js forwards it to BACKEND_URL (see next.config.ts).
const BASE = "/api/backend";
const DEFAULT_HEADERS: Record<string, string> = {
  "ngrok-skip-browser-warning": "true",
  Accept: "application/json",
};
const JSON_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  "ngrok-skip-browser-warning": "true",
  Accept: "application/json",
};

// Routes in backend/app/main.py. If a route changes, change only the paths in this file.
//   GET  /health
//   GET  /cases?limit=200                 returns {items: [...]}
//   GET  /cases/{id}                      case plus approvals plus events
//   GET  /cases/{id}/file                 the uploaded file (needs the patch for Suve)
//   POST /cases                           multipart: file, uploaded_by
//   POST /cases/{id}/analyze              runs detection + the agent and stores the diagnosis (case stays pending)
//   POST /cases/{id}/approve              json: decided_by, reason
//   POST /cases/{id}/reject               json: decided_by, reason (required)
//   POST /cases/{id}/retry-task
// The diagnosis is posted by the AI pipeline (POST /cases/{id}/diagnosis), never by the frontend.

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("ngrok-skip-browser-warning")) {
    headers.set("ngrok-skip-browser-warning", "true");
  }
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }

  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
  });

  if (!res.ok) {
    // The backend sends {"detail": "..."} for clear errors and a list for validation errors.
    let msg = res.statusText;
    try {
      const text = await res.text();
      try {
        const j = JSON.parse(text);
        msg =
          typeof j.detail === "string"
            ? j.detail
            : Array.isArray(j.detail)
              ? j.detail.map((d: { msg?: string }) => d.msg).join(", ")
              : JSON.stringify(j);
      } catch {
        if (text.includes("<!DOCTYPE") || text.includes("<html")) {
          msg = "Received HTML from upstream tunnel instead of JSON. Ensure ngrok-skip-browser-warning is active.";
        } else if (text.trim()) {
          msg = text.slice(0, 200);
        }
      }
    } catch {
      /* response had no readable body */
    }
    throw new Error(`${res.status}: ${msg}`);
  }

  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    if (text.includes("<!DOCTYPE") || text.includes("<html")) {
      throw new Error(
        "Received HTML interstitial warning from Ngrok instead of valid JSON. Upstream Ngrok warning was not bypassed."
      );
    }
    throw err;
  }
}

export const fileUrl = (id: string) => `${BASE}/cases/${id}/file`;

export interface Health {
  state: "mock" | "online" | "offline";
  evolus_mode?: string;
  in_memory_db?: boolean;
  analyzer_mode?: string;
  reasoner?: string | null;
  llm_provider?: string | null;
  llm_answered?: boolean;
  detector?: string | null;
}

export async function getHealth(): Promise<Health> {
  if (MOCK) return { state: "mock" };
  try {
    const r = await fetch(`${BASE}/health`, {
      cache: "no-store",
      headers: {
        "ngrok-skip-browser-warning": "true",
        Accept: "application/json",
      },
    });
    if (!r.ok) return { state: "offline" };
    return { ...(await r.json()), state: "online" };
  } catch {
    return { state: "offline" };
  }
}

export async function listCases(): Promise<Case[]> {
  if (MOCK) return mockList();
  return normalizeList(await http<unknown>("/cases?limit=200"));
}

export async function getCase(id: string): Promise<Case> {
  if (MOCK) return mockGet(id);
  return normalizeCase(await http<unknown>(`/cases/${id}`));
}

export async function uploadCase(file: File, uploadedBy: string): Promise<Case> {
  if (MOCK) return mockUpload(uploadedBy);
  const form = new FormData();
  form.append("file", file);
  form.append("uploaded_by", uploadedBy);
  return normalizeCase(await http<unknown>("/cases", { method: "POST", body: form }));
}

// Runs the vision detector and the AI agent on an uploaded case; the diagnosis is stored on the case.
// Can take several seconds (the LLM answers), so callers should not block the screen on it.
export async function analyzeCase(id: string): Promise<Case> {
  if (MOCK) return mockGet(id); // the mock upload adds its own diagnosis after a few seconds
  return normalizeCase(await http<unknown>(`/cases/${id}/analyze`, { method: "POST" }));
}

// Returns the case. status is "task_created" or "task_failed". The approval is saved either way.
export async function approveCase(id: string, decidedBy: string, reason?: string): Promise<Case> {
  if (MOCK) return mockApprove(id, decidedBy);
  return normalizeCase(await http<unknown>(`/cases/${id}/approve`, {
    method: "POST", headers: JSON_HEADERS,
    body: JSON.stringify({ decided_by: decidedBy, reason: reason || null }),
  }));
}

export async function rejectCase(id: string, decidedBy: string, reason: string): Promise<Case> {
  if (MOCK) return mockReject(id, decidedBy, reason);
  return normalizeCase(await http<unknown>(`/cases/${id}/reject`, {
    method: "POST", headers: JSON_HEADERS,
    body: JSON.stringify({ decided_by: decidedBy, reason }),
  }));
}

export async function retryTask(id: string): Promise<Case> {
  if (MOCK) return mockRetry(id);
  return normalizeCase(await http<unknown>(`/cases/${id}/retry-task`, { method: "POST" }));
}