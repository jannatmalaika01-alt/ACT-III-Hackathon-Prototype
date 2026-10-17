import type { Case, DetectorAnswer, DualDetector, EvolusResult } from "./types";

/** "self-dploy/Qwen3.8-Flash-Next" -> "Qwen3.8-Flash-Next" */
export const modelName = (n?: string | null) => (n ?? "").split("/").pop() || "vision model";
export const pretty = (label?: string | null) => (label ?? "").replace(/_/g, " ");
export const pct = (v: number | null | undefined) => (v == null ? "-" : `${Math.round(v * 100)}%`);

export interface Proposer { name: string; confidence: number | null; advisory: boolean }
export interface Candidate { label: string; proposers: Proposer[] }

/** The case needs a person: the pipeline flagged it UNVERIFIED (detectors disagree, or the detection is not confident). */
export const isUnverified = (c: Case) => !!c.defect && c.diagnosis_raw?.needs_manual_verification === true;

export const dualOf = (c: Case): DualDetector | null => c.diagnosis_raw?.dual_detector ?? null;

/** Rows for the votes card, in the order vision model 1, vision model 2, CNN. */
export function detectorRows(d: DualDetector): { key: string; name: string; advisory: boolean; answer: DetectorAnswer | undefined }[] {
  const advisory = d.mode === "vision_primary";
  const rows = [
    { key: "vision_model", name: modelName(d.vision_model?.name), advisory: false, answer: d.vision_model },
    { key: "vision_model_2", name: modelName(d.vision_model_2?.name), advisory: false, answer: d.vision_model_2 },
    { key: "cnn", name: "CNN", advisory, answer: d.cnn },
  ];
  return rows.filter((r) => r.answer !== undefined);
}

/** Which detector gave the final answer, in words ("the CNN", "Qwen3.8-Flash-Next"). */
export function answerSource(d: DualDetector): string | null {
  const from = d.answer?.from;
  if (!from) return null;
  if (from === "cnn") return "the CNN";
  return modelName((from === "vision_model" ? d.vision_model : d.vision_model_2)?.name);
}

/** Every label a detector proposed, with who proposed it. Used for the "Unverified" headline. */
export function candidatesOf(c: Case): Candidate[] {
  const d = dualOf(c);
  const out = new Map<string, Proposer[]>();
  if (d) {
    for (const r of detectorRows(d)) {
      if (r.answer?.label) {
        out.set(r.answer.label, [...(out.get(r.answer.label) ?? []), { name: r.name, confidence: r.answer.confidence, advisory: r.advisory }]);
      }
    }
  } else {
    const label = c.diagnosis_raw?.defect_report?.defect_type ?? c.defect?.split(" at ")[0];
    if (label) out.set(label.replace(/ /g, "_"), [{ name: "Image detector (CNN)", confidence: c.confidence, advisory: false }]);
  }
  return [...out.entries()].map(([label, proposers]) => ({ label, proposers }));
}

/** Why a detector has no answer, in plain words. */
export function failureText(a?: DetectorAnswer): string {
  const e = (a?.error ?? "").toLowerCase();
  if (e.includes("no answer within") || e.includes("timeout")) return "timed out";
  if (e.includes("parse") || e.includes("visionanswer")) return "gave an unusable answer";
  if (e.includes("not configured")) return "not configured";
  return "failed";
}

export type Tone = "ok" | "warn" | "bad";

/** The vote outcome in one line: agreed / tie broken by the CNN / disagree / unverified. */
export function voteOutcome(d: DualDetector): { tone: Tone; text: string } {
  const label = pretty(d.answer?.label);
  const dec = d.decision ?? "";
  if (d.mode === "vision_primary") {
    if (dec === "vision models agree") {
      return { tone: "ok", text: `Agreed: both vision models say ${label}.${d.cnn_dissent ? " The CNN disagrees; it is shown as a note only." : ""}` };
    }
    if (dec.startsWith("CNN breaks the tie")) {
      return { tone: "ok", text: `Tie broken by the CNN: the vision models disagree, the CNN (${pct(d.cnn?.confidence)} sure) matched ${label}.` };
    }
    if (dec.startsWith("CNN not confident enough")) {
      return { tone: "warn", text: `Unverified: the vision models disagree and the CNN (${pct(d.cnn?.confidence)}) was not confident enough to break the tie.` };
    }
    if (dec === "3-way split") return { tone: "warn", text: "Unverified: all three detectors gave different answers." };
    if (dec.startsWith("vision models disagree")) return { tone: "warn", text: "Unverified: the vision models disagree and the CNN cannot break the tie." };
    if (dec.startsWith("vision model and CNN agree")) return { tone: "ok", text: `One vision model failed; the other and the CNN agree on ${label}.` };
    if (dec.startsWith("vision model and CNN disagree")) return { tone: "warn", text: "Unverified: one vision model failed and the other disagrees with the CNN." };
    if (dec.startsWith("only one detector answered")) return { tone: "warn", text: "Unverified: only one vision model answered." };
    if (dec.startsWith("both vision models failed")) return { tone: "bad", text: "Unverified: both vision models failed, only the CNN answered (low trust)." };
  }
  if (d.needs_human_review) return { tone: "warn", text: `Unverified: ${d.note ?? "the detectors disagree"}` };
  return { tone: "ok", text: d.agreement === "agree" ? `Agreed: ${label}.` : `${d.note ?? "Decided by the " + d.mode + " rule"}` };
}

export interface ParsedCitation { document: string | null; page: string | null; quote: string | null; raw: string }

/** The Evolus agent writes: `document: X — section: Page 2: Title — quote: "..."`. Pull the parts out; keep the raw text as a fallback. */
export function parseCitation(raw: string): ParsedCitation {
  const doc = raw.match(/document:\s*(.+?)(?:\s+[\u2014-]\s+section:|\s+[\u2014-]\s+quote:|$)/i)?.[1]?.trim() ?? null;
  const section = raw.match(/section:\s*(.+?)(?:\s+[\u2014-]\s+quote:|$)/i)?.[1]?.trim() ?? null;
  const quote = raw.match(/quote:\s*["\u201c]?(.+?)["\u201d]?\s*$/i)?.[1]?.trim() ?? null;
  const page = (section ?? raw).match(/page\s*(\d+)/i)?.[1] ?? null;
  return { document: doc, page, quote, raw };
}

export function evolusSeconds(e: Partial<EvolusResult>): number | null {
  const s = e.evolus_seconds ?? e.duration_seconds;
  return s == null ? null : Math.round(s);
}

export const duration = (s: number) => (s >= 120 ? `${s} s (${Math.floor(s / 60)} min ${s % 60} s)` : `${s} s`);
