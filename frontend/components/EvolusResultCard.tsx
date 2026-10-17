import { AlertTriangle, Check, ExternalLink, Pencil } from "lucide-react";
import { SeverityBadge } from "@/components/Badges";
import { duration, evolusSeconds, parseCitation, pct } from "@/lib/detectors";
import type { EvolusResult, Severity, VerifiedCitation } from "@/lib/types";

const pill = "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium";
const SEVERITIES = ["low", "medium", "high", "critical"];

const STATUS: Record<VerifiedCitation["status"], { label: string; cls: string }> = {
  verified: { label: "verified", cls: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" },
  corrected: { label: "corrected to source", cls: "border-sky-400/30 bg-sky-400/10 text-sky-300" },
  unverified: { label: "unverified", cls: "border-red-400/40 bg-red-400/10 text-red-300" },
};

/** Results stored before the citation check existed only have the agent's raw lines: show those, marked as unchecked. */
function legacyCitations(raw: string[]): VerifiedCitation[] {
  return raw.map((r) => {
    const c = parseCitation(r);
    return { document: c.document, page: c.page ? Number(c.page) : null, quote: c.quote, status: "unverified", raw: r, note: "not checked against our manuals" };
  });
}

/** What the Evolus agent answered after the approval: severity, safety precautions, recommended steps, each citation
 *  (document, page, quote), its confidence, and whether it wants a person to look. */
export default function EvolusResultCard({ taskId, result, evolusUrl }: { taskId: string | null; result: EvolusResult; evolusUrl?: string }) {
  const p = result.parsed;
  const seconds = evolusSeconds(result);
  const sev = (p.severity ?? "").toLowerCase();
  const confidence = p.confidence != null ? pct(p.confidence) : p.confidence_text ?? "not given";
  const cites = result.citations ?? legacyCitations(p.citation);
  const unverified = cites.filter((c) => c.status === "unverified").length;
  const forcedByCheck = unverified > 0 && result.agent_needs_human_review === false;   // the agent said no review, our check says yes
  return (
    <section className="card animate-fade-up p-5" data-testid="evolus-result">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-slate-300">Evolus agent: maintenance task</h2>
        <span className={`${pill} border-emerald-400/30 bg-emerald-400/10 text-emerald-300`}>Task {taskId ?? result.execution_id}</span>
        {seconds != null && <span className="text-xs text-slate-400">agent answered in {duration(seconds)}</span>}
        {result.resumed && <span className="text-xs text-slate-500">(picked up an earlier run)</span>}
        {evolusUrl && (
          <a href={evolusUrl} target="_blank" rel="noreferrer" className="ml-auto flex items-center gap-1 text-xs text-emerald-300 underline">
            Open Evolus <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-300">
        <span className="text-slate-500">Severity</span>
        {SEVERITIES.includes(sev) ? <SeverityBadge severity={sev as Severity} /> : <span>{p.severity ?? "not given"}</span>}
        <span className="ml-2 text-slate-500">Confidence</span>
        <span className="font-medium text-slate-200">{confidence}</span>
        {p.needs_human_review ? (
          <span className={`${pill} ml-2 border-amber-400/30 bg-amber-400/10 text-amber-300`}><AlertTriangle className="h-3 w-3" /> needs human review{forcedByCheck ? " (citation check)" : ""}</span>
        ) : (
          <span className={`${pill} ml-2 border-emerald-400/30 bg-emerald-400/10 text-emerald-300`}><Check className="h-3 w-3" /> no human review needed</span>
        )}
      </div>

      {p.defect && <p className="mb-2 text-sm text-slate-200"><span className="text-slate-500">Defect: </span>{p.defect}</p>}
      {p.summary && <p className="mb-2 text-sm text-slate-300">{p.summary}</p>}
      {p.probable_cause && <p className="mb-4 text-sm text-slate-300"><span className="text-slate-500">Probable cause: </span>{p.probable_cause}</p>}

      {p.safety_precautions.length > 0 && (
        <div className="mb-4">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Safety precautions</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">{p.safety_precautions.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
      )}
      {p.recommended_steps.length > 0 && (
        <div className="mb-4">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Recommended steps</h3>
          <ol className="list-decimal space-y-1 pl-5 text-sm text-slate-300">{p.recommended_steps.map((x, i) => <li key={i}>{x}</li>)}</ol>
        </div>
      )}
      {unverified > 0 && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-red-400/40 bg-red-400/10 p-3 text-sm text-red-200" data-testid="citation-warning">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            {unverified === 1 ? "1 quote" : `${unverified} quotes`} from the agent could not be found in our manuals. Do not rely on
            {unverified === 1 ? " it" : " them"}: a person must check the answer against the manual before anyone acts on it.
          </span>
        </div>
      )}
      {cites.length > 0 && (
        <div className="mb-3">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Citations, checked against our manuals</h3>
          <ul className="space-y-2">
            {cites.map((c, i) => (
              <li key={i} className={`rounded-lg border-l-2 bg-white/5 p-3 text-sm ${c.status === "unverified" ? "border-red-400/70" : "border-cyan-400/60"}`}
                  data-testid="evolus-citation" data-status={c.status}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-cyan-300">{c.document ? <>{c.document}{c.page != null && <> · page {c.page}</>}</> : c.raw}</span>
                  <span className={`${pill} text-[11px] ${STATUS[c.status].cls}`} data-testid="citation-status">
                    {c.status === "verified" ? <Check className="h-3 w-3" /> : c.status === "corrected" ? <Pencil className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                    {STATUS[c.status].label}
                  </span>
                </div>
                {c.quote && <blockquote className={`mt-1 text-xs italic ${c.status === "unverified" ? "text-red-200/90" : "text-slate-300"}`}>“{c.quote}”</blockquote>}
                {c.status === "corrected" && c.agent_quote && (
                  <p className="mt-1 text-[11px] text-slate-500">The agent wrote: “{c.agent_quote}” The quote above is the manual&apos;s exact text.</p>
                )}
                {c.claimed && c.note && <p className="mt-1 text-[11px] text-slate-500">{c.note}.</p>}
                {c.status === "unverified" && c.note && <p className="mt-1 text-[11px] text-red-300/80">{c.note}.</p>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.notes.length > 0 && <p className="mb-2 text-xs text-slate-500">Reading the reply: {result.notes.join("; ")}.</p>}
      <details className="text-xs text-slate-400">
        <summary className="cursor-pointer">Show the raw reply of the agent</summary>
        <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-3 text-slate-300">{result.raw_text}</pre>
      </details>
    </section>
  );
}
