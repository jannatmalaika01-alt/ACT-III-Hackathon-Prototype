import { AlertTriangle, Check, ExternalLink } from "lucide-react";
import { SeverityBadge } from "@/components/Badges";
import { duration, evolusSeconds, parseCitation, pct } from "@/lib/detectors";
import type { EvolusResult, Severity } from "@/lib/types";

const pill = "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium";
const SEVERITIES = ["low", "medium", "high", "critical"];

/** What the Evolus agent answered after the approval: severity, safety precautions, recommended steps, each citation
 *  (document, page, quote), its confidence, and whether it wants a person to look. */
export default function EvolusResultCard({ taskId, result, evolusUrl }: { taskId: string | null; result: EvolusResult; evolusUrl?: string }) {
  const p = result.parsed;
  const seconds = evolusSeconds(result);
  const sev = (p.severity ?? "").toLowerCase();
  const confidence = p.confidence != null ? pct(p.confidence) : p.confidence_text ?? "not given";
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
          <span className={`${pill} ml-2 border-amber-400/30 bg-amber-400/10 text-amber-300`}><AlertTriangle className="h-3 w-3" /> needs human review</span>
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
      {p.citation.length > 0 && (
        <div className="mb-3">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Citations from the manuals library</h3>
          <ul className="space-y-2">
            {p.citation.map((raw, i) => {
              const c = parseCitation(raw);
              return (
                <li key={i} className="rounded-lg border-l-2 border-cyan-400/60 bg-white/5 p-3 text-sm" data-testid="evolus-citation">
                  <div className="text-cyan-300">{c.document ? <>{c.document}{c.page && <> · page {c.page}</>}</> : c.raw}</div>
                  {c.quote && <blockquote className="mt-1 text-xs italic text-slate-300">“{c.quote}”</blockquote>}
                </li>
              );
            })}
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
