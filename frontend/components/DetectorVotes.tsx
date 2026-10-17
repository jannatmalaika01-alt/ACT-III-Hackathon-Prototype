import { AlertTriangle, Check, CircleSlash, Info } from "lucide-react";
import { detectorRows, failureText, pct, pretty, voteOutcome, type Tone } from "@/lib/detectors";
import type { DualDetector } from "@/lib/types";

const tone: Record<Tone, string> = {
  ok: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
  warn: "border-amber-400/30 bg-amber-400/10 text-amber-200",
  bad: "border-red-400/30 bg-red-400/10 text-red-200",
};
const chip = "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium";

/** One row per detector: what it said and how sure it was. The CNN is an advisory note in vision_primary mode, not a voter.
 *  A detector that failed or timed out is marked clearly. Confidences are never averaged. */
export default function DetectorVotes({ d }: { d: DualDetector }) {
  const outcome = voteOutcome(d);
  const winner = d.needs_human_review ? null : d.answer?.label ?? null;
  return (
    <section className="card animate-fade-up p-5" data-testid="detector-votes">
      <h2 className="mb-3 text-sm font-semibold text-slate-300">Detector votes</h2>

      <div className={`mb-4 flex items-start gap-2 rounded-lg border p-3 text-sm ${tone[outcome.tone]}`} data-testid="vote-outcome">
        {outcome.tone === "ok" ? <Check className="mt-0.5 h-4 w-4 shrink-0" /> : <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />}
        <span>{outcome.text}</span>
      </div>

      <ul className="divide-y divide-white/5 rounded-lg border border-white/10">
        {detectorRows(d).map(({ key, name, advisory, answer }) => {
          const failed = !answer?.label;
          const agrees = !failed && winner !== null && answer?.label === winner;
          return (
            <li key={key} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2.5 text-sm" data-testid={`row-${key}`}>
              <div className="w-44 shrink-0">
                <div className="font-medium text-slate-200">{name}</div>
                <div className="text-[11px] text-slate-500">{advisory ? "advisory note, not a vote" : "votes"}</div>
              </div>
              {failed ? (
                <span className={`${chip} border-red-400/30 bg-red-400/10 text-red-300`}>
                  <CircleSlash className="h-3 w-3" /> {failureText(answer)} (no answer)
                </span>
              ) : (
                <>
                  <span className="min-w-28 font-semibold capitalize text-slate-100">{pretty(answer?.label)}</span>
                  <span className="text-xs tabular-nums text-slate-400">{pct(answer?.confidence)} confident</span>
                  {advisory && agrees && d.decision.startsWith("CNN breaks the tie") ? (
                    <span className={`${chip} border-emerald-400/30 bg-emerald-400/10 text-emerald-300`}><Check className="h-3 w-3" /> broke the tie</span>
                  ) : advisory ? (
                    <span className={`${chip} border-slate-400/30 bg-slate-400/10 text-slate-300`}><Info className="h-3 w-3" /> note, low trust</span>
                  ) : agrees ? (
                    <span className={`${chip} border-emerald-400/30 bg-emerald-400/10 text-emerald-300`}><Check className="h-3 w-3" /> in the answer</span>
                  ) : null}
                </>
              )}
            </li>
          );
        })}
      </ul>

      <p className="mt-3 text-xs text-slate-400">
        {d.suggestion && <>{d.suggestion}. </>}
        The confidence of each model is shown separately and is never averaged.
        {d.cnn_dissent && <> The CNN was trained on drawings and is weak on real photos, so it only leaves a note.</>}
      </p>
      {(d.warnings?.length ?? 0) > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-amber-300/90">
          {d.warnings!.map((w) => <li key={w}>⚠ {w}</li>)}
        </ul>
      )}
    </section>
  );
}
