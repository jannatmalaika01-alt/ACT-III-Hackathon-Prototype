import type { Severity, Status } from "@/lib/types";

const pill = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium";

const statusCls: Record<Status, string> = {
  pending: "border-amber-400/30 bg-amber-400/10 text-amber-300",
  approved: "border-sky-400/30 bg-sky-400/10 text-sky-300",
  rejected: "border-red-400/30 bg-red-400/10 text-red-300",
  task_created: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
  task_failed: "border-orange-400/40 bg-orange-500/15 text-orange-300",
};
const statusText: Record<Status, string> = {
  pending: "Pending", approved: "Approved", rejected: "Rejected",
  task_created: "Task created", task_failed: "Task failed",
};
const sevCls: Record<Severity, string> = {
  low: "border-slate-400/30 bg-slate-400/10 text-slate-300",
  medium: "border-sky-400/30 bg-sky-400/10 text-sky-300",
  high: "border-orange-400/30 bg-orange-400/10 text-orange-300",
  critical: "border-red-400/40 bg-red-500/15 text-red-300",
};

export const StatusBadge = ({ status }: { status: Status }) => (
  <span className={`${pill} ${statusCls[status]}`}>{statusText[status]}</span>
);

export const SeverityBadge = ({ severity }: { severity: Severity | null }) =>
  severity ? <span className={`${pill} capitalize ${sevCls[severity]}`}>{severity}</span>
           : <span className="text-xs text-slate-500">-</span>;

const colorFor = (v: number) => (v >= 0.85 ? "#34d399" : v >= 0.7 ? "#fbbf24" : "#f87171");

export function ConfidenceBar({ value }: { value: number | null }) {
  if (value == null) return <span className="text-xs text-slate-500">-</span>;
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded bg-white/10">
        <div className="h-1.5 rounded" style={{ width: `${pct}%`, background: colorFor(value) }} />
      </div>
      <span className="text-xs text-slate-400">{pct}%</span>
    </div>
  );
}

export function ConfidenceRing({ value }: { value: number | null }) {
  const r = 34, c = 2 * Math.PI * r, v = value ?? 0;
  return (
    <div className="relative h-24 w-24">
      <svg viewBox="0 0 80 80" className="-rotate-90">
        <circle cx="40" cy="40" r={r} fill="none" stroke="#ffffff14" strokeWidth="7" />
        <circle cx="40" cy="40" r={r} fill="none" stroke={colorFor(v)} strokeWidth="7"
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - v)}
          style={{ transition: "stroke-dashoffset .8s ease" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-bold">{value == null ? "n/a" : `${Math.round(v * 100)}%`}</span>
        <span className="text-[10px] uppercase text-slate-400">confidence</span>
      </div>
    </div>
  );
}