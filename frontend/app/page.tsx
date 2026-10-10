"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Clock, FilePlus2, Search, ShieldCheck, Wrench } from "lucide-react";
import Shell from "@/components/Shell";
import { ConfidenceBar, SeverityBadge, StatusBadge } from "@/components/Badges";
import { listCases } from "@/lib/api";
import { hasDiagnosis, LOW_CONFIDENCE, missingForApproval, shortId, type Case, type Status } from "@/lib/types";

type Filter = "all" | Status;
const FILTERS: [Filter, string][] = [
  ["all", "All"], ["pending", "Pending"], ["task_created", "Task created"],
  ["task_failed", "Task failed"], ["rejected", "Rejected"],
];

function StatCard({ icon: I, label, value, hint, alert }: {
  icon: typeof Clock; label: string; value: string | number; hint?: string; alert?: boolean;
}) {
  return (
    <div className="card animate-fade-up p-4">
      <div className="mb-3 flex items-center justify-between text-slate-400">
        <span className="text-xs uppercase tracking-wide">{label}</span>
        <I className="h-4 w-4" />
      </div>
      <div className={`text-3xl font-bold ${alert ? "text-orange-300" : ""}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export default function Dashboard() {
  const [cases, setCases] = useState<Case[] | null>(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");

  const load = useCallback(() => listCases().then(setCases).catch((e) => setError(e.message)), []);
  useEffect(() => { load(); }, [load]);

  // Some case is still waiting for the AI: refresh until the diagnosis arrives
  useEffect(() => {
    if (!cases?.some((c) => c.status === "pending" && !c.defect)) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [cases, load]);

  const s = useMemo(() => {
    const all = cases ?? [];
    const n = (f: (c: Case) => boolean) => all.filter(f).length;
    const diagnosed = all.filter(hasDiagnosis);
    const types: Record<string, number> = {};
    diagnosed.forEach((c) => { types[c.defect!] = (types[c.defect!] ?? 0) + 1; });
    const cited = diagnosed.filter((c) => c.source_manual && c.source_page).length;
    return {
      total: all.length,
      waiting: n((c) => c.status === "pending" && hasDiagnosis(c)),
      awaitingAI: n((c) => c.status === "pending" && !hasDiagnosis(c)),
      done: n((c) => c.status === "task_created" || c.status === "approved"),
      failed: n((c) => c.status === "task_failed"),
      rejected: n((c) => c.status === "rejected"),
      review: n((c) => c.status === "pending" && hasDiagnosis(c) &&
        ((c.confidence ?? 1) < LOW_CONFIDENCE || missingForApproval(c).length > 0)),
      citedPct: diagnosed.length ? Math.round((cited / diagnosed.length) * 100) : 0,
      types: Object.entries(types).sort((a, b) => b[1] - a[1]),
    };
  }, [cases]);

  const rows = (cases ?? []).filter(
    (c) => (filter === "all" || c.status === filter) &&
      `${c.id} ${c.defect ?? ""} ${c.file_name}`.toLowerCase().includes(q.toLowerCase())
  );
  const maxType = Math.max(1, ...s.types.map((t) => t[1]));
  const pct = (v: number) => (s.total ? (v / s.total) * 100 : 0);

  return (
    <Shell>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-slate-400">Every AI recommendation needs a human decision.</p>
        </div>
        <Link href="/upload" className="flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400">
          <FilePlus2 className="h-4 w-4" /> New case
        </Link>
      </div>

      {error && <p className="mb-4 rounded-lg border border-red-400/30 bg-red-400/10 p-3 text-sm text-red-300">{error}</p>}

      {s.review > 0 && (
        <div className="mb-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-3 text-sm text-amber-200">
          ⚠ {s.review} pending case{s.review > 1 ? "s need" : " needs"} careful review (low confidence or no manual source).
        </div>
      )}
      {s.failed > 0 && (
        <div className="mb-4 rounded-lg border border-orange-400/30 bg-orange-400/10 p-3 text-sm text-orange-200">
          ⚠ {s.failed} approved case{s.failed > 1 ? "s have" : " has"} no Evolus task yet. Open the case and press Retry.
        </div>
      )}

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {!cases ? (
          Array.from({ length: 4 }).map((_, i) => <div key={i} className="skeleton h-28" />)
        ) : (
          <>
            <StatCard icon={Clock} label="Waiting for approval" value={s.waiting} hint={`${s.awaitingAI} waiting for the AI`} />
            <StatCard icon={Wrench} label="Evolus tasks" value={s.done} hint={`${s.rejected} rejected`} />
            <StatCard icon={AlertTriangle} label="Failed tasks" value={s.failed} hint="need a retry" alert={s.failed > 0} />
            <StatCard icon={ShieldCheck} label="Cited from manuals" value={`${s.citedPct}%`} hint={`${s.total} cases total`} />
          </>
        )}
      </div>

      {cases && (
        <div className="mb-6 grid gap-3 md:grid-cols-2">
          <div className="card p-4">
            <h2 className="mb-3 text-sm font-semibold">Defects by type</h2>
            <div className="space-y-2">
              {s.types.length === 0 && <p className="text-xs text-slate-500">No diagnosed cases yet.</p>}
              {s.types.map(([t, n]) => (
                <div key={t} className="flex items-center gap-3 text-xs">
                  <span className="w-28 truncate capitalize text-slate-300">{t.replace("_", " ")}</span>
                  <div className="h-2 flex-1 rounded bg-white/10">
                    <div className="h-2 rounded bg-cyan-400" style={{ width: `${(n / maxType) * 100}%` }} />
                  </div>
                  <span className="w-4 text-right text-slate-400">{n}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card p-4">
            <h2 className="mb-3 text-sm font-semibold">Case outcomes</h2>
            <div className="flex h-3 overflow-hidden rounded bg-white/10">
              <div className="bg-emerald-400" style={{ width: `${pct(s.done)}%` }} />
              <div className="bg-amber-400" style={{ width: `${pct(s.waiting + s.awaitingAI)}%` }} />
              <div className="bg-orange-400" style={{ width: `${pct(s.failed)}%` }} />
              <div className="bg-red-400" style={{ width: `${pct(s.rejected)}%` }} />
            </div>
            <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-300">
              <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-400" />Task created</span>
              <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400" />Pending</span>
              <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-orange-400" />Task failed</span>
              <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-red-400" />Rejected</span>
            </div>
          </div>
        </div>
      )}

      <div className="card overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 p-3">
          <div className="flex flex-wrap gap-1">
            {FILTERS.map(([f, label]) => (
              <button key={f} onClick={() => setFilter(f)}
                className={`rounded-md px-3 py-1.5 text-xs transition ${
                  filter === f ? "bg-cyan-400/15 text-cyan-300" : "text-slate-400 hover:bg-white/5"}`}>
                {label}
              </button>
            ))}
          </div>
          <div className="relative">
            <Search className="absolute left-2.5 top-2 h-4 w-4 text-slate-500" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search..."
              className="w-52 rounded-md border border-white/10 bg-white/5 py-1.5 pl-8 pr-2 text-xs outline-none focus:border-cyan-400" />
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>{["Case", "Defect", "Confidence", "Severity", "Status", "Evolus task"].map((h) => <th key={h} className="px-4 py-3 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody>
              {!cases && Array.from({ length: 4 }).map((_, i) => (
                <tr key={i}><td colSpan={6} className="px-4 py-2"><div className="skeleton h-8" /></td></tr>
              ))}
              {cases && rows.length === 0 && <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500">No cases found.</td></tr>}
              {rows.map((c) => (
                <tr key={c.id} className="border-t border-white/5 transition hover:bg-white/5">
                  <td className="px-4 py-3">
                    <Link href={`/cases/${c.id}`} className="font-medium text-cyan-300 hover:underline">{shortId(c.id)}</Link>
                    <div className="text-xs text-slate-500">{c.file_name}</div>
                  </td>
                  <td className="px-4 py-3 capitalize">
                    {c.defect ? c.defect.replace("_", " ") : <span className="animate-pulse text-slate-500">Analysing...</span>}
                  </td>
                  <td className="px-4 py-3"><ConfidenceBar value={c.confidence} /></td>
                  <td className="px-4 py-3"><SeverityBadge severity={c.severity} /></td>
                  <td className="px-4 py-3"><StatusBadge status={c.status} /></td>
                  <td className="px-4 py-3 text-slate-400">{c.evolus_task_id ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Shell>
  );
}