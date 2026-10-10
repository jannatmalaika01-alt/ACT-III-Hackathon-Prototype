"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, BookOpen, Check, Code2, Copy, ExternalLink, FileText, History, Loader2, RotateCcw, X } from "lucide-react";
import Shell from "@/components/Shell";
import Pipeline from "@/components/Pipeline";
import { useToast } from "@/components/Toast";
import { ConfidenceRing, SeverityBadge, StatusBadge } from "@/components/Badges";
import { approveCase, fileUrl, getCase, rejectCase, retryTask } from "@/lib/api";
import { getUser } from "@/lib/auth";
import { EVOLUS_URL, IS_MOCK } from "@/lib/config";
import { hasDiagnosis, inputKind, LOW_CONFIDENCE, missingForApproval, shortId, type Case } from "@/lib/types";

export default function CasePage() {
  const { id } = useParams<{ id: string }>();
  const toast = useToast();
  const [c, setC] = useState<Case | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [showJson, setShowJson] = useState(false);
  const [imgOk, setImgOk] = useState(true);

  const load = useCallback(() => getCase(id).then(setC).catch((e) => setError(e.message)), [id]);
  useEffect(() => { load(); }, [load]);

  // INTEGRATION(JANNAT, TITUS): the AI diagnosis is posted by the backend after upload.
  // Until it arrives the AI fields are empty, so we check again every 2 seconds.
  const awaiting = c?.status === "pending" && !c.defect;
  useEffect(() => {
    if (!awaiting) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [awaiting, load]);

  async function act(kind: "approve" | "reject" | "retry") {
    setBusy(true);
    const by = getUser() ?? "unknown";
    try {
      if (kind === "reject") {
        await rejectCase(id, by, reason);
        toast("info", "Case rejected");
      } else {
        const r = kind === "approve" ? await approveCase(id, by) : await retryTask(id);
        if (r.status === "task_failed") toast("error", "Approval saved, but the Evolus task failed. You can retry.");
        else toast("success", `Evolus task created: ${r.evolus_task_id}`);
      }
      setRejecting(false);
      setReason("");
      await load();
    } catch (e) {
      toast("error", (e as Error).message);
    }
    setBusy(false);
  }

  if (!c) {
    return (
      <Shell>
        {error ? <p className="text-red-400">{error}</p> : (
          <div className="space-y-3"><div className="skeleton h-10 w-48" /><div className="skeleton h-24" /><div className="skeleton h-64" /></div>
        )}
      </Shell>
    );
  }

  const diag = hasDiagnosis(c);
  const missing = diag ? missingForApproval(c) : [];
  const lowConf = diag && (c.confidence ?? 1) < LOW_CONFIDENCE;
  const canDecide = c.status === "pending" && diag;
  const isImage = c.content_type.startsWith("image/");
  const rejection = c.approvals?.find((a) => a.decision === "rejected");
  const analysisError = [...(c.events ?? [])].reverse().find((e) => e.note?.startsWith("analysis failed"))?.note;

  return (
    <Shell>
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-white">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>
      <div className="mb-4 mt-2 flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-2xl font-bold">Case {shortId(c.id)}</h1>
          <p className="text-xs text-slate-500">{inputKind(c.content_type)} · {c.file_name} · uploaded by {c.uploaded_by}</p>
        </div>
        <StatusBadge status={c.status} />
        <SeverityBadge severity={c.severity} />
        <button onClick={() => setShowJson((v) => !v)}
          className="ml-auto flex items-center gap-1.5 rounded-md border border-white/10 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/5">
          <Code2 className="h-3.5 w-3.5" /> Raw JSON
        </button>
      </div>

      <div className="mb-4"><Pipeline c={c} /></div>

      {(lowConf || missing.length > 0) && (
        <div className="mb-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-3 text-sm text-amber-200">
          ⚠ Review carefully:
          {lowConf && " detection confidence is low (may be a false positive)."}
          {missing.length > 0 && ` the backend will not allow approval because these fields are missing: ${missing.join(", ")}.`}
        </div>
      )}

      {/* Use this panel to debug: compare it with the Case type in lib/types.ts */}
      {showJson && (
        <div className="card animate-fade-up relative mb-4 p-4">
          <button onClick={() => { navigator.clipboard.writeText(JSON.stringify(c, null, 2)); toast("info", "JSON copied"); }}
            className="absolute right-3 top-3 text-slate-400 hover:text-white"><Copy className="h-4 w-4" /></button>
          <pre className="max-h-72 overflow-auto text-xs text-slate-300">{JSON.stringify(c, null, 2)}</pre>
        </div>
      )}

      {awaiting ? (
        <section className="card relative mb-4 overflow-hidden p-10 text-center">
          {!analysisError && <div className="scan-line" />}
          {!analysisError && <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-cyan-300" />}
          <p className="font-medium">{analysisError ? "The AI could not analyse this file" : "The AI is analysing your file..."}</p>
          <p className="mt-1 text-sm text-slate-400">{analysisError ?? "This page updates by itself."}</p>
        </section>
      ) : (
        <div className="grid gap-4 lg:grid-cols-5">
          <div className="space-y-4 lg:col-span-3">
            <section className="card animate-fade-up p-5">
              <h2 className="mb-4 text-sm font-semibold text-slate-300">Detected issue</h2>
              {/* The file comes from GET /cases/{id}/file (patch for Suve). Hidden if the route is missing. */}
              {!IS_MOCK && isImage && imgOk && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={fileUrl(c.id)} alt={c.file_name} onError={() => setImgOk(false)} className="mb-4 max-h-64 rounded-lg" />
              )}
              {!IS_MOCK && !isImage && (
                <a href={fileUrl(c.id)} target="_blank" rel="noreferrer"
                  className="mb-4 inline-flex items-center gap-2 text-sm text-cyan-300 underline">
                  <FileText className="h-4 w-4" /> Open uploaded file
                </a>
              )}
              <div className="flex items-center gap-5">
                <ConfidenceRing value={c.confidence} />
                <div className="text-xl font-semibold capitalize">{c.defect?.replace("_", " ")}</div>
              </div>
              <p className="mt-4 text-sm leading-relaxed text-slate-300">{c.explanation}</p>
            </section>

            <section className="card animate-fade-up p-5">
              <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-300">
                <BookOpen className="h-4 w-4" /> Source from manuals / SOPs
              </h2>
              {/* INTEGRATION(TITUS): the backend stores one citation (manual and page). To show the exact
                  quote, add a source_excerpt field to DiagnosisIn and display it here. */}
              {c.source_manual && c.source_page ? (
                <div className="rounded-lg border-l-2 border-cyan-400/60 bg-white/5 p-3 text-sm text-cyan-300">
                  {c.source_manual} · page {c.source_page}
                </div>
              ) : (
                <p className="text-sm text-red-300">No citation. A human must verify this recommendation.</p>
              )}
            </section>
          </div>

          <section className="card animate-fade-up h-fit space-y-4 p-5 lg:col-span-2 lg:sticky lg:top-6">
            <h2 className="text-sm font-semibold text-slate-300">Recommended action</h2>
            {/* INTEGRATION(JANNAT): this text is the recommendation written by the AI pipeline */}
            <p className="rounded-lg bg-white/5 p-3 text-sm leading-relaxed">{c.recommended_action}</p>

            {canDecide && !rejecting && (
              <div className="flex gap-3">
                <button disabled={busy || missing.length > 0} onClick={() => act("approve")}
                  className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-emerald-500 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:opacity-40">
                  <Check className="h-4 w-4" /> {busy ? "Working..." : "Approve"}
                </button>
                <button disabled={busy} onClick={() => setRejecting(true)}
                  className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-red-400/40 py-2.5 text-sm font-semibold text-red-300 transition hover:bg-red-400/10 disabled:opacity-50">
                  <X className="h-4 w-4" /> Reject
                </button>
              </div>
            )}

            {canDecide && rejecting && (
              <div className="space-y-2">
                <textarea rows={3} value={reason} onChange={(e) => setReason(e.target.value)}
                  placeholder="Why reject? A reason is required."
                  className="w-full rounded-lg border border-white/10 bg-white/5 p-2 text-sm outline-none focus:border-red-400" />
                <div className="flex gap-3">
                  <button disabled={busy || !reason.trim()} onClick={() => act("reject")}
                    className="flex-1 rounded-lg bg-red-500 py-2 text-sm font-semibold disabled:opacity-50">Confirm reject</button>
                  <button onClick={() => setRejecting(false)} className="flex-1 rounded-lg border border-white/15 py-2 text-sm">Cancel</button>
                </div>
              </div>
            )}

            {/* INTEGRATION(EVOLUS, KARIM): evolus_task_id is the id of the task created in Evolus */}
            {c.status === "task_created" && (
              <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/10 p-3 text-sm text-emerald-200">
                ✅ Evolus task created: <b>{c.evolus_task_id}</b>
                {EVOLUS_URL && (
                  <a href={EVOLUS_URL} target="_blank" rel="noreferrer" className="mt-2 flex items-center gap-1 text-xs text-emerald-300 underline">
                    Open Evolus <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </div>
            )}

            {c.status === "task_failed" && (
              <div className="space-y-3 rounded-lg border border-orange-400/30 bg-orange-400/10 p-3 text-sm text-orange-200">
                <p>Your approval is saved, but the Evolus task failed (attempts: {c.task_attempts}).</p>
                <p className="text-xs text-orange-300/80">{c.task_error}</p>
                <button disabled={busy} onClick={() => act("retry")}
                  className="flex w-full items-center justify-center gap-2 rounded-lg bg-orange-500 py-2 text-sm font-semibold text-slate-950 hover:bg-orange-400 disabled:opacity-50">
                  <RotateCcw className="h-4 w-4" /> {busy ? "Retrying..." : "Retry task"}
                </button>
              </div>
            )}

            {c.status === "rejected" && (
              <div className="rounded-lg border border-red-400/30 bg-red-400/10 p-3 text-sm text-red-200">
                Rejected by {rejection?.decided_by ?? "unknown"}. Reason: {rejection?.reason ?? "-"}
              </div>
            )}
          </section>
        </div>
      )}

      {(c.events?.length ?? 0) > 0 && (
        <section className="card mt-4 p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-300"><History className="h-4 w-4" /> Activity</h2>
          <ul className="space-y-2 text-sm">
            {c.events!.map((e) => (
              <li key={e.id} className="flex flex-wrap items-baseline gap-x-3 text-slate-300">
                <span className="text-xs text-slate-500">{new Date(e.created_at).toLocaleString()}</span>
                <span className="rounded bg-white/5 px-1.5 py-0.5 text-[11px] text-slate-400">{e.source}</span>
                <span>{e.note ?? `${e.from_status} to ${e.to_status}`}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </Shell>
  );
}