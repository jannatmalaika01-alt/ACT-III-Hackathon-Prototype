"use client";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

/** Shown while the Evolus agent prepares the maintenance task. Counts the time up from when the run started (taken from
 *  the server, so it survives a page reload) and never blocks the rest of the page. */
export default function EvolusProgress({ since }: { since: string | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const started = since ? new Date(since).getTime() : NaN;
  const seconds = Number.isFinite(started) ? Math.max(0, Math.floor((now - started) / 1000)) : null;
  return (
    <div role="status" aria-live="polite" className="space-y-2 rounded-lg border border-cyan-400/30 bg-cyan-400/10 p-4 text-sm text-cyan-100">
      <div className="flex items-center gap-2 font-medium">
        <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
        Evolus agent is preparing the maintenance task, this can take a few minutes
      </div>
      <p className="text-xs text-cyan-200/80">
        {seconds !== null && <>Elapsed: <b className="tabular-nums">{mmss(seconds)}</b>. </>}
        Your approval is saved. You can keep using this page or leave it; the result appears here by itself.
        {seconds !== null && seconds > 300 && " The shared AI service is busy, so this is slower than usual. We keep waiting for up to 10 minutes."}
      </p>
    </div>
  );
}
