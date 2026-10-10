import { AlertTriangle, Check, Loader2, X } from "lucide-react";
import type { Case } from "@/lib/types";

type S = "done" | "current" | "todo" | "warn" | "fail";

const style: Record<S, string> = {
  done: "border-emerald-400 bg-emerald-400/15 text-emerald-300",
  current: "border-cyan-400 bg-cyan-400/15 text-cyan-300",
  todo: "border-white/15 bg-white/5 text-slate-500",
  warn: "border-amber-400 bg-amber-400/15 text-amber-300",
  fail: "border-red-400 bg-red-400/15 text-red-300",
};

function Icon({ s, n }: { s: S; n: number }) {
  if (s === "done") return <Check className="h-4 w-4" />;
  if (s === "current") return <Loader2 className="h-4 w-4 animate-spin" />;
  if (s === "warn") return <AlertTriangle className="h-4 w-4" />;
  if (s === "fail") return <X className="h-4 w-4" />;
  return <span className="text-xs">{n}</span>;
}

export default function Pipeline({ c }: { c: Case }) {
  const diag = !!c.defect;
  const cited = !!(c.source_manual && c.source_page);
  const decided = ["approved", "task_created", "task_failed"].includes(c.status);
  const steps: [string, S][] = [
    ["Uploaded", "done"],
    ["AI diagnosis", diag ? "done" : "current"],
    ["Manual source", !diag ? "todo" : cited ? "done" : "warn"],
    ["Human decision", c.status === "rejected" ? "fail" : decided ? "done" : diag ? "current" : "todo"],
    ["Evolus task", c.status === "task_created" ? "done" : c.status === "task_failed" ? "fail" : c.status === "approved" ? "current" : "todo"],
  ];
  return (
    <div className="card overflow-x-auto p-4">
      <ol className="flex min-w-[560px] items-center">
        {steps.map(([label, s], i) => (
          <li key={label} className="flex flex-1 items-center last:flex-none">
            <div className="flex flex-col items-center gap-1.5">
              <div className={`flex h-9 w-9 items-center justify-center rounded-full border ${style[s]}`}>
                <Icon s={s} n={i + 1} />
              </div>
              <span className="whitespace-nowrap text-xs text-slate-300">{label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`mx-2 mb-5 h-px flex-1 ${s === "done" ? "bg-emerald-400/50" : "bg-white/10"}`} />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}