"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Cpu, Factory, FilePlus2, LayoutDashboard, LogOut } from "lucide-react";
import { getUser, logout } from "@/lib/auth";
import { getHealth, type Health } from "@/lib/api";
import { MODEL_LABEL } from "@/lib/config";

const links = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "New case", icon: FilePlus2 },
];

const dot = { mock: "bg-amber-400", online: "bg-emerald-400", offline: "bg-red-400", checking: "bg-slate-500" };
const dotText = { mock: "Demo data (mock)", online: "Backend online", offline: "Backend offline", checking: "Checking backend..." };

export default function Shell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const path = usePathname();
  const [user, setUser] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    const u = getUser();
    if (!u) router.replace("/login");
    else setUser(u);
  }, [router]);

  // INTEGRATION(KARIM): needs GET /health. A green dot means the connection works.
  useEffect(() => {
    if (!user) return;
    let alive = true;
    const check = () => getHealth().then((h) => alive && setHealth(h));
    check();
    const t = setInterval(check, 15000);
    return () => { alive = false; clearInterval(t); };
  }, [user]);

  if (!user) return null;

  const out = () => { logout(); router.replace("/login"); };
  const active = (h: string) => (h === "/" ? path === "/" : path.startsWith(h));
  const state = health?.state ?? "checking";
  // INTEGRATION(KRISHIV): the name comes from LLM_MODEL in the backend .env, so it cannot lie.
  const model = health?.reasoner
    ? `${health.reasoner}${health.llm_provider ? ` (${health.llm_provider})` : ""}`
    : MODEL_LABEL;

  return (
    <div className="min-h-screen md:pl-64">
      <aside className="fixed inset-y-0 left-0 hidden w-64 flex-col border-r border-white/10 bg-black/30 p-4 md:flex">
        <div className="mb-8 flex items-center gap-2 px-2">
          <div className="rounded-lg bg-cyan-400/15 p-2 text-cyan-300"><Factory className="h-5 w-5" /></div>
          <div>
            <div className="text-sm font-bold leading-tight">Factory Agent</div>
            <div className="text-[11px] text-slate-400">Quality &amp; Maintenance</div>
          </div>
        </div>
        <nav className="space-y-1">
          {links.map(({ href, label, icon: I }) => (
            <Link key={href} href={href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                active(href) ? "bg-cyan-400/10 text-cyan-300" : "text-slate-300 hover:bg-white/5"}`}>
              <I className="h-4 w-4" /> {label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto space-y-3">
          <div className="card flex items-start gap-2 p-3 text-xs text-slate-300">
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
            <div>
              <div>{model}</div>
              {health?.detector && <div className="mt-1 text-slate-500">Detector: {health.detector}</div>}
            </div>
          </div>
          <div className="space-y-1 px-2 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${dot[state]}`} /> {dotText[state]}
            </div>
            {state === "online" && health?.evolus_mode && (
              <div>
                Evolus:{" "}
                {health.evolus_mode === "mcp"
                  ? <span className="text-emerald-400">live (MCP)</span>
                  : <span className="text-amber-400">mock tasks</span>}
              </div>
            )}
            {health?.in_memory_db && <div className="text-amber-400">Memory database, data resets on restart</div>}
          </div>
          <div className="flex items-center justify-between px-2 text-sm">
            <span className="text-slate-300">{user}</span>
            <button onClick={out} title="Logout" className="text-slate-400 hover:text-white"><LogOut className="h-4 w-4" /></button>
          </div>
        </div>
      </aside>

      <header className="flex items-center justify-between border-b border-white/10 bg-black/30 px-4 py-3 md:hidden">
        <span className="font-bold">Factory Agent</span>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/">Dashboard</Link>
          <Link href="/upload">New</Link>
          <button onClick={out}><LogOut className="h-4 w-4" /></button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl p-4 md:p-8">{children}</main>
    </div>
  );
}