"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Factory, Lock, User } from "lucide-react";
import { login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [error, setError] = useState("");

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (login(u, p)) router.replace("/");
    else setError("Wrong username or password");
  }

  const input = "w-full rounded-lg border border-white/10 bg-white/5 py-2.5 pl-10 pr-3 text-sm outline-none focus:border-cyan-400";

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="card animate-fade-up w-full max-w-sm space-y-4 p-8">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-400/15 p-3 text-cyan-300"><Factory className="h-6 w-6" /></div>
          <div>
            <h1 className="text-lg font-bold">Factory Agent</h1>
            <p className="text-xs text-slate-400">AI quality &amp; maintenance</p>
          </div>
        </div>
        <div className="relative">
          <User className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
          <input className={input} placeholder="Username" value={u} onChange={(e) => setU(e.target.value)} />
        </div>
        <div className="relative">
          <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
          <input className={input} type="password" placeholder="Password" value={p} onChange={(e) => setP(e.target.value)} />
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button className="w-full rounded-lg bg-cyan-500 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400">
          Log in
        </button>
        <p className="text-center text-xs text-slate-500">Demo login: admin / factory123</p>
      </form>
    </div>
  );
}