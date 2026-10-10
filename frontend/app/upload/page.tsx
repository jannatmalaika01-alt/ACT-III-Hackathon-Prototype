"use client";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { Activity, Camera, FileText, UploadCloud } from "lucide-react";
import Shell from "@/components/Shell";
import { useToast } from "@/components/Toast";
import { uploadCase } from "@/lib/api";
import { getUser } from "@/lib/auth";
import { MAX_UPLOAD_MB } from "@/lib/config";

// INTEGRATION(KARIM): the backend accepts only jpeg, png, webp, pdf, txt, csv and json.
// The cards below only change the file picker filter. The backend does not receive an input type.
const TYPES = [
  { id: "image", label: "Machine photo", hint: "JPG, PNG, WEBP", accept: "image/jpeg,image/png,image/webp", icon: Camera },
  { id: "sensor", label: "Sensor report", hint: "CSV, JSON, TXT", accept: ".csv,.json,.txt", icon: Activity },
  { id: "document", label: "Maintenance document", hint: "PDF, TXT", accept: ".pdf,.txt", icon: FileText },
] as const;

export default function UploadPage() {
  const router = useRouter();
  const toast = useToast();
  const inputRef = useRef<HTMLInputElement>(null);
  const [type, setType] = useState<(typeof TYPES)[number]["id"]>("image");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);

  const cfg = TYPES.find((t) => t.id === type)!;
  const preview = useMemo(
    () => (file && file.type.startsWith("image/") ? URL.createObjectURL(file) : null),
    [file]
  );

  function pick(f: File | null) {
    if (f && f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast("error", `File is larger than ${MAX_UPLOAD_MB} MB`);
      return;
    }
    setFile(f);
  }

  async function submit() {
    if (!file) return;
    setBusy(true);
    try {
      // INTEGRATION(KARIM, TITUS): this only stores the file. The AI diagnosis arrives a little later
      // (the case page checks every 2 seconds), because the backend analyses in the background.
      const c = await uploadCase(file, getUser() ?? "unknown");
      toast("success", "File uploaded. The AI is analysing it.");
      router.push(`/cases/${c.id}`);
    } catch (e) {
      toast("error", (e as Error).message);
      setBusy(false);
    }
  }

  return (
    <Shell>
      <h1 className="text-2xl font-bold">New case</h1>
      <p className="mb-6 text-sm text-slate-400">Upload a machine photo, sensor report or maintenance document.</p>

      <div className="grid max-w-3xl gap-3 sm:grid-cols-3">
        {TYPES.map(({ id, label, hint, icon: I }) => (
          <button key={id} disabled={busy} onClick={() => { setType(id); setFile(null); }}
            className={`card p-4 text-left transition ${type === id ? "border-cyan-400/60 bg-cyan-400/10" : "hover:bg-white/5"}`}>
            <I className={`mb-2 h-5 w-5 ${type === id ? "text-cyan-300" : "text-slate-400"}`} />
            <div className="text-sm font-medium">{label}</div>
            <div className="text-xs text-slate-500">{hint}</div>
          </button>
        ))}
      </div>

      <div className="card mt-4 max-w-3xl p-6">
        <div
          onClick={() => !busy && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0] ?? null); }}
          className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition ${
            drag ? "border-cyan-400 bg-cyan-400/10" : "border-white/15 hover:border-white/30"}`}
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="preview" className="max-h-56 rounded-lg" />
          ) : (
            <>
              <UploadCloud className="mb-2 h-9 w-9 text-slate-400" />
              <p className="text-sm">{file ? file.name : "Drag a file here or click to browse"}</p>
              <p className="text-xs text-slate-500">{cfg.hint}, max {MAX_UPLOAD_MB} MB</p>
            </>
          )}
          <input ref={inputRef} type="file" accept={cfg.accept} className="hidden"
            onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        </div>
        {file && preview && <p className="mt-2 text-xs text-slate-400">{file.name}</p>}

        <button onClick={submit} disabled={!file || busy}
          className="mt-4 w-full rounded-lg bg-cyan-500 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-40">
          {busy ? "Uploading..." : "Upload and analyse"}
        </button>
      </div>
    </Shell>
  );
}