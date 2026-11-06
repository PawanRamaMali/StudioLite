"use client";
import { useCallback, useEffect, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { UserCircle2, Upload, Download, AlertTriangle, Info } from "lucide-react";
import {
  uploadImage,
  identityStatus,
  generateWithIdentity,
  getJob,
  getDownloadUrl,
  type IdentityStatus,
  type ImageUploadResponse,
} from "@/lib/api";

const DIM_OPTIONS = [
  { w: 1024, h: 1024, label: "1024×1024 (1:1)" },
  { w: 832, h: 1216, label: "832×1216 (portrait)" },
  { w: 1216, h: 832, label: "1216×832 (landscape)" },
  { w: 896, h: 1152, label: "896×1152 (4:5 feed)" },
];

export default function IdentityPanel() {
  const [ref, setRef] = useState<ImageUploadResponse | null>(null);
  const [status, setStatus] = useState<IdentityStatus | null>(null);
  const [prompt, setPrompt] = useState("cinematic portrait, soft light, 35mm film, sharp focus");
  const [negative, setNegative] = useState("low quality, deformed, extra fingers, watermark");
  const [dims, setDims] = useState(DIM_OPTIONS[0]);
  const [idScale, setIdScale] = useState(0.8);
  const [guidance, setGuidance] = useState(5.0);
  const [steps, setSteps] = useState(25);
  const [seed, setSeed] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    identityStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const onRef = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    try {
      setRef(await uploadImage(file));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const start = async () => {
    if (!ref) {
      setError("Upload a reference face first.");
      return;
    }
    if (!prompt.trim()) {
      setError("Enter a prompt.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Queueing...");
    try {
      const job = await generateWithIdentity({
        reference_image: ref.image_path,
        prompt,
        negative_prompt: negative,
        width: dims.w,
        height: dims.h,
        id_scale: idScale,
        guidance_scale: guidance,
        num_inference_steps: steps,
        seed: seed ? Number(seed) : null,
      });
      const poll = setInterval(async () => {
        try {
          const j = await getJob(job.job_id);
          setProgress(Math.min(Math.round(j.progress || 0), 100));
          setProgressMsg(j.message || "Working...");
          if (j.status === "completed") {
            clearInterval(poll);
            const out = (j.result as { output_path?: string } | undefined)?.output_path;
            if (out) setResult(out);
            setRunning(false);
          } else if (j.status === "failed") {
            clearInterval(poll);
            setError(j.error || "Generation failed");
            setRunning(false);
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch (e) {
      setError((e as Error).message);
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <UserCircle2 className="w-6 h-6 text-indigo-400" />
        <h1 className="text-2xl font-bold">Identity-Anchored Image (PuLID)</h1>
      </div>
      <p className="text-sm text-zinc-400 -mt-4">
        Generate an SDXL scene image that keeps a specific person&apos;s face across
        prompts. Uses PuLID when installed; falls back to IP-Adapter conditioning
        on plain SDXL. The fallback produces usable identity carry-over but
        significantly weaker than PuLID proper.
      </p>

      {status && !status.pulid_available && (
        <Card className="border-sky-500/40 bg-sky-500/5">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-sky-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-sky-200">
              <div className="font-semibold mb-1">
                PuLID not installed — running IP-Adapter fallback.
              </div>
              <pre className="whitespace-pre-wrap text-xs text-sky-200/80 font-mono">
                {status.install_hint}
              </pre>
            </div>
          </div>
        </Card>
      )}
      {status && status.pulid_available && (
        <div className="text-xs text-green-400">PuLID is ready.</div>
      )}

      <Card>
        <CardTitle>1. Reference face</CardTitle>
        <label className="flex items-center gap-3 px-4 py-6 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-indigo-500 transition">
          <Upload className="w-5 h-5 text-zinc-400" />
          <input type="file" accept="image/png,image/jpeg,image/jpg" className="hidden"
                 onChange={(e) => onRef(e.target.files?.[0] || null)} />
          <div className="flex-1">
            <div className="font-medium">
              {ref ? ref.image_path.split(/[\\/]/).pop() : "JPG or PNG portrait (no webp)"}
            </div>
            {ref && (
              <div className="text-xs text-zinc-500 mt-1 truncate">{ref.image_path}</div>
            )}
          </div>
        </label>
        {ref && (
          <img src={ref.url} alt="reference" className="mt-3 h-32 rounded border border-zinc-800 object-cover" />
        )}
      </Card>

      <Card>
        <CardTitle>2. Prompt</CardTitle>
        <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)}
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
        <label className="block mt-3 text-xs text-zinc-400">Negative prompt</label>
        <textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)}
                  className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
      </Card>

      <Card>
        <CardTitle>3. Options</CardTitle>
        <div className="space-y-4">
          <div>
            <label className="text-xs text-zinc-400">Dimensions</label>
            <div className="grid grid-cols-2 gap-2 mt-1">
              {DIM_OPTIONS.map((d) => (
                <button key={d.label}
                        onClick={() => setDims(d)}
                        className={`text-left p-2 rounded border text-sm transition ${
                          dims.w === d.w && dims.h === d.h
                            ? "bg-indigo-600/15 border-indigo-500 text-indigo-200"
                            : "border-zinc-800 hover:border-zinc-700 text-zinc-300"
                        }`}>
                  {d.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs text-zinc-400">
              Identity strength: {idScale.toFixed(2)} (0.6 balances, 1.0 strict)
            </label>
            <input type="range" min={0.0} max={1.5} step={0.05}
                   value={idScale}
                   onChange={(e) => setIdScale(Number(e.target.value))}
                   className="w-full mt-1" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-zinc-400">Guidance: {guidance.toFixed(1)}</label>
              <input type="range" min={1.0} max={10.0} step={0.1}
                     value={guidance}
                     onChange={(e) => setGuidance(Number(e.target.value))}
                     className="w-full mt-1" />
            </div>
            <div>
              <label className="text-xs text-zinc-400">Steps: {steps}</label>
              <input type="range" min={4} max={60} step={1}
                     value={steps}
                     onChange={(e) => setSteps(Number(e.target.value))}
                     className="w-full mt-1" />
            </div>
          </div>
          <div>
            <label className="text-xs text-zinc-400">Seed (blank = random)</label>
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)}
                   placeholder="random"
                   className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
          </div>
        </div>
      </Card>

      <Card>
        <CardTitle>4. Generate</CardTitle>
        <Button onClick={start} disabled={!ref || !prompt.trim() || running} variant="primary">
          {running ? "Generating..." : "Generate image"}
        </Button>
        {running && (
          <div className="mt-4">
            <div className="h-2 bg-zinc-900 rounded overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="text-xs text-zinc-400 mt-2">{progress}% — {progressMsg}</div>
          </div>
        )}
        {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
        {result && (
          <div className="mt-4 space-y-3">
            <img src={getDownloadUrl(result)} alt="generated"
                 className="w-full rounded-lg border border-zinc-700" />
            <a href={getDownloadUrl(result)} target="_blank" rel="noreferrer"
               className="inline-flex items-center gap-2 text-sm text-green-400 hover:underline">
              <Download className="w-4 h-4" /> Download image
            </a>
          </div>
        )}
      </Card>

      {status && !status.pulid_available && (
        <Card>
          <div className="flex items-start gap-2 text-xs text-amber-300/80">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>
              Running IP-Adapter fallback — identity carry-over will be noticeably
              weaker than PuLID. Install PuLID weights to upgrade (see the status
              box above).
            </span>
          </div>
        </Card>
      )}
    </div>
  );
}
