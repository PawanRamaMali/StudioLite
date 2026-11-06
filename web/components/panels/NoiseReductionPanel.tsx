"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Download, Volume2 } from "lucide-react";
import {
  denoiseAudio,
  getJob,
  type DenoiseMethod,
  type Job,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function NoiseReductionPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [method, setMethod] = useState<DenoiseMethod>("afftdn");
  // UI stores noise reduction as positive dB (0..60), converted at submit.
  const [reductionDb, setReductionDb] = useState(25);
  const [floorDb, setFloorDb] = useState(-40);
  const [smoothing, setSmoothing] = useState(0.002);
  const [format, setFormat] = useState<"wav" | "mp3">("wav");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => clearPoll, []);

  const onFile = useCallback((f: File | null) => {
    setFile(f);
    setJob(null);
    setError(null);
  }, []);

  const onStart = async () => {
    if (!file || job) return;
    setError(null);
    try {
      const j = await denoiseAudio(file, {
        method,
        noise_reduction_db: -Math.abs(reductionDb),
        noise_floor_db: floorDb,
        smoothing,
        output_format: format,
      });
      setJob(j);
      clearPoll();
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getJob(j.job_id);
          setJob(latest);
          if (latest.status === "completed" || latest.status === "failed" || latest.status === "cancelled") {
            clearPoll();
          }
        } catch (e) {
          setError((e as Error).message);
          clearPoll();
        }
      }, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = !!job && (job.status === "running" || job.status === "queued");
  const audioUrl = job?.status === "completed"
    ? (job.result as { audio_url?: string } | undefined)?.audio_url
    : undefined;
  const fullUrl = audioUrl ? `${API_BASE}${audioUrl}` : "";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Noise Reduction</h1>
        <p className="text-zinc-400 mt-1">
          Strip hum, hiss, and background noise. FFT mode (afftdn) is fast and
          handles broadband noise; NLM mode (anlmdn) smooths stationary hiss
          more gently at the cost of render time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file"
                accept="audio/*,video/*"
                className="hidden"
                disabled={busy}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-10 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                {file ? (
                  <>
                    <p className="text-sm text-zinc-300 truncate">{file.name}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {(file.size / (1024 * 1024)).toFixed(1)} MB &mdash; click to replace
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-zinc-400">Drop audio here or click to upload</p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Method</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setMethod("afftdn")}
                disabled={busy}
                className={`px-3 py-2 rounded-md text-sm border transition-colors ${
                  method === "afftdn"
                    ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                    : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >FFT (fast)</button>
              <button
                onClick={() => setMethod("anlmdn")}
                disabled={busy}
                className={`px-3 py-2 rounded-md text-sm border transition-colors ${
                  method === "anlmdn"
                    ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                    : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                }`}
              >NLM (gentle)</button>
            </div>

            {method === "afftdn" ? (
              <div className="space-y-3 mt-3">
                <div>
                  <label className="text-xs text-zinc-400">
                    Reduction: {reductionDb} dB
                  </label>
                  <input
                    type="range" min={6} max={60} step={1}
                    value={reductionDb}
                    disabled={busy}
                    onChange={(e) => setReductionDb(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400">
                    Noise floor: {floorDb} dB
                  </label>
                  <input
                    type="range" min={-80} max={-20} step={1}
                    value={floorDb}
                    disabled={busy}
                    onChange={(e) => setFloorDb(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                  <p className="text-[11px] text-zinc-500 mt-1">
                    Anything quieter than this is treated as noise.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3 mt-3">
                <div>
                  <label className="text-xs text-zinc-400">
                    Smoothing: {smoothing.toFixed(4)} s
                  </label>
                  <input
                    type="range" min={0.0005} max={0.01} step={0.0005}
                    value={smoothing}
                    disabled={busy}
                    onChange={(e) => setSmoothing(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Output</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {(["wav", "mp3"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  disabled={busy}
                  className={`px-3 py-2 rounded-md text-xs uppercase border transition-colors ${
                    format === f
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >{f}</button>
              ))}
            </div>
          </Card>

          <Button size="lg" className="w-full" disabled={!file || busy} onClick={onStart}>
            <Zap className="w-5 h-5 mr-2" />
            {busy ? `Processing - ${Math.round((job?.progress || 0) * 100)}%` : "Remove noise"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px] flex items-center justify-center">
          <div className="text-center w-full px-6">
            {job && job.status === "completed" && fullUrl ? (
              <>
                <Volume2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Denoised</p>
                <audio key={fullUrl} src={fullUrl} controls className="w-full mt-4" />
                <Button
                  size="sm" variant="secondary" className="mt-4"
                  onClick={() => window.open(fullUrl, "_blank", "noopener,noreferrer")}
                >
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
              </>
            ) : job && job.status === "failed" ? (
              <p className="text-sm text-red-400">{job.error || "Denoise failed"}</p>
            ) : (
              <p className="text-sm text-zinc-500">Output appears here once processing completes.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
