"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ArrowUpCircle, Upload, Zap, Download } from "lucide-react";
import {
  uploadEditVideo,
  upscaleVideo,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type Job,
} from "@/lib/api";

interface Preset {
  id: string;
  name: string;
  desc: string;
  badge: "Fast" | "Quality" | "Ultra" | "Anime";
  scale: 1 | 2 | 3 | 4;
}

const PRESETS: Preset[] = [
  { id: "fast_2x",    name: "Fast 2x",    desc: "Lanczos interpolation", badge: "Fast",    scale: 2 },
  { id: "quality_2x", name: "Quality 2x", desc: "Real-ESRGAN",           badge: "Quality", scale: 2 },
  { id: "ultra_4x",   name: "Ultra 4x",   desc: "Real-ESRGAN 4x",        badge: "Ultra",   scale: 4 },
  { id: "anime_4x",   name: "Anime 4x",   desc: "Anime-optimized",       badge: "Anime",   scale: 4 },
];

export default function UpscalePanel() {
  const [preset, setPreset] = useState<Preset>(PRESETS[1]);
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => clearPoll, []);

  const onFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setJob(null);
    setUploading(true);
    try {
      const res = await uploadEditVideo(file);
      setUpload(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }, []);

  const onStart = async () => {
    if (!upload || job) return;
    setError(null);
    try {
      const j = await upscaleVideo({
        video_path: upload.video_path,
        scale: preset.scale,
        preset: preset.id,
      });
      setJob(j);
      // Poll every 2 s until the job leaves running/queued.
      clearPoll();
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getJob(j.job_id);
          setJob(latest);
          if (latest.status === "completed" || latest.status === "failed" || latest.status === "cancelled") {
            clearPoll();
          }
        } catch (err) {
          setError((err as Error).message);
          clearPoll();
        }
      }, 2000);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = uploading || (job && (job.status === "running" || job.status === "queued"));

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Upscale Video</h1>
        <p className="text-zinc-400 mt-1">Enhance video resolution up to 4K with AI super-resolution</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file"
                accept="video/*"
                className="hidden"
                disabled={!!busy}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-12 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-10 h-10 text-zinc-500 mx-auto mb-3" />
                {upload ? (
                  <>
                    <p className="text-sm text-zinc-300 truncate">{upload.filename}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {(upload.size_bytes / (1024 * 1024)).toFixed(1)} MB - click to replace
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-zinc-400">
                    {uploading ? "Uploading…" : "Drop video here or click to upload"}
                  </p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Upscale Preset</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPreset(p)}
                  disabled={!!busy}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    preset.id === p.id
                      ? "border-indigo-500 bg-indigo-500/10"
                      : "border-zinc-700 hover:border-zinc-600"
                  } ${busy ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{p.name}</span>
                    <Badge
                      variant={
                        p.badge === "Ultra"
                          ? "warning"
                          : p.badge === "Quality"
                          ? "info"
                          : "default"
                      }
                    >
                      {p.badge}
                    </Badge>
                  </div>
                  <p className="text-[10px] text-zinc-500 mt-1">{p.desc}</p>
                </button>
              ))}
            </div>
          </Card>

          <Button
            size="lg"
            className="w-full"
            disabled={!upload || !!busy}
            onClick={onStart}
          >
            <Zap className="w-5 h-5 mr-2" />
            {job && (job.status === "running" || job.status === "queued")
              ? `Upscaling - ${Math.round((job.progress || 0) * 100)}%`
              : "Upscale Video"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px] flex items-center justify-center">
          <div className="text-center w-full px-6">
            {job && job.status === "completed" ? (
              <>
                <ArrowUpCircle className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Upscale complete</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message || `Job ${job.job_id.slice(0, 8)}`}
                </p>
                <Button
                  size="sm"
                  variant="secondary"
                  className="mt-4"
                  onClick={() =>
                    window.open(jobDownloadUrl(job.job_id), "_blank", "noopener,noreferrer")
                  }
                >
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
              </>
            ) : job && (job.status === "failed" || job.status === "cancelled") ? (
              <>
                <ArrowUpCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">
                  Upscale {job.status}
                </p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <ArrowUpCircle className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
                <p className="text-sm text-zinc-300">
                  {job.status === "queued" ? "Queued…" : "Rendering…"}
                </p>
                <div className="mt-3 mx-auto max-w-xs h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all"
                    style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
                  />
                </div>
                {job.message && (
                  <p className="text-[11px] text-zinc-500 mt-2 truncate">
                    {job.message}
                  </p>
                )}
              </>
            ) : (
              <>
                <ArrowUpCircle className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Upscaled video will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">
                  Pick a preset and hit Upscale to start.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
