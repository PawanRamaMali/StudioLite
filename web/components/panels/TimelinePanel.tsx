"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  Film, Upload, Play, Trash2, Download, Scissors,
  ChevronLeft, ChevronRight, Loader2, AlertCircle,
} from "lucide-react";
import {
  uploadEditVideo,
  renderTimeline,
  getJob,
  jobDownloadUrl,
  getLicenseStatus,
  type EditUploadResponse,
  type Job,
  type LicenseStatus,
} from "@/lib/api";

// A clip on the timeline: source video + in/out points in seconds.
// duration is the source file's total length, cached so trim sliders
// can bound themselves without probing the file every render.
interface Clip {
  id: string;
  name: string;
  video_path: string;
  duration: number;   // source duration
  in_point: number;
  out_point: number;
}

const CODECS = [
  { id: "h264", label: "H.264", desc: "Widest compatibility (mp4)" },
  { id: "h265", label: "H.265 / HEVC", desc: "Smaller files, newer decoders" },
  { id: "prores", label: "ProRes", desc: "Post-friendly (mov)" },
] as const;

const QUALITIES = [
  { id: "high", label: "High" },
  { id: "medium", label: "Medium" },
  { id: "low", label: "Low" },
] as const;

interface Resolution { label: string; width: number; height: number; }
const RESOLUTIONS: readonly Resolution[] = [
  { label: "1080p (1920×1080)", width: 1920, height: 1080 },
  { label: "720p (1280×720)", width: 1280, height: 720 },
  { label: "4K (3840×2160)", width: 3840, height: 2160 },
];

type CodecId = typeof CODECS[number]["id"];
type QualityId = typeof QUALITIES[number]["id"];

function fmt(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.0";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

async function probeDuration(url: string): Promise<number> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => resolve(Number.isFinite(v.duration) ? v.duration : 0);
    v.onerror = () => resolve(0);
    v.src = url;
  });
}

export default function TimelinePanel() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [codec, setCodec] = useState<CodecId>("h264");
  const [quality, setQuality] = useState<QualityId>("high");
  const [resolution, setResolution] = useState(RESOLUTIONS[0]);
  const [fps, setFps] = useState(30);
  const [license, setLicense] = useState<LicenseStatus | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };
  useEffect(() => clearPoll, []);

  // License status controls the watermark toggle: only tiers with
  // watermark_removal (or Pro/Studio) can turn it off.
  useEffect(() => {
    getLicenseStatus().then(setLicense).catch(() => setLicense(null));
  }, []);
  const canRemoveWatermark =
    !!license &&
    license.valid &&
    (license.tier === "pro" ||
      license.tier === "studio" ||
      license.features?.includes("watermark_removal"));
  const [wantWatermark, setWantWatermark] = useState(true);
  const applyWatermark = canRemoveWatermark ? wantWatermark : true;

  const selected = clips.find((c) => c.id === selectedId) || null;
  const totalDuration = clips.reduce((acc, c) => acc + (c.out_point - c.in_point), 0);

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    const localUrl = URL.createObjectURL(file);
    try {
      const [saved, duration] = await Promise.all([
        uploadEditVideo(file),
        probeDuration(localUrl),
      ]);
      const clip: Clip = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: file.name,
        video_path: saved.video_path,
        duration: duration || 10,
        in_point: 0,
        out_point: duration || 10,
      };
      setClips((prev) => [...prev, clip]);
      setSelectedId(clip.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      URL.revokeObjectURL(localUrl);
      setUploading(false);
    }
  }, []);

  const move = (id: string, delta: -1 | 1) => {
    setClips((prev) => {
      const idx = prev.findIndex((c) => c.id === id);
      if (idx < 0) return prev;
      const target = idx + delta;
      if (target < 0 || target >= prev.length) return prev;
      const copy = prev.slice();
      [copy[idx], copy[target]] = [copy[target], copy[idx]];
      return copy;
    });
  };

  const removeClip = (id: string) => {
    setClips((prev) => prev.filter((c) => c.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  const updateTrim = (id: string, patch: Partial<Pick<Clip, "in_point" | "out_point">>) => {
    setClips((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        let inp = patch.in_point ?? c.in_point;
        let outp = patch.out_point ?? c.out_point;
        inp = Math.max(0, Math.min(inp, c.duration - 0.1));
        outp = Math.max(inp + 0.1, Math.min(outp, c.duration));
        return { ...c, in_point: inp, out_point: outp };
      })
    );
  };

  const onRender = async () => {
    if (clips.length === 0) return;
    setError(null);
    try {
      const j = await renderTimeline({
        clips: clips.map((c) => ({
          video_path: c.video_path,
          in_point: c.in_point,
          out_point: c.out_point,
        })),
        fps,
        width: resolution.width,
        height: resolution.height,
        codec,
        quality,
        apply_watermark: applyWatermark,
      });
      setJob(j);
      clearPoll();
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getJob(j.job_id);
          setJob(latest);
          if (["completed", "failed", "cancelled"].includes(latest.status)) {
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

  const rendering = !!job && (job.status === "running" || job.status === "queued");
  const busy = uploading || rendering;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Timeline Editor</h1>
        <p className="text-zinc-400 mt-1">
          Sequence clips, trim in/out, and export with a codec preset
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {/* Timeline row */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm">Timeline</CardTitle>
              <div className="text-xs text-zinc-500">
                {clips.length} clip{clips.length === 1 ? "" : "s"} · {fmt(totalDuration)}
              </div>
            </div>

            <div className="space-y-2">
              {clips.length === 0 && (
                <div className="text-center text-xs text-zinc-500 py-6 border border-dashed border-zinc-800 rounded-md">
                  Upload a video to add it to the timeline.
                </div>
              )}
              {clips.map((c, i) => {
                const dur = c.out_point - c.in_point;
                return (
                  <div
                    key={c.id}
                    onClick={() => setSelectedId(c.id)}
                    className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-all ${
                      selectedId === c.id
                        ? "border-indigo-500 bg-indigo-500/10"
                        : "border-zinc-700 hover:border-zinc-600"
                    }`}
                  >
                    <div className="w-6 text-[10px] text-zinc-500 text-center">{i + 1}</div>
                    <Film className="w-4 h-4 text-indigo-300 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium truncate">{c.name}</div>
                      <div className="text-[10px] text-zinc-500">
                        {fmt(c.in_point)} → {fmt(c.out_point)} · {fmt(dur)}
                      </div>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); move(c.id, -1); }}
                      disabled={i === 0}
                      className="p-1 rounded hover:bg-zinc-800 disabled:opacity-30"
                      title="Move up"
                    ><ChevronLeft className="w-3 h-3 rotate-90" /></button>
                    <button
                      onClick={(e) => { e.stopPropagation(); move(c.id, 1); }}
                      disabled={i === clips.length - 1}
                      className="p-1 rounded hover:bg-zinc-800 disabled:opacity-30"
                      title="Move down"
                    ><ChevronRight className="w-3 h-3 rotate-90" /></button>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeClip(c.id); }}
                      className="p-1 rounded text-red-400 hover:bg-red-900/40"
                      title="Remove"
                    ><Trash2 className="w-3 h-3" /></button>
                  </div>
                );
              })}
            </div>

            <div className="mt-3">
              <input
                ref={fileRef}
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleUpload(f);
                  e.target.value = "";
                }}
              />
              <Button
                variant="secondary"
                onClick={() => fileRef.current?.click()}
                disabled={busy}
                className="w-full"
              >
                <Upload className="w-4 h-4 mr-2" />
                {uploading ? "Uploading…" : "Add Clip"}
              </Button>
            </div>
          </Card>

          {/* Trim editor */}
          {selected && (
            <Card>
              <div className="flex items-center gap-2 mb-3">
                <Scissors className="w-4 h-4 text-indigo-300" />
                <CardTitle className="text-sm mb-0">Trim: {selected.name}</CardTitle>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-zinc-500">
                    In point: {fmt(selected.in_point)}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={selected.duration}
                    step={0.1}
                    value={selected.in_point}
                    onChange={(e) =>
                      updateTrim(selected.id, { in_point: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-zinc-500">
                    Out point: {fmt(selected.out_point)}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={selected.duration}
                    step={0.1}
                    value={selected.out_point}
                    onChange={(e) =>
                      updateTrim(selected.id, { out_point: Number(e.target.value) })
                    }
                    className="w-full accent-indigo-500"
                  />
                </div>
                <p className="text-xs text-zinc-500">
                  Clip length: {fmt(selected.out_point - selected.in_point)}
                </p>
              </div>
            </Card>
          )}

          {/* Preview / render output */}
          <Card className="min-h-[240px] flex items-center justify-center">
            {rendering ? (
              <div className="text-center w-full max-w-md">
                <Loader2 className="w-8 h-8 text-indigo-400 mx-auto mb-3 animate-spin" />
                <p className="text-sm text-indigo-400">
                  {job?.message || "Rendering…"}
                </p>
                <div className="mt-3 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all"
                    style={{ width: `${Math.round((job?.progress || 0))}%` }}
                  />
                </div>
                <p className="text-[10px] text-zinc-500 mt-1">
                  {Math.round(job?.progress || 0)}%
                </p>
              </div>
            ) : job && job.status === "completed" ? (
              <div className="text-center">
                <Film className="w-10 h-10 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Export ready</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message}
                </p>
                <Button
                  size="sm"
                  className="mt-4"
                  onClick={() =>
                    window.open(jobDownloadUrl(job.job_id), "_blank", "noopener,noreferrer")
                  }
                >
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
              </div>
            ) : job && (job.status === "failed" || job.status === "cancelled") ? (
              <div className="text-center">
                <AlertCircle className="w-10 h-10 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">Render {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </div>
            ) : (
              <div className="text-center">
                <Film className="w-10 h-10 text-zinc-700 mx-auto mb-2" />
                <p className="text-sm text-zinc-500">Add clips and press Render</p>
              </div>
            )}
          </Card>
        </div>

        {/* Export settings */}
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Codec</CardTitle>
            <div className="space-y-2">
              {CODECS.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setCodec(c.id)}
                  disabled={busy}
                  className={`w-full p-2.5 rounded-lg border text-left transition-all ${
                    codec === c.id
                      ? "border-indigo-500 bg-indigo-500/10"
                      : "border-zinc-700 hover:border-zinc-600"
                  }`}
                >
                  <div className="text-xs font-medium">{c.label}</div>
                  <div className="text-[10px] text-zinc-500">{c.desc}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Quality</CardTitle>
            <div className="grid grid-cols-3 gap-2">
              {QUALITIES.map((q) => (
                <button
                  key={q.id}
                  onClick={() => setQuality(q.id)}
                  disabled={busy}
                  className={`p-2 rounded-lg border text-xs font-medium transition-all ${
                    quality === q.id
                      ? "border-indigo-500 bg-indigo-500/10 text-indigo-300"
                      : "border-zinc-700 hover:border-zinc-600"
                  }`}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Canvas</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] uppercase tracking-wide text-zinc-500">
                  Resolution
                </label>
                <select
                  value={resolution.label}
                  onChange={(e) => {
                    const r = RESOLUTIONS.find((x) => x.label === e.target.value);
                    if (r) setResolution(r);
                  }}
                  disabled={busy}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 mt-1"
                >
                  {RESOLUTIONS.map((r) => (
                    <option key={r.label} value={r.label}>{r.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[10px] uppercase tracking-wide text-zinc-500">
                  FPS: {fps}
                </label>
                <input
                  type="range"
                  min={12}
                  max={60}
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  disabled={busy}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Watermark</CardTitle>
            {canRemoveWatermark ? (
              <label className="flex items-center gap-2 text-xs text-zinc-300">
                <input
                  type="checkbox"
                  checked={wantWatermark}
                  onChange={(e) => setWantWatermark(e.target.checked)}
                  className="accent-indigo-500"
                />
                Apply StudioLite watermark
              </label>
            ) : (
              <div className="text-xs text-zinc-500">
                Free-tier exports include a StudioLite watermark. Install a
                Pro/Studio license to remove it.
              </div>
            )}
          </Card>

          <Button
            className="w-full"
            size="lg"
            onClick={onRender}
            disabled={busy || clips.length === 0}
          >
            <Play className="w-4 h-4 mr-2" />
            {rendering ? `Rendering ${Math.round(job?.progress || 0)}%` : "Render"}
          </Button>

          {error && (
            <div className="text-xs text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
