"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Smartphone, Upload, Download } from "lucide-react";
import {
  uploadEditVideo,
  exportSocial,
  listFormats,
  getJob,
  getDownloadUrl,
  type EditUploadResponse,
  type VideoFormatPreset,
  type SocialFitMode,
} from "@/lib/api";

const FIT_MODES: { id: SocialFitMode; label: string; hint: string }[] = [
  { id: "pad_blur", label: "Blurred fill", hint: "Fit the source, fill empty bars with a blurred copy. Best default." },
  { id: "pad_black", label: "Black bars", hint: "Fit the source, add solid black bars. Classic letterbox look." },
  { id: "crop", label: "Crop to fill", hint: "Zoom in and crop edges so the frame is completely filled." },
  { id: "stretch", label: "Stretch", hint: "Stretch to the target aspect. Usually bad - don't use unless intentional." },
];

export default function SocialExportPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [formats, setFormats] = useState<VideoFormatPreset[]>([]);
  const [formatId, setFormatId] = useState<string>("reels");
  const [fitMode, setFitMode] = useState<SocialFitMode>("pad_blur");
  const [normalize, setNormalize] = useState(true);
  const [subtitlePath, setSubtitlePath] = useState("");
  const [maxDuration, setMaxDuration] = useState<string>("");
  const [captionSize, setCaptionSize] = useState<number>(4.5);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>("");
  const [resultPath, setResultPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listFormats().then((r) => setFormats(r.formats || [])).catch(() => setFormats([]));
  }, []);

  const selected = useMemo(
    () => formats.find((f) => f.id === formatId) || null,
    [formats, formatId],
  );

  const onFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setResultPath(null);
    setUploading(true);
    try {
      const r = await uploadEditVideo(file);
      setUpload(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }, []);

  const start = async () => {
    if (!upload) {
      setError("Upload a video first.");
      return;
    }
    setError(null);
    setResultPath(null);
    setRunning(true);
    setProgress(0);
    setStatus("Queueing export...");

    try {
      const job = await exportSocial({
        video_path: upload.video_path,
        format_id: formatId,
        fit_mode: fitMode,
        subtitle_path: subtitlePath || undefined,
        normalize_loudness: normalize,
        max_duration_s: maxDuration ? Number(maxDuration) : null,
        caption_font_size_pct: captionSize / 100,
      });

      const poll = setInterval(async () => {
        try {
          const j = await getJob(job.job_id);
          setProgress(Math.min(Math.round(j.progress || 0), 100));
          setStatus(j.message || "Working...");
          if (j.status === "completed") {
            clearInterval(poll);
            const out = (j.result as { output_path?: string } | undefined)?.output_path;
            if (out) setResultPath(out);
            setRunning(false);
          } else if (j.status === "failed") {
            clearInterval(poll);
            setError(j.error || "Export failed");
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
        <Smartphone className="w-6 h-6 text-indigo-400" />
        <h1 className="text-2xl font-bold">Social Export</h1>
      </div>
      <p className="text-sm text-zinc-400 -mt-4">
        Reshape a finished video for Reels, Shorts, TikTok, or feed posts. Handles
        aspect, duration cap, loudness normalization, optional burned captions,
        and H.264 under the platform's bitrate ceiling.
      </p>

      {/* Upload */}
      <Card>
        <CardTitle>1. Source video</CardTitle>
        <label className="flex items-center gap-3 px-4 py-6 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-indigo-500 transition">
          <Upload className="w-5 h-5 text-zinc-400" />
          <input type="file" accept="video/*" className="hidden"
                 onChange={(e) => onFile(e.target.files?.[0] || null)} />
          <div className="flex-1">
            <div className="font-medium">
              {upload ? upload.filename : (uploading ? "Uploading..." : "Click to upload a video")}
            </div>
            {upload && (
              <div className="text-xs text-zinc-500 mt-1 truncate">{upload.video_path}</div>
            )}
          </div>
        </label>
      </Card>

      {/* Format */}
      <Card>
        <CardTitle>2. Target platform</CardTitle>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
          {formats.map((f) => (
            <button key={f.id}
                    onClick={() => setFormatId(f.id)}
                    className={`text-left p-3 rounded-lg border transition ${
                      formatId === f.id
                        ? "bg-indigo-600/15 border-indigo-500 text-indigo-200"
                        : "border-zinc-800 hover:border-zinc-700 text-zinc-300"
                    }`}>
              <div className="font-semibold text-sm">{f.label}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">
                {f.width}x{f.height} · {f.target_fps}fps · ≤{Math.round(f.max_duration_s)}s
              </div>
            </button>
          ))}
        </div>
        {selected && (
          <div className="mt-3 text-xs text-zinc-500 italic">{selected.notes}</div>
        )}
      </Card>

      {/* Fit mode */}
      <Card>
        <CardTitle>3. How to fit the frame</CardTitle>
        <div className="grid grid-cols-2 gap-2">
          {FIT_MODES.map((m) => (
            <button key={m.id}
                    onClick={() => setFitMode(m.id)}
                    className={`text-left p-3 rounded-lg border transition ${
                      fitMode === m.id
                        ? "bg-indigo-600/15 border-indigo-500 text-indigo-200"
                        : "border-zinc-800 hover:border-zinc-700 text-zinc-300"
                    }`}>
              <div className="font-semibold text-sm">{m.label}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">{m.hint}</div>
            </button>
          ))}
        </div>
      </Card>

      {/* Options */}
      <Card>
        <CardTitle>4. Options</CardTitle>
        <div className="space-y-4">
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={normalize}
                   onChange={(e) => setNormalize(e.target.checked)} />
            Normalize loudness to {selected?.audio_lufs ?? -14} LUFS (recommended)
          </label>

          <div>
            <label className="text-xs text-zinc-400">Burn subtitles (optional server path to .srt/.vtt)</label>
            <input type="text" value={subtitlePath}
                   onChange={(e) => setSubtitlePath(e.target.value)}
                   placeholder=".mp/captions/clip.srt"
                   className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-zinc-400">Max duration (seconds, blank = platform cap)</label>
              <input type="number" value={maxDuration} min={1} max={600}
                     onChange={(e) => setMaxDuration(e.target.value)}
                     placeholder={`${selected?.max_duration_s ?? 60}`}
                     className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
            </div>
            <div>
              <label className="text-xs text-zinc-400">Caption size (% of height)</label>
              <input type="number" value={captionSize} min={1.5} max={12} step={0.5}
                     onChange={(e) => setCaptionSize(Number(e.target.value))}
                     className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
            </div>
          </div>
        </div>
      </Card>

      {/* Run */}
      <Card>
        <CardTitle>5. Export</CardTitle>
        <Button onClick={start} disabled={!upload || running} variant="primary">
          {running ? "Exporting..." : "Export for " + (selected?.label ?? formatId)}
        </Button>

        {running && (
          <div className="mt-4">
            <div className="h-2 bg-zinc-900 rounded overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="text-xs text-zinc-400 mt-2">{progress}% — {status}</div>
          </div>
        )}

        {error && (
          <div className="mt-3 text-sm text-red-400">{error}</div>
        )}

        {resultPath && (
          <div className="mt-4 flex items-center gap-3">
            <Download className="w-4 h-4 text-green-400" />
            <a href={getDownloadUrl(resultPath)} target="_blank" rel="noreferrer"
               className="text-sm text-green-400 hover:underline">
              Download exported video
            </a>
          </div>
        )}
      </Card>
    </div>
  );
}
