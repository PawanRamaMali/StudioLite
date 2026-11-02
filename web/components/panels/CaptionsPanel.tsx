"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Captions, Upload, Zap, Download } from "lucide-react";
import {
  uploadEditVideo,
  captionVideo,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type Job,
} from "@/lib/api";

type Position = "top" | "middle" | "bottom";

const POSITIONS: { id: Position; label: string }[] = [
  { id: "top",    label: "Top" },
  { id: "middle", label: "Middle" },
  { id: "bottom", label: "Bottom" },
];

const FONT_COLORS = [
  { id: "white",   label: "White",   swatch: "#ffffff" },
  { id: "yellow",  label: "Yellow",  swatch: "#ffff00" },
  { id: "cyan",    label: "Cyan",    swatch: "#00ffff" },
  { id: "#ff8800", label: "Orange",  swatch: "#ff8800" },
];

export default function CaptionsPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [fontSize, setFontSize] = useState(24);
  const [fontColor, setFontColor] = useState("white");
  const [outlineWidth, setOutlineWidth] = useState(2);
  const [position, setPosition] = useState<Position>("bottom");
  const [bold, setBold] = useState(false);
  const [italic, setItalic] = useState(false);

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
      const j = await captionVideo({
        video_path: upload.video_path,
        font_size: fontSize,
        font_color: fontColor,
        outline_width: outlineWidth,
        position,
        bold,
        italic,
      });
      setJob(j);
      clearPoll();
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getJob(j.job_id);
          setJob(latest);
          if (
            latest.status === "completed" ||
            latest.status === "failed" ||
            latest.status === "cancelled"
          ) {
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
        <h1 className="text-3xl font-bold gradient-text">Auto Captions</h1>
        <p className="text-zinc-400 mt-1">
          Transcribe speech with faster-whisper and burn subtitles into the video.
        </p>
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
            <CardTitle className="text-sm mb-3">Style</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Size: {fontSize}px</label>
                <input
                  type="range"
                  min={10}
                  max={64}
                  value={fontSize}
                  disabled={!!busy}
                  onChange={(e) => setFontSize(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Outline: {outlineWidth}px</label>
                <input
                  type="range"
                  min={0}
                  max={6}
                  value={outlineWidth}
                  disabled={!!busy}
                  onChange={(e) => setOutlineWidth(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>

              <div>
                <div className="text-xs text-zinc-400 mb-1">Text Color</div>
                <div className="flex gap-2">
                  {FONT_COLORS.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setFontColor(c.id)}
                      disabled={!!busy}
                      className={`flex items-center gap-2 px-2 py-1 rounded border text-[11px] transition-colors ${
                        fontColor === c.id
                          ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                          : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                      }`}
                    >
                      <span
                        className="w-3 h-3 rounded-sm border border-zinc-700"
                        style={{ background: c.swatch }}
                      />
                      {c.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-xs text-zinc-400 mb-1">Position</div>
                <div className="flex gap-2">
                  {POSITIONS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setPosition(p.id)}
                      disabled={!!busy}
                      className={`px-3 py-1 rounded border text-[11px] transition-colors ${
                        position === p.id
                          ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                          : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-4 pt-1">
                <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={bold}
                    onChange={(e) => setBold(e.target.checked)}
                    disabled={!!busy}
                    className="accent-indigo-500"
                  />
                  Bold
                </label>
                <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={italic}
                    onChange={(e) => setItalic(e.target.checked)}
                    disabled={!!busy}
                    className="accent-indigo-500"
                  />
                  Italic
                </label>
              </div>
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
              ? `Captioning - ${Math.round((job.progress || 0) * 100)}%`
              : "Transcribe & Burn Captions"}
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
                <Captions className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Captions burned in</p>
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
                <Captions className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">Captioning {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <Captions className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
                <p className="text-sm text-zinc-300">
                  {job.status === "queued" ? "Queued…" : "Working…"}
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
                <Captions className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Captioned video will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">
                  Upload a video with speech and hit the button to start.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
