"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Wand2, Upload, Zap, Download } from "lucide-react";
import {
  uploadEditVideo,
  uploadImage,
  chromaKeyRemove,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type ImageUploadResponse,
  type Job,
  type ChromaKeyMode,
} from "@/lib/api";

const MODES: { id: ChromaKeyMode; label: string; detail: string }[] = [
  { id: "transparent", label: "Transparent",  detail: "VP9+alpha .webm with the key as transparency." },
  { id: "color",       label: "Solid color",  detail: "Composite over a flat background color." },
  { id: "image",       label: "Image",        detail: "Composite over a still image." },
  { id: "video",       label: "Video",        detail: "Composite over another video clip." },
];

const PRESETS = [
  { id: "green",   label: "Green",   swatch: "#00ff00" },
  { id: "blue",    label: "Blue",    swatch: "#0000ff" },
  { id: "red",     label: "Red",     swatch: "#ff0000" },
  { id: "cyan",    label: "Cyan",    swatch: "#00ffff" },
  { id: "magenta", label: "Magenta", swatch: "#ff00ff" },
];

export default function ChromaKeyPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<ChromaKeyMode>("transparent");
  const [keyColor, setKeyColor] = useState("green");
  const [similarity, setSimilarity] = useState(0.15);
  const [blend, setBlend] = useState(0.08);
  const [outWidth, setOutWidth] = useState(1280);
  const [outHeight, setOutHeight] = useState(720);
  const [fps, setFps] = useState(30);
  const [audio, setAudio] = useState<"copy" | "drop">("copy");
  const [backgroundColor, setBackgroundColor] = useState("#000000");

  // Image background and video background each need their own upload
  // because the server-side upload endpoints validate the file type.
  const [imageBg, setImageBg] = useState<ImageUploadResponse | null>(null);
  const [imageBgName, setImageBgName] = useState("");
  const [imageBgUploading, setImageBgUploading] = useState(false);
  const [videoBg, setVideoBg] = useState<EditUploadResponse | null>(null);
  const [videoBgUploading, setVideoBgUploading] = useState(false);

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
      setUpload(await uploadEditVideo(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }, []);

  const onImageBg = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setImageBgUploading(true);
    try {
      setImageBg(await uploadImage(file));
      setImageBgName(file.name);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setImageBgUploading(false);
    }
  }, []);

  const onVideoBg = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setVideoBgUploading(true);
    try {
      setVideoBg(await uploadEditVideo(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setVideoBgUploading(false);
    }
  }, []);

  const onStart = async () => {
    if (!upload || job) return;
    setError(null);

    let bgPath: string | undefined;
    if (mode === "image") {
      if (!imageBg) {
        setError("Upload a background image for image mode.");
        return;
      }
      bgPath = imageBg.image_path;
    } else if (mode === "video") {
      if (!videoBg) {
        setError("Upload a background video for video mode.");
        return;
      }
      bgPath = videoBg.video_path;
    }

    try {
      // The server normalizes hex (#RRGGBB) to 0xRRGGBB on its own, so
      // sending the HTML colour-picker value as-is works for both key
      // and background.
      const j = await chromaKeyRemove({
        video_path: upload.video_path,
        mode,
        key_color: keyColor,
        similarity,
        blend,
        out_width: outWidth,
        out_height: outHeight,
        fps,
        audio,
        background_color: backgroundColor,
        background_path: bgPath,
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
      }, 1800);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = uploading || (job && (job.status === "running" || job.status === "queued"));
  const isWebm = job?.result?.mode === "transparent";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Chroma Key</h1>
        <p className="text-zinc-400 mt-1">
          Remove a green (or any) screen and either keep the alpha, or
          composite over a color, image, or second video.
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
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-10 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                {upload ? (
                  <>
                    <p className="text-sm text-zinc-300 truncate">{upload.filename}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {(upload.size_bytes / (1024 * 1024)).toFixed(1)} MB - click to replace
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-zinc-400">
                    {uploading ? "Uploading..." : "Drop video here or click to upload"}
                  </p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Output mode</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  disabled={!!busy}
                  title={m.detail}
                  className={`px-2 py-2 rounded border text-[11px] transition-colors ${
                    mode === m.id
                      ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                      : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-zinc-500 mt-2">
              {MODES.find((m) => m.id === mode)?.detail}
            </p>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Key color</CardTitle>
            <div className="flex gap-2 flex-wrap mb-3">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setKeyColor(p.id)}
                  disabled={!!busy}
                  className={`flex items-center gap-2 px-2 py-1 rounded border text-[11px] transition-colors ${
                    keyColor === p.id
                      ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                      : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  <span
                    className="w-3 h-3 rounded-sm border border-zinc-700"
                    style={{ background: p.swatch }}
                  />
                  {p.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-400">Custom:</label>
              <input
                type="color"
                value={keyColor.startsWith("#") ? keyColor : "#00ff00"}
                disabled={!!busy}
                onChange={(e) => setKeyColor(e.target.value)}
                className="h-7 w-10 bg-transparent"
              />
              <input
                type="text"
                value={keyColor}
                disabled={!!busy}
                onChange={(e) => setKeyColor(e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs"
                placeholder="#RRGGBB or name"
              />
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Key tolerance</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">
                  Similarity: {(similarity * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min={0} max={1} step={0.01} value={similarity}
                  disabled={!!busy}
                  onChange={(e) => setSimilarity(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  How wide a range around the key is removed. Too high eats
                  the foreground.
                </p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">
                  Blend: {(blend * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min={0} max={1} step={0.01} value={blend}
                  disabled={!!busy}
                  onChange={(e) => setBlend(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  Softens the edge. Helps with hair and motion blur.
                </p>
              </div>
            </div>
          </Card>

          {mode === "color" && (
            <Card>
              <CardTitle className="text-sm mb-3">Background color</CardTitle>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={backgroundColor.startsWith("#") ? backgroundColor : "#000000"}
                  disabled={!!busy}
                  onChange={(e) => setBackgroundColor(e.target.value)}
                  className="h-7 w-10 bg-transparent"
                />
                <input
                  type="text"
                  value={backgroundColor}
                  disabled={!!busy}
                  onChange={(e) => setBackgroundColor(e.target.value)}
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs"
                  placeholder="#000000 or 0x000000"
                />
              </div>
            </Card>
          )}

          {mode === "image" && (
            <Card>
              <CardTitle className="text-sm mb-3">Background image</CardTitle>
              <label className="block cursor-pointer">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  disabled={!!busy}
                  onChange={(e) => onImageBg(e.target.files?.[0] ?? null)}
                />
                <div className="border border-dashed border-zinc-700 rounded p-3 text-center hover:border-indigo-500/50 transition-colors">
                  {imageBg ? (
                    <p className="text-xs text-zinc-300 truncate">{imageBgName || "background.png"}</p>
                  ) : (
                    <p className="text-xs text-zinc-400">
                      {imageBgUploading ? "Uploading..." : "Click to upload background image"}
                    </p>
                  )}
                </div>
              </label>
            </Card>
          )}

          {mode === "video" && (
            <Card>
              <CardTitle className="text-sm mb-3">Background video</CardTitle>
              <label className="block cursor-pointer">
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  disabled={!!busy}
                  onChange={(e) => onVideoBg(e.target.files?.[0] ?? null)}
                />
                <div className="border border-dashed border-zinc-700 rounded p-3 text-center hover:border-indigo-500/50 transition-colors">
                  {videoBg ? (
                    <p className="text-xs text-zinc-300 truncate">{videoBg.filename}</p>
                  ) : (
                    <p className="text-xs text-zinc-400">
                      {videoBgUploading ? "Uploading..." : "Click to upload background video"}
                    </p>
                  )}
                </div>
              </label>
            </Card>
          )}

          {mode !== "transparent" && (
            <Card>
              <CardTitle className="text-sm mb-3">Output size</CardTitle>
              <div className="flex gap-3">
                <div className="flex-1">
                  <label className="text-xs text-zinc-400">Width: {outWidth}</label>
                  <input
                    type="range" min={240} max={3840} step={20} value={outWidth}
                    disabled={!!busy}
                    onChange={(e) => setOutWidth(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-zinc-400">Height: {outHeight}</label>
                  <input
                    type="range" min={240} max={2160} step={20} value={outHeight}
                    disabled={!!busy}
                    onChange={(e) => setOutHeight(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div className="w-16">
                  <label className="text-xs text-zinc-400">FPS: {fps}</label>
                  <input
                    type="range" min={12} max={60} step={1} value={fps}
                    disabled={!!busy}
                    onChange={(e) => setFps(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            </Card>
          )}

          <Card>
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm mb-0">Audio</CardTitle>
              <div className="flex gap-2">
                {(["copy", "drop"] as const).map((a) => (
                  <button
                    key={a}
                    onClick={() => setAudio(a)}
                    disabled={!!busy}
                    className={`px-3 py-1 rounded border text-[11px] transition-colors ${
                      audio === a
                        ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                        : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                    }`}
                  >
                    {a === "copy" ? "Keep" : "Drop"}
                  </button>
                ))}
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
              ? `Keying - ${Math.round((job.progress || 0) * 100)}%`
              : "Remove key"}
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
                <Wand2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">
                  {isWebm ? "Keyed to transparent WebM" : "Composite ready"}
                </p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message || `Job ${job.job_id.slice(0, 8)}`}
                </p>
                <video
                  src={`${jobDownloadUrl(job.job_id)}&t=${Date.now()}`}
                  controls
                  className="mt-4 mx-auto max-h-64 rounded border border-zinc-800 bg-[linear-gradient(45deg,#333_25%,transparent_25%),linear-gradient(-45deg,#333_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#333_75%),linear-gradient(-45deg,transparent_75%,#333_75%)] bg-[length:16px_16px]"
                />
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
                <Wand2 className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">Chroma key {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <Wand2 className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
                <p className="text-sm text-zinc-300">
                  {job.status === "queued" ? "Queued..." : "Keying..."}
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
                <Wand2 className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Keyed video will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">
                  Pick a key color + mode, then press Remove key.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
