"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Image as ImageIcon, Upload, Zap, Download } from "lucide-react";
import {
  uploadEditVideo,
  uploadImage,
  thumbnailVideo,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type ImageUploadResponse,
  type Job,
  type ThumbnailMode,
  type ThumbnailOverlayPosition,
} from "@/lib/api";

const MODES: { id: ThumbnailMode; label: string; detail: string }[] = [
  { id: "single",   label: "Single frame",   detail: "One PNG at a timestamp." },
  { id: "multiple", label: "Multiple frames", detail: "A zip of PNGs at several timestamps." },
  { id: "grid",     label: "Contact sheet",  detail: "A single PNG tiled from evenly-spaced frames." },
];

const POSITIONS: { id: ThumbnailOverlayPosition; label: string }[] = [
  { id: "top-left",     label: "TL" },
  { id: "top-right",    label: "TR" },
  { id: "bottom-left",  label: "BL" },
  { id: "bottom-right", label: "BR" },
  { id: "center",       label: "Center" },
];

// Parse `"1, 5.5, 12"` -> [1, 5.5, 12]. Invalid entries drop out quietly;
// the server still rejects an empty list with a 500-equivalent job failure
// only when mode=multiple, so we refuse to start one from here.
function parseTimestampList(raw: string): number[] {
  return raw
    .split(/[,\s]+/)
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n >= 0);
}

export default function ThumbnailsPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<ThumbnailMode>("single");
  const [timestamp, setTimestamp] = useState(0);
  const [timestampsRaw, setTimestampsRaw] = useState("0, 2, 5, 10");
  const [cols, setCols] = useState(3);
  const [rows, setRows] = useState(3);
  const [width, setWidth] = useState(1280);

  const [useOverlay, setUseOverlay] = useState(false);
  const [overlay, setOverlay] = useState<ImageUploadResponse | null>(null);
  const [overlayName, setOverlayName] = useState<string>("");
  const [overlayUploading, setOverlayUploading] = useState(false);
  const [overlayPosition, setOverlayPosition] = useState<ThumbnailOverlayPosition>("bottom-right");
  const [overlayOpacity, setOverlayOpacity] = useState(1);

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

  // Overlay files are PNGs/JPEGs - use the image upload endpoint,
  // which validates the file type and stashes the bytes on disk.
  const onOverlayFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setOverlayUploading(true);
    try {
      setOverlay(await uploadImage(file));
      setOverlayName(file.name);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setOverlayUploading(false);
    }
  }, []);

  const onStart = async () => {
    if (!upload || job) return;
    setError(null);

    const parsed = parseTimestampList(timestampsRaw);
    if (mode === "multiple" && parsed.length === 0) {
      setError("Enter at least one timestamp for multiple mode.");
      return;
    }

    try {
      const j = await thumbnailVideo({
        video_path: upload.video_path,
        mode,
        timestamp,
        timestamps: mode === "multiple" ? parsed : undefined,
        cols,
        rows,
        width,
        overlay_path: useOverlay && overlay ? overlay.image_path : undefined,
        overlay_position: overlayPosition,
        overlay_opacity: overlayOpacity,
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
      }, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = uploading || (job && (job.status === "running" || job.status === "queued"));
  const buttonLabel =
    job && (job.status === "running" || job.status === "queued")
      ? `Working - ${Math.round((job.progress || 0) * 100)}%`
      : mode === "multiple"
        ? "Extract frames"
        : mode === "grid"
          ? "Build contact sheet"
          : "Extract thumbnail";

  // The completed job's result carries output_paths (and the chosen mode),
  // but the download endpoint already resolves to the right single file
  // (png for single/grid, zip for multiple) so we just hit it.
  const isZip = job?.result?.mode === "multiple";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Thumbnail Generator</h1>
        <p className="text-zinc-400 mt-1">
          Extract single frames, a zipped set of frames, or a contact sheet.
          Overlay a PNG logo in any corner.
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
            <CardTitle className="text-sm mb-3">Mode</CardTitle>
            <div className="grid grid-cols-3 gap-2">
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
            <CardTitle className="text-sm mb-3">Options</CardTitle>
            <div className="space-y-3">
              {mode === "single" && (
                <div>
                  <label className="text-xs text-zinc-400">
                    Timestamp (seconds): {timestamp}
                  </label>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    value={timestamp}
                    disabled={!!busy}
                    onChange={(e) => setTimestamp(Number(e.target.value))}
                    className="w-full mt-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm"
                  />
                </div>
              )}
              {mode === "multiple" && (
                <div>
                  <label className="text-xs text-zinc-400">
                    Timestamps (comma or space separated)
                  </label>
                  <input
                    type="text"
                    value={timestampsRaw}
                    disabled={!!busy}
                    onChange={(e) => setTimestampsRaw(e.target.value)}
                    className="w-full mt-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm"
                    placeholder="0, 2.5, 5, 10"
                  />
                  <p className="text-[11px] text-zinc-500 mt-1">
                    One PNG per timestamp, packed into a zip.
                  </p>
                </div>
              )}
              {mode === "grid" && (
                <div className="flex gap-3">
                  <div className="flex-1">
                    <label className="text-xs text-zinc-400">Columns: {cols}</label>
                    <input
                      type="range" min={1} max={6} value={cols}
                      disabled={!!busy}
                      onChange={(e) => setCols(Number(e.target.value))}
                      className="w-full mt-1 accent-indigo-500"
                    />
                  </div>
                  <div className="flex-1">
                    <label className="text-xs text-zinc-400">Rows: {rows}</label>
                    <input
                      type="range" min={1} max={6} value={rows}
                      disabled={!!busy}
                      onChange={(e) => setRows(Number(e.target.value))}
                      className="w-full mt-1 accent-indigo-500"
                    />
                  </div>
                </div>
              )}
              <div>
                <label className="text-xs text-zinc-400">Width: {width}px</label>
                <input
                  type="range" min={240} max={3840} step={20} value={width}
                  disabled={!!busy}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  {mode === "grid"
                    ? "Width of the full contact sheet."
                    : "Width of each thumbnail; height scales."}
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-2">
              <CardTitle className="text-sm mb-0">Overlay (optional)</CardTitle>
              <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useOverlay}
                  onChange={(e) => setUseOverlay(e.target.checked)}
                  disabled={!!busy}
                  className="accent-indigo-500"
                />
                Enable
              </label>
            </div>
            {useOverlay && (
              <div className="space-y-3">
                <label className="block cursor-pointer">
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    className="hidden"
                    disabled={!!busy}
                    onChange={(e) => onOverlayFile(e.target.files?.[0] ?? null)}
                  />
                  <div className="border border-dashed border-zinc-700 rounded-lg p-3 text-center hover:border-indigo-500/50 transition-colors">
                    {overlay ? (
                      <p className="text-xs text-zinc-300 truncate">{overlayName || "overlay.png"}</p>
                    ) : (
                      <p className="text-xs text-zinc-400">
                        {overlayUploading ? "Uploading..." : "Click to upload PNG/JPEG overlay"}
                      </p>
                    )}
                  </div>
                </label>
                <div>
                  <div className="text-xs text-zinc-400 mb-1">Position</div>
                  <div className="flex gap-2 flex-wrap">
                    {POSITIONS.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => setOverlayPosition(p.id)}
                        disabled={!!busy}
                        className={`px-2 py-1 rounded border text-[11px] transition-colors ${
                          overlayPosition === p.id
                            ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                            : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                        }`}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-zinc-400">
                    Opacity: {(overlayOpacity * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range" min={0} max={1} step={0.05} value={overlayOpacity}
                    disabled={!!busy}
                    onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            )}
          </Card>

          <Button
            size="lg"
            className="w-full"
            disabled={!upload || !!busy || (useOverlay && !overlay)}
            onClick={onStart}
          >
            <Zap className="w-5 h-5 mr-2" />
            {buttonLabel}
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
                <ImageIcon className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">
                  {isZip ? "Frames packed into a zip" : "Thumbnail ready"}
                </p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message || `Job ${job.job_id.slice(0, 8)}`}
                </p>
                {!isZip && (
                  <img
                    src={`${jobDownloadUrl(job.job_id)}&t=${Date.now()}`}
                    alt="thumbnail preview"
                    className="mt-4 mx-auto max-h-64 rounded border border-zinc-800"
                  />
                )}
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
                <ImageIcon className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">Thumbnail job {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <ImageIcon className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
                <p className="text-sm text-zinc-300">
                  {job.status === "queued" ? "Queued..." : "Working..."}
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
                <ImageIcon className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Thumbnail will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">
                  Pick a mode and timestamps, then press the button.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
