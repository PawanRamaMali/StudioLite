"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Film, Upload, Zap, Download } from "lucide-react";
import {
  uploadEditVideo,
  gifVideo,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type Job,
} from "@/lib/api";

// Rough bytes-per-pixel heuristic for lanczos+palette GIF - tracks pretty
// well with observed output so the user can gauge filesize before hitting
// render. Not precise enough to drive a cap, just a hint.
function estimateGifBytes(width: number, duration: number, fps: number): number {
  const frames = Math.max(1, Math.round(duration * fps));
  const bytesPerFrame = Math.max(1024, Math.round(width * (width / 1.78) * 0.5));
  return frames * bytesPerFrame;
}

function prettyBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function GifPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [startTime, setStartTime] = useState(0);
  const [duration, setDuration] = useState(3);
  const [fps, setFps] = useState(12);
  const [width, setWidth] = useState(480);

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

  const onStart = async () => {
    if (!upload || job) return;
    setError(null);
    try {
      const j = await gifVideo({
        video_path: upload.video_path,
        start_time: startTime,
        duration,
        fps,
        width,
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
  const estimated = estimateGifBytes(width, duration, fps);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">GIF Creator</h1>
        <p className="text-zinc-400 mt-1">
          Trim a clip to an animated GIF with a palettegen/paletteuse pass -
          small enough to share, sharp enough to not look blurry.
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
            <CardTitle className="text-sm mb-3">Trim</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Start: {startTime.toFixed(1)}s</label>
                <input
                  type="range" min={0} max={300} step={0.1} value={startTime}
                  disabled={!!busy}
                  onChange={(e) => setStartTime(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Duration: {duration.toFixed(1)}s</label>
                <input
                  type="range" min={0.5} max={30} step={0.1} value={duration}
                  disabled={!!busy}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  GIFs over 10 seconds get big quickly - consider trimming.
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Quality</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">FPS: {fps}</label>
                <input
                  type="range" min={4} max={30} step={1} value={fps}
                  disabled={!!busy}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  12-15 fps is the sweet spot - above 24 fps the file doubles
                  for a difference most viewers won't notice.
                </p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Width: {width}px</label>
                <input
                  type="range" min={120} max={1280} step={20} value={width}
                  disabled={!!busy}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div className="text-[11px] text-zinc-500">
                Estimated size: <span className="text-zinc-300">~{prettyBytes(estimated)}</span>
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
              ? `Rendering - ${Math.round((job.progress || 0) * 100)}%`
              : "Create GIF"}
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
                <Film className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">GIF ready</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message || `Job ${job.job_id.slice(0, 8)}`}
                </p>
                <img
                  src={`${jobDownloadUrl(job.job_id)}&t=${Date.now()}`}
                  alt="gif preview"
                  className="mt-4 mx-auto max-h-64 rounded border border-zinc-800"
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
                <Film className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">GIF job {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <Film className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
                <p className="text-sm text-zinc-300">
                  {job.status === "queued" ? "Queued..." : "Rendering..."}
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
                <Film className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">GIF will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">
                  Pick a start + duration and press Create.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
