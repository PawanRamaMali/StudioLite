"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Scissors, Clock } from "lucide-react";
import {
  uploadEditVideo,
  detectScenes,
  getJob,
  type EditUploadResponse,
  type Job,
  type Scene,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function fmt(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function SceneDetectPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(10);
  const [minLen, setMinLen] = useState(1.0);
  const [withThumbs, setWithThumbs] = useState(true);
  const [thumbWidth, setThumbWidth] = useState(320);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
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
      const j = await detectScenes({
        video_path: upload.video_path,
        threshold,
        min_scene_length: minLen,
        extract_thumbnails: withThumbs,
        thumb_width: thumbWidth,
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

  const busy = uploading || (!!job && (job.status === "running" || job.status === "queued"));
  const result = job?.status === "completed" ? (job.result as unknown as {
    scene_count: number;
    cut_count: number;
    duration?: number | null;
    scenes: Scene[];
  }) : null;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Scene Detection</h1>
        <p className="text-zinc-400 mt-1">
          Find hard cuts in a video using ffmpeg&apos;s scdet filter. Lower
          the threshold to catch soft transitions; raise the minimum scene
          length to collapse strobe flashes into a single cut.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file" accept="video/*" className="hidden"
                disabled={busy}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-10 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                {upload ? (
                  <>
                    <p className="text-sm text-zinc-300 truncate">{upload.filename}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {(upload.size_bytes / (1024 * 1024)).toFixed(1)} MB &mdash; click to replace
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
            <CardTitle className="text-sm mb-3">Detection</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">
                  Threshold: {threshold.toFixed(1)}
                </label>
                <input
                  type="range" min={1} max={40} step={0.5}
                  value={threshold} disabled={busy}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  scdet default is 10. Lower = more cuts.
                </p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">
                  Min scene length: {minLen.toFixed(1)}s
                </label>
                <input
                  type="range" min={0} max={10} step={0.1}
                  value={minLen} disabled={busy}
                  onChange={(e) => setMinLen(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <label className="flex items-center gap-2 text-sm text-zinc-300">
                <input
                  type="checkbox" checked={withThumbs}
                  onChange={(e) => setWithThumbs(e.target.checked)}
                  disabled={busy}
                />
                Extract thumbnail per scene
              </label>
              {withThumbs && (
                <div>
                  <label className="text-xs text-zinc-400">
                    Thumbnail width: {thumbWidth}px
                  </label>
                  <input
                    type="range" min={120} max={640} step={20}
                    value={thumbWidth} disabled={busy}
                    onChange={(e) => setThumbWidth(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              )}
            </div>
          </Card>

          <Button size="lg" className="w-full" disabled={!upload || busy} onClick={onStart}>
            <Zap className="w-5 h-5 mr-2" />
            {busy ? `Scanning - ${Math.round((job?.progress || 0) * 100)}%` : "Detect scenes"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px]">
          {result ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Scissors className="w-4 h-4" />
                  {result.scene_count} scenes
                  <span className="text-zinc-500 font-normal">
                    &middot; {result.cut_count} cuts
                  </span>
                </CardTitle>
                {result.duration && (
                  <span className="text-[11px] text-zinc-500 flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {fmt(result.duration)}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2 max-h-[600px] overflow-y-auto">
                {result.scenes.map((sc) => (
                  <div
                    key={sc.index}
                    className="rounded-md border border-zinc-800 overflow-hidden bg-zinc-900/60"
                  >
                    {sc.thumbnail_url ? (
                      <img
                        src={`${API_BASE}${sc.thumbnail_url}`}
                        alt={`Scene ${sc.index + 1}`}
                        className="w-full aspect-video object-cover"
                      />
                    ) : (
                      <div className="w-full aspect-video flex items-center justify-center text-zinc-600 text-xs">
                        no thumbnail
                      </div>
                    )}
                    <div className="px-2 py-1.5">
                      <div className="text-[11px] text-indigo-300">
                        #{sc.index + 1}
                      </div>
                      <div className="text-[10px] text-zinc-500">
                        {fmt(sc.start)} &rarr; {fmt(sc.end)}
                      </div>
                      <div className="text-[10px] text-zinc-600">
                        {sc.duration.toFixed(1)}s
                        {sc.score > 0 && <span className="ml-1">&middot; score {sc.score.toFixed(1)}</span>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : job && job.status === "failed" ? (
            <div className="h-full flex items-center justify-center">
              <p className="text-sm text-red-400">{job.error || "Detection failed"}</p>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <p className="text-sm text-zinc-500">
                Scene breakdown appears here once detection completes.
              </p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
