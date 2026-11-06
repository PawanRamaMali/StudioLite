"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Download, Grid3x3, X } from "lucide-react";
import {
  uploadEditVideo,
  buildCollage,
  getJob,
  jobDownloadUrl,
  type CollageLayout,
  type EditUploadResponse,
  type Job,
} from "@/lib/api";

type LayoutMeta = {
  id: CollageLayout;
  label: string;
  minClips: number;
  maxClips: number;
  description: string;
};

const LAYOUTS: LayoutMeta[] = [
  { id: "hstack_2", label: "Side by side (2)",     minClips: 2, maxClips: 2, description: "Two clips, left and right." },
  { id: "vstack_2", label: "Top / bottom (2)",     minClips: 2, maxClips: 2, description: "Two clips stacked vertically." },
  { id: "hstack_3", label: "Triple row",           minClips: 2, maxClips: 3, description: "Up to three clips in a row." },
  { id: "vstack_3", label: "Triple column",        minClips: 2, maxClips: 3, description: "Up to three clips in a column." },
  { id: "grid_2x2", label: "2 x 2 grid",           minClips: 2, maxClips: 4, description: "Quad-split layout for up to four clips." },
  { id: "grid_3x3", label: "3 x 3 grid",           minClips: 2, maxClips: 9, description: "Classic Brady Bunch grid." },
  { id: "pip",      label: "Picture in picture",   minClips: 2, maxClips: 2, description: "Base clip with a corner inset." },
];

const PIP_CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"] as const;
type PipCorner = (typeof PIP_CORNERS)[number];

export default function CollagePanel() {
  const [layout, setLayout] = useState<CollageLayout>("hstack_2");
  const [clips, setClips] = useState<EditUploadResponse[]>([]);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [cellWidth, setCellWidth] = useState(640);
  const [cellHeight, setCellHeight] = useState(360);
  const [fps, setFps] = useState(30);
  const [audio, setAudio] = useState<"first" | "mix" | "drop">("first");
  const [background, setBackground] = useState("black");
  const [pipPosition, setPipPosition] = useState<PipCorner>("bottom-right");
  const [pipScale, setPipScale] = useState(0.28);
  const [crf, setCrf] = useState(20);

  const meta = LAYOUTS.find((l) => l.id === layout) ?? LAYOUTS[0];

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => clearPoll, []);

  const onAddFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      const uploaded: EditUploadResponse[] = [];
      for (const f of Array.from(files)) {
        if (clips.length + uploaded.length >= meta.maxClips) break;
        try {
          uploaded.push(await uploadEditVideo(f));
        } catch (e) {
          setError((e as Error).message);
          break;
        }
      }
      setClips([...clips, ...uploaded]);
    } finally {
      setUploading(false);
    }
  }, [clips, meta.maxClips]);

  const removeAt = (i: number) => {
    const next = clips.slice();
    next.splice(i, 1);
    setClips(next);
    setJob(null);
  };

  const moveClip = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= clips.length) return;
    const next = clips.slice();
    [next[i], next[j]] = [next[j], next[i]];
    setClips(next);
  };

  const onStart = async () => {
    if (clips.length < meta.minClips || job) return;
    setError(null);
    try {
      const j = await buildCollage({
        video_paths: clips.map((c) => c.video_path),
        layout,
        cell_width: cellWidth,
        cell_height: cellHeight,
        fps,
        background,
        pip_position: pipPosition,
        pip_scale: pipScale,
        audio,
        crf,
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
  const canStart = clips.length >= meta.minClips && clips.length <= meta.maxClips && !busy;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Collage</h1>
        <p className="text-zinc-400 mt-1">
          Tile clips into grids, split screens or picture-in-picture. Each
          cell is scaled and padded to a common size so mismatched aspect
          ratios still look clean.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Layout</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {LAYOUTS.map((l) => (
                <button
                  key={l.id}
                  onClick={() => { setLayout(l.id); setJob(null); }}
                  disabled={busy}
                  className={`px-3 py-2 rounded-md text-xs border transition-colors text-left ${
                    layout === l.id
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <div className="font-medium">{l.label}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">{l.description}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">
              Clips ({clips.length} / {meta.maxClips})
            </CardTitle>
            <label className="block cursor-pointer">
              <input
                type="file" accept="video/*" multiple
                className="hidden"
                disabled={busy || clips.length >= meta.maxClips}
                onChange={(e) => onAddFiles(e.target.files)}
              />
              <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                clips.length >= meta.maxClips
                  ? "border-zinc-800 text-zinc-600"
                  : "border-zinc-700 hover:border-indigo-500/50 text-zinc-400"
              }`}>
                <Upload className="w-6 h-6 mx-auto mb-2" />
                <p className="text-sm">
                  {clips.length >= meta.maxClips
                    ? `Max ${meta.maxClips} clips for this layout`
                    : (uploading ? "Uploading..." : "Add clips")}
                </p>
              </div>
            </label>

            {clips.length > 0 && (
              <div className="space-y-2 mt-3">
                {clips.map((c, i) => (
                  <div key={`${c.video_path}-${i}`} className="flex items-center gap-2 bg-zinc-900/60 border border-zinc-800 rounded-md px-3 py-2">
                    <span className="text-[11px] text-zinc-500 w-6">#{i + 1}</span>
                    <span className="flex-1 text-sm text-zinc-300 truncate">{c.filename}</span>
                    <button
                      onClick={() => moveClip(i, -1)}
                      disabled={i === 0 || busy}
                      className="text-xs text-zinc-500 hover:text-zinc-300 disabled:opacity-30"
                    >up</button>
                    <button
                      onClick={() => moveClip(i, 1)}
                      disabled={i === clips.length - 1 || busy}
                      className="text-xs text-zinc-500 hover:text-zinc-300 disabled:opacity-30"
                    >dn</button>
                    <button
                      onClick={() => removeAt(i)}
                      disabled={busy}
                      className="text-zinc-500 hover:text-red-400"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Output</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">
                  Cell width: {cellWidth}px
                </label>
                <input
                  type="range" min={240} max={1920} step={20}
                  value={cellWidth} disabled={busy}
                  onChange={(e) => setCellWidth(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">
                  Cell height: {cellHeight}px
                </label>
                <input
                  type="range" min={160} max={1080} step={20}
                  value={cellHeight} disabled={busy}
                  onChange={(e) => setCellHeight(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">FPS: {fps}</label>
                <input
                  type="range" min={15} max={60} step={1}
                  value={fps} disabled={busy}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Quality (CRF): {crf}</label>
                <input
                  type="range" min={14} max={32} step={1}
                  value={crf} disabled={busy}
                  onChange={(e) => setCrf(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Background color</label>
                <input
                  type="text" value={background}
                  disabled={busy}
                  onChange={(e) => setBackground(e.target.value)}
                  className="w-full mt-1 bg-zinc-900/60 border border-zinc-800 rounded-md px-2 py-1.5 text-sm text-zinc-200"
                />
              </div>
            </div>
          </Card>

          {layout === "pip" && (
            <Card>
              <CardTitle className="text-sm mb-3">PIP options</CardTitle>
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {PIP_CORNERS.map((c) => (
                    <button
                      key={c}
                      onClick={() => setPipPosition(c)}
                      disabled={busy}
                      className={`px-2 py-1.5 rounded text-xs border ${
                        pipPosition === c
                          ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                          : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                      }`}
                    >{c.replace("-", " ")}</button>
                  ))}
                </div>
                <div>
                  <label className="text-xs text-zinc-400">
                    Inset size: {(pipScale * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range" min={0.1} max={0.5} step={0.01}
                    value={pipScale} disabled={busy}
                    onChange={(e) => setPipScale(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            </Card>
          )}

          <Card>
            <CardTitle className="text-sm mb-3">Audio</CardTitle>
            <div className="grid grid-cols-3 gap-2">
              {(["first", "mix", "drop"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setAudio(m)}
                  disabled={busy}
                  className={`px-3 py-2 rounded-md text-xs border ${
                    audio === m
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                  }`}
                >
                  {m === "first" ? "Primary clip" : m === "mix" ? "Mix all" : "Silent"}
                </button>
              ))}
            </div>
          </Card>

          <Button size="lg" className="w-full" disabled={!canStart} onClick={onStart}>
            <Zap className="w-5 h-5 mr-2" />
            {busy ? `Rendering - ${Math.round((job?.progress || 0) * 100)}%` : "Render collage"}
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
                <Grid3x3 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Collage ready</p>
                <video
                  key={jobDownloadUrl(job.job_id)}
                  src={jobDownloadUrl(job.job_id)}
                  controls
                  className="mt-4 mx-auto max-h-80 rounded border border-zinc-800"
                />
                <Button
                  size="sm" variant="secondary" className="mt-3"
                  onClick={() => window.open(jobDownloadUrl(job.job_id), "_blank", "noopener,noreferrer")}
                >
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
              </>
            ) : job && job.status === "failed" ? (
              <p className="text-sm text-red-400">{job.error || "Collage failed"}</p>
            ) : (
              <p className="text-sm text-zinc-500">
                Pick a layout, add {meta.minClips} to {meta.maxClips} clips, then render.
              </p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
