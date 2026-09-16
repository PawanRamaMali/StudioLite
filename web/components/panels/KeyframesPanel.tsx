"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { KeyRound, Upload, Play, ArrowRight, Download, X } from "lucide-react";
import {
  uploadImage,
  animateKeyframes,
  getJob,
  jobDownloadUrl,
  type ImageUploadResponse,
  type Job,
} from "@/lib/api";

const EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out", "bounce"];
const METHODS = [
  {
    id: "blend",
    name: "Alpha Blend",
    desc: "Fast, no AI required",
    enabled: true,
  },
  {
    id: "i2v",
    name: "AI Image-to-Video",
    desc: "Uses GPU, best quality (not yet wired)",
    enabled: false,
  },
  {
    id: "interpolate",
    name: "Frame Interpolation",
    desc: "RIFE-based, smooth (not yet wired)",
    enabled: false,
  },
];

function KeyframeSlot({
  label,
  upload,
  uploading,
  disabled,
  onFile,
  onClear,
}: {
  label: string;
  upload: ImageUploadResponse | null;
  uploading: boolean;
  disabled: boolean;
  onFile: (file: File) => void;
  onClear: () => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <Card className="col-span-2 min-h-[200px] flex items-center justify-center relative">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {upload ? (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={upload.url}
            alt={label}
            className="max-h-[190px] max-w-full rounded-md object-contain"
          />
          <button
            title="Remove"
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
            className="absolute top-2 right-2 bg-zinc-900/80 rounded-full p-1 hover:bg-zinc-800 text-zinc-300"
          >
            <X className="w-3 h-3" />
          </button>
        </>
      ) : (
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="text-center cursor-pointer hover:opacity-80 transition-opacity"
        >
          <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
          <p className="text-sm text-zinc-400 font-medium">{label}</p>
          <p className="text-xs text-zinc-600">
            {uploading ? "Uploading…" : "Upload image"}
          </p>
        </button>
      )}
    </Card>
  );
}

export default function KeyframesPanel() {
  const [easing, setEasing] = useState("ease_in_out");
  const [method, setMethod] = useState("blend");
  const [frames, setFrames] = useState(30);
  const [fps, setFps] = useState(24);
  const [startImg, setStartImg] = useState<ImageUploadResponse | null>(null);
  const [endImg, setEndImg] = useState<ImageUploadResponse | null>(null);
  const [uploading, setUploading] = useState<null | "start" | "end">(null);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };
  useEffect(() => clearPoll, []);

  const putFile = useCallback(async (which: "start" | "end", file: File) => {
    setUploading(which);
    setError(null);
    try {
      const res = await uploadImage(file);
      (which === "start" ? setStartImg : setEndImg)(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(null);
    }
  }, []);

  const onGenerate = async () => {
    if (!startImg || !endImg) return;
    setError(null);
    try {
      const j = await animateKeyframes({
        start_image_path: startImg.image_path,
        end_image_path: endImg.image_path,
        frames,
        fps,
        easing,
        method,
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

  const busy =
    !!uploading || (job && (job.status === "running" || job.status === "queued"));
  const disabled = !!busy;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Keyframe Animation</h1>
        <p className="text-zinc-400 mt-1">Create smooth transitions between keyframe images</p>
      </div>

      <div className="grid grid-cols-5 gap-4 mb-6 items-center">
        <KeyframeSlot
          label="Start Keyframe"
          upload={startImg}
          uploading={uploading === "start"}
          disabled={disabled}
          onFile={(f) => putFile("start", f)}
          onClear={() => setStartImg(null)}
        />
        <div className="flex justify-center">
          <div className="w-10 h-10 rounded-full bg-indigo-600/20 flex items-center justify-center">
            <ArrowRight className="w-5 h-5 text-indigo-400" />
          </div>
        </div>
        <KeyframeSlot
          label="End Keyframe"
          upload={endImg}
          uploading={uploading === "end"}
          disabled={disabled}
          onFile={(f) => putFile("end", f)}
          onClear={() => setEndImg(null)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Method</CardTitle>
            <div className="space-y-2">
              {METHODS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => m.enabled && setMethod(m.id)}
                  disabled={!m.enabled || disabled}
                  className={`w-full p-2.5 rounded-lg border text-left transition-all ${
                    method === m.id
                      ? "border-indigo-500 bg-indigo-500/10"
                      : "border-zinc-700 hover:border-zinc-600"
                  } ${!m.enabled ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <div className="text-xs font-medium">{m.name}</div>
                  <div className="text-[10px] text-zinc-500">{m.desc}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Settings</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Easing</label>
                <select
                  value={easing}
                  onChange={(e) => setEasing(e.target.value)}
                  disabled={disabled}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-1"
                >
                  {EASINGS.map((e) => (
                    <option key={e} value={e}>
                      {e.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Frames: {frames}</label>
                <input
                  type="range"
                  min={10}
                  max={120}
                  value={frames}
                  disabled={disabled}
                  onChange={(e) => setFrames(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">FPS: {fps}</label>
                <input
                  type="range"
                  min={8}
                  max={30}
                  value={fps}
                  disabled={disabled}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <p className="text-xs text-zinc-500">
                Duration: {(frames / fps).toFixed(1)}s
              </p>
            </div>
          </Card>

          <Button
            size="lg"
            className="w-full"
            disabled={disabled || !startImg || !endImg}
            onClick={onGenerate}
          >
            <Play className="w-5 h-5 mr-2" />
            {job && (job.status === "running" || job.status === "queued")
              ? `Rendering ${Math.round((job.progress || 0) * 100)}%`
              : "Generate Animation"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <div className="lg:col-span-2">
          <Card className="min-h-[300px] flex items-center justify-center">
            <div className="text-center w-full px-6">
              {job && job.status === "completed" ? (
                <>
                  <KeyRound className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                  <p className="text-sm text-zinc-300">Animation ready</p>
                  <p className="text-[11px] text-zinc-500 mt-1">
                    {job.message || `Job ${job.job_id.slice(0, 8)}`}
                  </p>
                  <Button
                    size="sm"
                    variant="secondary"
                    className="mt-4"
                    onClick={() =>
                      window.open(
                        jobDownloadUrl(job.job_id),
                        "_blank",
                        "noopener,noreferrer"
                      )
                    }
                  >
                    <Download className="w-4 h-4 mr-1" /> Download
                  </Button>
                </>
              ) : job && (job.status === "failed" || job.status === "cancelled") ? (
                <>
                  <KeyRound className="w-12 h-12 text-red-500 mx-auto mb-3" />
                  <p className="text-sm text-red-300">
                    Animation {job.status}
                  </p>
                  <p className="text-[11px] text-zinc-500 mt-1">
                    {job.error || job.message}
                  </p>
                </>
              ) : job ? (
                <>
                  <KeyRound className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
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
                  <KeyRound className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                  <p className="text-zinc-500 text-sm">
                    Generated animation will appear here
                  </p>
                  <p className="text-zinc-600 text-xs mt-1">
                    Upload two keyframes, pick a preset, hit Generate.
                  </p>
                </>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
