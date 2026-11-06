"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  Images, Upload, Zap, Download, X, ArrowUp, ArrowDown, Music,
} from "lucide-react";
import {
  uploadImage,
  uploadEditAudio,
  buildSlideshow,
  getJob,
  jobDownloadUrl,
  type ImageUploadResponse,
  type EditAudioUploadResponse,
  type Job,
  type SlideshowTransition,
  type SlideshowKenBurns,
} from "@/lib/api";

type Slide = {
  image_path: string;
  filename: string;
  duration: number;
};

const TRANSITIONS: { id: SlideshowTransition; label: string }[] = [
  { id: "none",       label: "Cut" },
  { id: "fade",       label: "Fade" },
  { id: "dissolve",   label: "Dissolve" },
  { id: "wipeleft",   label: "Wipe L" },
  { id: "wiperight",  label: "Wipe R" },
  { id: "slideleft",  label: "Slide L" },
  { id: "slideright", label: "Slide R" },
  { id: "circleopen", label: "Circle" },
  { id: "pixelize",   label: "Pixelate" },
];

const KENBURNS: { id: SlideshowKenBurns; label: string }[] = [
  { id: "none",      label: "Static" },
  { id: "zoom-in",   label: "Zoom in" },
  { id: "zoom-out",  label: "Zoom out" },
  { id: "pan-left",  label: "Pan left" },
  { id: "pan-right", label: "Pan right" },
];

const RESOLUTIONS = [
  { label: "1080p", width: 1920, height: 1080 },
  { label: "720p",  width: 1280, height: 720 },
  { label: "4K",    width: 3840, height: 2160 },
  { label: "9:16",  width: 1080, height: 1920 },
];

export default function SlideshowPanel() {
  const [slides, setSlides] = useState<Slide[]>([]);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [resIdx, setResIdx] = useState(0);
  const [fps, setFps] = useState(30);
  const [defaultDuration, setDefaultDuration] = useState(3);
  const [transition, setTransition] = useState<SlideshowTransition>("fade");
  const [transitionDuration, setTransitionDuration] = useState(0.5);
  const [kenburns, setKenburns] = useState<SlideshowKenBurns>("none");

  const [audio, setAudio] = useState<EditAudioUploadResponse | null>(null);
  const [audioUploading, setAudioUploading] = useState(false);
  const [audioVolume, setAudioVolume] = useState(1);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };
  useEffect(() => clearPoll, []);

  const addImages = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setError(null);
    setUploadingCount(files.length);
    try {
      const results: Slide[] = [];
      for (const file of Array.from(files)) {
        const resp: ImageUploadResponse = await uploadImage(file);
        results.push({
          image_path: resp.image_path,
          filename: file.name,
          duration: defaultDuration,
        });
      }
      setSlides((cur) => [...cur, ...results]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploadingCount(0);
    }
  }, [defaultDuration]);

  const addAudio = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setAudioUploading(true);
    try {
      setAudio(await uploadEditAudio(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setAudioUploading(false);
    }
  }, []);

  const removeSlide = (idx: number) =>
    setSlides((cur) => cur.filter((_, i) => i !== idx));
  const moveSlide = (from: number, to: number) =>
    setSlides((cur) => {
      if (to < 0 || to >= cur.length) return cur;
      const next = cur.slice();
      const [s] = next.splice(from, 1);
      next.splice(to, 0, s);
      return next;
    });
  const setSlideDuration = (idx: number, dur: number) =>
    setSlides((cur) =>
      cur.map((s, i) => (i === idx ? { ...s, duration: dur } : s)),
    );

  const onStart = async () => {
    if (slides.length === 0 || job) return;
    setError(null);
    // xfade needs the transition shorter than the shortest clip.
    if (transition !== "none") {
      const minDur = Math.min(...slides.map((s) => s.duration));
      if (transitionDuration >= minDur) {
        setError(`Transition (${transitionDuration}s) must be shorter than the shortest slide (${minDur}s).`);
        return;
      }
    }
    try {
      const res = RESOLUTIONS[resIdx];
      const j = await buildSlideshow({
        images: slides.map((s) => s.image_path),
        durations: slides.map((s) => s.duration),
        per_image_duration: defaultDuration,
        width: res.width,
        height: res.height,
        fps,
        transition,
        transition_duration: transitionDuration,
        kenburns,
        audio_path: audio?.audio_path,
        audio_volume: audioVolume,
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

  const busy =
    uploadingCount > 0 ||
    (job && (job.status === "running" || job.status === "queued"));

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Slideshow Creator</h1>
        <p className="text-zinc-400 mt-1">
          Turn an ordered list of images into a video with xfade transitions,
          optional Ken Burns motion, and a background music track.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                disabled={!!busy}
                onChange={(e) => addImages(e.target.files)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-400">
                  {uploadingCount > 0
                    ? `Uploading ${uploadingCount} image(s)...`
                    : "Drop images here or click to add (multi-select is fine)"}
                </p>
              </div>
            </label>
          </Card>

          {slides.length > 0 && (
            <Card>
              <CardTitle className="text-sm mb-3">
                Slides ({slides.length})
              </CardTitle>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {slides.map((s, i) => (
                  <div
                    key={`${s.image_path}-${i}`}
                    className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded px-2 py-1"
                  >
                    <span className="text-xs text-zinc-500 w-6">{i + 1}</span>
                    <span className="text-xs text-zinc-300 flex-1 truncate">
                      {s.filename}
                    </span>
                    <input
                      type="number"
                      min={0.5}
                      max={60}
                      step={0.5}
                      value={s.duration}
                      disabled={!!busy}
                      onChange={(e) =>
                        setSlideDuration(i, Number(e.target.value) || 1)
                      }
                      className="w-14 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-xs"
                    />
                    <span className="text-[10px] text-zinc-500">sec</span>
                    <button
                      onClick={() => moveSlide(i, i - 1)}
                      disabled={!!busy || i === 0}
                      className="p-1 text-zinc-500 hover:text-zinc-200 disabled:opacity-20"
                      title="Move up"
                    >
                      <ArrowUp className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => moveSlide(i, i + 1)}
                      disabled={!!busy || i === slides.length - 1}
                      className="p-1 text-zinc-500 hover:text-zinc-200 disabled:opacity-20"
                      title="Move down"
                    >
                      <ArrowDown className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => removeSlide(i)}
                      disabled={!!busy}
                      className="p-1 text-zinc-500 hover:text-red-400"
                      title="Remove"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </Card>
          )}

          <Card>
            <CardTitle className="text-sm mb-3">Video</CardTitle>
            <div className="space-y-3">
              <div>
                <div className="text-xs text-zinc-400 mb-1">Resolution</div>
                <div className="flex gap-2 flex-wrap">
                  {RESOLUTIONS.map((r, i) => (
                    <button
                      key={r.label}
                      onClick={() => setResIdx(i)}
                      disabled={!!busy}
                      className={`px-2 py-1 rounded border text-[11px] transition-colors ${
                        i === resIdx
                          ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                          : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                      }`}
                    >
                      {r.label} ({r.width}x{r.height})
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-1">
                  <label className="text-xs text-zinc-400">FPS: {fps}</label>
                  <input
                    type="range" min={12} max={60} step={1} value={fps}
                    disabled={!!busy}
                    onChange={(e) => setFps(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-zinc-400">
                    Default slide: {defaultDuration}s
                  </label>
                  <input
                    type="range" min={1} max={10} step={0.5} value={defaultDuration}
                    disabled={!!busy}
                    onChange={(e) => setDefaultDuration(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Transition</CardTitle>
            <div className="flex gap-2 flex-wrap mb-3">
              {TRANSITIONS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTransition(t.id)}
                  disabled={!!busy}
                  className={`px-2 py-1 rounded border text-[11px] transition-colors ${
                    transition === t.id
                      ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                      : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            {transition !== "none" && (
              <div>
                <label className="text-xs text-zinc-400">
                  Transition length: {transitionDuration.toFixed(2)}s
                </label>
                <input
                  type="range" min={0.1} max={3} step={0.05} value={transitionDuration}
                  disabled={!!busy}
                  onChange={(e) => setTransitionDuration(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Camera motion</CardTitle>
            <div className="flex gap-2 flex-wrap">
              {KENBURNS.map((k) => (
                <button
                  key={k.id}
                  onClick={() => setKenburns(k.id)}
                  disabled={!!busy}
                  className={`px-2 py-1 rounded border text-[11px] transition-colors ${
                    kenburns === k.id
                      ? "border-indigo-500 bg-indigo-500/10 text-zinc-200"
                      : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  {k.label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-zinc-500 mt-2">
              Applies to every slide. Static is cheapest; Ken Burns adds a
              zoompan pass per image.
            </p>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Background music (optional)</CardTitle>
            <label className="block cursor-pointer">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                disabled={!!busy}
                onChange={(e) => addAudio(e.target.files?.[0] ?? null)}
              />
              <div className="border border-dashed border-zinc-700 rounded-lg p-3 text-center hover:border-indigo-500/50 transition-colors">
                {audio ? (
                  <p className="text-xs text-zinc-300 truncate">
                    <Music className="w-3 h-3 inline mr-1" />
                    {audio.filename}
                  </p>
                ) : (
                  <p className="text-xs text-zinc-400">
                    {audioUploading ? "Uploading..." : "Click to attach an audio track"}
                  </p>
                )}
              </div>
            </label>
            {audio && (
              <div className="mt-3">
                <label className="text-xs text-zinc-400">
                  Volume: {(audioVolume * 100).toFixed(0)}%
                </label>
                <input
                  type="range" min={0} max={2} step={0.05} value={audioVolume}
                  disabled={!!busy}
                  onChange={(e) => setAudioVolume(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
            )}
          </Card>

          <Button
            size="lg"
            className="w-full"
            disabled={slides.length === 0 || !!busy}
            onClick={onStart}
          >
            <Zap className="w-5 h-5 mr-2" />
            {job && (job.status === "running" || job.status === "queued")
              ? `Rendering - ${Math.round((job.progress || 0) * 100)}%`
              : "Render slideshow"}
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
                <Images className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Slideshow ready</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.message || `Job ${job.job_id.slice(0, 8)}`}
                </p>
                <video
                  src={`${jobDownloadUrl(job.job_id)}&t=${Date.now()}`}
                  controls
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
                <Images className="w-12 h-12 text-red-500 mx-auto mb-3" />
                <p className="text-sm text-red-300">Slideshow {job.status}</p>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {job.error || job.message}
                </p>
              </>
            ) : job ? (
              <>
                <Images className="w-12 h-12 text-indigo-500 mx-auto mb-3 animate-pulse" />
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
                <Images className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">
                  Slideshow video will appear here
                </p>
                <p className="text-zinc-600 text-xs mt-1">
                  Add images, pick a transition, press render.
                </p>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
