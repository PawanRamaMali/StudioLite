"use client";

import { useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  AlertCircle, Download, FileAudio, FlipHorizontal, FlipVertical, Loader2, Music,
  Palette, PictureInPicture2, Repeat2, RotateCw, Scissors, Shield, SlidersHorizontal,
  Gauge, Image as ImageIcon, Upload, Video, VolumeX, Wand2, Zap,
} from "lucide-react";
import {
  backgroundMusicVideo, colorCorrectVideo, compressVideo, downloadBlob, extractVideoAudio,
  getDownloadUrl, getJob, gifVideo, loopVideo, pictureInPictureVideo, regionEffectVideo,
  removeVideoAudio, reverseVideo, rotateFlipVideo, stabilizeVideo, uploadEditAudio,
  speedVideo, thumbnailVideo, uploadEditVideo,
  type EditAudioUploadResponse, type EditUploadResponse, type Job,
} from "@/lib/api";

const STYLE_FILTERS = [
  { id: "grayscale", name: "Grayscale", desc: "Black & white", preview: "from-gray-400 to-gray-600" },
  { id: "sepia", name: "Sepia", desc: "Warm vintage", preview: "from-amber-700 to-yellow-600" },
  { id: "invert", name: "Invert", desc: "Negative colors", preview: "from-white to-gray-800" },
  { id: "warm", name: "Warm", desc: "Orange tones", preview: "from-orange-400 to-red-400" },
  { id: "cool", name: "Cool", desc: "Blue tones", preview: "from-blue-400 to-cyan-400" },
  { id: "vintage", name: "Vintage", desc: "Retro faded", preview: "from-amber-600 to-orange-800" },
  { id: "high_contrast", name: "High Contrast", desc: "Bold contrast", preview: "from-black to-white" },
];

const EDIT_OPS = [
  { id: "remove", name: "Remove Object", desc: "Inpaint and remove using AI" },
  { id: "blur", name: "Blur Region", desc: "Censor or soften an area" },
  { id: "brighten", name: "Brighten", desc: "Increase brightness of area" },
  { id: "darken", name: "Darken", desc: "Decrease brightness of area" },
  { id: "color_shift", name: "Color Shift", desc: "Change color tint" },
  { id: "grayscale", name: "Grayscale Region", desc: "Desaturate specific area" },
];

const UTILITIES = [
  { id: "extract", name: "Extract Audio", icon: FileAudio, desc: "Save the audio track" },
  { id: "mute", name: "Remove Audio", icon: VolumeX, desc: "Export a silent MP4" },
  { id: "compress", name: "Compress", icon: Zap, desc: "Reduce file size" },
  { id: "rotate", name: "Rotate / Flip", icon: RotateCw, desc: "Change orientation" },
  { id: "reverse", name: "Reverse", icon: Repeat2, desc: "Play backward" },
  { id: "loop", name: "Loop", icon: Repeat2, desc: "Repeat clip" },
  { id: "stabilize", name: "Stabilize", icon: SlidersHorizontal, desc: "Reduce camera shake" },
  { id: "color", name: "Color", icon: Palette, desc: "Adjust image tone" },
  { id: "privacy", name: "Privacy Blur", icon: Shield, desc: "Blur or pixelate a region" },
  { id: "pip", name: "Picture-in-Picture", icon: PictureInPicture2, desc: "Overlay a second video" },
  { id: "music", name: "Background Music", icon: Music, desc: "Mix in an audio track" },
  { id: "speed", name: "Speed", icon: Gauge, desc: "Slow down or speed up" },
  { id: "gif", name: "GIF", icon: ImageIcon, desc: "Export an animated GIF" },
  { id: "thumbnail", name: "Thumbnail", icon: ImageIcon, desc: "Extract a PNG frame" },
] as const;

type Mode = "utilities" | "filters" | "region";
type UtilityId = typeof UTILITIES[number]["id"];
type AudioFormat = "wav" | "mp3";
type CompressPreset = "high" | "medium" | "low";
type FlipMode = "none" | "horizontal" | "vertical";
type RotateValue = 0 | 90 | 180 | 270;
type RegionMode = "blur" | "pixelate";
type PipPosition = "top_left" | "top_right" | "bottom_left" | "bottom_right" | "center";

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function EditPanel() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const secondaryFileInputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<Mode>("utilities");
  const [selectedFilter, setSelectedFilter] = useState("grayscale");
  const [selectedOp, setSelectedOp] = useState("blur");
  const [selectedUtility, setSelectedUtility] = useState<UtilityId>("extract");
  const [uploaded, setUploaded] = useState<EditUploadResponse | null>(null);
  const [overlayVideo, setOverlayVideo] = useState<EditUploadResponse | null>(null);
  const [musicAudio, setMusicAudio] = useState<EditAudioUploadResponse | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [secondaryName, setSecondaryName] = useState("");
  const [sourcePreview, setSourcePreview] = useState("");
  const [audioFormat, setAudioFormat] = useState<AudioFormat>("wav");
  const [compressPreset, setCompressPreset] = useState<CompressPreset>("medium");
  const [rotate, setRotate] = useState<RotateValue>(0);
  const [flip, setFlip] = useState<FlipMode>("none");
  const [reverseAudio, setReverseAudio] = useState(false);
  const [loopCount, setLoopCount] = useState(2);
  const [shakiness, setShakiness] = useState(5);
  const [accuracy, setAccuracy] = useState(9);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [saturation, setSaturation] = useState(1);
  const [hue, setHue] = useState(0);
  const [regionMode, setRegionMode] = useState<RegionMode>("blur");
  const [regionX, setRegionX] = useState(50);
  const [regionY, setRegionY] = useState(50);
  const [regionW, setRegionW] = useState(200);
  const [regionH, setRegionH] = useState(120);
  const [regionStrength, setRegionStrength] = useState(12);
  const [pipPosition, setPipPosition] = useState<PipPosition>("bottom_right");
  const [pipSize, setPipSize] = useState(25);
  const [pipMargin, setPipMargin] = useState(24);
  const [musicVolume, setMusicVolume] = useState(0.35);
  const [videoVolume, setVideoVolume] = useState(1);
  const [speed, setSpeed] = useState(1.25);
  const [speedKeepAudio, setSpeedKeepAudio] = useState(true);
  const [gifStart, setGifStart] = useState(0);
  const [gifDuration, setGifDuration] = useState(3);
  const [gifFps, setGifFps] = useState(12);
  const [gifWidth, setGifWidth] = useState(480);
  const [thumbTimestamp, setThumbTimestamp] = useState(0);
  const [thumbWidth, setThumbWidth] = useState(1280);
  const [job, setJob] = useState<Job | null>(null);
  const [resultUrl, setResultUrl] = useState("");
  const [resultKind, setResultKind] = useState<"video" | "audio" | "image" | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleUpload(file: File) {
    setLoading(true);
    setLoadingMsg(`Uploading ${file.name}...`);
    setError(null);
    setJob(null);
    setResultUrl("");
    setResultKind(null);
    try {
      const saved = await uploadEditVideo(file);
      if (sourcePreview) URL.revokeObjectURL(sourcePreview);
      setUploaded(saved);
      setSourceName(file.name);
      setSourcePreview(URL.createObjectURL(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  }

  async function handleSecondaryUpload(file: File) {
    setLoading(true);
    setLoadingMsg(`Uploading ${file.name}...`);
    setError(null);
    try {
      if (selectedUtility === "pip") {
        const saved = await uploadEditVideo(file);
        setOverlayVideo(saved);
      } else if (selectedUtility === "music") {
        const saved = await uploadEditAudio(file);
        setMusicAudio(saved);
      }
      setSecondaryName(file.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  }

  async function poll(jobId: string) {
    while (true) {
      await sleep(700);
      const next = await getJob(jobId);
      setJob(next);
      setLoadingMsg(next.message || "Processing...");
      if (next.status === "completed") return next;
      if (next.status === "failed") throw new Error(next.error || next.message || "Job failed");
    }
  }

  async function handleRunUtility() {
    if (!uploaded) {
      setError("Upload a video first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResultUrl("");
    setResultKind(null);
    setLoadingMsg("Starting job...");
    try {
      let started: Job;
      const video_path = uploaded.video_path;
      if (selectedUtility === "extract") {
        started = await extractVideoAudio({ video_path, format: audioFormat });
      } else if (selectedUtility === "mute") {
        started = await removeVideoAudio({ video_path });
      } else if (selectedUtility === "compress") {
        started = await compressVideo({ video_path, preset: compressPreset });
      } else if (selectedUtility === "rotate") {
        started = await rotateFlipVideo({ video_path, rotate, flip });
      } else if (selectedUtility === "reverse") {
        started = await reverseVideo({ video_path, include_audio: reverseAudio });
      } else if (selectedUtility === "loop") {
        started = await loopVideo({ video_path, count: loopCount });
      } else if (selectedUtility === "stabilize") {
        started = await stabilizeVideo({ video_path, shakiness, accuracy });
      } else if (selectedUtility === "color") {
        started = await colorCorrectVideo({ video_path, brightness, contrast, saturation, hue });
      } else if (selectedUtility === "privacy") {
        started = await regionEffectVideo({
          video_path,
          x: regionX,
          y: regionY,
          width: regionW,
          height: regionH,
          mode: regionMode,
          strength: regionStrength,
        });
      } else if (selectedUtility === "pip") {
        if (!overlayVideo) throw new Error("Upload an overlay video first.");
        started = await pictureInPictureVideo({
          video_path,
          overlay_path: overlayVideo.video_path,
          position: pipPosition,
          size_percent: pipSize,
          margin: pipMargin,
        });
      } else if (selectedUtility === "music") {
        if (!musicAudio) throw new Error("Upload a music file first.");
        started = await backgroundMusicVideo({
          video_path,
          audio_path: musicAudio.audio_path,
          music_volume: musicVolume,
          video_volume: videoVolume,
        });
      } else if (selectedUtility === "speed") {
        started = await speedVideo({ video_path, speed, keep_audio: speedKeepAudio });
      } else if (selectedUtility === "gif") {
        started = await gifVideo({
          video_path,
          start_time: gifStart,
          duration: gifDuration,
          fps: gifFps,
          width: gifWidth,
        });
      } else {
        started = await thumbnailVideo({ video_path, timestamp: thumbTimestamp, width: thumbWidth });
      }
      setJob(started);
      const finished = await poll(started.job_id);
      setResultKind(selectedUtility === "extract" ? "audio" : selectedUtility === "gif" || selectedUtility === "thumbnail" ? "image" : "video");
      setResultUrl(`${getDownloadUrl(finished.job_id)}?t=${Date.now()}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Utility job failed");
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  }

  const progress = Math.round(Number(job?.progress || 0));
  const resultName =
    selectedUtility === "extract" ? `extracted_audio.${audioFormat}` :
    selectedUtility === "gif" ? "clip.gif" :
    selectedUtility === "thumbnail" ? "thumbnail.png" :
    `${selectedUtility}_${sourceName || "video.mp4"}`;
  const needsSecondaryUpload = selectedUtility === "pip" || selectedUtility === "music";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Editor</h1>
        <p className="text-zinc-400 mt-1">Fast local video utilities and focused edit tools</p>
      </div>

      <div className="flex flex-wrap gap-2 mb-6">
        {[
          { id: "utilities", label: "Utilities", count: UTILITIES.length },
          { id: "filters", label: "Style Filters", count: STYLE_FILTERS.length },
          { id: "region", label: "Region Edit", count: EDIT_OPS.length },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setMode(tab.id as Mode)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === tab.id ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {tab.label} <Badge className="ml-1.5">{tab.count}</Badge>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <Card>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleUpload(file);
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-full border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-indigo-500/30 transition-colors"
            >
              <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
              <p className="text-sm text-zinc-400">{sourceName || "Drop video here or click to upload"}</p>
              <p className="text-xs text-zinc-600 mt-1">MP4, MOV, AVI, MKV, WEBM</p>
            </button>
          </Card>

          {needsSecondaryUpload && (
            <Card>
              <input
                ref={secondaryFileInputRef}
                type="file"
                accept={selectedUtility === "pip" ? "video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm" : "audio/mpeg,audio/wav,audio/aac,audio/flac,audio/ogg"}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void handleSecondaryUpload(file);
                }}
              />
              <button
                type="button"
                onClick={() => secondaryFileInputRef.current?.click()}
                className="w-full border border-zinc-700 rounded-lg p-4 text-left hover:border-indigo-500/30 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {selectedUtility === "pip" ? <Video className="w-5 h-5 text-zinc-500" /> : <Music className="w-5 h-5 text-zinc-500" />}
                  <div className="min-w-0">
                    <p className="text-sm text-zinc-300 truncate">{secondaryName || (selectedUtility === "pip" ? "Upload overlay video" : "Upload music file")}</p>
                    <p className="text-xs text-zinc-600">{selectedUtility === "pip" ? "Used as the inset video" : "Mixed with the video audio"}</p>
                  </div>
                </div>
              </button>
            </Card>
          )}

          {mode === "utilities" ? (
            <>
              <Card>
                <CardTitle className="text-sm mb-3">Utility</CardTitle>
                <div className="grid grid-cols-1 gap-2">
                  {UTILITIES.map((utility) => {
                    const Icon = utility.icon;
                    return (
                      <button
                        key={utility.id}
                        onClick={() => setSelectedUtility(utility.id)}
                        className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${
                          selectedUtility === utility.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                        }`}
                      >
                        <Icon className="w-4 h-4 text-indigo-300" />
                        <span className="flex-1 min-w-0">
                          <span className="block text-sm font-medium">{utility.name}</span>
                          <span className="block text-[10px] text-zinc-500">{utility.desc}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </Card>
              <UtilityControls
                selectedUtility={selectedUtility}
                audioFormat={audioFormat}
                setAudioFormat={setAudioFormat}
                compressPreset={compressPreset}
                setCompressPreset={setCompressPreset}
                rotate={rotate}
                setRotate={setRotate}
                flip={flip}
                setFlip={setFlip}
                reverseAudio={reverseAudio}
                setReverseAudio={setReverseAudio}
                loopCount={loopCount}
                setLoopCount={setLoopCount}
                shakiness={shakiness}
                setShakiness={setShakiness}
                accuracy={accuracy}
                setAccuracy={setAccuracy}
                brightness={brightness}
                setBrightness={setBrightness}
                contrast={contrast}
                setContrast={setContrast}
                saturation={saturation}
                setSaturation={setSaturation}
                hue={hue}
                setHue={setHue}
                regionMode={regionMode}
                setRegionMode={setRegionMode}
                regionX={regionX}
                setRegionX={setRegionX}
                regionY={regionY}
                setRegionY={setRegionY}
                regionW={regionW}
                setRegionW={setRegionW}
                regionH={regionH}
                setRegionH={setRegionH}
                regionStrength={regionStrength}
                setRegionStrength={setRegionStrength}
                pipPosition={pipPosition}
                setPipPosition={setPipPosition}
                pipSize={pipSize}
                setPipSize={setPipSize}
                pipMargin={pipMargin}
                setPipMargin={setPipMargin}
                musicVolume={musicVolume}
                setMusicVolume={setMusicVolume}
                videoVolume={videoVolume}
                setVideoVolume={setVideoVolume}
                speed={speed}
                setSpeed={setSpeed}
                speedKeepAudio={speedKeepAudio}
                setSpeedKeepAudio={setSpeedKeepAudio}
                gifStart={gifStart}
                setGifStart={setGifStart}
                gifDuration={gifDuration}
                setGifDuration={setGifDuration}
                gifFps={gifFps}
                setGifFps={setGifFps}
                gifWidth={gifWidth}
                setGifWidth={setGifWidth}
                thumbTimestamp={thumbTimestamp}
                setThumbTimestamp={setThumbTimestamp}
                thumbWidth={thumbWidth}
                setThumbWidth={setThumbWidth}
              />
              <Button className="w-full" onClick={handleRunUtility} disabled={loading || !uploaded}>
                {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</> : <><Wand2 className="w-4 h-4 mr-2" /> Run Utility</>}
              </Button>
            </>
          ) : mode === "filters" ? (
            <LegacyFilters selectedFilter={selectedFilter} setSelectedFilter={setSelectedFilter} />
          ) : (
            <LegacyRegion selectedOp={selectedOp} setSelectedOp={setSelectedOp} />
          )}
        </div>

        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <Card className="min-h-[400px]">
            {loading ? (
              <div className="min-h-[360px] flex items-center justify-center">
                <div className="text-center w-full max-w-md">
                  <Loader2 className="w-10 h-10 text-indigo-400 mx-auto mb-3 animate-spin" />
                  <p className="text-indigo-400 text-sm font-medium">{loadingMsg || "Processing..."}</p>
                  {job && (
                    <div className="mt-4">
                      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${Math.max(progress, 6)}%` }} />
                      </div>
                      <p className="text-[10px] text-zinc-500 mt-1">{progress}%</p>
                    </div>
                  )}
                </div>
              </div>
            ) : resultUrl ? (
              <div>
                {resultKind === "audio" ? (
                  <audio controls className="w-full" src={resultUrl} />
                ) : resultKind === "image" ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img className="w-full rounded-lg bg-black max-h-[520px] object-contain" src={resultUrl} alt="" />
                ) : (
                  <video controls className="w-full rounded-lg bg-black max-h-[520px]" src={resultUrl} />
                )}
                <div className="flex justify-end mt-4">
                  <Button onClick={() => void downloadBlob(resultUrl, resultName)}>
                    <Download className="w-4 h-4 mr-2" /> Download
                  </Button>
                </div>
              </div>
            ) : sourcePreview ? (
              <video controls className="w-full rounded-lg bg-black max-h-[520px]" src={sourcePreview} />
            ) : (
              <div className="min-h-[360px] flex items-center justify-center">
                <div className="text-center">
                  <Scissors className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                  <p className="text-zinc-500 text-sm">Upload a video to start editing</p>
                  <p className="text-zinc-600 text-xs mt-1">Utilities run locally through the StudioLite API</p>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function UtilityControls({
  selectedUtility, audioFormat, setAudioFormat, compressPreset, setCompressPreset,
  rotate, setRotate, flip, setFlip, reverseAudio, setReverseAudio, loopCount, setLoopCount,
  shakiness, setShakiness, accuracy, setAccuracy,
  brightness, setBrightness, contrast, setContrast, saturation, setSaturation, hue, setHue,
  regionMode, setRegionMode, regionX, setRegionX, regionY, setRegionY, regionW, setRegionW,
  regionH, setRegionH, regionStrength, setRegionStrength,
  pipPosition, setPipPosition, pipSize, setPipSize, pipMargin, setPipMargin,
  musicVolume, setMusicVolume, videoVolume, setVideoVolume,
  speed, setSpeed, speedKeepAudio, setSpeedKeepAudio,
  gifStart, setGifStart, gifDuration, setGifDuration, gifFps, setGifFps, gifWidth, setGifWidth,
  thumbTimestamp, setThumbTimestamp, thumbWidth, setThumbWidth,
}: {
  selectedUtility: UtilityId;
  audioFormat: AudioFormat;
  setAudioFormat: (v: AudioFormat) => void;
  compressPreset: CompressPreset;
  setCompressPreset: (v: CompressPreset) => void;
  rotate: RotateValue;
  setRotate: (v: RotateValue) => void;
  flip: FlipMode;
  setFlip: (v: FlipMode) => void;
  reverseAudio: boolean;
  setReverseAudio: (v: boolean) => void;
  loopCount: number;
  setLoopCount: (v: number) => void;
  shakiness: number;
  setShakiness: (v: number) => void;
  accuracy: number;
  setAccuracy: (v: number) => void;
  brightness: number;
  setBrightness: (v: number) => void;
  contrast: number;
  setContrast: (v: number) => void;
  saturation: number;
  setSaturation: (v: number) => void;
  hue: number;
  setHue: (v: number) => void;
  regionMode: RegionMode;
  setRegionMode: (v: RegionMode) => void;
  regionX: number;
  setRegionX: (v: number) => void;
  regionY: number;
  setRegionY: (v: number) => void;
  regionW: number;
  setRegionW: (v: number) => void;
  regionH: number;
  setRegionH: (v: number) => void;
  regionStrength: number;
  setRegionStrength: (v: number) => void;
  pipPosition: PipPosition;
  setPipPosition: (v: PipPosition) => void;
  pipSize: number;
  setPipSize: (v: number) => void;
  pipMargin: number;
  setPipMargin: (v: number) => void;
  musicVolume: number;
  setMusicVolume: (v: number) => void;
  videoVolume: number;
  setVideoVolume: (v: number) => void;
  speed: number;
  setSpeed: (v: number) => void;
  speedKeepAudio: boolean;
  setSpeedKeepAudio: (v: boolean) => void;
  gifStart: number;
  setGifStart: (v: number) => void;
  gifDuration: number;
  setGifDuration: (v: number) => void;
  gifFps: number;
  setGifFps: (v: number) => void;
  gifWidth: number;
  setGifWidth: (v: number) => void;
  thumbTimestamp: number;
  setThumbTimestamp: (v: number) => void;
  thumbWidth: number;
  setThumbWidth: (v: number) => void;
}) {
  if (selectedUtility === "extract") {
    return (
      <Card>
        <CardTitle className="text-sm mb-3">Audio Format</CardTitle>
        <Segmented options={["wav", "mp3"]} value={audioFormat} onChange={(v) => setAudioFormat(v as AudioFormat)} />
      </Card>
    );
  }
  if (selectedUtility === "compress") {
    return (
      <Card>
        <CardTitle className="text-sm mb-3">Quality</CardTitle>
        <Segmented options={["high", "medium", "low"]} value={compressPreset} onChange={(v) => setCompressPreset(v as CompressPreset)} />
      </Card>
    );
  }
  if (selectedUtility === "rotate") {
    return (
      <Card className="space-y-4">
        <div>
          <CardTitle className="text-sm mb-3">Rotation</CardTitle>
          <Segmented options={["0", "90", "180", "270"]} value={String(rotate)} onChange={(v) => setRotate(Number(v) as RotateValue)} />
        </div>
        <div>
          <CardTitle className="text-sm mb-3">Flip</CardTitle>
          <div className="grid grid-cols-3 gap-2">
            {[
              { id: "none", icon: RotateCw, label: "None" },
              { id: "horizontal", icon: FlipHorizontal, label: "Horizontal" },
              { id: "vertical", icon: FlipVertical, label: "Vertical" },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  type="button"
                  title={item.label}
                  onClick={() => setFlip(item.id as FlipMode)}
                  className={`h-10 rounded-lg border flex items-center justify-center transition-all ${
                    flip === item.id ? "border-indigo-500 bg-indigo-500/10 text-indigo-300" : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </button>
              );
            })}
          </div>
        </div>
      </Card>
    );
  }
  if (selectedUtility === "reverse") {
    return (
      <Card>
        <label className="flex items-center gap-3 text-sm text-zinc-300">
          <input type="checkbox" checked={reverseAudio} onChange={(e) => setReverseAudio(e.target.checked)} className="accent-indigo-500" />
          Reverse audio too
        </label>
      </Card>
    );
  }
  if (selectedUtility === "loop") {
    return (
      <Card>
        <label className="text-[10px] uppercase tracking-wide text-zinc-500">Loop Count</label>
        <input
          type="number"
          min={2}
          max={20}
          value={loopCount}
          onChange={(e) => setLoopCount(Math.max(2, Math.min(20, Number(e.target.value) || 2)))}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 mt-1"
        />
      </Card>
    );
  }
  if (selectedUtility === "stabilize") {
    return (
      <Card className="space-y-3">
        <NumberField label="Shakiness" value={shakiness} min={1} max={10} step={1} onChange={setShakiness} />
        <NumberField label="Accuracy" value={accuracy} min={1} max={15} step={1} onChange={setAccuracy} />
      </Card>
    );
  }
  if (selectedUtility === "color") {
    return (
      <Card className="space-y-3">
        <NumberField label="Brightness" value={brightness} min={-1} max={1} step={0.05} onChange={setBrightness} />
        <NumberField label="Contrast" value={contrast} min={0} max={3} step={0.05} onChange={setContrast} />
        <NumberField label="Saturation" value={saturation} min={0} max={3} step={0.05} onChange={setSaturation} />
        <NumberField label="Hue" value={hue} min={-180} max={180} step={5} onChange={setHue} />
      </Card>
    );
  }
  if (selectedUtility === "privacy") {
    return (
      <Card className="space-y-3">
        <div>
          <CardTitle className="text-sm mb-3">Effect</CardTitle>
          <Segmented options={["blur", "pixelate"]} value={regionMode} onChange={(v) => setRegionMode(v as RegionMode)} />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <NumberField label="X" value={regionX} min={0} max={10000} step={1} onChange={setRegionX} />
          <NumberField label="Y" value={regionY} min={0} max={10000} step={1} onChange={setRegionY} />
          <NumberField label="Width" value={regionW} min={1} max={10000} step={1} onChange={setRegionW} />
          <NumberField label="Height" value={regionH} min={1} max={10000} step={1} onChange={setRegionH} />
        </div>
        <NumberField label="Strength" value={regionStrength} min={2} max={80} step={1} onChange={setRegionStrength} />
      </Card>
    );
  }
  if (selectedUtility === "pip") {
    return (
      <Card className="space-y-3">
        <div>
          <CardTitle className="text-sm mb-3">Position</CardTitle>
          <Segmented
            options={["top_left", "top_right", "bottom_left", "bottom_right", "center"]}
            value={pipPosition}
            onChange={(v) => setPipPosition(v as PipPosition)}
          />
        </div>
        <NumberField label="Size %" value={pipSize} min={10} max={80} step={1} onChange={setPipSize} />
        <NumberField label="Margin" value={pipMargin} min={0} max={200} step={1} onChange={setPipMargin} />
      </Card>
    );
  }
  if (selectedUtility === "music") {
    return (
      <Card className="space-y-3">
        <NumberField label="Music Volume" value={musicVolume} min={0} max={2} step={0.05} onChange={setMusicVolume} />
        <NumberField label="Video Volume" value={videoVolume} min={0} max={2} step={0.05} onChange={setVideoVolume} />
      </Card>
    );
  }
  if (selectedUtility === "speed") {
    return (
      <Card className="space-y-3">
        <NumberField label="Speed" value={speed} min={0.25} max={4} step={0.05} onChange={setSpeed} />
        <label className="flex items-center gap-3 text-sm text-zinc-300">
          <input type="checkbox" checked={speedKeepAudio} onChange={(e) => setSpeedKeepAudio(e.target.checked)} className="accent-indigo-500" />
          Keep audio in sync
        </label>
      </Card>
    );
  }
  if (selectedUtility === "gif") {
    return (
      <Card className="space-y-3">
        <NumberField label="Start Time" value={gifStart} min={0} max={100000} step={0.1} onChange={setGifStart} />
        <NumberField label="Duration" value={gifDuration} min={0.1} max={30} step={0.1} onChange={setGifDuration} />
        <NumberField label="FPS" value={gifFps} min={4} max={30} step={1} onChange={setGifFps} />
        <NumberField label="Width" value={gifWidth} min={120} max={1280} step={10} onChange={setGifWidth} />
      </Card>
    );
  }
  if (selectedUtility === "thumbnail") {
    return (
      <Card className="space-y-3">
        <NumberField label="Timestamp" value={thumbTimestamp} min={0} max={100000} step={0.1} onChange={setThumbTimestamp} />
        <NumberField label="Width" value={thumbWidth} min={120} max={3840} step={10} onChange={setThumbWidth} />
      </Card>
    );
  }
  return (
    <Card>
      <p className="text-sm text-zinc-400">This utility has no extra controls.</p>
    </Card>
  );
}

function NumberField({
  label, value, min, max, step, onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const next = Number(e.target.value);
          onChange(Math.max(min, Math.min(max, Number.isFinite(next) ? next : value)));
        }}
        className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 mt-1"
      />
    </label>
  );
}

function Segmented({ options, value, onChange }: { options: string[]; value: string; onChange: (value: string) => void }) {
  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onChange(option)}
          className={`px-3 py-2 rounded-lg border text-xs font-medium capitalize transition-all ${
            value === option ? "border-indigo-500 bg-indigo-500/10 text-indigo-300" : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

function LegacyFilters({ selectedFilter, setSelectedFilter }: { selectedFilter: string; setSelectedFilter: (v: string) => void }) {
  return (
    <>
      <Card>
        <CardTitle className="text-sm mb-3">Select Filter</CardTitle>
        <div className="space-y-2">
          {STYLE_FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setSelectedFilter(f.id)}
              className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-left transition-all ${
                selectedFilter === f.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
              }`}
            >
              <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${f.preview} flex-shrink-0`} />
              <div>
                <div className="text-xs font-medium">{f.name}</div>
                <div className="text-[10px] text-zinc-500">{f.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </Card>
      <Button className="w-full" disabled>
        <Wand2 className="w-4 h-4 mr-2" /> Apply Filter
      </Button>
    </>
  );
}

function LegacyRegion({ selectedOp, setSelectedOp }: { selectedOp: string; setSelectedOp: (v: string) => void }) {
  return (
    <>
      <Card>
        <CardTitle className="text-sm mb-3">Edit Operation</CardTitle>
        <div className="space-y-2">
          {EDIT_OPS.map((op) => (
            <button
              key={op.id}
              onClick={() => setSelectedOp(op.id)}
              className={`w-full p-2.5 rounded-lg border text-left transition-all ${
                selectedOp === op.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
              }`}
            >
              <div className="text-xs font-medium">{op.name}</div>
              <div className="text-[10px] text-zinc-500">{op.desc}</div>
            </button>
          ))}
        </div>
      </Card>
      <Card>
        <CardTitle className="text-sm mb-3">Region</CardTitle>
        <div className="grid grid-cols-2 gap-2">
          {["X", "Y", "Width", "Height"].map((label, i) => (
            <div key={label}>
              <label className="text-[10px] text-zinc-500">{label}</label>
              <input
                type="number"
                defaultValue={[50, 50, 200, 200][i]}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 mt-0.5"
              />
            </div>
          ))}
        </div>
      </Card>
      <Button className="w-full" disabled>
        <Wand2 className="w-4 h-4 mr-2" /> Apply Edit
      </Button>
    </>
  );
}
