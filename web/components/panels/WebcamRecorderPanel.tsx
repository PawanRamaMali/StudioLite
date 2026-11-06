"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Camera, Circle, Download, Upload, Square, Mic, MicOff, Video as VideoIcon } from "lucide-react";
import { uploadEditVideo, type EditUploadResponse } from "@/lib/api";

type Resolution = { label: string; width: number; height: number };
const RESOLUTIONS: Resolution[] = [
  { label: "1080p", width: 1920, height: 1080 },
  { label: "720p",  width: 1280, height: 720 },
  { label: "480p",  width: 854,  height: 480 },
];

type VideoBits = { label: string; kbps: number };
const BITRATES: VideoBits[] = [
  { label: "Low",    kbps: 1500 },
  { label: "Medium", kbps: 3500 },
  { label: "High",   kbps: 8000 },
];

function pickSupportedMime(): string {
  // VP9/Opus produces smaller files on Chromium; VP8 has broader support.
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];
  if (typeof window === "undefined" || !("MediaRecorder" in window)) return "video/webm";
  for (const c of candidates) {
    try {
      if ((window as unknown as { MediaRecorder: { isTypeSupported?: (s: string) => boolean } }).MediaRecorder.isTypeSupported?.(c)) {
        return c;
      }
    } catch { /* fall through */ }
  }
  return "video/webm";
}

function fmtTime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export default function WebcamRecorderPanel() {
  const previewRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [resolution, setResolution] = useState<Resolution>(RESOLUTIONS[1]);
  const [bitrate, setBitrate] = useState<VideoBits>(BITRATES[1]);
  const [micOn, setMicOn] = useState(true);
  const [facing, setFacing] = useState<"user" | "environment">("user");

  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  useEffect(() => () => {
    stopStream();
    if (tickRef.current) clearInterval(tickRef.current);
    if (blobUrl) URL.revokeObjectURL(blobUrl);
  }, [stopStream, blobUrl]);

  const startCamera = async () => {
    setError(null);
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
          facingMode: facing,
        },
        audio: micOn,
      });
      streamRef.current = stream;
      setCameraActive(true);
      if (previewRef.current) {
        previewRef.current.srcObject = stream;
        await previewRef.current.play().catch(() => { /* play can reject on autoplay policy */ });
      }
    } catch (e) {
      setError((e as Error).message || "Could not access camera");
    }
  };

  const startRecording = () => {
    if (!streamRef.current) return;
    setError(null);
    chunksRef.current = [];
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    setBlobUrl(null);
    setBlob(null);
    setUpload(null);
    const mimeType = pickSupportedMime();
    try {
      const rec = new MediaRecorder(streamRef.current, {
        mimeType,
        videoBitsPerSecond: bitrate.kbps * 1000,
      });
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
      };
      rec.onstop = () => {
        const b = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(b);
        setBlob(b);
        setBlobUrl(url);
      };
      rec.start(500);
      recorderRef.current = rec;
      startedAtRef.current = Date.now();
      setElapsed(0);
      if (tickRef.current) clearInterval(tickRef.current);
      tickRef.current = setInterval(() => setElapsed(Date.now() - startedAtRef.current), 250);
      setRecording(true);
    } catch (e) {
      setError((e as Error).message || "Could not start recorder");
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (tickRef.current) { clearInterval(tickRef.current); tickRef.current = null; }
    setRecording(false);
  };

  const download = () => {
    if (!blob || !blobUrl) return;
    const ext = blob.type.includes("mp4") ? "mp4" : "webm";
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = `webcam_${new Date().toISOString().replace(/[:.]/g, "-")}.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const uploadRecording = async () => {
    if (!blob) return;
    const ext = blob.type.includes("mp4") ? "mp4" : "webm";
    const file = new File([blob], `webcam_${Date.now()}.${ext}`, { type: blob.type });
    setUploading(true);
    try {
      const r = await uploadEditVideo(file);
      setUpload(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const supported = typeof window !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Webcam Recorder</h1>
        <p className="text-zinc-400 mt-1">
          Record from your camera and microphone, review the clip, then
          download or send it into StudioLite&apos;s editor pipeline.
        </p>
      </div>

      {!supported && (
        <div className="mb-4 text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-3">
          Your browser doesn&apos;t expose getUserMedia. Try a modern Chromium
          or Firefox and make sure the page is served over HTTPS or localhost.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Camera</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Resolution</label>
                <div className="grid grid-cols-3 gap-2">
                  {RESOLUTIONS.map((r) => (
                    <button
                      key={r.label}
                      onClick={() => setResolution(r)}
                      disabled={recording}
                      className={`px-3 py-2 rounded-md text-xs border ${
                        resolution.label === r.label
                          ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                          : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                      }`}
                    >{r.label}</button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Bitrate</label>
                <div className="grid grid-cols-3 gap-2">
                  {BITRATES.map((b) => (
                    <button
                      key={b.label}
                      onClick={() => setBitrate(b)}
                      disabled={recording}
                      className={`px-3 py-2 rounded-md text-xs border ${
                        bitrate.label === b.label
                          ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                          : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                      }`}
                    >{b.label}</button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setMicOn(!micOn)}
                  disabled={recording}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-xs border ${
                    micOn
                      ? "bg-emerald-600/15 border-emerald-500/40 text-emerald-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-500"
                  }`}
                >
                  {micOn ? <Mic className="w-3.5 h-3.5" /> : <MicOff className="w-3.5 h-3.5" />}
                  {micOn ? "Mic on" : "Mic off"}
                </button>
                <button
                  onClick={() => setFacing(facing === "user" ? "environment" : "user")}
                  disabled={recording}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-xs border bg-zinc-900/50 border-zinc-800 text-zinc-400"
                >
                  <VideoIcon className="w-3.5 h-3.5" />
                  {facing === "user" ? "Front" : "Rear"}
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              <Button
                size="sm"
                variant={cameraActive ? "secondary" : "primary"}
                onClick={cameraActive ? stopStream : startCamera}
                disabled={!supported || recording}
              >
                <Camera className="w-4 h-4 mr-1" />
                {cameraActive ? "Stop camera" : "Start camera"}
              </Button>
              <Button
                size="sm"
                variant={recording ? "danger" : "primary"}
                disabled={!cameraActive}
                onClick={recording ? stopRecording : startRecording}
              >
                {recording ? (
                  <><Square className="w-4 h-4 mr-1" /> Stop ({fmtTime(elapsed)})</>
                ) : (
                  <><Circle className="w-4 h-4 mr-1 fill-current" /> Record</>
                )}
              </Button>
            </div>
          </Card>

          {blob && (
            <Card>
              <CardTitle className="text-sm mb-3">Clip ready</CardTitle>
              <p className="text-[11px] text-zinc-500 mb-3">
                {(blob.size / (1024 * 1024)).toFixed(1)} MB &middot; {blob.type}
              </p>
              <div className="grid grid-cols-2 gap-2">
                <Button size="sm" variant="secondary" onClick={download}>
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
                <Button size="sm" onClick={uploadRecording} disabled={uploading || !!upload}>
                  <Upload className="w-4 h-4 mr-1" />
                  {uploading ? "Uploading..." : upload ? "Uploaded" : "Send to editor"}
                </Button>
              </div>
              {upload && (
                <p className="text-[11px] text-emerald-400 mt-2">
                  Saved as {upload.filename} &mdash; available in the editor panels.
                </p>
              )}
            </Card>
          )}

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px] flex items-center justify-center">
          {cameraActive || blobUrl ? (
            <div className="w-full">
              {cameraActive && !blobUrl && (
                <>
                  <video ref={previewRef} muted playsInline autoPlay className="w-full rounded-md bg-black" />
                  <p className="text-[11px] text-zinc-500 mt-2 text-center">
                    Live preview {recording && <span className="text-red-400">&middot; recording</span>}
                  </p>
                </>
              )}
              {blobUrl && (
                <>
                  <video src={blobUrl} controls playsInline className="w-full rounded-md bg-black" />
                  <p className="text-[11px] text-zinc-500 mt-2 text-center">Review the recording above.</p>
                </>
              )}
            </div>
          ) : (
            <p className="text-sm text-zinc-500 text-center px-8">
              Start the camera to see a live preview.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
