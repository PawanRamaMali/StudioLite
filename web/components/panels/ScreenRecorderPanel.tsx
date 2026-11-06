"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Monitor, Circle, Download, Upload, Square, Mic, MicOff, Volume2 } from "lucide-react";
import { uploadEditVideo, type EditUploadResponse } from "@/lib/api";

type VideoBits = { label: string; kbps: number };
const BITRATES: VideoBits[] = [
  { label: "Low (1.5M)",   kbps: 1500 },
  { label: "Medium (4M)",  kbps: 4000 },
  { label: "High (8M)",    kbps: 8000 },
  { label: "Lossless-ish", kbps: 20000 },
];

function pickSupportedMime(): string {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
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

/** Merge a mic MediaStream's audio into the display stream. Returns the
 * combined stream and a cleanup function; both tracks are tied to the
 * same AudioContext so stopping one stops all. */
function mergeDisplayAndMic(display: MediaStream, mic: MediaStream): {
  stream: MediaStream;
  cleanup: () => void;
} {
  type GlobalWithWebkit = typeof window & { webkitAudioContext?: typeof AudioContext };
  const g = window as GlobalWithWebkit;
  const AC: typeof AudioContext = g.AudioContext ?? g.webkitAudioContext!;
  const ctx = new AC();
  const dest = ctx.createMediaStreamDestination();
  const sources: MediaStreamAudioSourceNode[] = [];
  if (display.getAudioTracks().length > 0) {
    const s = ctx.createMediaStreamSource(new MediaStream(display.getAudioTracks()));
    s.connect(dest);
    sources.push(s);
  }
  if (mic.getAudioTracks().length > 0) {
    const s = ctx.createMediaStreamSource(new MediaStream(mic.getAudioTracks()));
    s.connect(dest);
    sources.push(s);
  }
  const combined = new MediaStream();
  display.getVideoTracks().forEach((t) => combined.addTrack(t));
  dest.stream.getAudioTracks().forEach((t) => combined.addTrack(t));
  return {
    stream: combined,
    cleanup: () => {
      sources.forEach((s) => { try { s.disconnect(); } catch { /* already gone */ } });
      try { ctx.close(); } catch { /* already closed */ }
    },
  };
}

export default function ScreenRecorderPanel() {
  const previewRef = useRef<HTMLVideoElement>(null);
  const displayStreamRef = useRef<MediaStream | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const combinedStreamRef = useRef<MediaStream | null>(null);
  const cleanupAudioRef = useRef<(() => void) | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [bitrate, setBitrate] = useState<VideoBits>(BITRATES[1]);
  const [micOn, setMicOn] = useState(false);
  const [systemAudio, setSystemAudio] = useState(true);

  const [capturing, setCapturing] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopStreams = useCallback(() => {
    displayStreamRef.current?.getTracks().forEach((t) => t.stop());
    displayStreamRef.current = null;
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    combinedStreamRef.current = null;
    cleanupAudioRef.current?.();
    cleanupAudioRef.current = null;
    setCapturing(false);
  }, []);

  useEffect(() => () => {
    stopStreams();
    if (tickRef.current) clearInterval(tickRef.current);
    if (blobUrl) URL.revokeObjectURL(blobUrl);
  }, [stopStreams, blobUrl]);

  const startCapture = async () => {
    setError(null);
    try {
      stopStreams();
      // Chromium supports an `audio: true` on getDisplayMedia for tab/system
      // audio; Firefox does not. We request it and silently drop if denied.
      const display = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: systemAudio,
      });
      displayStreamRef.current = display;
      let combined: MediaStream = display;
      if (micOn) {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
        micStreamRef.current = mic;
        const merged = mergeDisplayAndMic(display, mic);
        combined = merged.stream;
        cleanupAudioRef.current = merged.cleanup;
      }
      combinedStreamRef.current = combined;
      // Tab-close auto-stops the stream. React to that so the UI
      // doesn't look stuck capturing.
      display.getVideoTracks()[0]?.addEventListener("ended", () => {
        stopStreams();
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
          recorderRef.current.stop();
        }
      });
      setCapturing(true);
      if (previewRef.current) {
        previewRef.current.srcObject = display;
        await previewRef.current.play().catch(() => { /* autoplay reject */ });
      }
    } catch (e) {
      setError((e as Error).message || "Could not start screen capture");
    }
  };

  const startRecording = () => {
    const stream = combinedStreamRef.current;
    if (!stream) return;
    setError(null);
    chunksRef.current = [];
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    setBlobUrl(null);
    setBlob(null);
    setUpload(null);
    const mimeType = pickSupportedMime();
    try {
      const rec = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: bitrate.kbps * 1000,
      });
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
      };
      rec.onstop = () => {
        const b = new Blob(chunksRef.current, { type: mimeType });
        setBlob(b);
        setBlobUrl(URL.createObjectURL(b));
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
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = `screen_${new Date().toISOString().replace(/[:.]/g, "-")}.webm`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const uploadRecording = async () => {
    if (!blob) return;
    const file = new File([blob], `screen_${Date.now()}.webm`, { type: blob.type });
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

  const supported = typeof window !== "undefined" && !!navigator.mediaDevices?.getDisplayMedia;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Screen Recorder</h1>
        <p className="text-zinc-400 mt-1">
          Capture a window, tab, or your whole screen with optional mic and
          system audio. Recordings land as .webm and can go straight into
          the editor.
        </p>
      </div>

      {!supported && (
        <div className="mb-4 text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-3">
          Screen capture (getDisplayMedia) isn&apos;t available here. Use a
          recent Chromium or Firefox over HTTPS or localhost.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Capture</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Bitrate</label>
                <div className="grid grid-cols-2 gap-2">
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
              <div className="flex gap-3">
                <button
                  onClick={() => setSystemAudio(!systemAudio)}
                  disabled={recording}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-xs border ${
                    systemAudio
                      ? "bg-emerald-600/15 border-emerald-500/40 text-emerald-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-500"
                  }`}
                >
                  <Volume2 className="w-3.5 h-3.5" />
                  System audio
                </button>
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
                  Microphone
                </button>
              </div>
              <p className="text-[11px] text-zinc-500">
                System-audio capture is Chromium-only. On Firefox, mic audio
                still works.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              <Button
                size="sm"
                variant={capturing ? "secondary" : "primary"}
                onClick={capturing ? stopStreams : startCapture}
                disabled={!supported || recording}
              >
                <Monitor className="w-4 h-4 mr-1" />
                {capturing ? "Stop capture" : "Pick source"}
              </Button>
              <Button
                size="sm"
                variant={recording ? "danger" : "primary"}
                disabled={!capturing}
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
          {capturing || blobUrl ? (
            <div className="w-full">
              {capturing && !blobUrl && (
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
              Start capture to pick a window, tab, or screen to record.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
