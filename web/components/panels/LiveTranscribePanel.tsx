"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Mic, MonitorSpeaker, Radio, Square, Download, Loader2, AlertCircle, Languages,
  Copy, Check, Scissors, Trash2,
} from "lucide-react";

type Segment = { start: number; end: number; text: string };
type Block = { id: string; segments: Segment[] };
type ServerMsg =
  | { type: "ready"; session: string; device: string }
  | { type: "status"; stage: "loading_model" | "model_ready"; model?: string }
  | ({ type: "final" | "partial" } & Segment)
  | { type: "complete"; transcripts: { txt: string; srt: string; json: string } }
  | { type: "pong" }
  | { type: "error"; message: string };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");
const SAMPLE_RATE = 16000;

// Server-side flush is bounded to 20s; give the round-trip and any queued
// decode a comfortable margin so we don't fall back to the on-disk transcript
// while the server was legitimately still finalizing.
const STOP_TIMEOUT_MS = 30000;
const PING_INTERVAL_MS = 25000;

/**
 * Turn the raw getUserMedia / getDisplayMedia exceptions into actionable text.
 * The browser messages ("Permission denied by user") don't tell the user how to
 * recover, so we map the DOMException name to a concrete next step.
 */
function describeStartError(e: unknown, useMic: boolean, useDesktop: boolean): string {
  const name = (e && typeof e === "object" && "name" in e) ? String((e as { name: unknown }).name) : "";
  const raw = e instanceof Error ? e.message : "Failed to start";
  const src = useMic && useDesktop ? "microphone / desktop audio" : useDesktop ? "desktop audio" : "microphone";

  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return (
        `${src} access was blocked. Click the tune/lock icon at the left of Chrome's address bar, ` +
        `set Microphone (and Screen share) to "Allow", then reload and try again. ` +
        `Also check Windows Settings › Privacy & security › Microphone is on for Chrome.`
      );
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone was found. Plug in or enable a mic (Windows Settings › Sound › Input), then retry.";
    case "NotReadableError":
      return "The microphone is in use by another app (Zoom, Teams, OBS...). Close it and retry.";
    case "OverconstrainedError":
      return "The selected audio device can't meet the requested settings. Try a different input device.";
    case "AbortError":
      return "The audio capture request was dismissed. Click Start again and accept the prompt.";
    default:
      // getDisplayMedia is only offered over a secure context.
      if (typeof window !== "undefined" && !window.isSecureContext) {
        return "Audio capture needs a secure context. Open the app via http://localhost or http://127.0.0.1 (not a LAN IP).";
      }
      return raw;
  }
}

function buildFallbackDownloads(session: string) {
  const base = `/static/transcripts/${session}`;
  return {
    txt: `${API_BASE}${base}.txt`,
    srt: `${API_BASE}${base}.srt`,
    json: `${API_BASE}${base}.json`,
  };
}

const MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"];
const LANGUAGES: { code: string | ""; label: string }[] = [
  { code: "", label: "Auto-detect" },
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "hi", label: "Hindi" },
  { code: "zh", label: "Chinese" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "ar", label: "Arabic" },
  { code: "ru", label: "Russian" },
];

function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

export default function LiveTranscribePanel() {
  const [useMic, setUseMic] = useState(true);
  const [useDesktop, setUseDesktop] = useState(false);
  const [model, setModel] = useState("base");
  const [language, setLanguage] = useState("");
  const [translate, setTranslate] = useState(false);

  const [status, setStatus] = useState<"idle" | "connecting" | "loading-model" | "recording" | "stopping" | "complete" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [device, setDevice] = useState<string>("");
  const [elapsed, setElapsed] = useState(0);
  const [finals, setFinals] = useState<Segment[]>([]);
  const [partial, setPartial] = useState<Segment | null>(null);
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [downloads, setDownloads] = useState<{ txt: string; srt: string; json: string } | null>(null);
  const [copied, setCopied] = useState<"all" | "last5" | "split" | `block-${string}` | null>(null);

  // Mic device selection + live input-level meter, so the user can pick the
  // right input and confirm it's actually picking up sound before recording.
  const [inputDevices, setInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [micDeviceId, setMicDeviceId] = useState<string>("");
  const [metering, setMetering] = useState(false);
  const [micLevel, setMicLevel] = useState(0); // RMS, ~0..0.3 for speech
  const [micPeak, setMicPeak] = useState(0);   // decaying peak hold

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const desktopStreamRef = useRef<MediaStream | null>(null);
  const micSrcRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const desktopSrcRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const startTsRef = useRef<number>(0);
  const tickRef = useRef<number | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<string>("");
  const completeRef = useRef<boolean>(false);
  const pingRef = useRef<number | null>(null);
  const stopWatchdogRef = useRef<number | null>(null);
  const meterStreamRef = useRef<MediaStream | null>(null);
  const meterCtxRef = useRef<AudioContext | null>(null);
  const meterRafRef = useRef<number | null>(null);
  const micPeakRef = useRef<number>(0);

  const clearTimers = useCallback(() => {
    if (pingRef.current !== null) {
      window.clearInterval(pingRef.current);
      pingRef.current = null;
    }
    if (stopWatchdogRef.current !== null) {
      window.clearTimeout(stopWatchdogRef.current);
      stopWatchdogRef.current = null;
    }
  }, []);

  const refreshInputDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setInputDevices(all.filter((d) => d.kind === "audioinput"));
    } catch { /* ignore */ }
  }, []);

  const stopMeter = useCallback(() => {
    if (meterRafRef.current !== null) {
      cancelAnimationFrame(meterRafRef.current);
      meterRafRef.current = null;
    }
    meterStreamRef.current?.getTracks().forEach((t) => t.stop());
    meterStreamRef.current = null;
    meterCtxRef.current?.close().catch(() => { /* ignore */ });
    meterCtxRef.current = null;
    micPeakRef.current = 0;
    setMicLevel(0);
    setMicPeak(0);
    setMetering(false);
  }, []);

  const startMeter = useCallback(async () => {
    // Tear down any previous meter first (e.g. device switch).
    if (meterRafRef.current !== null) { cancelAnimationFrame(meterRafRef.current); meterRafRef.current = null; }
    meterStreamRef.current?.getTracks().forEach((t) => t.stop());
    meterCtxRef.current?.close().catch(() => { /* ignore */ });
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: micDeviceId ? { deviceId: { exact: micDeviceId } } : true,
        video: false,
      });
      meterStreamRef.current = stream;
      // Labels are only populated once a stream has been granted.
      refreshInputDevices();
      const ctx = new AudioContext();
      meterCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      src.connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      setMetering(true);
      const loop = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / buf.length);
        micPeakRef.current = Math.max(rms, micPeakRef.current * 0.92);
        setMicLevel(rms);
        setMicPeak(micPeakRef.current);
        meterRafRef.current = requestAnimationFrame(loop);
      };
      loop();
    } catch (e) {
      setErrorMsg(describeStartError(e, true, false));
      setMetering(false);
    }
  }, [micDeviceId, refreshInputDevices]);

  const toggleMeter = useCallback(() => {
    if (metering) stopMeter();
    else startMeter();
  }, [metering, startMeter, stopMeter]);

  // Populate the device list on mount and whenever devices change.
  useEffect(() => {
    refreshInputDevices();
    const md = navigator.mediaDevices;
    if (md && "addEventListener" in md) {
      md.addEventListener("devicechange", refreshInputDevices);
      return () => md.removeEventListener("devicechange", refreshInputDevices);
    }
  }, [refreshInputDevices]);

  const finalizeWithFallback = useCallback((reason?: string) => {
    if (completeRef.current) return;
    completeRef.current = true;
    clearTimers();
    if (sessionRef.current) {
      setDownloads(buildFallbackDownloads(sessionRef.current));
    }
    setStatus("complete");
    setPartial(null);
    if (reason) {
      setErrorMsg(reason);
    }
    try { wsRef.current?.close(); } catch { /* ignore */ }
  }, [clearTimers]);

  // Auto-scroll transcript view
  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [finals, partial]);

  const cleanup = useCallback(() => {
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    clearTimers();
    try { workletNodeRef.current?.disconnect(); } catch { /* ignore */ }
    workletNodeRef.current = null;
    try { micSrcRef.current?.disconnect(); } catch { /* ignore */ }
    micSrcRef.current = null;
    try { desktopSrcRef.current?.disconnect(); } catch { /* ignore */ }
    desktopSrcRef.current = null;
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    desktopStreamRef.current?.getTracks().forEach((t) => t.stop());
    desktopStreamRef.current = null;
    audioCtxRef.current?.close().catch(() => { /* ignore */ });
    audioCtxRef.current = null;
  }, []);

  const start = useCallback(async () => {
    if (!useMic && !useDesktop) {
      setErrorMsg("Enable Microphone, Desktop Audio, or both before starting.");
      return;
    }
    setErrorMsg(null);
    setFinals([]);
    setPartial(null);
    setBlocks([]);
    setDownloads(null);
    completeRef.current = false;
    setStatus("connecting");
    stopMeter(); // free the device before the real capture opens it

    try {
      // 1. Capture streams
      if (useMic) {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            ...(micDeviceId ? { deviceId: { exact: micDeviceId } } : {}),
          },
          video: false,
        });
      }
      if (useDesktop) {
        // Some browsers require video:true; we'll request it then drop the video tracks.
        const ds = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: { displaySurface: "monitor" } as MediaTrackConstraints,
        });
        ds.getVideoTracks().forEach((t) => t.stop());
        if (ds.getAudioTracks().length === 0) {
          ds.getTracks().forEach((t) => t.stop());
          throw new Error("Desktop audio capture not granted. In the share dialog, tick 'Share audio'.");
        }
        desktopStreamRef.current = ds;
      }

      // 2. AudioContext at 16kHz so we don't have to resample.
      // Browsers will resample input streams to this rate automatically.
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ctx;
      await ctx.audioWorklet.addModule("/pcm-worklet.js");
      // After the getUserMedia prompt resolves the click gesture is often
      // considered "consumed", so Chrome opens the AudioContext in "suspended"
      // state. The worklet then never runs and the first Start reliably fails
      // while the second (with permissions already granted) works. Explicit
      // resume here is the fix.
      if (ctx.state === "suspended") {
        try { await ctx.resume(); } catch { /* ignore — worklet still connects */ }
      }

      const workletNode = new AudioWorkletNode(ctx, "pcm-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
      });
      workletNodeRef.current = workletNode;

      // 3. Open WebSocket, with one retry — the very first connect after
      // cold-start sometimes fires `onerror` before the server finishes
      // registering the route, and the second attempt then works instantly.
      const session = `live-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      sessionRef.current = session;
      const params = new URLSearchParams({ session, model });
      if (language) params.set("language", language);
      if (translate) params.set("translate", "1");
      const wsUrl = `${WS_BASE}/api/v1/transcribe/live?${params.toString()}`;

      const openWs = () => new Promise<WebSocket>((resolve, reject) => {
        const sock = new WebSocket(wsUrl);
        sock.binaryType = "arraybuffer";
        const to = window.setTimeout(() => {
          try { sock.close(); } catch { /* ignore */ }
          reject(new Error("WebSocket connect timeout"));
        }, 8000);
        sock.onopen = () => { window.clearTimeout(to); resolve(sock); };
        sock.onerror = () => {
          window.clearTimeout(to);
          try { sock.close(); } catch { /* ignore */ }
          reject(new Error("WebSocket connection failed"));
        };
      });

      let ws: WebSocket;
      try {
        ws = await openWs();
      } catch (firstErr) {
        // Brief backoff then a single retry.
        await new Promise((r) => window.setTimeout(r, 400));
        try {
          ws = await openWs();
        } catch {
          throw firstErr;
        }
      }
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg: ServerMsg = JSON.parse(ev.data);
          if (msg.type === "ready") {
            setDevice(msg.device);
          } else if (msg.type === "status") {
            if (msg.stage === "loading_model") {
              setStatus("loading-model");
            } else if (msg.stage === "model_ready") {
              setStatus("recording");
              startTsRef.current = Date.now();
              setElapsed(0);
              if (tickRef.current === null) {
                tickRef.current = window.setInterval(() => {
                  setElapsed((Date.now() - startTsRef.current) / 1000);
                }, 250);
              }
              if (pingRef.current === null) {
                pingRef.current = window.setInterval(() => {
                  const sock = wsRef.current;
                  if (sock && sock.readyState === WebSocket.OPEN) {
                    try { sock.send("ping"); } catch { /* ignore */ }
                  }
                }, PING_INTERVAL_MS);
              }
            }
          } else if (msg.type === "final") {
            setFinals((prev) => [...prev, { start: msg.start, end: msg.end, text: msg.text }]);
            setPartial(null);
          } else if (msg.type === "partial") {
            setPartial({ start: msg.start, end: msg.end, text: msg.text });
          } else if (msg.type === "complete") {
            completeRef.current = true;
            clearTimers();
            const base = API_BASE;
            setDownloads({
              txt: base + msg.transcripts.txt,
              srt: base + msg.transcripts.srt,
              json: base + msg.transcripts.json,
            });
            setStatus("complete");
            setPartial(null);
          } else if (msg.type === "error") {
            setErrorMsg(msg.message);
            setStatus("error");
          }
          // pong messages are intentionally ignored
        } catch {
          /* ignore */
        }
      };
      ws.onerror = () => {
        // If we already finalized, ignore. Otherwise fall back to the on-disk file.
        if (!completeRef.current && sessionRef.current) {
          finalizeWithFallback(
            "Network/WebSocket error. Recovered transcript from server-side save."
          );
        }
      };
      ws.onclose = () => {
        if (!completeRef.current) {
          if (sessionRef.current && (status === "recording" || status === "stopping" || status === "connecting" || status === "loading-model")) {
            finalizeWithFallback(
              "Connection closed before the server confirmed save. Showing the transcript that was written to disk."
            );
          } else {
            setStatus((s) => (s === "error" ? "error" : "complete"));
          }
        }
        cleanup();
      };

      workletNode.port.onmessage = (ev) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(ev.data);
      };

      // 4. Build the audio graph: mic + desktop -> worklet
      if (micStreamRef.current) {
        micSrcRef.current = ctx.createMediaStreamSource(micStreamRef.current);
        micSrcRef.current.connect(workletNode);
      }
      if (desktopStreamRef.current) {
        desktopSrcRef.current = ctx.createMediaStreamSource(desktopStreamRef.current);
        desktopSrcRef.current.connect(workletNode);
      }

      // Status now driven by server messages (loading_model -> model_ready).
      // Once model_ready arrives we start the timer.
    } catch (e) {
      setErrorMsg(describeStartError(e, useMic, useDesktop));
      setStatus("error");
      cleanup();
    }
  }, [useMic, useDesktop, model, language, translate, status, micDeviceId, cleanup, clearTimers, finalizeWithFallback, stopMeter]);

  const stop = useCallback(() => {
    if (status !== "recording") return;
    setStatus("stopping");
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send("stop"); } catch { /* ignore */ }
    }
    // Audio capture stops once cleanup runs (on ws close). Trim audio early so server stops getting samples.
    try { workletNodeRef.current?.disconnect(); } catch { /* ignore */ }

    // Watchdog: if the server doesn't confirm save within STOP_TIMEOUT_MS,
    // fall back to the on-disk transcript so the UI never strands.
    if (stopWatchdogRef.current !== null) {
      window.clearTimeout(stopWatchdogRef.current);
    }
    stopWatchdogRef.current = window.setTimeout(() => {
      if (!completeRef.current) {
        finalizeWithFallback(
          "Server didn't confirm save in time. Showing the transcript that was written to disk."
        );
      }
    }, STOP_TIMEOUT_MS);
  }, [status, finalizeWithFallback]);

  // Cleanup on unmount
  useEffect(() => () => {
    try { wsRef.current?.close(); } catch { /* ignore */ }
    cleanup();
    stopMeter();
  }, [cleanup, stopMeter]);

  const copyText = useCallback(async (text: string, kind: "all" | "last5" | `block-${string}`) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      window.setTimeout(() => setCopied((c) => (c === kind ? null : c)), 1500);
    } catch {
      // Fallback for non-secure contexts
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); setCopied(kind); window.setTimeout(() => setCopied(null), 1500); } catch { /* ignore */ }
      document.body.removeChild(ta);
    }
  }, []);

  // Reset the panel back to a fresh start: drop every previous block, the
  // current live transcript, cached download links, and any lingering error
  // — leaves the audio-source and model selections intact.
  const clearAll = useCallback(() => {
    setFinals([]);
    setPartial(null);
    setBlocks([]);
    setDownloads(null);
    setErrorMsg(null);
    setDevice("");
    setElapsed(0);
    setCopied(null);
  }, []);

  // Snapshot the current live text into a fresh block, then reset the live
  // area so subsequent captions accumulate from this point onward.
  const splitBlock = useCallback(() => {
    if (finals.length === 0) return;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setBlocks((prev) => [...prev, { id, segments: finals }]);
    setFinals([]);
    setPartial(null);
    setCopied("split");
    window.setTimeout(() => setCopied((c) => (c === "split" ? null : c)), 1200);
  }, [finals]);

  const allText = finals.map((s) => s.text).join(" ");
  const last5Text = finals.slice(-5).map((s) => s.text).join(" ");

  const downloadFile = useCallback(async (url: string, filename: string) => {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = filename;
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (e) {
      setErrorMsg(e instanceof Error ? `Download failed: ${e.message}` : "Download failed");
    }
  }, []);

  const recording = status === "recording";
  const loadingModel = status === "loading-model";
  const active = recording || loadingModel || status === "stopping";
  const busy = status === "connecting" || status === "stopping";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Live Transcribe</h1>
        <p className="text-zinc-400 mt-1">
          Stream microphone and/or desktop audio to whisper for live captions. Transcripts auto-save to <code className="text-zinc-300 bg-zinc-800/60 px-1.5 py-0.5 rounded">.mp/transcripts/</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: settings */}
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Audio Sources</CardTitle>
            <div className="space-y-2">
              <SourceToggle
                label="Microphone"
                description="Your input device"
                icon={<Mic className="w-4 h-4" />}
                enabled={useMic}
                onToggle={() => !recording && setUseMic((v) => !v)}
                disabled={recording || busy}
              />
              <SourceToggle
                label="Desktop Audio"
                description="System / tab audio (Chrome / Edge)"
                icon={<MonitorSpeaker className="w-4 h-4" />}
                enabled={useDesktop}
                onToggle={() => !recording && setUseDesktop((v) => !v)}
                disabled={recording || busy}
              />
            </div>
            {useMic && (
              <div className="mt-3 pt-3 border-t border-zinc-800 space-y-2">
                <label className="text-[10px] uppercase tracking-wide text-zinc-500">Input device</label>
                <select
                  value={micDeviceId}
                  disabled={recording || busy}
                  onChange={(e) => {
                    setMicDeviceId(e.target.value);
                    if (metering) startMeter(); // re-open meter on the newly chosen device
                  }}
                  className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
                >
                  <option value="">System default</option>
                  {inputDevices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </option>
                  ))}
                </select>

                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={toggleMeter}
                    disabled={recording || busy}
                    title="Open the mic and watch the level — confirm it picks up your voice before recording"
                  >
                    <Mic className="w-3.5 h-3.5 mr-1.5" />
                    {metering ? "Stop test" : "Test mic"}
                  </Button>
                  <div className="relative flex-1 h-2.5 bg-zinc-800 rounded-full overflow-hidden">
                    {/* good-range guide marker */}
                    <div className="absolute inset-y-0 left-[8%] right-[30%] bg-green-500/10" />
                    <div
                      className="h-full rounded-full transition-[width] duration-75"
                      style={{
                        width: `${Math.min(100, micPeak * 300)}%`,
                        background:
                          micPeak * 300 < 8 ? "#71717a" : micPeak * 300 > 85 ? "#f59e0b" : "#22c55e",
                      }}
                    />
                  </div>
                </div>
                {metering && (
                  <p className="text-[10px] text-zinc-500 leading-relaxed">
                    {micPeak * 300 < 8
                      ? "Barely any signal — speak up, pick another device above, or raise the level in Windows Sound settings."
                      : micPeak * 300 > 85
                      ? "Loud — the mic is working (watch for clipping)."
                      : "Good — the mic is picking up your voice. You can Start now."}
                  </p>
                )}
              </div>
            )}
            {useDesktop && (
              <p className="text-[10px] text-amber-400/80 mt-3 leading-relaxed">
                When prompted, choose a screen/tab and tick <strong>&ldquo;Share audio&rdquo;</strong>. Firefox/Safari may not capture system audio.
              </p>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Model</CardTitle>
            <select
              value={model}
              disabled={recording || busy}
              onChange={(e) => setModel(e.target.value)}
              className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed">
              On CPU, prefer <code className="text-zinc-300">tiny</code> or <code className="text-zinc-300">base</code> for realtime feel. Larger models are accurate but lag.
            </p>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Language</CardTitle>
            <select
              value={language}
              disabled={recording || busy}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{l.label}</option>
              ))}
            </select>
            <label className="flex items-center gap-2 mt-3 text-xs text-zinc-300 cursor-pointer">
              <input
                type="checkbox"
                checked={translate}
                disabled={recording || busy}
                onChange={(e) => setTranslate(e.target.checked)}
                className="accent-indigo-500"
              />
              <Languages className="w-3.5 h-3.5" />
              Translate to English
            </label>
          </Card>

          {!active && (
            <div className="flex gap-2">
              <Button className="flex-1" size="lg" onClick={start} disabled={busy}>
                {status === "connecting" ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Connecting...</>
                ) : (
                  <><Radio className="w-4 h-4 mr-2" /> Start Live Transcribe</>
                )}
              </Button>
              {(finals.length > 0 || blocks.length > 0 || downloads || errorMsg) && (
                <Button
                  size="lg"
                  variant="secondary"
                  onClick={clearAll}
                  disabled={busy}
                  title="Clear the transcript, all blocks, and download links"
                >
                  <Trash2 className="w-4 h-4 mr-2" /> Clear
                </Button>
              )}
            </div>
          )}
          {active && (
            <Button className="w-full" size="lg" variant="danger" onClick={stop} disabled={status === "stopping"}>
              {status === "stopping" ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving transcript...</>
              ) : loadingModel ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading model... (cancel)</>
              ) : (
                <><Square className="w-4 h-4 mr-2" /> Stop &amp; Save</>
              )}
            </Button>
          )}

          {errorMsg && (
            <Card className="border-red-500/30 bg-red-500/5">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 leading-relaxed">{errorMsg}</p>
              </div>
            </Card>
          )}
        </div>

        {/* Right column: live output */}
        <div className="lg:col-span-2 space-y-4">
          {blocks.map((b, bi) => {
            const first = b.segments[0];
            const last = b.segments[b.segments.length - 1];
            const range = first && last ? `${fmtClock(first.start)} – ${fmtClock(last.end)}` : "";
            const blockText = b.segments.map((s) => s.text).join(" ");
            const kind: `block-${string}` = `block-${b.id}`;
            return (
              <Card key={b.id}>
                <div className="flex items-center justify-between mb-2 gap-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    <Scissors className="w-3.5 h-3.5 text-indigo-400" />
                    <CardTitle className="text-sm">Block {bi + 1}</CardTitle>
                    <span className="text-[10px] font-mono text-zinc-500">{range}</span>
                    <span className="text-[10px] text-zinc-500">
                      · {b.segments.length} segment{b.segments.length === 1 ? "" : "s"}
                    </span>
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => copyText(blockText, kind)}
                    title="Copy this block without timestamps"
                  >
                    {copied === kind ? (
                      <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                    ) : (
                      <><Copy className="w-3.5 h-3.5 mr-1.5" /> Copy</>
                    )}
                  </Button>
                </div>
                <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-3 max-h-[220px] overflow-y-auto space-y-1">
                  {b.segments.map((s, i) => (
                    <p key={i} className="text-sm text-zinc-300 leading-relaxed">
                      <span className="text-[10px] font-mono text-zinc-600 mr-2">[{fmtClock(s.start)}]</span>
                      {s.text}
                    </p>
                  ))}
                </div>
              </Card>
            );
          })}
          <Card className="min-h-[420px] flex flex-col">
            <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <CardTitle className="text-sm">Live Transcript</CardTitle>
                {recording && (
                  <span className="flex items-center gap-1.5 text-xs text-red-400">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    REC
                  </span>
                )}
                {device && <Badge>{device.toUpperCase()}</Badge>}
                <span className="text-[10px] text-zinc-500">
                  {finals.length} segment{finals.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={splitBlock}
                  disabled={finals.length === 0 || !active}
                  title="Move the current live text into a fresh block above. New captions will accumulate in a clean area from now on."
                >
                  {copied === "split" ? (
                    <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Split</>
                  ) : (
                    <><Scissors className="w-3.5 h-3.5 mr-1.5" /> Split</>
                  )}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => copyText(last5Text, "last5")}
                  disabled={finals.length === 0}
                  title="Copy the last 5 segments without timestamps"
                >
                  {copied === "last5" ? (
                    <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                  ) : (
                    <><Copy className="w-3.5 h-3.5 mr-1.5" /> Last 5</>
                  )}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => copyText(allText, "all")}
                  disabled={finals.length === 0}
                  title="Copy the entire transcript without timestamps"
                >
                  {copied === "all" ? (
                    <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                  ) : (
                    <><Copy className="w-3.5 h-3.5 mr-1.5" /> Copy All</>
                  )}
                </Button>
                <span className="text-xs font-mono text-zinc-400 ml-1">{fmtClock(elapsed)}</span>
              </div>
            </div>

            <div
              ref={transcriptRef}
              className="flex-1 overflow-y-auto bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 space-y-2 min-h-[320px] max-h-[60vh]"
            >
              {finals.length === 0 && !partial && loadingModel && (
                <div className="text-center mt-12">
                  <Loader2 className="w-6 h-6 text-indigo-400 mx-auto mb-2 animate-spin" />
                  <p className="text-zinc-300 text-xs font-medium">Loading whisper model on the server...</p>
                  <p className="text-zinc-600 text-[10px] mt-1">First run downloads ~140 MB. Audio is buffering and will be transcribed once ready.</p>
                </div>
              )}
              {finals.length === 0 && !partial && !loadingModel && (
                <p className="text-zinc-600 text-xs text-center mt-12">
                  Captions will appear here once recording starts.
                </p>
              )}
              {finals.map((s, i) => (
                <p key={i} className="text-sm text-zinc-200 leading-relaxed">
                  <span className="text-[10px] font-mono text-zinc-600 mr-2">[{fmtClock(s.start)}]</span>
                  {s.text}
                </p>
              ))}
              {partial && (
                <p className="text-sm italic text-zinc-500 leading-relaxed">
                  <span className="text-[10px] font-mono text-zinc-700 mr-2">[{fmtClock(partial.start)}]</span>
                  {partial.text}
                </p>
              )}
            </div>
          </Card>

          {downloads && (
            <Card className="border-green-500/20 bg-green-500/5">
              <p className="text-green-400 text-sm font-medium mb-3">Transcript saved</p>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(downloads.txt, downloads.txt.split("/").pop() || "transcript.txt")}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .txt
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(downloads.srt, downloads.srt.split("/").pop() || "transcript.srt")}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .srt
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(downloads.json, downloads.json.split("/").pop() || "transcript.json")}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .json
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceToggle({
  label, description, icon, enabled, onToggle, disabled,
}: {
  label: string;
  description: string;
  icon: React.ReactNode;
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={`w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg border transition-all text-left ${
        enabled
          ? "bg-indigo-600/10 border-indigo-500/30 text-indigo-300"
          : "bg-zinc-800/40 border-zinc-700 text-zinc-300 hover:border-zinc-600"
      } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
    >
      <div className="flex items-center gap-2.5">
        <div className={enabled ? "text-indigo-400" : "text-zinc-500"}>{icon}</div>
        <div>
          <p className="text-xs font-medium">{label}</p>
          <p className="text-[10px] text-zinc-500">{description}</p>
        </div>
      </div>
      <div
        className={`w-9 h-5 rounded-full p-0.5 transition-colors ${
          enabled ? "bg-indigo-500" : "bg-zinc-700"
        }`}
      >
        <div
          className={`w-4 h-4 rounded-full bg-white transition-transform ${
            enabled ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </div>
    </button>
  );
}
