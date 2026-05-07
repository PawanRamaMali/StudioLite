"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Mic, MonitorSpeaker, Radio, Square, Download, Loader2, AlertCircle, Languages,
  Copy, Check,
} from "lucide-react";

type Segment = { start: number; end: number; text: string };
type ServerMsg =
  | { type: "ready"; session: string; device: string }
  | { type: "status"; stage: "loading_model" | "model_ready"; model?: string }
  | ({ type: "final" | "partial" } & Segment)
  | { type: "complete"; transcripts: { txt: string; srt: string; json: string } }
  | { type: "error"; message: string };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");
const SAMPLE_RATE = 16000;

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
  const [downloads, setDownloads] = useState<{ txt: string; srt: string; json: string } | null>(null);
  const [copied, setCopied] = useState<"all" | "last5" | null>(null);

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

  // Auto-scroll transcript view
  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [finals, partial]);

  const cleanup = useCallback(() => {
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
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
    setDownloads(null);
    setStatus("connecting");

    try {
      // 1. Capture streams
      if (useMic) {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
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

      const workletNode = new AudioWorkletNode(ctx, "pcm-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
      });
      workletNodeRef.current = workletNode;

      // 3. Open WebSocket
      const session = `live-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const params = new URLSearchParams({ session, model });
      if (language) params.set("language", language);
      if (translate) params.set("translate", "1");
      const ws = new WebSocket(`${WS_BASE}/api/v1/transcribe/live?${params.toString()}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        const to = window.setTimeout(() => reject(new Error("WebSocket connect timeout")), 8000);
        ws.onopen = () => { window.clearTimeout(to); resolve(); };
        ws.onerror = () => { window.clearTimeout(to); reject(new Error("WebSocket connection failed")); };
      });

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
            }
          } else if (msg.type === "final") {
            setFinals((prev) => [...prev, { start: msg.start, end: msg.end, text: msg.text }]);
            setPartial(null);
          } else if (msg.type === "partial") {
            setPartial({ start: msg.start, end: msg.end, text: msg.text });
          } else if (msg.type === "complete") {
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
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (status === "recording" || status === "stopping" || status === "connecting") {
          setStatus((s) => (s === "complete" ? "complete" : s === "error" ? "error" : "complete"));
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
      const msg = e instanceof Error ? e.message : "Failed to start";
      setErrorMsg(msg);
      setStatus("error");
      cleanup();
    }
  }, [useMic, useDesktop, model, language, translate, status, cleanup]);

  const stop = useCallback(() => {
    if (status !== "recording") return;
    setStatus("stopping");
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send("stop"); } catch { /* ignore */ }
    }
    // Audio capture stops once cleanup runs (on ws close). Trim audio early so server stops getting samples.
    try { workletNodeRef.current?.disconnect(); } catch { /* ignore */ }
  }, [status]);

  // Cleanup on unmount
  useEffect(() => () => {
    try { wsRef.current?.close(); } catch { /* ignore */ }
    cleanup();
  }, [cleanup]);

  const copyText = useCallback(async (text: string, kind: "all" | "last5") => {
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
            <Button className="w-full" size="lg" onClick={start} disabled={busy}>
              {status === "connecting" ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Connecting...</>
              ) : (
                <><Radio className="w-4 h-4 mr-2" /> Start Live Transcribe</>
              )}
            </Button>
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
