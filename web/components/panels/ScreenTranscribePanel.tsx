"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  MonitorPlay, ScanText, Square, Download, Loader2, AlertCircle,
  Copy, Check, Server,
} from "lucide-react";

type Line = { t: number; text: string; conf: number; bbox?: number[][] };
type FrameEvent = { type: "frame"; frame_skipped: boolean; reason?: string; phash_diff?: number; lines?: number };
type ServerMsg =
  | { type: "ready"; session: string; source: string; device: string }
  | { type: "status"; stage: "loading_model" | "model_ready" }
  | ({ type: "line" } & Line)
  | FrameEvent
  | { type: "complete"; transcripts: { txt: string; json: string }; stats: Record<string, number> }
  | { type: "error"; message: string };

type Monitor = { index: number; label: string; left: number; top: number; width: number; height: number };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

const FPS_OPTIONS = [
  { v: 0.5, label: "0.5 fps (low)" },
  { v: 1,   label: "1 fps (balanced)" },
  { v: 2,   label: "2 fps (responsive)" },
];

function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`
    : `${m}:${String(r).padStart(2, "0")}`;
}

export default function ScreenTranscribePanel() {
  const [source, setSource] = useState<"browser" | "local">("browser");
  const [fps, setFps] = useState(1);
  const [confidence, setConfidence] = useState(0.6);
  const [diff, setDiff] = useState(4);
  const [maxRepeats, setMaxRepeats] = useState(2);
  const [dropGarbage, setDropGarbage] = useState(true);
  const [monitors, setMonitors] = useState<Monitor[]>([]);
  const [monitor, setMonitor] = useState<number>(1);

  const [status, setStatus] = useState<"idle" | "connecting" | "loading-model" | "running" | "stopping" | "complete" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [device, setDevice] = useState<string>("");
  const [elapsed, setElapsed] = useState(0);
  const [lines, setLines] = useState<Line[]>([]);
  const [framesSeen, setFramesSeen] = useState(0);
  const [framesSkipped, setFramesSkipped] = useState(0);
  const [downloads, setDownloads] = useState<{ txt: string; json: string } | null>(null);
  const [copied, setCopied] = useState<"all" | "last5" | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewRef = useRef<HTMLCanvasElement | null>(null);
  const sendIntervalRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const startTsRef = useRef<number>(0);
  const feedRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
  }, [lines]);

  // Pull the server-side monitor list once, so the dropdown is ready when the
  // user toggles to "Local capture".
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/v1/screen/monitors`)
      .then((r) => r.json())
      .then((data: { monitors: Monitor[] }) => {
        if (cancelled) return;
        const mons = data.monitors || [];
        setMonitors(mons);
        // Default to first physical monitor (index 1) if present
        const first = mons.find((m) => m.index >= 1) || mons[0];
        if (first) setMonitor(first.index);
      })
      .catch(() => { /* server-side capture just won't be available */ });
    return () => { cancelled = true; };
  }, []);

  const cleanup = useCallback(() => {
    if (sendIntervalRef.current !== null) {
      window.clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      try { videoRef.current.srcObject = null; } catch { /* ignore */ }
    }
  }, []);

  const start = useCallback(async () => {
    setErrorMsg(null);
    setLines([]);
    setDownloads(null);
    setFramesSeen(0);
    setFramesSkipped(0);
    setElapsed(0);
    setStatus("connecting");

    try {
      const session = `screen-${Date.now()}`;
      const qp = new URLSearchParams({
        session,
        source,
        confidence: String(confidence),
        diff: String(diff),
        max_repeats: String(maxRepeats),
        drop_garbage: dropGarbage ? "1" : "0",
      });
      if (source === "local") {
        qp.set("monitor", String(monitor));
        qp.set("fps", String(fps));
      }
      const ws = new WebSocket(`${WS_BASE}/api/v1/screen/live?${qp.toString()}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        let msg: ServerMsg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        switch (msg.type) {
          case "ready":
            setDevice(msg.device);
            break;
          case "status":
            if (msg.stage === "loading_model") setStatus("loading-model");
            else if (msg.stage === "model_ready") {
              setStatus("running");
              startTsRef.current = Date.now();
              tickRef.current = window.setInterval(() => {
                setElapsed((Date.now() - startTsRef.current) / 1000);
              }, 250);
            }
            break;
          case "line":
            setLines((prev) => [...prev, { t: msg.t, text: msg.text, conf: msg.conf, bbox: msg.bbox }]);
            break;
          case "frame":
            setFramesSeen((n) => n + 1);
            if (msg.frame_skipped) setFramesSkipped((n) => n + 1);
            break;
          case "complete":
            setDownloads(msg.transcripts);
            setStatus("complete");
            cleanup();
            break;
          case "error":
            setErrorMsg(msg.message);
            setStatus("error");
            cleanup();
            break;
        }
      };

      ws.onerror = () => {
        setErrorMsg("WebSocket error - is the API running on :8000?");
        setStatus("error");
        cleanup();
      };

      ws.onclose = () => {
        if (status === "running" || status === "loading-model" || status === "connecting") {
          setStatus((s) => (s === "complete" ? s : "complete"));
        }
        cleanup();
      };

      if (source === "browser") {
        // Capture the screen / window / tab the user picks
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
        streamRef.current = stream;

        const video = document.createElement("video");
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
        videoRef.current = video;

        const canvas = canvasRef.current || document.createElement("canvas");
        canvasRef.current = canvas;

        // Sample frames at the chosen fps, encode JPEG, ship over WS
        const intervalMs = Math.max(100, Math.round(1000 / fps));
        const sample = async () => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const v = videoRef.current;
          if (!v || v.readyState < 2 || v.videoWidth === 0) return;
          // Cap encode resolution so we don't push 4K JPEGs over the socket.
          // OCR accuracy is fine at 1280-wide; downscale anything larger.
          const targetW = Math.min(1600, v.videoWidth);
          const scale = targetW / v.videoWidth;
          canvas.width = targetW;
          canvas.height = Math.round(v.videoHeight * scale);
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height);

          // Mirror to the small live preview
          const prev = previewRef.current;
          if (prev) {
            const pCtx = prev.getContext("2d");
            if (pCtx) {
              prev.width = 240;
              prev.height = Math.round((canvas.height / canvas.width) * 240);
              pCtx.drawImage(canvas, 0, 0, prev.width, prev.height);
            }
          }

          canvas.toBlob(async (blob) => {
            if (!blob) return;
            const buf = await blob.arrayBuffer();
            try { ws.send(buf); } catch { /* WS may have closed */ }
          }, "image/jpeg", 0.72);
        };
        sendIntervalRef.current = window.setInterval(sample, intervalMs);

        // Browser sometimes stops the share when the user clicks "Stop sharing"
        // in the browser chrome. Catch that and clean up.
        stream.getVideoTracks()[0]?.addEventListener("ended", () => {
          if (ws.readyState === WebSocket.OPEN) {
            try { ws.send("stop"); } catch { /* ignore */ }
          }
        });
      }
      // For source=local, the server samples on its own; we just receive lines.
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to start screen transcribe";
      setErrorMsg(msg);
      setStatus("error");
      cleanup();
    }
  }, [source, fps, confidence, diff, maxRepeats, dropGarbage, monitor, status, cleanup]);

  const stop = useCallback(() => {
    if (!["running", "loading-model"].includes(status)) return;
    setStatus("stopping");
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send("stop"); } catch { /* ignore */ }
    }
    // Stop pushing more frames immediately
    if (sendIntervalRef.current !== null) {
      window.clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
  }, [status]);

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
      // ignore fallback failure
    }
  }, []);

  const allText = useMemo(() => lines.map((l) => l.text).join("\n"), [lines]);
  const last5Text = useMemo(() => lines.slice(-5).map((l) => l.text).join("\n"), [lines]);

  const downloadFile = useCallback(async (url: string, filename: string) => {
    try {
      const res = await fetch(API_BASE + url, { cache: "no-store" });
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

  const running = status === "running";
  const loadingModel = status === "loading-model";
  const active = running || loadingModel || status === "stopping";
  const busy = status === "connecting" || status === "stopping";
  const skipRate = framesSeen > 0 ? Math.round((framesSkipped / framesSeen) * 100) : 0;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Screen Transcribe</h1>
        <p className="text-zinc-400 mt-1">
          Read text shown on your screen live. Frames are OCR&apos;d at the chosen rate;
          a perceptual-hash diff skips frames that haven&apos;t changed. Results auto-save to{" "}
          <code className="text-zinc-300 bg-zinc-800/60 px-1.5 py-0.5 rounded">.mp/screen_transcripts/</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: settings */}
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Capture Source</CardTitle>
            <div className="space-y-2">
              <SourceToggle
                label="Browser screen-share"
                description="Pick a window/tab/screen. Works over network."
                icon={<MonitorPlay className="w-4 h-4" />}
                enabled={source === "browser"}
                onToggle={() => !active && setSource("browser")}
                disabled={active}
              />
              <SourceToggle
                label="Local capture (server)"
                description="Backend grabs frames directly. Local-only."
                icon={<Server className="w-4 h-4" />}
                enabled={source === "local"}
                onToggle={() => !active && monitors.length > 0 && setSource("local")}
                disabled={active || monitors.length === 0}
              />
            </div>
            {source === "local" && monitors.length > 0 && (
              <div className="mt-3">
                <label className="text-[10px] text-zinc-500 block mb-1">Monitor</label>
                <select
                  value={monitor}
                  disabled={active}
                  onChange={(e) => setMonitor(Number(e.target.value))}
                  className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
                >
                  {monitors.map((m) => (
                    <option key={m.index} value={m.index}>
                      {m.label} - {m.width}x{m.height}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Sampling rate</CardTitle>
            <select
              value={fps}
              disabled={active}
              onChange={(e) => setFps(Number(e.target.value))}
              className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
            >
              {FPS_OPTIONS.map((o) => (
                <option key={o.v} value={o.v}>{o.label}</option>
              ))}
            </select>
            <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed">
              Higher rates catch fast changes but cost CPU. The frame-diff skips redundant frames
              automatically.
            </p>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Smart filters</CardTitle>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[11px] text-zinc-400">Confidence floor</label>
                  <span className="text-[11px] font-mono text-zinc-300">{confidence.toFixed(2)}</span>
                </div>
                <input
                  type="range" min={0.1} max={0.95} step={0.05}
                  value={confidence}
                  disabled={active}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <p className="text-[10px] text-zinc-500 mt-1">Drop boxes below this confidence.</p>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[11px] text-zinc-400">Frame-diff threshold</label>
                  <span className="text-[11px] font-mono text-zinc-300">{diff}</span>
                </div>
                <input
                  type="range" min={0} max={16} step={1}
                  value={diff}
                  disabled={active}
                  onChange={(e) => setDiff(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <p className="text-[10px] text-zinc-500 mt-1">Skip OCR when perceptual-hash diff is below this (0 = always OCR).</p>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[11px] text-zinc-400">Suppress repeats after</label>
                  <span className="text-[11px] font-mono text-zinc-300">{maxRepeats}x</span>
                </div>
                <input
                  type="range" min={1} max={10} step={1}
                  value={maxRepeats}
                  disabled={active}
                  onChange={(e) => setMaxRepeats(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <p className="text-[10px] text-zinc-500 mt-1">Stop emitting a line after it appears this many times (kills browser/Office chrome).</p>
              </div>
              <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={dropGarbage}
                  disabled={active}
                  onChange={(e) => setDropGarbage(e.target.checked)}
                  className="accent-indigo-500"
                />
                Drop URLs, icon misreads, and symbol-only lines
              </label>
            </div>
          </Card>

          {!active && (
            <Button className="w-full" size="lg" onClick={start} disabled={busy}>
              {status === "connecting" ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Connecting...</>
              ) : (
                <><ScanText className="w-4 h-4 mr-2" /> Start Screen Transcribe</>
              )}
            </Button>
          )}
          {active && (
            <Button className="w-full" size="lg" variant="danger" onClick={stop} disabled={status === "stopping"}>
              {status === "stopping" ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving...</>
              ) : loadingModel ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading OCR model... (cancel)</>
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

          {source === "browser" && (
            <Card>
              <CardTitle className="text-sm mb-2">Live preview</CardTitle>
              <canvas
                ref={previewRef}
                className="w-full border border-zinc-800 rounded bg-zinc-950"
              />
              {!active && (
                <p className="text-[10px] text-zinc-500 mt-2">Preview shows once you start sharing.</p>
              )}
            </Card>
          )}
        </div>

        {/* Right column: live output */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="min-h-[420px] flex flex-col">
            <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <CardTitle className="text-sm">Detected text</CardTitle>
                {running && (
                  <span className="flex items-center gap-1.5 text-xs text-red-400">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    LIVE
                  </span>
                )}
                {device && <Badge>{device.toUpperCase()}</Badge>}
                <span className="text-[10px] text-zinc-500">
                  {lines.length} line{lines.length === 1 ? "" : "s"}
                </span>
                {framesSeen > 0 && (
                  <span className="text-[10px] text-zinc-500" title="Frames OCR'd / frames skipped via phash diff">
                    {framesSeen - framesSkipped} OCR&apos;d / {framesSkipped} skipped ({skipRate}%)
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary" size="sm"
                  onClick={() => copyText(last5Text, "last5")}
                  disabled={lines.length === 0}
                  title="Copy the last 5 detected lines"
                >
                  {copied === "last5" ? (
                    <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                  ) : (
                    <><Copy className="w-3.5 h-3.5 mr-1.5" /> Last 5</>
                  )}
                </Button>
                <Button
                  variant="secondary" size="sm"
                  onClick={() => copyText(allText, "all")}
                  disabled={lines.length === 0}
                  title="Copy everything"
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
              ref={feedRef}
              className="flex-1 overflow-y-auto bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 space-y-1 min-h-[320px] max-h-[60vh]"
            >
              {lines.length === 0 && loadingModel && (
                <div className="text-center mt-12">
                  <Loader2 className="w-6 h-6 text-indigo-400 mx-auto mb-2 animate-spin" />
                  <p className="text-zinc-300 text-xs font-medium">Loading RapidOCR models on the server...</p>
                  <p className="text-zinc-600 text-[10px] mt-1">First run downloads ~80 MB of ONNX weights.</p>
                </div>
              )}
              {lines.length === 0 && !loadingModel && (
                <p className="text-zinc-600 text-xs text-center mt-12">
                  Detected text will appear here once capture starts.
                </p>
              )}
              {lines.map((l, i) => (
                <p key={i} className="text-sm text-zinc-200 leading-snug">
                  <span className="text-[10px] font-mono text-zinc-600 mr-2">[{fmtClock(l.t)}]</span>
                  <span className="text-[10px] font-mono text-zinc-700 mr-2">{Math.round(l.conf * 100)}%</span>
                  {l.text}
                </p>
              ))}
            </div>
          </Card>

          {downloads && (
            <Card className="border-green-500/20 bg-green-500/5">
              <p className="text-green-400 text-sm font-medium mb-3">Transcript saved</p>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="secondary" size="sm"
                  onClick={() => downloadFile(downloads.txt, downloads.txt.split("/").pop() || "screen.txt")}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .txt
                </Button>
                <Button
                  variant="secondary" size="sm"
                  onClick={() => downloadFile(downloads.json, downloads.json.split("/").pop() || "screen.json")}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .json (boxes + conf)
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
      <div className={`w-9 h-5 rounded-full p-0.5 transition-colors ${enabled ? "bg-indigo-500" : "bg-zinc-700"}`}>
        <div className={`w-4 h-4 rounded-full bg-white transition-transform ${enabled ? "translate-x-4" : "translate-x-0"}`} />
      </div>
    </button>
  );
}
