"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  FileVideo, Upload, Loader2, AlertCircle, Languages, Download,
  Copy, Check, Mic, ScanText, X,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"];
const LANGUAGES: { code: string; label: string }[] = [
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

type JobStatus = "pending" | "running" | "completed" | "failed";

interface JobResponse {
  job_id: string;
  status: JobStatus;
  progress?: number;
  message?: string;
  error?: string | null;
  result?: {
    session_id?: string;
    duration?: number | null;
    audio_transcripts?: { txt: string; srt: string; json: string } | null;
    screen_transcripts?: { txt: string; json: string } | null;
  } | null;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtDuration(secs?: number | null): string {
  if (!secs || secs <= 0) return "";
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`
    : `${m}:${String(r).padStart(2, "0")}`;
}

export default function VideoTranscribePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [doAudio, setDoAudio] = useState(true);
  const [doScreen, setDoScreen] = useState(true);
  const [model, setModel] = useState("base");
  const [language, setLanguage] = useState("");
  const [translate, setTranslate] = useState(false);
  const [fps, setFps] = useState(1);
  const [confidence, setConfidence] = useState(0.6);

  const [status, setStatus] = useState<"idle" | "uploading" | "running" | "complete" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [jobMessage, setJobMessage] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<JobResponse["result"]>(null);
  const [audioText, setAudioText] = useState<string>("");
  const [screenText, setScreenText] = useState<string>("");
  const [copied, setCopied] = useState<"audio" | "screen" | null>(null);

  const pollRef = useRef<number | null>(null);
  const dropRef = useRef<HTMLLabelElement | null>(null);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => clearPoll(), [clearPoll]);

  const onPickFile = useCallback((f: File | null) => {
    setFile(f);
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setAudioText("");
    setScreenText("");
    setErrorMsg(null);
    setJobMessage("");
  }, []);

  const onDrop = useCallback((ev: React.DragEvent<HTMLLabelElement>) => {
    ev.preventDefault();
    if (status === "uploading" || status === "running") return;
    const f = ev.dataTransfer.files?.[0];
    if (f && f.type.startsWith("video/")) onPickFile(f);
    else if (f) setErrorMsg("Please drop a video file (mp4, mov, mkv, webm, etc).");
  }, [onPickFile, status]);

  const fetchText = useCallback(async (url: string): Promise<string> => {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) return "";
      return await res.text();
    } catch { return ""; }
  }, []);

  const startTranscribe = useCallback(async () => {
    if (!file) {
      setErrorMsg("Pick a video file first.");
      return;
    }
    if (!doAudio && !doScreen) {
      setErrorMsg("Enable Audio, Screen OCR, or both.");
      return;
    }
    setErrorMsg(null);
    setProgress(0);
    setResult(null);
    setAudioText("");
    setScreenText("");
    setStatus("uploading");
    setJobMessage("Uploading video...");

    try {
      const form = new FormData();
      form.append("file", file);
      const params = new URLSearchParams({
        audio_model: model,
        language,
        translate: translate ? "true" : "false",
        do_audio: doAudio ? "true" : "false",
        do_screen: doScreen ? "true" : "false",
        fps: String(fps),
        confidence: String(confidence),
        drop_garbage: "true",
      });
      const res = await fetch(`${API_BASE}/api/v1/video/transcribe?${params.toString()}`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Upload failed: HTTP ${res.status} ${txt.slice(0, 200)}`);
      }
      const initial: JobResponse = await res.json();
      const jobId = initial.job_id;
      setStatus("running");

      clearPoll();
      pollRef.current = window.setInterval(async () => {
        try {
          const r = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`, { cache: "no-store" });
          if (!r.ok) return;
          const data: JobResponse = await r.json();
          if (typeof data.progress === "number") setProgress(data.progress);
          if (data.message) setJobMessage(data.message);
          if (data.status === "completed") {
            clearPoll();
            setProgress(100);
            setResult(data.result || null);
            setStatus("complete");
            const r2 = data.result || {};
            if (r2.audio_transcripts?.txt) {
              fetchText(`${API_BASE}${r2.audio_transcripts.txt}`).then(setAudioText);
            }
            if (r2.screen_transcripts?.txt) {
              fetchText(`${API_BASE}${r2.screen_transcripts.txt}`).then(setScreenText);
            }
          } else if (data.status === "failed") {
            clearPoll();
            setStatus("error");
            setErrorMsg(data.error || data.message || "Transcription failed.");
          }
        } catch {
          // Transient poll error — keep retrying.
        }
      }, 1500);
    } catch (e) {
      setStatus("error");
      setErrorMsg(e instanceof Error ? e.message : "Failed to start.");
    }
  }, [file, doAudio, doScreen, model, language, translate, fps, confidence, clearPoll, fetchText]);

  const cancel = useCallback(() => {
    clearPoll();
    setStatus("idle");
    setProgress(0);
    setJobMessage("");
  }, [clearPoll]);

  const copyText = useCallback(async (text: string, kind: "audio" | "screen") => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      window.setTimeout(() => setCopied((c) => (c === kind ? null : c)), 1500);
    } catch {
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

  const busy = status === "uploading" || status === "running";
  const session = result?.session_id || "";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Transcribe</h1>
        <p className="text-zinc-400 mt-1">
          Upload a recorded video. We&apos;ll transcribe the audio with whisper and OCR the screen separately, then give you both transcripts.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: settings */}
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Input Video</CardTitle>
            <label
              ref={dropRef}
              htmlFor="video-file-input"
              onDragOver={(e) => { e.preventDefault(); }}
              onDrop={onDrop}
              className={`block border-2 border-dashed rounded-lg px-4 py-6 text-center cursor-pointer transition-colors ${
                busy
                  ? "border-zinc-800 bg-zinc-900/40 cursor-not-allowed"
                  : "border-zinc-700 hover:border-indigo-500/50 bg-zinc-900/40"
              }`}
            >
              <input
                id="video-file-input"
                type="file"
                accept="video/*"
                disabled={busy}
                className="hidden"
                onChange={(e) => onPickFile(e.target.files?.[0] || null)}
              />
              {file ? (
                <div className="text-left">
                  <div className="flex items-center gap-2">
                    <FileVideo className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                    <span className="text-xs text-zinc-200 truncate" title={file.name}>{file.name}</span>
                    {!busy && (
                      <button
                        type="button"
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onPickFile(null); }}
                        className="ml-auto text-zinc-500 hover:text-zinc-300"
                        aria-label="Clear file"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                  <p className="text-[10px] text-zinc-500 mt-1">{fmtBytes(file.size)}</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-zinc-400">
                  <Upload className="w-5 h-5" />
                  <span className="text-xs">Click or drop a video file</span>
                  <span className="text-[10px] text-zinc-600">mp4, mov, mkv, webm, ...</span>
                </div>
              )}
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Outputs</CardTitle>
            <div className="space-y-2">
              <ToggleRow
                label="Audio transcript"
                description="Whisper over the audio track"
                icon={<Mic className="w-4 h-4" />}
                enabled={doAudio}
                onToggle={() => !busy && setDoAudio((v) => !v)}
                disabled={busy}
              />
              <ToggleRow
                label="Screen OCR"
                description="Read text shown on screen"
                icon={<ScanText className="w-4 h-4" />}
                enabled={doScreen}
                onToggle={() => !busy && setDoScreen((v) => !v)}
                disabled={busy}
              />
            </div>
          </Card>

          {doAudio && (
            <Card>
              <CardTitle className="text-sm mb-3">Audio Settings</CardTitle>
              <label className="block text-[10px] text-zinc-500 mb-1">Whisper model</label>
              <select
                value={model}
                disabled={busy}
                onChange={(e) => setModel(e.target.value)}
                className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 disabled:opacity-50"
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>

              <label className="block text-[10px] text-zinc-500 mb-1 mt-3">Language</label>
              <select
                value={language}
                disabled={busy}
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
                  disabled={busy}
                  onChange={(e) => setTranslate(e.target.checked)}
                  className="accent-indigo-500"
                />
                <Languages className="w-3.5 h-3.5" />
                Translate to English
              </label>
            </Card>
          )}

          {doScreen && (
            <Card>
              <CardTitle className="text-sm mb-3">Screen OCR Settings</CardTitle>
              <label className="block text-[10px] text-zinc-500 mb-1">Sample rate: <span className="text-zinc-300">{fps} fps</span></label>
              <input
                type="range"
                min={0.2}
                max={4}
                step={0.2}
                value={fps}
                disabled={busy}
                onChange={(e) => setFps(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
                Higher = more thorough but slower. 1 fps is fine for slides; bump to 2–3 for fast-changing screens.
              </p>

              <label className="block text-[10px] text-zinc-500 mb-1 mt-3">Confidence floor: <span className="text-zinc-300">{confidence.toFixed(2)}</span></label>
              <input
                type="range"
                min={0.3}
                max={0.95}
                step={0.05}
                value={confidence}
                disabled={busy}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
                Drop low-confidence OCR boxes. 0.6 is the live default.
              </p>
            </Card>
          )}

          {status !== "complete" && (
            <Button className="w-full" size="lg" onClick={startTranscribe} disabled={busy || !file}>
              {busy ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> {status === "uploading" ? "Uploading..." : "Transcribing..."}</>
              ) : (
                <><FileVideo className="w-4 h-4 mr-2" /> Transcribe Video</>
              )}
            </Button>
          )}
          {status === "complete" && (
            <Button className="w-full" size="lg" variant="secondary" onClick={() => { onPickFile(null); }}>
              Transcribe another video
            </Button>
          )}
          {busy && (
            <Button className="w-full" size="sm" variant="ghost" onClick={cancel}>
              Stop watching (job keeps running)
            </Button>
          )}

          {errorMsg && (
            <Card className="border-red-500/30 bg-red-500/5">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 leading-relaxed break-words">{errorMsg}</p>
              </div>
            </Card>
          )}
        </div>

        {/* Right column: progress + results */}
        <div className="lg:col-span-2 space-y-4">
          {(busy || status === "complete") && (
            <Card>
              <div className="flex items-center justify-between mb-2 gap-2">
                <CardTitle className="text-sm">Progress</CardTitle>
                <div className="flex items-center gap-2 text-[11px] text-zinc-400">
                  {session && <Badge>{session}</Badge>}
                  {result?.duration ? <span>video {fmtDuration(result.duration)}</span> : null}
                  <span>{progress}%</span>
                </div>
              </div>
              <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              {jobMessage && (
                <p className="text-[11px] text-zinc-400 mt-2 leading-relaxed">{jobMessage}</p>
              )}
            </Card>
          )}

          {result?.audio_transcripts && (
            <Card>
              <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <Mic className="w-4 h-4 text-indigo-400" />
                  <CardTitle className="text-sm">Audio Transcript</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => copyText(audioText, "audio")}
                    disabled={!audioText}
                  >
                    {copied === "audio" ? (
                      <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                    ) : (
                      <><Copy className="w-3.5 h-3.5 mr-1.5" /> Copy</>
                    )}
                  </Button>
                </div>
              </div>
              <div className="bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 max-h-[40vh] overflow-y-auto whitespace-pre-wrap text-sm text-zinc-200 leading-relaxed">
                {audioText || <span className="text-zinc-600">Loading transcript...</span>}
              </div>
              <div className="grid grid-cols-3 gap-2 mt-3">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(`${API_BASE}${result.audio_transcripts!.txt}`, `${session}.txt`)}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .txt
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(`${API_BASE}${result.audio_transcripts!.srt}`, `${session}.srt`)}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .srt
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(`${API_BASE}${result.audio_transcripts!.json}`, `${session}.json`)}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .json
                </Button>
              </div>
            </Card>
          )}

          {result?.screen_transcripts && (
            <Card>
              <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <ScanText className="w-4 h-4 text-emerald-400" />
                  <CardTitle className="text-sm">Screen OCR Transcript</CardTitle>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => copyText(screenText, "screen")}
                    disabled={!screenText}
                  >
                    {copied === "screen" ? (
                      <><Check className="w-3.5 h-3.5 mr-1.5 text-green-400" /> Copied</>
                    ) : (
                      <><Copy className="w-3.5 h-3.5 mr-1.5" /> Copy</>
                    )}
                  </Button>
                </div>
              </div>
              <div className="bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 max-h-[40vh] overflow-y-auto whitespace-pre-wrap text-sm text-zinc-200 leading-relaxed font-mono">
                {screenText || <span className="text-zinc-600 font-sans">Loading transcript...</span>}
              </div>
              <div className="grid grid-cols-2 gap-2 mt-3">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(`${API_BASE}${result.screen_transcripts!.txt}`, `${session}-screen.txt`)}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .txt
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => downloadFile(`${API_BASE}${result.screen_transcripts!.json}`, `${session}-screen.json`)}
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> .json
                </Button>
              </div>
            </Card>
          )}

          {!busy && status !== "complete" && (
            <Card className="border-dashed border-zinc-800 bg-zinc-900/30">
              <p className="text-xs text-zinc-500 text-center py-8 leading-relaxed">
                Pick a video, choose what to transcribe, and hit <span className="text-zinc-300">Transcribe Video</span>.<br />
                The audio leg uses whisper; the screen leg samples frames and runs OCR.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ToggleRow({
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
