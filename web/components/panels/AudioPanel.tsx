"use client";

import { useEffect, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Music, Mic, Headphones, Volume2, Loader2, Download, Upload, RotateCcw,
} from "lucide-react";
import {
  generateTTS, generateSFX, downloadBlob,
  isolateVoice, normalizeAudio, getAudioVoices,
  type PersonaInfo, type VoiceInfo, type Job,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SFX_CATEGORIES: Record<string, string[]> = {
  Nature: ["rain", "thunder", "wind", "ocean_waves"],
  Impact: ["explosion", "glass_break"],
  Ambient: ["fire_crackling", "bird_chirp", "clock_tick"],
  Human: ["heartbeat", "footsteps", "typing"],
  Motion: ["whoosh", "car_engine", "door_close"],
};

const TTS_SAMPLES = [
  "A brave dog embarks on an epic adventure through the mountains.",
  "In a world where technology rules, one human dares to dream.",
  "The ocean whispered secrets that only the bravest sailors could hear.",
  "Welcome to StudioLite, your personal AI video creation studio.",
];

interface SfxResultMeta {
  engine: string; method: string; sfx_type: string; details: string;
  duration: number; channels: number; sample_rate: number;
}
interface TTSResultMeta {
  voice: string; persona: string | null; persona_label: string | null;
  speed: number; pitch: number; volume: number; output_format: string;
}
interface IsolateResult {
  vocals_url: string; background_url: string; method: string;
}
interface NormalizeResult {
  audio_url: string; target_db: number;
}

async function pollJob(jobId: string, onMsg?: (m: string) => void): Promise<{ status: "completed" | "failed"; result?: Record<string, unknown>; error?: string }> {
  while (true) {
    await new Promise((r) => setTimeout(r, 700));
    const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
    const j = await res.json();
    if (onMsg && j.message) onMsg(j.message);
    if (j.status === "completed") return { status: "completed", result: j.result };
    if (j.status === "failed") return { status: "failed", error: j.error };
  }
}

export default function AudioPanel() {
  const [mode, setMode] = useState<"sfx" | "tts" | "isolate" | "normalize">("sfx");

  // SFX state
  const [sfxMode, setSfxMode] = useState<"preset" | "prompt">("preset");
  const [selectedSfx, setSelectedSfx] = useState("rain");
  const [sfxPrompt, setSfxPrompt] = useState("");
  const [sfxDuration, setSfxDuration] = useState(2);
  const [sfxResult, setSfxResult] = useState<{ url: string; meta: SfxResultMeta } | null>(null);

  // TTS state
  const [ttsText, setTtsText] = useState(TTS_SAMPLES[0]);
  const [personas, setPersonas] = useState<PersonaInfo[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>("audiobook");
  const [voiceOverride, setVoiceOverride] = useState<string>("");
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(0);
  const [volume, setVolume] = useState(1.0);
  const [outputFormat, setOutputFormat] = useState<"wav" | "mp3">("wav");
  const [ttsResult, setTtsResult] = useState<{ url: string; meta: TTSResultMeta } | null>(null);

  // Isolate / Normalize state
  const [isoFile, setIsoFile] = useState<File | null>(null);
  const [isoResult, setIsoResult] = useState<IsolateResult | null>(null);
  const [normFile, setNormFile] = useState<File | null>(null);
  const [normTargetDb, setNormTargetDb] = useState(-3);
  const [normResult, setNormResult] = useState<NormalizeResult | null>(null);

  // Shared
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAudioVoices()
      .then((data) => { setPersonas(data.personas); setVoices(data.voices); })
      .catch(() => { /* keep defaults; UI still works without the catalog */ });
  }, []);

  const activePersona = personas.find((p) => p.id === selectedPersona);

  // Apply persona's defaults to the sliders
  const applyPersonaDefaults = () => {
    if (!activePersona) return;
    setVoiceOverride("");
    setSpeed(activePersona.speed);
    setPitch(activePersona.pitch);
    setVolume(activePersona.volume);
  };

  const resetSliders = () => {
    setSpeed(1.0); setPitch(0); setVolume(1.0);
  };

  const clearResults = () => {
    setSfxResult(null); setTtsResult(null); setIsoResult(null); setNormResult(null); setError(null);
  };

  // ---- SFX ------------------------------------------------------------------
  const handleGenerateSFX = async () => {
    if (sfxMode === "prompt" && !sfxPrompt.trim()) {
      setError("Enter a prompt or switch to Preset mode."); return;
    }
    const label = sfxMode === "prompt" ? sfxPrompt.trim() : selectedSfx.replace(/_/g, " ");
    setLoading(true); setLoadingMsg(`Generating ${label}...`);
    setError(null); setSfxResult(null);
    try {
      const job = await generateSFX(
        sfxMode === "prompt"
          ? { sfx_type: "", prompt: sfxPrompt.trim(), duration: sfxDuration }
          : { sfx_type: selectedSfx, duration: sfxDuration }
      );
      const out = await pollJob(job.job_id, setLoadingMsg);
      if (out.status === "completed") {
        const r = (out.result || {}) as Record<string, unknown>;
        setSfxResult({
          url: `${API_BASE}/api/v1/jobs/${job.job_id}/download?t=${Date.now()}`,
          meta: {
            engine: String(r.engine || "unknown"),
            method: String(r.method || "Procedural synthesis"),
            sfx_type: String(r.sfx_type || selectedSfx),
            details: String(r.details || ""),
            duration: Number(r.duration ?? sfxDuration),
            channels: Number(r.channels ?? 1),
            sample_rate: Number(r.sample_rate ?? 22050),
          },
        });
      } else {
        setError(out.error || "SFX generation failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start SFX job");
    } finally {
      setLoading(false); setLoadingMsg("");
    }
  };

  // ---- TTS ------------------------------------------------------------------
  const handleGenerateTTS = async () => {
    if (!ttsText.trim()) return;
    setLoading(true); setLoadingMsg("Synthesizing speech...");
    setError(null); setTtsResult(null);
    try {
      const job = await generateTTS({
        text: ttsText.trim(),
        engine: "piper",
        voice: voiceOverride || (activePersona?.voice ?? "Amy"),
        persona: selectedPersona,
        speed, pitch, volume,
        output_format: outputFormat,
      });
      const out = await pollJob(job.job_id, setLoadingMsg);
      if (out.status === "completed") {
        const r = (out.result || {}) as Record<string, unknown>;
        setTtsResult({
          url: `${API_BASE}/api/v1/jobs/${job.job_id}/download?t=${Date.now()}`,
          meta: {
            voice: String(r.voice || "Amy"),
            persona: r.persona ? String(r.persona) : null,
            persona_label: r.persona_label ? String(r.persona_label) : null,
            speed: Number(r.speed ?? 1.0),
            pitch: Number(r.pitch ?? 0),
            volume: Number(r.volume ?? 1.0),
            output_format: String(r.output_format || "wav"),
          },
        });
      } else {
        setError(out.error || "TTS failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
    } finally {
      setLoading(false); setLoadingMsg("");
    }
  };

  // ---- Isolate --------------------------------------------------------------
  const handleIsolate = async () => {
    if (!isoFile) { setError("Pick an audio/video file first."); return; }
    setLoading(true); setLoadingMsg(`Uploading ${isoFile.name}...`);
    setError(null); setIsoResult(null);
    try {
      const job: Job = await isolateVoice(isoFile);
      setLoadingMsg("Separating vocals from background...");
      const out = await pollJob(job.job_id, setLoadingMsg);
      if (out.status === "completed") {
        const r = (out.result || {}) as Record<string, unknown>;
        setIsoResult({
          vocals_url: r.vocals_url ? `${API_BASE}${r.vocals_url}` : "",
          background_url: r.background_url ? `${API_BASE}${r.background_url}` : "",
          method: String(r.method || "ffmpeg_filter"),
        });
      } else {
        setError(out.error || "Isolation failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false); setLoadingMsg("");
    }
  };

  // ---- Normalize ------------------------------------------------------------
  const handleNormalize = async () => {
    if (!normFile) { setError("Pick an audio file first."); return; }
    setLoading(true); setLoadingMsg(`Uploading ${normFile.name}...`);
    setError(null); setNormResult(null);
    try {
      const job: Job = await normalizeAudio(normFile, normTargetDb);
      setLoadingMsg(`Normalizing to ${normTargetDb} dB...`);
      const out = await pollJob(job.job_id, setLoadingMsg);
      if (out.status === "completed") {
        const r = (out.result || {}) as Record<string, unknown>;
        setNormResult({
          audio_url: r.audio_url ? `${API_BASE}${r.audio_url}` : "",
          target_db: Number(r.target_db ?? normTargetDb),
        });
      } else {
        setError(out.error || "Normalize failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false); setLoadingMsg("");
    }
  };

  const modes = [
    { id: "sfx" as const, label: "Sound Effects", icon: Music },
    { id: "tts" as const, label: "Text to Speech", icon: Mic },
    { id: "isolate" as const, label: "Voice Isolation", icon: Headphones },
    { id: "normalize" as const, label: "Normalize", icon: Volume2 },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Audio Studio</h1>
        <p className="text-zinc-400 mt-1">Generate sound effects and speech, separate vocals, normalize audio.</p>
      </div>

      <div className="flex gap-2 mb-6 flex-wrap">
        {modes.map((m) => {
          const Icon = m.icon;
          return (
            <button key={m.id} onClick={() => { setMode(m.id); clearResults(); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === m.id ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}>
              <Icon className="w-4 h-4" /> {m.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          {mode === "sfx" && (
            <SFXControls
              sfxMode={sfxMode} setSfxMode={setSfxMode}
              selectedSfx={selectedSfx} setSelectedSfx={setSelectedSfx}
              sfxPrompt={sfxPrompt} setSfxPrompt={setSfxPrompt}
              sfxDuration={sfxDuration} setSfxDuration={setSfxDuration}
              loading={loading} onGenerate={handleGenerateSFX}
            />
          )}

          {mode === "tts" && (
            <TTSControls
              ttsText={ttsText} setTtsText={setTtsText}
              personas={personas} voices={voices}
              selectedPersona={selectedPersona} setSelectedPersona={setSelectedPersona}
              voiceOverride={voiceOverride} setVoiceOverride={setVoiceOverride}
              speed={speed} setSpeed={setSpeed}
              pitch={pitch} setPitch={setPitch}
              volume={volume} setVolume={setVolume}
              outputFormat={outputFormat} setOutputFormat={setOutputFormat}
              applyPersonaDefaults={applyPersonaDefaults} resetSliders={resetSliders}
              loading={loading} onGenerate={handleGenerateTTS}
            />
          )}

          {mode === "isolate" && (
            <FileUploadCard
              icon={<Headphones className="w-8 h-8 text-zinc-500 mx-auto mb-2" />}
              title="Voice Isolation"
              description="Separate vocals from background. Uses Demucs if available, else FFmpeg bandpass."
              accept="audio/*,video/*"
              file={isoFile} setFile={setIsoFile}
              actionLabel="Isolate Voice"
              onAction={handleIsolate}
              loading={loading}
            />
          )}

          {mode === "normalize" && (
            <FileUploadCard
              icon={<Volume2 className="w-8 h-8 text-zinc-500 mx-auto mb-2" />}
              title="Normalize Volume"
              description="Peak-normalize an audio file to a target dB level."
              accept="audio/*,video/*"
              file={normFile} setFile={setNormFile}
              actionLabel="Normalize"
              onAction={handleNormalize}
              loading={loading}
              extra={
                <div>
                  <label className="text-xs text-zinc-400 mt-3 block">
                    Target dB: <span className="text-indigo-400 font-mono">{normTargetDb.toFixed(1)}</span>
                  </label>
                  <input type="range" min={-20} max={0} step={0.5} value={normTargetDb}
                    onChange={(e) => setNormTargetDb(Number(e.target.value))} className="w-full mt-1 accent-indigo-500" />
                  <p className="text-[10px] text-zinc-500 mt-1">
                    -3 dB is the typical target for clean playback; -14 LUFS is broadcast loudness.
                  </p>
                </div>
              }
            />
          )}
        </div>

        {/* Result area */}
        <div className="lg:col-span-2">
          {loading && (
            <Card className="min-h-[200px] flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-10 h-10 text-indigo-400 mx-auto mb-3 animate-spin" />
                <p className="text-indigo-400 text-sm font-medium">{loadingMsg || "Processing..."}</p>
              </div>
            </Card>
          )}

          {!loading && error && (
            <Card className="border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm">{error}</p>
            </Card>
          )}

          {!loading && !error && sfxResult && mode === "sfx" && (
            <SFXResult result={sfxResult} fallbackName={selectedSfx} fallbackDuration={sfxDuration} />
          )}

          {!loading && !error && ttsResult && mode === "tts" && (
            <TTSResult result={ttsResult} />
          )}

          {!loading && !error && isoResult && mode === "isolate" && (
            <IsolateResultCard result={isoResult} sourceName={isoFile?.name || "input"} />
          )}

          {!loading && !error && normResult && mode === "normalize" && (
            <NormalizeResultCard result={normResult} sourceName={normFile?.name || "input"} />
          )}

          {!loading && !error && !sfxResult && !ttsResult && !isoResult && !normResult && (
            <Card className="min-h-[300px] flex items-center justify-center">
              <div className="text-center">
                <Music className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Generated audio will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">Pick a tool, configure, and click Generate</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ===== Sub-components ======================================================

function SFXControls({
  sfxMode, setSfxMode, selectedSfx, setSelectedSfx,
  sfxPrompt, setSfxPrompt, sfxDuration, setSfxDuration,
  loading, onGenerate,
}: {
  sfxMode: "preset" | "prompt"; setSfxMode: (m: "preset" | "prompt") => void;
  selectedSfx: string; setSelectedSfx: (s: string) => void;
  sfxPrompt: string; setSfxPrompt: (s: string) => void;
  sfxDuration: number; setSfxDuration: (d: number) => void;
  loading: boolean; onGenerate: () => void;
}) {
  return (
    <>
      <Card>
        <div className="flex gap-1 mb-3">
          <button onClick={() => setSfxMode("preset")}
            className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              sfxMode === "preset" ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}>Preset</button>
          <button onClick={() => setSfxMode("prompt")}
            className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              sfxMode === "prompt" ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}>Prompt</button>
        </div>
        {sfxMode === "preset" ? (
          <>
            <CardTitle className="text-sm mb-3">Sound Effects Library</CardTitle>
            {Object.entries(SFX_CATEGORIES).map(([cat, effects]) => (
              <div key={cat} className="mb-3">
                <p className="text-[10px] text-zinc-500 uppercase font-semibold mb-1.5">{cat}</p>
                <div className="flex flex-wrap gap-1.5">
                  {effects.map((sfx) => (
                    <button key={sfx} onClick={() => setSelectedSfx(sfx)}
                      className={`px-2.5 py-1 rounded-full text-xs transition-all ${
                        selectedSfx === sfx ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                      }`}>
                      {sfx.replace(/_/g, " ")}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </>
        ) : (
          <>
            <CardTitle className="text-sm mb-3">Describe the sound</CardTitle>
            <textarea value={sfxPrompt} onChange={(e) => setSfxPrompt(e.target.value)}
              placeholder="e.g. heavy rain on a tin roof, distant thunder rumble, crackling campfire..."
              className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
            <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed">
              Keywords (rain, thunder, wind, fire, footsteps...) match a procedural preset.
              For true text-to-audio AI, install AudioLDM2 (~3 GB).
            </p>
          </>
        )}
      </Card>
      <Card>
        <label className="text-xs text-zinc-400">
          Duration: <span className="text-indigo-400 font-mono">{sfxDuration}s</span>
        </label>
        <input type="range" min={0.5} max={15} step={0.5} value={sfxDuration}
          onChange={(e) => setSfxDuration(Number(e.target.value))} className="w-full mt-1 accent-indigo-500" />
      </Card>
      <Button className="w-full" onClick={onGenerate} disabled={loading}>
        {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...</> : <><Music className="w-4 h-4 mr-2" /> Generate SFX</>}
      </Button>
    </>
  );
}

function TTSControls({
  ttsText, setTtsText,
  personas, voices,
  selectedPersona, setSelectedPersona,
  voiceOverride, setVoiceOverride,
  speed, setSpeed, pitch, setPitch, volume, setVolume,
  outputFormat, setOutputFormat,
  applyPersonaDefaults, resetSliders,
  loading, onGenerate,
}: {
  ttsText: string; setTtsText: (s: string) => void;
  personas: PersonaInfo[]; voices: VoiceInfo[];
  selectedPersona: string; setSelectedPersona: (s: string) => void;
  voiceOverride: string; setVoiceOverride: (s: string) => void;
  speed: number; setSpeed: (n: number) => void;
  pitch: number; setPitch: (n: number) => void;
  volume: number; setVolume: (n: number) => void;
  outputFormat: "wav" | "mp3"; setOutputFormat: (f: "wav" | "mp3") => void;
  applyPersonaDefaults: () => void; resetSliders: () => void;
  loading: boolean; onGenerate: () => void;
}) {
  const persona = personas.find((p) => p.id === selectedPersona);
  return (
    <>
      <Card>
        <CardTitle className="text-sm mb-3">Persona</CardTitle>
        <select value={selectedPersona}
          onChange={(e) => setSelectedPersona(e.target.value)}
          className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50">
          {personas.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
        {persona && (
          <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed italic">{persona.description}</p>
        )}
        <div className="flex gap-2 mt-3">
          <Button variant="ghost" size="sm" className="flex-1" onClick={applyPersonaDefaults}>
            <RotateCcw className="w-3 h-3 mr-1.5" /> Use Persona Defaults
          </Button>
        </div>
      </Card>

      <Card>
        <CardTitle className="text-sm mb-3">Voice</CardTitle>
        <p className="text-[10px] text-zinc-500 mb-2">
          Default for this persona: <span className="text-zinc-300 font-mono">{persona?.voice || "Amy"}</span>
        </p>
        <select value={voiceOverride}
          onChange={(e) => setVoiceOverride(e.target.value)}
          className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:ring-2 focus:ring-indigo-500/50">
          <option value="">— Use persona default —</option>
          {voices.map((v) => (
            <option key={v.id} value={v.id}>
              {v.id} · {v.accent} · {v.gender} ({v.quality})
            </option>
          ))}
        </select>
        <p className="text-[10px] text-amber-500/70 mt-2 leading-relaxed">
          First use of a new voice downloads ~30–80 MB from HuggingFace; subsequent runs are fast.
        </p>
      </Card>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <CardTitle className="text-sm">Voice Effects</CardTitle>
          <Button variant="ghost" size="sm" onClick={resetSliders}>
            <RotateCcw className="w-3 h-3 mr-1.5" /> Reset
          </Button>
        </div>
        <Slider label="Speed" value={speed} min={0.5} max={2.0} step={0.05}
          onChange={setSpeed} format={(v) => `${v.toFixed(2)}x`} />
        <Slider label="Pitch" value={pitch} min={-12} max={12} step={1}
          onChange={setPitch} format={(v) => `${v > 0 ? "+" : ""}${v} semitones`} />
        <Slider label="Volume" value={volume} min={0.0} max={2.0} step={0.05}
          onChange={setVolume} format={(v) => `${v.toFixed(2)}x`} />
      </Card>

      <Card>
        <CardTitle className="text-sm mb-3">Output Format</CardTitle>
        <div className="flex gap-2">
          {(["wav", "mp3"] as const).map((f) => (
            <button key={f} onClick={() => setOutputFormat(f)}
              className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                outputFormat === f ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-zinc-500 mt-2">
          WAV is uncompressed (best quality). MP3 (192 kbps) is ~10× smaller.
        </p>
      </Card>

      <Card>
        <CardTitle className="text-sm mb-2">Text</CardTitle>
        <p className="text-[10px] text-zinc-500 mb-2">Sample texts:</p>
        <div className="space-y-1 mb-3">
          {TTS_SAMPLES.map((s, i) => (
            <button key={i} onClick={() => setTtsText(s)}
              className={`w-full text-left px-2.5 py-1.5 rounded-lg text-[11px] transition-all ${
                ttsText === s ? "bg-indigo-600/15 text-indigo-300 border border-indigo-500/20"
                              : "text-zinc-400 hover:bg-zinc-800"
              }`}>
              {s.length > 65 ? s.slice(0, 65) + "..." : s}
            </button>
          ))}
        </div>
        <textarea value={ttsText} onChange={(e) => setTtsText(e.target.value)}
          placeholder="Enter text to speak..."
          className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
        <p className="text-[10px] text-zinc-600 mt-1 text-right">
          {ttsText.length} chars · ~{Math.max(1, Math.round(ttsText.length / 15))}s of speech
        </p>
      </Card>

      <Button className="w-full" onClick={onGenerate} disabled={loading || !ttsText.trim()}>
        {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Synthesizing...</> : <><Mic className="w-4 h-4 mr-2" /> Generate Speech</>}
      </Button>
    </>
  );
}

function Slider({ label, value, min, max, step, onChange, format }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (n: number) => void; format: (v: number) => string;
}) {
  return (
    <div className="mb-3">
      <label className="text-xs text-zinc-400 flex items-center justify-between">
        <span>{label}</span>
        <span className="text-indigo-400 font-mono">{format(value)}</span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-indigo-500" />
    </div>
  );
}

function FileUploadCard({
  icon, title, description, accept, file, setFile,
  actionLabel, onAction, loading, extra,
}: {
  icon: React.ReactNode; title: string; description: string; accept: string;
  file: File | null; setFile: (f: File | null) => void;
  actionLabel: string; onAction: () => void;
  loading: boolean; extra?: React.ReactNode;
}) {
  return (
    <Card>
      <CardTitle className="text-sm mb-3">{title}</CardTitle>
      <p className="text-xs text-zinc-400 mb-3">{description}</p>
      <label className="block border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-zinc-500 transition-colors cursor-pointer">
        <input type="file" accept={accept}
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="hidden" />
        {icon}
        <p className="text-xs text-zinc-300 break-all">
          {file ? file.name : "Click to choose a file"}
        </p>
        <p className="text-[10px] text-zinc-600 mt-1">
          {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP3, WAV, MP4, MOV..."}
        </p>
      </label>
      {extra}
      <Button className="w-full mt-3" onClick={onAction} disabled={loading || !file}>
        {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</> : <><Upload className="w-4 h-4 mr-2" /> {actionLabel}</>}
      </Button>
    </Card>
  );
}

function SFXResult({ result, fallbackName, fallbackDuration }: {
  result: { url: string; meta: SfxResultMeta };
  fallbackName: string; fallbackDuration: number;
}) {
  const { url, meta } = result;
  return (
    <Card className="border-green-500/20 bg-green-500/5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="text-green-400 text-sm font-medium">
            {(meta.sfx_type || fallbackName).replace(/_/g, " ")} ({(meta.duration ?? fallbackDuration).toFixed(1)}s)
          </p>
          <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
            {meta.method} · {meta.engine} · {meta.channels === 2 ? "stereo" : "mono"} @ {meta.sample_rate} Hz
          </p>
        </div>
        <Badge>{meta.engine}</Badge>
      </div>
      <audio key={url} src={url} controls autoPlay className="w-full mb-3" />
      {meta.details && (
        <p className="text-[10px] text-zinc-600 mb-3 italic leading-relaxed">{meta.details}</p>
      )}
      <Button variant="secondary" className="w-full"
        onClick={() => downloadBlob(url, `${meta.sfx_type}_${meta.duration.toFixed(1)}s.wav`)}>
        <Download className="w-4 h-4 mr-2" /> Download
      </Button>
    </Card>
  );
}

function TTSResult({ result }: { result: { url: string; meta: TTSResultMeta } }) {
  const { url, meta } = result;
  const ext = meta.output_format || "wav";
  return (
    <Card className="border-green-500/20 bg-green-500/5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="text-green-400 text-sm font-medium">
            {meta.persona_label || "Speech"} · {meta.voice}
          </p>
          <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed font-mono">
            {meta.speed.toFixed(2)}x speed · {meta.pitch > 0 ? "+" : ""}{meta.pitch} st pitch · {meta.volume.toFixed(2)}x vol · {ext.toUpperCase()}
          </p>
        </div>
        <Badge>{meta.persona || "custom"}</Badge>
      </div>
      <audio key={url} src={url} controls autoPlay className="w-full mb-3" />
      <Button variant="secondary" className="w-full"
        onClick={() => downloadBlob(url, `tts_${meta.voice}_${meta.persona || "custom"}.${ext}`)}>
        <Download className="w-4 h-4 mr-2" /> Download {ext.toUpperCase()}
      </Button>
    </Card>
  );
}

function IsolateResultCard({ result, sourceName }: { result: IsolateResult; sourceName: string }) {
  const baseName = sourceName.replace(/\.[^.]+$/, "");
  return (
    <div className="space-y-4">
      <Card className="border-green-500/20 bg-green-500/5">
        <p className="text-green-400 text-sm font-medium mb-1">Voice isolation complete</p>
        <p className="text-[10px] text-zinc-500">Method: {result.method}</p>
      </Card>
      {result.vocals_url && (
        <Card>
          <CardTitle className="text-sm mb-2">🎤 Vocals</CardTitle>
          <audio key={result.vocals_url} src={result.vocals_url} controls className="w-full mb-3" />
          <Button variant="secondary" size="sm" className="w-full"
            onClick={() => downloadBlob(result.vocals_url, `${baseName}_vocals.wav`)}>
            <Download className="w-3.5 h-3.5 mr-1.5" /> Download Vocals
          </Button>
        </Card>
      )}
      {result.background_url && (
        <Card>
          <CardTitle className="text-sm mb-2">🎵 Background</CardTitle>
          <audio key={result.background_url} src={result.background_url} controls className="w-full mb-3" />
          <Button variant="secondary" size="sm" className="w-full"
            onClick={() => downloadBlob(result.background_url, `${baseName}_background.wav`)}>
            <Download className="w-3.5 h-3.5 mr-1.5" /> Download Background
          </Button>
        </Card>
      )}
      {result.method === "ffmpeg_filter" && (
        <Card className="border-amber-500/20 bg-amber-500/5">
          <p className="text-[11px] text-amber-300 leading-relaxed">
            Using ffmpeg bandpass filter (200–3000 Hz) — fast but rough.
            For ML-quality separation install demucs: <code className="text-amber-200 bg-zinc-900/60 px-1 py-0.5 rounded">pip install demucs</code>
          </p>
        </Card>
      )}
    </div>
  );
}

function NormalizeResultCard({ result, sourceName }: { result: NormalizeResult; sourceName: string }) {
  const baseName = sourceName.replace(/\.[^.]+$/, "");
  return (
    <Card className="border-green-500/20 bg-green-500/5">
      <p className="text-green-400 text-sm font-medium mb-1">Normalized</p>
      <p className="text-[10px] text-zinc-500 mb-3">Target peak: {result.target_db.toFixed(1)} dB</p>
      <audio key={result.audio_url} src={result.audio_url} controls autoPlay className="w-full mb-3" />
      <Button variant="secondary" className="w-full"
        onClick={() => downloadBlob(result.audio_url, `${baseName}_normalized.wav`)}>
        <Download className="w-4 h-4 mr-2" /> Download Normalized
      </Button>
    </Card>
  );
}
