"use client";

import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Music, Mic, Headphones, Volume2, Waves, Loader2, Download, Play } from "lucide-react";
import { generateTTS, generateSFX, downloadBlob } from "@/lib/api";

const SFX_CATEGORIES: Record<string, string[]> = {
  "Nature": ["rain", "thunder", "wind", "ocean_waves"],
  "Impact": ["explosion", "glass_break"],
  "Ambient": ["fire_crackling", "bird_chirp", "clock_tick"],
  "Human": ["heartbeat", "footsteps", "typing"],
  "Motion": ["whoosh", "car_engine", "door_close"],
};

const TTS_SAMPLES = [
  "A brave dog embarks on an epic adventure through the mountains.",
  "In a world where technology rules, one human dares to dream.",
  "The ocean whispered secrets that only the bravest sailors could hear.",
  "Welcome to StudioLite, your personal AI video creation studio.",
];

interface SfxResultMeta {
  engine: string;
  method: string;
  sfx_type: string;
  details: string;
  duration: number;
  channels: number;
  sample_rate: number;
}

export default function AudioPanel() {
  const [mode, setMode] = useState<"sfx" | "tts" | "isolate" | "normalize">("sfx");
  const [sfxMode, setSfxMode] = useState<"preset" | "prompt">("preset");
  const [selectedSfx, setSelectedSfx] = useState("rain");
  const [sfxPrompt, setSfxPrompt] = useState("");
  const [sfxDuration, setSfxDuration] = useState(2);
  const [ttsText, setTtsText] = useState(TTS_SAMPLES[0]);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultMeta, setResultMeta] = useState<SfxResultMeta | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateSFX = async () => {
    if (sfxMode === "prompt" && !sfxPrompt.trim()) {
      setError("Enter a prompt or switch to Preset mode.");
      return;
    }
    const label = sfxMode === "prompt" ? sfxPrompt.trim() : selectedSfx.replace(/_/g, " ");
    setLoading(true);
    setLoadingMsg(`Generating ${label}...`);
    setError(null);
    setResultUrl(null);
    setResultMeta(null);
    try {
      const job = await generateSFX(
        sfxMode === "prompt"
          ? { sfx_type: "", prompt: sfxPrompt.trim(), duration: sfxDuration }
          : { sfx_type: selectedSfx, duration: sfxDuration }
      );
      setLoadingMsg(`Job ${job.job_id} - synthesizing...`);

      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${apiBase}/api/v1/jobs/${job.job_id}`);
          const status = await res.json();
          setLoadingMsg(status.message || "Processing...");
          if (status.status === "completed") {
            clearInterval(poll);
            setResultUrl(`${apiBase}/api/v1/jobs/${job.job_id}/download?t=${Date.now()}`);
            const r = status.result || {};
            setResultMeta({
              engine: r.engine || "unknown",
              method: r.method || "Procedural synthesis",
              sfx_type: r.sfx_type || selectedSfx,
              details: r.details || "",
              duration: r.duration || sfxDuration,
              channels: r.channels || 1,
              sample_rate: r.sample_rate || 22050,
            });
            setLoading(false);
            setLoadingMsg("");
          } else if (status.status === "failed") {
            clearInterval(poll);
            setError(status.error || "SFX generation failed");
            setLoading(false);
            setLoadingMsg("");
          }
        } catch { /* keep polling */ }
      }, 800);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start SFX job");
      setLoading(false);
      setLoadingMsg("");
    }
  };

  const handleGenerateTTS = async () => {
    if (!ttsText.trim()) return;
    setLoading(true);
    setLoadingMsg("Generating speech with Piper TTS...");
    setError(null);
    setResultUrl(null);
    try {
      const job = await generateTTS({ text: ttsText.trim(), engine: "piper" });
      setLoadingMsg(`Job started: ${job.job_id} - Processing...`);

      // Poll
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/jobs/${job.job_id}`);
          const status = await res.json();
          setLoadingMsg(status.message || "Processing...");
          if (status.status === "completed") {
            clearInterval(poll);
            setResultUrl(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/jobs/${job.job_id}/download`);
            setLoading(false);
          } else if (status.status === "failed") {
            clearInterval(poll);
            setError(status.error || "TTS failed");
            setLoading(false);
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
      setLoading(false);
    }
  };

  const modes = [
    { id: "sfx", label: "Sound Effects", icon: Music },
    { id: "tts", label: "Text to Speech", icon: Mic },
    { id: "isolate", label: "Voice Isolation", icon: Headphones },
    { id: "normalize", label: "Normalize", icon: Volume2 },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Audio Studio</h1>
        <p className="text-zinc-400 mt-1">Generate sound effects, speech, isolate vocals, and process audio</p>
      </div>

      <div className="flex gap-2 mb-6 flex-wrap">
        {modes.map((m) => {
          const Icon = m.icon;
          return (
            <button key={m.id} onClick={() => { setMode(m.id as typeof mode); setError(null); setResultUrl(null); }}
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
            <>
              <Card>
                <div className="flex gap-1 mb-3">
                  <button onClick={() => setSfxMode("preset")}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      sfxMode === "preset" ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                    }`}>
                    Preset
                  </button>
                  <button onClick={() => setSfxMode("prompt")}
                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      sfxMode === "prompt" ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                    }`}>
                    Prompt
                  </button>
                </div>
                {sfxMode === "preset" && (
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
                )}
                {sfxMode === "prompt" && (
                  <>
                    <CardTitle className="text-sm mb-3">Describe the sound</CardTitle>
                    <textarea value={sfxPrompt} onChange={(e) => setSfxPrompt(e.target.value)}
                      placeholder="e.g. heavy rain on a tin roof, distant thunder rumble, crackling campfire..."
                      className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
                    <p className="text-[10px] text-zinc-500 mt-2 leading-relaxed">
                      Keywords (rain, thunder, wind, fire, footsteps, etc.) are matched to a procedural preset.
                      For a true text-to-audio AI model, install AudioLDM2 (~3 GB).
                    </p>
                  </>
                )}
              </Card>
              <Card>
                <label className="text-xs text-zinc-400">Duration: <span className="text-indigo-400 font-mono">{sfxDuration}s</span></label>
                <input type="range" min={0.5} max={15} step={0.5} value={sfxDuration}
                  onChange={(e) => setSfxDuration(Number(e.target.value))} className="w-full mt-1 accent-indigo-500" />
              </Card>
              <Button className="w-full" onClick={handleGenerateSFX} disabled={loading}>
                {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...</> : <><Music className="w-4 h-4 mr-2" /> Generate SFX</>}
              </Button>
            </>
          )}

          {mode === "tts" && (
            <>
              <Card>
                <CardTitle className="text-sm mb-3">Text to Speech</CardTitle>
                <p className="text-xs text-zinc-500 mb-2">Sample texts:</p>
                <div className="space-y-1 mb-3">
                  {TTS_SAMPLES.map((s, i) => (
                    <button key={i} onClick={() => setTtsText(s)}
                      className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-all ${
                        ttsText === s ? "bg-indigo-600/15 text-indigo-300 border border-indigo-500/20" : "text-zinc-400 hover:bg-zinc-800"
                      }`}>
                      {s.slice(0, 60)}...
                    </button>
                  ))}
                </div>
                <textarea value={ttsText} onChange={(e) => setTtsText(e.target.value)}
                  placeholder="Enter text to convert to speech..."
                  className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
              </Card>
              <Button className="w-full" onClick={handleGenerateTTS} disabled={loading || !ttsText.trim()}>
                {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating Speech...</> : <><Mic className="w-4 h-4 mr-2" /> Generate Speech</>}
              </Button>
            </>
          )}

          {mode === "isolate" && (
            <Card>
              <CardTitle className="text-sm mb-3">Voice Isolation</CardTitle>
              <p className="text-xs text-zinc-400 mb-3">Separate vocals from background noise/music</p>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-zinc-500 transition-colors cursor-pointer">
                <Headphones className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Upload audio or video file</p>
                <p className="text-[10px] text-zinc-600 mt-1">MP3, WAV, MP4, MOV</p>
              </div>
              <Button className="w-full mt-3" disabled={true}>
                <Headphones className="w-4 h-4 mr-2" /> Isolate Voice
              </Button>
              <p className="text-[10px] text-zinc-600 mt-1 text-center">File upload requires Streamlit UI</p>
            </Card>
          )}

          {mode === "normalize" && (
            <Card>
              <CardTitle className="text-sm mb-3">Normalize Volume</CardTitle>
              <p className="text-xs text-zinc-400 mb-3">Adjust audio to consistent volume level</p>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-zinc-500 transition-colors cursor-pointer">
                <Volume2 className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Upload audio file</p>
              </div>
              <label className="text-xs text-zinc-400 mt-3 block">Target dB: -3.0</label>
              <input type="range" min={-10} max={0} step={0.5} defaultValue={-3} className="w-full mt-1 accent-indigo-500" />
              <Button className="w-full mt-3" disabled={true}>
                <Volume2 className="w-4 h-4 mr-2" /> Normalize
              </Button>
              <p className="text-[10px] text-zinc-600 mt-1 text-center">File upload requires Streamlit UI</p>
            </Card>
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

          {resultUrl && !loading && (
            <Card className="border-green-500/20 bg-green-500/5">
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <p className="text-green-400 text-sm font-medium">
                    {mode === "sfx"
                      ? `${(resultMeta?.sfx_type || selectedSfx).replace(/_/g, " ")} (${(resultMeta?.duration ?? sfxDuration).toFixed(1)}s)`
                      : "Audio Generated!"}
                  </p>
                  {mode === "sfx" && resultMeta && (
                    <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
                      {resultMeta.method} · {resultMeta.engine} · {resultMeta.channels === 2 ? "stereo" : "mono"} @ {resultMeta.sample_rate} Hz
                    </p>
                  )}
                </div>
                {mode === "sfx" && resultMeta && (
                  <Badge>{resultMeta.engine}</Badge>
                )}
              </div>
              <audio key={resultUrl} src={resultUrl} controls autoPlay className="w-full mb-3" />
              {mode === "sfx" && resultMeta?.details && (
                <p className="text-[10px] text-zinc-600 mb-3 italic leading-relaxed">{resultMeta.details}</p>
              )}
              <Button
                variant="secondary"
                className="w-full"
                onClick={() => {
                  const fname = mode === "sfx"
                    ? `${(resultMeta?.sfx_type || selectedSfx)}_${(resultMeta?.duration ?? sfxDuration).toFixed(1)}s.wav`
                    : "tts.wav";
                  downloadBlob(resultUrl, fname).catch((e) => setError(`Download failed: ${e.message}`));
                }}
              >
                <Download className="w-4 h-4 mr-2" /> Download Audio
              </Button>
            </Card>
          )}

          {error && (
            <Card className="border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm">{error}</p>
            </Card>
          )}

          {!loading && !resultUrl && !error && (
            <Card className="min-h-[300px] flex items-center justify-center">
              <div className="text-center">
                <Music className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Generated audio will appear here</p>
                <p className="text-zinc-600 text-xs mt-1">Select a tool and click Generate</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
