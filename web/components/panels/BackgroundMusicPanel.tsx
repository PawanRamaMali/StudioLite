"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Download, Music2, Mic } from "lucide-react";
import {
  mixBackgroundMusic,
  getJob,
  type Job,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function BackgroundMusicPanel() {
  const [voice, setVoice] = useState<File | null>(null);
  const [music, setMusic] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [musicVol, setMusicVol] = useState(0.3);
  const [voiceVol, setVoiceVol] = useState(1.0);
  const [duck, setDuck] = useState(true);
  const [duckThreshold, setDuckThreshold] = useState(0.05);
  const [duckRatio, setDuckRatio] = useState(8);
  const [fadeIn, setFadeIn] = useState(1.0);
  const [fadeOut, setFadeOut] = useState(2.0);
  const [loopMusic, setLoopMusic] = useState(true);
  const [format, setFormat] = useState<"wav" | "mp3">("wav");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => clearPoll, []);

  const onVoice = useCallback((f: File | null) => {
    setVoice(f); setJob(null); setError(null);
  }, []);
  const onMusic = useCallback((f: File | null) => {
    setMusic(f); setJob(null); setError(null);
  }, []);

  const onStart = async () => {
    if (!voice || !music || job) return;
    setError(null);
    try {
      const j = await mixBackgroundMusic(voice, music, {
        music_volume: musicVol,
        voice_volume: voiceVol,
        duck,
        duck_threshold: duckThreshold,
        duck_ratio: duckRatio,
        fade_in: fadeIn,
        fade_out: fadeOut,
        loop_music: loopMusic,
        output_format: format,
      });
      setJob(j);
      clearPoll();
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getJob(j.job_id);
          setJob(latest);
          if (latest.status === "completed" || latest.status === "failed" || latest.status === "cancelled") {
            clearPoll();
          }
        } catch (e) {
          setError((e as Error).message);
          clearPoll();
        }
      }, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const busy = !!job && (job.status === "running" || job.status === "queued");
  const audioUrl = job?.status === "completed"
    ? (job.result as { audio_url?: string } | undefined)?.audio_url
    : undefined;
  const fullUrl = audioUrl ? `${API_BASE}${audioUrl}` : "";

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Background Music</h1>
        <p className="text-zinc-400 mt-1">
          Mix a music bed under a voice or dialog track. Sidechain ducking
          dips the music while the voice plays so dialogue stays intelligible.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3 flex items-center gap-2">
              <Mic className="w-4 h-4" /> Voice / dialog
            </CardTitle>
            <label className="block cursor-pointer">
              <input
                type="file" accept="audio/*,video/*" className="hidden"
                disabled={busy}
                onChange={(e) => onVoice(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                {voice ? (
                  <p className="text-sm text-zinc-300 truncate">{voice.name}</p>
                ) : (
                  <p className="text-sm text-zinc-400">Primary track</p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3 flex items-center gap-2">
              <Music2 className="w-4 h-4" /> Music bed
            </CardTitle>
            <label className="block cursor-pointer">
              <input
                type="file" accept="audio/*" className="hidden"
                disabled={busy}
                onChange={(e) => onMusic(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                {music ? (
                  <p className="text-sm text-zinc-300 truncate">{music.name}</p>
                ) : (
                  <p className="text-sm text-zinc-400">Music to layer underneath</p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Levels</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Voice: {voiceVol.toFixed(2)}x</label>
                <input
                  type="range" min={0} max={2} step={0.05}
                  value={voiceVol} disabled={busy}
                  onChange={(e) => setVoiceVol(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Music: {musicVol.toFixed(2)}x</label>
                <input
                  type="range" min={0} max={2} step={0.05}
                  value={musicVol} disabled={busy}
                  onChange={(e) => setMusicVol(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  0.2-0.4 is a typical bed level under speech.
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Ducking</CardTitle>
            <label className="flex items-center gap-2 text-sm text-zinc-300">
              <input
                type="checkbox" checked={duck}
                onChange={(e) => setDuck(e.target.checked)}
                disabled={busy}
              />
              Dip music when voice is present
            </label>
            {duck && (
              <div className="space-y-3 mt-3">
                <div>
                  <label className="text-xs text-zinc-400">
                    Threshold: {duckThreshold.toFixed(3)}
                  </label>
                  <input
                    type="range" min={0.01} max={0.3} step={0.01}
                    value={duckThreshold} disabled={busy}
                    onChange={(e) => setDuckThreshold(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400">
                    Ratio: {duckRatio.toFixed(1)}:1
                  </label>
                  <input
                    type="range" min={2} max={20} step={0.5}
                    value={duckRatio} disabled={busy}
                    onChange={(e) => setDuckRatio(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Fades &amp; length</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Fade in: {fadeIn.toFixed(1)}s</label>
                <input
                  type="range" min={0} max={10} step={0.1}
                  value={fadeIn} disabled={busy}
                  onChange={(e) => setFadeIn(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Fade out: {fadeOut.toFixed(1)}s</label>
                <input
                  type="range" min={0} max={10} step={0.1}
                  value={fadeOut} disabled={busy}
                  onChange={(e) => setFadeOut(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500"
                />
              </div>
              <label className="flex items-center gap-2 text-sm text-zinc-300">
                <input
                  type="checkbox" checked={loopMusic}
                  onChange={(e) => setLoopMusic(e.target.checked)}
                  disabled={busy}
                />
                Loop music if shorter than voice
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(["wav", "mp3"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFormat(f)}
                    disabled={busy}
                    className={`px-3 py-2 rounded-md text-xs uppercase border transition-colors ${
                      format === f
                        ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                        : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                    }`}
                  >{f}</button>
                ))}
              </div>
            </div>
          </Card>

          <Button
            size="lg" className="w-full"
            disabled={!voice || !music || busy}
            onClick={onStart}
          >
            <Zap className="w-5 h-5 mr-2" />
            {busy ? `Mixing - ${Math.round((job?.progress || 0) * 100)}%` : "Mix tracks"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px] flex items-center justify-center">
          <div className="text-center w-full px-6">
            {job && job.status === "completed" && fullUrl ? (
              <>
                <Music2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <p className="text-sm text-zinc-300">Mixed track</p>
                <audio key={fullUrl} src={fullUrl} controls className="w-full mt-4" />
                <Button
                  size="sm" variant="secondary" className="mt-3"
                  onClick={() => window.open(fullUrl, "_blank", "noopener,noreferrer")}
                >
                  <Download className="w-4 h-4 mr-1" /> Download
                </Button>
              </>
            ) : job && job.status === "failed" ? (
              <p className="text-sm text-red-400">{job.error || "Mix failed"}</p>
            ) : (
              <p className="text-sm text-zinc-500">Mixed output appears here once processing completes.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
