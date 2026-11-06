"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Download, SlidersHorizontal } from "lucide-react";
import {
  equalizerAudio,
  getJob,
  EQ_BANDS_HZ,
  type EqPreset,
  type Job,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const PRESETS: { id: EqPreset; label: string }[] = [
  { id: "flat",         label: "Flat" },
  { id: "bass_boost",   label: "Bass boost" },
  { id: "bass_cut",     label: "Bass cut" },
  { id: "vocal_boost",  label: "Vocal boost" },
  { id: "treble_boost", label: "Treble boost" },
  { id: "treble_cut",   label: "Treble cut" },
  { id: "rock",         label: "Rock" },
  { id: "pop",          label: "Pop" },
  { id: "podcast",      label: "Podcast" },
  { id: "radio",        label: "Radio" },
];

const PRESET_GAINS: Record<EqPreset, number[]> = {
  flat:         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  bass_boost:   [8, 7, 5, 3, 0, 0, 0, 0, 0, 0],
  bass_cut:     [-10, -8, -5, -2, 0, 0, 0, 0, 0, 0],
  vocal_boost:  [-2, -2, 0, 2, 4, 5, 4, 2, 0, 0],
  treble_boost: [0, 0, 0, 0, 0, 2, 4, 6, 7, 7],
  treble_cut:   [0, 0, 0, 0, 0, -2, -5, -7, -9, -10],
  rock:         [4, 3, 2, 1, -1, -1, 2, 4, 5, 5],
  pop:          [-2, -1, 0, 2, 4, 4, 2, 0, -1, -2],
  podcast:      [-6, -4, -2, 0, 3, 4, 3, 1, 0, -2],
  radio:        [-4, -2, 0, 1, 2, 3, 2, 1, 0, -4],
};

function bandLabel(hz: number): string {
  return hz >= 1000 ? `${hz / 1000}k` : `${hz}`;
}

export default function EqualizerPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [preset, setPreset] = useState<EqPreset>("flat");
  const [gains, setGains] = useState<number[]>(PRESET_GAINS.flat);
  const [custom, setCustom] = useState(false);
  const [format, setFormat] = useState<"wav" | "mp3">("wav");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => clearPoll, []);

  const onFile = useCallback((f: File | null) => {
    setFile(f); setJob(null); setError(null);
  }, []);

  const pickPreset = (p: EqPreset) => {
    setPreset(p);
    setGains(PRESET_GAINS[p]);
    setCustom(false);
  };

  const updateBand = (i: number, g: number) => {
    const next = [...gains];
    next[i] = g;
    setGains(next);
    setCustom(true);
  };

  const onStart = async () => {
    if (!file || job) return;
    setError(null);
    try {
      const j = await equalizerAudio(file, {
        preset: custom ? "flat" : preset,
        gains_db: custom ? gains : undefined,
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
        <h1 className="text-3xl font-bold gradient-text">Audio Equalizer</h1>
        <p className="text-zinc-400 mt-1">
          Shape the sound with a 10-band graphic EQ. Pick a preset or push the
          sliders yourself &mdash; gains clamp at &plusmn;18 dB.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file"
                accept="audio/*,video/*"
                className="hidden"
                disabled={busy}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-10 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                {file ? (
                  <p className="text-sm text-zinc-300 truncate">{file.name}</p>
                ) : (
                  <p className="text-sm text-zinc-400">Drop audio here or click to upload</p>
                )}
              </div>
            </label>
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Presets</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => pickPreset(p.id)}
                  disabled={busy}
                  className={`px-3 py-1.5 rounded-md text-xs border transition-colors ${
                    !custom && preset === p.id
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >{p.label}</button>
              ))}
            </div>
            {custom && (
              <p className="text-[11px] text-amber-400 mt-2">
                Custom curve active &mdash; sliders override presets
              </p>
            )}
          </Card>

          <Card>
            <CardTitle className="text-sm mb-3">Output</CardTitle>
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
          </Card>

          <Button size="lg" className="w-full" disabled={!file || busy} onClick={onStart}>
            <Zap className="w-5 h-5 mr-2" />
            {busy ? `Equalizing - ${Math.round((job?.progress || 0) * 100)}%` : "Apply EQ"}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3 flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4" /> 10-band graphic EQ
            </CardTitle>
            <div className="grid grid-cols-10 gap-1 items-end h-56">
              {EQ_BANDS_HZ.map((hz, i) => (
                <div key={hz} className="flex flex-col items-center gap-1 h-full">
                  <input
                    type="range"
                    min={-18} max={18} step={0.5}
                    value={gains[i]}
                    disabled={busy}
                    onChange={(e) => updateBand(i, Number(e.target.value))}
                    className="w-full accent-indigo-500"
                    style={{
                      writingMode: "vertical-lr" as unknown as "horizontal-tb",
                      WebkitAppearance: "slider-vertical",
                      height: "100%",
                    }}
                  />
                  <span className="text-[10px] text-zinc-500">{bandLabel(hz)}</span>
                  <span className="text-[10px] text-indigo-300">
                    {gains[i] > 0 ? "+" : ""}{gains[i].toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
            <Button
              size="sm" variant="ghost" className="mt-3"
              onClick={() => { setGains(PRESET_GAINS.flat); setPreset("flat"); setCustom(false); }}
              disabled={busy}
            >
              Reset to flat
            </Button>
          </Card>

          <Card className="min-h-[200px] flex items-center justify-center">
            <div className="text-center w-full px-6">
              {job && job.status === "completed" && fullUrl ? (
                <>
                  <audio key={fullUrl} src={fullUrl} controls className="w-full" />
                  <Button
                    size="sm" variant="secondary" className="mt-3"
                    onClick={() => window.open(fullUrl, "_blank", "noopener,noreferrer")}
                  >
                    <Download className="w-4 h-4 mr-1" /> Download
                  </Button>
                </>
              ) : job && job.status === "failed" ? (
                <p className="text-sm text-red-400">{job.error || "EQ failed"}</p>
              ) : (
                <p className="text-sm text-zinc-500">Output appears here once processing completes.</p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
