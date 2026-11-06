"use client";
import { useEffect, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Waves, Download, AlertTriangle } from "lucide-react";
import {
  ambientStatus,
  generateAmbient,
  getJob,
  getDownloadUrl,
  type AmbientStatus,
} from "@/lib/api";

const AMBIENT_PROMPTS: { label: string; prompt: string }[] = [
  { label: "Forest at dusk", prompt: "crickets, distant owl, rustling leaves, gentle wind through trees, calm forest at dusk" },
  { label: "City nighttime", prompt: "distant traffic hum, occasional car horn, muffled street conversations, light rain, urban night ambience" },
  { label: "Coffee shop", prompt: "milk steamer, muffled background chatter, clinking cups and saucers, soft indie music" },
  { label: "Ocean waves", prompt: "ocean waves breaking on sand, distant seagulls, light breeze, calm shoreline" },
  { label: "Footsteps gravel", prompt: "slow deliberate footsteps on gravel path, outdoor foley, crisp foot crunch" },
  { label: "Thunderstorm", prompt: "heavy rain on rooftop, distant thunder rolling, occasional close thunderclap, wind" },
  { label: "Office keyboard", prompt: "mechanical keyboard typing, soft mouse clicks, air conditioning hum, office ambience" },
  { label: "Campfire", prompt: "crackling campfire, wood popping, distant crickets, calm night" },
];

export default function AmbientAudioPanel() {
  const [prompt, setPrompt] = useState(AMBIENT_PROMPTS[0].prompt);
  const [negative, setNegative] = useState("low quality, distorted, noise, silence");
  const [status, setStatus] = useState<AmbientStatus | null>(null);
  const [backend, setBackend] = useState<"auto" | "audioldm2" | "stable_audio">("auto");
  const [duration, setDuration] = useState(5.0);
  const [steps, setSteps] = useState(50);
  const [guidance, setGuidance] = useState(3.5);
  const [seed, setSeed] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    ambientStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const start = async () => {
    if (!prompt.trim()) {
      setError("Enter a prompt.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Queueing...");
    try {
      const job = await generateAmbient({
        prompt,
        negative_prompt: negative,
        duration_s: duration,
        num_inference_steps: steps,
        guidance_scale: guidance,
        backend,
        seed: seed ? Number(seed) : null,
      });
      const poll = setInterval(async () => {
        try {
          const j = await getJob(job.job_id);
          setProgress(Math.min(Math.round(j.progress || 0), 100));
          setProgressMsg(j.message || "Working...");
          if (j.status === "completed") {
            clearInterval(poll);
            const out = (j.result as { output_path?: string } | undefined)?.output_path;
            if (out) setResult(out);
            setRunning(false);
          } else if (j.status === "failed") {
            clearInterval(poll);
            setError(j.error || "Ambient generation failed");
            setRunning(false);
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch (e) {
      setError((e as Error).message);
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Waves className="w-6 h-6 text-indigo-400" />
        <h1 className="text-2xl font-bold">Ambient &amp; Foley Audio</h1>
      </div>
      <p className="text-sm text-zinc-400 -mt-4">
        Text-to-audio for backgrounds and foley. Generates a WAV at the requested
        duration from AudioLDM2 (fast, ~2GB model) or Stable Audio Open (higher
        fidelity, 24GB VRAM). Layer it under a clip for depth or feed it to the
        Timeline.
      </p>

      {status && !status.available && (
        <Card className="border-yellow-500/40 bg-yellow-500/5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-200">
              <div className="font-semibold mb-1">No backend installed yet.</div>
              <pre className="whitespace-pre-wrap text-xs text-yellow-200/80 font-mono">
                {status.install_hint}
              </pre>
            </div>
          </div>
        </Card>
      )}
      {status && status.available && (
        <div className="text-xs text-green-400">Ready: {status.backends.join(", ")}</div>
      )}

      <Card>
        <CardTitle>Sample prompts</CardTitle>
        <div className="flex flex-wrap gap-2">
          {AMBIENT_PROMPTS.map((p) => (
            <button key={p.label} onClick={() => setPrompt(p.prompt)}
                    className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-full text-xs text-zinc-300">
              {p.label}
            </button>
          ))}
        </div>
      </Card>

      <Card>
        <CardTitle>Prompt</CardTitle>
        <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the sound..."
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
        <label className="block mt-3 text-xs text-zinc-400">Negative prompt</label>
        <textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)}
                  className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
      </Card>

      <Card>
        <CardTitle>Options</CardTitle>
        <div className="space-y-4">
          <div>
            <label className="text-xs text-zinc-400">Backend</label>
            <select value={backend}
                    onChange={(e) => setBackend(e.target.value as "auto" | "audioldm2" | "stable_audio")}
                    className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm">
              <option value="auto">Auto (prefers Stable Audio when installed)</option>
              <option value="audioldm2">AudioLDM2 (fast, ~2GB)</option>
              <option value="stable_audio">Stable Audio Open (24GB VRAM, higher fidelity)</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-zinc-400">Duration: {duration.toFixed(1)}s</label>
            <input type="range" min={1} max={30} step={0.5}
                   value={duration}
                   onChange={(e) => setDuration(Number(e.target.value))}
                   className="w-full mt-1" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-zinc-400">Steps: {steps}</label>
              <input type="range" min={4} max={100} step={1}
                     value={steps}
                     onChange={(e) => setSteps(Number(e.target.value))}
                     className="w-full mt-1" />
            </div>
            <div>
              <label className="text-xs text-zinc-400">Guidance: {guidance.toFixed(1)}</label>
              <input type="range" min={0.5} max={10.0} step={0.1}
                     value={guidance}
                     onChange={(e) => setGuidance(Number(e.target.value))}
                     className="w-full mt-1" />
            </div>
          </div>
          <div>
            <label className="text-xs text-zinc-400">Seed (blank = random)</label>
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)}
                   placeholder="random"
                   className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
          </div>
        </div>
      </Card>

      <Card>
        <CardTitle>Generate</CardTitle>
        <Button onClick={start}
                disabled={!prompt.trim() || running || (status !== null && !status.available)}
                variant="primary">
          {running ? "Generating..." : `Generate ${duration.toFixed(0)}s of audio`}
        </Button>
        {running && (
          <div className="mt-4">
            <div className="h-2 bg-zinc-900 rounded overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="text-xs text-zinc-400 mt-2">{progress}% — {progressMsg}</div>
          </div>
        )}
        {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
        {result && (
          <div className="mt-4 space-y-3">
            <audio src={getDownloadUrl(result)} controls className="w-full" />
            <a href={getDownloadUrl(result)} target="_blank" rel="noreferrer"
               className="inline-flex items-center gap-2 text-sm text-green-400 hover:underline">
              <Download className="w-4 h-4" /> Download WAV
            </a>
          </div>
        )}
      </Card>
    </div>
  );
}
