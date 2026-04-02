"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Sparkles, Play, Loader2, Download, Wand2, Camera, Sun, Film } from "lucide-react";
import { generateText2Video, getJob, getDownloadUrl } from "@/lib/api";

const STYLES = [
  { id: "none", name: "None" },
  { id: "cinematic_film", name: "Cinematic Film" },
  { id: "anime", name: "Anime" },
  { id: "photorealistic", name: "Photorealistic" },
  { id: "film_noir", name: "Film Noir" },
  { id: "dreamlike", name: "Dreamlike" },
  { id: "cyberpunk", name: "Cyberpunk" },
  { id: "watercolor", name: "Watercolor" },
  { id: "fantasy_epic", name: "Fantasy Epic" },
  { id: "vintage_80s", name: "Vintage 80s" },
  { id: "horror", name: "Horror" },
  { id: "stop_motion", name: "Stop Motion" },
];

const CAMERAS = [
  { id: "static", name: "Static" },
  { id: "slow_pan_right", name: "Pan Right" },
  { id: "slow_pan_left", name: "Pan Left" },
  { id: "zoom_in", name: "Zoom In" },
  { id: "zoom_out", name: "Zoom Out" },
  { id: "orbit_cw", name: "Orbit" },
  { id: "crane_up", name: "Crane Up" },
  { id: "dolly_forward", name: "Dolly In" },
  { id: "handheld", name: "Handheld" },
  { id: "steadicam", name: "Steadicam" },
];

const ENGINES = [
  { id: "wan", name: "Wan 2.1", badge: "Recommended", vram: "8GB+" },
  { id: "ltx", name: "LTX-Video", badge: "Fast", vram: "8GB+" },
  { id: "cogvideox", name: "CogVideoX", badge: "Versatile", vram: "8GB+" },
];

export default function GeneratePanel() {
  const [prompt, setPrompt] = useState("");
  const [engine, setEngine] = useState("wan");
  const [style, setStyle] = useState("none");
  const [camera, setCamera] = useState("static");
  const [steps, setSteps] = useState(30);
  const [frames, setFrames] = useState(49);
  const [guidance, setGuidance] = useState(5.0);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [resultJobId, setResultJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setGenerating(true);
    setProgress(0);
    setStatusMsg("Starting generation...");
    setError(null);
    setResultJobId(null);

    try {
      const job = await generateText2Video({
        prompt,
        engine,
        model: "1.3b",
        num_frames: frames,
        num_inference_steps: steps,
        guidance_scale: guidance,
        style_preset: style !== "none" ? style : undefined,
        camera_movement: camera !== "static" ? camera : undefined,
      });

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJob(job.job_id);
          setProgress(status.progress * 100);
          setStatusMsg(status.message || `${(status.progress * 100).toFixed(0)}%`);

          if (status.status === "completed") {
            clearInterval(pollInterval);
            setResultJobId(job.job_id);
            setGenerating(false);
          } else if (status.status === "failed") {
            clearInterval(pollInterval);
            setError(status.error || "Generation failed");
            setGenerating(false);
          }
        } catch {
          // Network error during poll, keep trying
        }
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start generation");
      setGenerating(false);
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Generator</h1>
        <p className="text-zinc-400 mt-1">Generate videos from text using state-of-the-art diffusion models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Controls */}
        <div className="lg:col-span-2 space-y-5">
          {/* Prompt */}
          <Card>
            <label className="block text-sm font-medium text-zinc-300 mb-2">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A majestic eagle soaring through a dramatic mountain landscape at golden hour, with snow-capped peaks in the background, cinematic camera tracking shot..."
              className="w-full h-32 bg-zinc-800/50 border border-zinc-700 rounded-lg px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 resize-none"
            />
          </Card>

          {/* Engine Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400" /> Engine
              </CardTitle>
            </CardHeader>
            <div className="grid grid-cols-3 gap-3">
              {ENGINES.map((eng) => (
                <button
                  key={eng.id}
                  onClick={() => setEngine(eng.id)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    engine === eng.id
                      ? "border-indigo-500 bg-indigo-500/10"
                      : "border-zinc-700 bg-zinc-800/30 hover:border-zinc-600"
                  }`}
                >
                  <div className="font-medium text-sm">{eng.name}</div>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant={eng.badge === "Recommended" ? "info" : "default"}>{eng.badge}</Badge>
                    <span className="text-[10px] text-zinc-500">{eng.vram}</span>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {/* Style & Camera */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Film className="w-3.5 h-3.5 text-indigo-400" /> Style
              </label>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50"
              >
                {STYLES.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </Card>

            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Camera className="w-3.5 h-3.5 text-indigo-400" /> Camera
              </label>
              <select
                value={camera}
                onChange={(e) => setCamera(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50"
              >
                {CAMERAS.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </Card>

            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Sun className="w-3.5 h-3.5 text-indigo-400" /> Frames
              </label>
              <select
                value={frames}
                onChange={(e) => setFrames(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50"
              >
                <option value={33}>33 (~1.4s)</option>
                <option value={49}>49 (~2s)</option>
                <option value={81}>81 (~3.4s)</option>
              </select>
            </Card>
          </div>

          {/* Advanced Settings */}
          <Card>
            <details>
              <summary className="text-sm font-medium text-zinc-300 cursor-pointer select-none">
                Advanced Settings
              </summary>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <label className="text-xs text-zinc-400">Inference Steps: {steps}</label>
                  <input
                    type="range" min={6} max={50} value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400">Guidance Scale: {guidance.toFixed(1)}</label>
                  <input
                    type="range" min={1} max={10} step={0.5} value={guidance}
                    onChange={(e) => setGuidance(Number(e.target.value))}
                    className="w-full mt-1 accent-indigo-500"
                  />
                </div>
              </div>
            </details>
          </Card>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={generating || !prompt.trim()}
            size="lg"
            className="w-full"
          >
            {generating ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating... {progress.toFixed(0)}%
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Generate Video
              </>
            )}
          </Button>

          {/* Progress */}
          {generating && (
            <Card className="overflow-hidden">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-300">{statusMsg}</span>
                <span className="text-sm font-mono text-indigo-400">{progress.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </Card>
          )}

          {error && (
            <Card className="border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm">{error}</p>
            </Card>
          )}
        </div>

        {/* Right: Result */}
        <div className="space-y-4">
          <Card className="min-h-[300px] flex flex-col items-center justify-center">
            {resultJobId ? (
              <div className="w-full">
                <video
                  src={getDownloadUrl(resultJobId)}
                  controls
                  autoPlay
                  loop
                  className="w-full rounded-lg"
                />
                <div className="mt-4 flex gap-2">
                  <a
                    href={getDownloadUrl(resultJobId)}
                    download
                    className="flex-1"
                  >
                    <Button variant="secondary" className="w-full">
                      <Download className="w-4 h-4 mr-2" /> Download
                    </Button>
                  </a>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Video className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Your generated video will appear here</p>
              </div>
            )}
          </Card>

          {/* Quick tips */}
          <Card>
            <h4 className="text-sm font-medium text-zinc-300 mb-2">Tips</h4>
            <ul className="text-xs text-zinc-500 space-y-1.5">
              <li>Be specific about camera movement and lighting</li>
              <li>Include style keywords like "cinematic, 4K, dramatic"</li>
              <li>Wan 1.3B works best for 8-12GB GPUs</li>
              <li>More frames = longer video but slower generation</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Video(props: React.SVGAttributes<SVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" />
      <rect x="2" y="6" width="14" height="12" rx="2" />
    </svg>
  );
}
