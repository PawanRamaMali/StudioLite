"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Sparkles, Play, Loader2, Download, Camera, Film, Zap } from "lucide-react";
import { generateText2Video, getJob, getDownloadUrl, downloadBlob } from "@/lib/api";

const SAMPLE_PROMPTS = [
  { label: "Cinematic Nature", prompt: "A majestic eagle soaring through dramatic mountain landscape at golden hour, snow-capped peaks, volumetric clouds, cinematic tracking shot, warm sunlight on detailed feathers, 4K quality" },
  { label: "Ocean Waves", prompt: "Crystal clear turquoise ocean waves rolling onto pristine white sand beach, aerial drone shot slowly descending, palm trees swaying, golden sunset reflections on water, serene atmosphere" },
  { label: "Space Journey", prompt: "Spaceship traveling through colorful nebula in deep space, giant gas planets with swirling storms, stars streaking past at warp speed, epic sci-fi cinematography, lens flares, particle effects" },
  { label: "Forest Magic", prompt: "Enchanted forest path in morning mist, golden sunlight filtering through ancient oak trees, fireflies and magical particles, a deer walking peacefully, fantasy fairy tale atmosphere" },
  { label: "City Night", prompt: "Busy city intersection at night, car light trails forming red and white streams, neon signs flickering, pedestrians in fast motion, wet pavement reflections after rain, timelapse style" },
  { label: "Underwater World", prompt: "Vibrant coral reef teeming with tropical fish, camera gliding through underwater paradise, sunbeams through crystal clear water, sea turtle swimming past, National Geographic documentary style" },
  { label: "Dancing Flames", prompt: "Mesmerizing campfire flames in slow motion against dark night sky, sparks floating upward like stars, warm orange and red colors, close-up macro with shallow depth of field, autumn forest" },
  { label: "Robot Awakening", prompt: "Humanoid robot in futuristic laboratory slowly awakening, LED lights flickering on across chrome body, camera orbiting as it takes first steps, holographic displays, Blade Runner aesthetic" },
];

const STYLES = [
  { id: "none", name: "No Style" },
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
  { id: "3d_render", name: "3D Render" },
  { id: "oil_painting", name: "Oil Painting" },
  { id: "documentary", name: "Documentary" },
];

const CAMERAS = [
  { id: "static", name: "Static (no movement)" },
  { id: "slow_pan_right", name: "Slow Pan Right" },
  { id: "slow_pan_left", name: "Slow Pan Left" },
  { id: "zoom_in", name: "Zoom In" },
  { id: "zoom_out", name: "Zoom Out" },
  { id: "orbit_cw", name: "Orbit Clockwise" },
  { id: "crane_up", name: "Crane Up" },
  { id: "dolly_forward", name: "Dolly Forward" },
  { id: "handheld", name: "Handheld" },
  { id: "steadicam", name: "Steadicam Follow" },
  { id: "tilt_up", name: "Tilt Up" },
  { id: "tilt_down", name: "Tilt Down" },
  { id: "whip_pan", name: "Whip Pan" },
];

const ENGINES = [
  { id: "wan", name: "Wan 2.1", model: "1.3b", badge: "Recommended", desc: "Best quality for 8-12GB GPUs" },
  { id: "ltx", name: "LTX-Video", model: "base", badge: "Fast", desc: "Fastest generation" },
  { id: "cogvideox", name: "CogVideoX", model: "2b", badge: "Versatile", desc: "Good quality, versatile" },
];

export default function GeneratePanel() {
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0].prompt);
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
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setGenerating(true);
    setProgress(0);
    setStatusMsg("Sending request to API...");
    setError(null);
    setResultJobId(null);

    const selectedEngine = ENGINES.find((e) => e.id === engine) || ENGINES[0];

    try {
      const job = await generateText2Video({
        prompt: prompt.trim(),
        engine: selectedEngine.id,
        model: selectedEngine.model,
        num_frames: frames,
        num_inference_steps: steps,
        guidance_scale: guidance,
      });

      setStatusMsg(`Job started: ${job.job_id}`);

      // Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const status = await getJob(job.job_id);
          const pct = Math.round((status.progress || 0) * 100);
          setProgress(pct);
          setStatusMsg(status.message || `Processing... ${pct}%`);

          if (status.status === "completed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setResultJobId(job.job_id);
            setGenerating(false);
            setStatusMsg("Complete!");
          } else if (status.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(status.error || "Generation failed");
            setGenerating(false);
          }
        } catch {
          // Network blip - keep polling
        }
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API. Is the server running on port 8000?");
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

          {/* Sample Prompts */}
          <Card>
            <CardHeader><CardTitle className="text-sm">Quick Start - Sample Prompts</CardTitle></CardHeader>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_PROMPTS.map((sp) => (
                <button
                  key={sp.label}
                  onClick={() => setPrompt(sp.prompt)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                    prompt === sp.prompt
                      ? "bg-indigo-600 text-white"
                      : "bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                  }`}
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </Card>

          {/* Prompt */}
          <Card>
            <label className="block text-sm font-medium text-zinc-300 mb-2">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the video you want to generate..."
              className="w-full h-32 bg-zinc-800/50 border border-zinc-700 rounded-lg px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 resize-none"
            />
            <p className="text-xs text-zinc-500 mt-1">{prompt.length} characters</p>
          </Card>

          {/* Engine Selection */}
          <Card>
            <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Sparkles className="w-4 h-4 text-indigo-400" /> Engine</CardTitle></CardHeader>
            <div className="grid grid-cols-3 gap-3">
              {ENGINES.map((eng) => (
                <button
                  key={eng.id}
                  onClick={() => setEngine(eng.id)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    engine === eng.id
                      ? "border-indigo-500 bg-indigo-500/10 shadow-lg shadow-indigo-500/10"
                      : "border-zinc-700 bg-zinc-800/30 hover:border-zinc-600"
                  }`}
                >
                  <div className="font-medium text-sm">{eng.name}</div>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant={eng.badge === "Recommended" ? "info" : eng.badge === "Fast" ? "success" : "default"}>{eng.badge}</Badge>
                  </div>
                  <p className="text-[10px] text-zinc-500 mt-1">{eng.desc}</p>
                </button>
              ))}
            </div>
          </Card>

          {/* Style & Camera Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Film className="w-3.5 h-3.5 text-indigo-400" /> Visual Style
              </label>
              <select value={style} onChange={(e) => setStyle(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50">
                {STYLES.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
              </select>
            </Card>
            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Camera className="w-3.5 h-3.5 text-indigo-400" /> Camera Movement
              </label>
              <select value={camera} onChange={(e) => setCamera(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50">
                {CAMERAS.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </Card>
            <Card>
              <label className="flex items-center gap-2 text-sm font-medium text-zinc-300 mb-2">
                <Zap className="w-3.5 h-3.5 text-indigo-400" /> Frames
              </label>
              <select value={frames} onChange={(e) => setFrames(Number(e.target.value))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50">
                <option value={33}>33 frames (~1.4s)</option>
                <option value={49}>49 frames (~2s)</option>
                <option value={81}>81 frames (~3.4s)</option>
              </select>
            </Card>
          </div>

          {/* Advanced Settings */}
          <Card>
            <details>
              <summary className="text-sm font-medium text-zinc-300 cursor-pointer select-none hover:text-zinc-100">
                Advanced Settings
              </summary>
              <div className="grid grid-cols-2 gap-6 mt-4">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Inference Steps: <span className="text-indigo-400 font-mono">{steps}</span></label>
                  <input type="range" min={6} max={50} value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="w-full accent-indigo-500" />
                  <div className="flex justify-between text-[10px] text-zinc-600"><span>Fast (6)</span><span>Quality (50)</span></div>
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Guidance Scale: <span className="text-indigo-400 font-mono">{guidance.toFixed(1)}</span></label>
                  <input type="range" min={1} max={10} step={0.5} value={guidance}
                    onChange={(e) => setGuidance(Number(e.target.value))}
                    className="w-full accent-indigo-500" />
                  <div className="flex justify-between text-[10px] text-zinc-600"><span>Creative (1)</span><span>Strict (10)</span></div>
                </div>
              </div>
            </details>
          </Card>

          {/* Generate Button */}
          <Button onClick={handleGenerate} disabled={generating || !prompt.trim()} size="lg" className="w-full">
            {generating ? (
              <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Generating... {progress}%</>
            ) : (
              <><Play className="w-5 h-5 mr-2" /> Generate Video</>
            )}
          </Button>

          {/* Progress Bar */}
          {generating && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-300">{statusMsg}</span>
                <span className="text-sm font-mono text-indigo-400">{progress}%</span>
              </div>
              <div className="h-2.5 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-indigo-600 via-purple-500 to-pink-500 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${Math.max(progress, 2)}%` }} />
              </div>
            </Card>
          )}

          {/* Error */}
          {error && (
            <Card className="border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm font-medium">Generation Failed</p>
              <p className="text-red-400/70 text-xs mt-1">{error}</p>
            </Card>
          )}
        </div>

        {/* Right: Result & Tips */}
        <div className="space-y-4">
          <Card className="min-h-[300px] flex flex-col items-center justify-center">
            {resultJobId ? (
              <div className="w-full">
                <video src={getDownloadUrl(resultJobId)} controls autoPlay loop muted
                  className="w-full rounded-lg border border-zinc-700" />
                <Button variant="secondary" className="w-full mt-3" onClick={() => {
                  downloadBlob(getDownloadUrl(resultJobId), `${resultJobId}.mp4`).catch(() => {});
                }}>
                  <Download className="w-4 h-4 mr-2" /> Download MP4
                </Button>
                <p className="text-xs text-zinc-500 mt-2 text-center">Job: {resultJobId}</p>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-16 h-16 rounded-2xl bg-zinc-800 flex items-center justify-center mx-auto mb-4">
                  <Play className="w-8 h-8 text-zinc-600" />
                </div>
                <p className="text-zinc-400 text-sm font-medium">No video yet</p>
                <p className="text-zinc-600 text-xs mt-1">Pick a prompt and click Generate</p>
              </div>
            )}
          </Card>

          {/* Tips */}
          <Card>
            <h4 className="text-sm font-medium text-zinc-300 mb-3">Tips for Better Results</h4>
            <ul className="text-xs text-zinc-500 space-y-2">
              <li className="flex gap-2"><span className="text-indigo-400">1.</span> Be specific about camera movement and lighting</li>
              <li className="flex gap-2"><span className="text-indigo-400">2.</span> Include style keywords: "cinematic, 4K, dramatic"</li>
              <li className="flex gap-2"><span className="text-indigo-400">3.</span> Wan 1.3B works best for 8-12GB GPUs</li>
              <li className="flex gap-2"><span className="text-indigo-400">4.</span> More frames = longer video but slower generation</li>
              <li className="flex gap-2"><span className="text-indigo-400">5.</span> Use Style presets to enhance your prompts automatically</li>
            </ul>
          </Card>

          {/* Config summary */}
          <Card className="bg-zinc-900/30">
            <h4 className="text-xs font-medium text-zinc-500 mb-2 uppercase tracking-wide">Current Config</h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between"><span className="text-zinc-500">Engine</span><span className="text-zinc-300">{ENGINES.find(e => e.id === engine)?.name}</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Style</span><span className="text-zinc-300">{STYLES.find(s => s.id === style)?.name}</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Camera</span><span className="text-zinc-300">{CAMERAS.find(c => c.id === camera)?.name}</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Frames</span><span className="text-zinc-300">{frames} (~{(frames/24).toFixed(1)}s)</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Steps</span><span className="text-zinc-300">{steps}</span></div>
              <div className="flex justify-between"><span className="text-zinc-500">Guidance</span><span className="text-zinc-300">{guidance.toFixed(1)}</span></div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
