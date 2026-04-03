"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { BookOpen, Plus, Trash2, GripVertical, Sparkles, Play, Loader2, ChevronDown, ChevronUp, Copy, Download } from "lucide-react";
import { generateStory, getJob, getDownloadUrl } from "@/lib/api";

interface Scene {
  id: string;
  title: string;
  visual: string;
  narration: string;
  duration: number;
}

const TRANSITIONS = [
  { id: "cut", name: "Hard Cut" },
  { id: "crossfade", name: "Crossfade" },
  { id: "fade_to_black", name: "Fade to Black" },
  { id: "fade_to_white", name: "Fade to White" },
  { id: "wipe_left", name: "Wipe Left" },
  { id: "wipe_right", name: "Wipe Right" },
  { id: "zoom", name: "Zoom" },
  { id: "glitch", name: "Glitch" },
];

const GENRES = ["Cinematic", "Sci-Fi", "Fantasy", "Horror", "Comedy", "Documentary", "Action", "Romance", "Mystery"];
const MOODS = ["Epic", "Calm", "Tense", "Mysterious", "Joyful", "Melancholic", "Energetic", "Dreamy"];

const STORY_IDEAS = [
  { label: "Space Adventure", concept: "A lone astronaut discovers alien life on Mars during a sandstorm, must decide whether to make contact or flee" },
  { label: "Dog Story", concept: "A mischievous husky left home alone destroys the living room, steals food, gets tangled in curtains, then acts innocent when owner returns" },
  { label: "Ocean Mystery", concept: "Deep sea divers discover an ancient underwater temple glowing with bioluminescent light, revealing secrets of a lost civilization" },
  { label: "Time Travel", concept: "A scientist accidentally sends herself 100 years into the future, finding Earth transformed into a lush paradise run by AI" },
  { label: "Mountain Journey", concept: "A golden retriever puppy's first hiking trip through snowy mountains, discovering streams, wildlife, and the summit at sunset" },
  { label: "Cyberpunk Chase", concept: "A hacker on the run through neon-lit streets, chased by drones, ducking through holographic markets and rain-soaked alleyways" },
];

export default function StoryPanel() {
  const [concept, setConcept] = useState("");
  const [genre, setGenre] = useState("Cinematic");
  const [mood, setMood] = useState("Epic");
  const [numScenes, setNumScenes] = useState(4);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [transition, setTransition] = useState("crossfade");
  const [narrationEnabled, setNarrationEnabled] = useState(true);
  const [musicEnabled, setMusicEnabled] = useState(false);

  // Loading states
  const [scriptLoading, setScriptLoading] = useState(false);
  const [scriptStatus, setScriptStatus] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStatus, setGenStatus] = useState("");
  const [resultJobId, setResultJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [expandedScene, setExpandedScene] = useState<number | null>(0);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleGenerateScript = async () => {
    if (!concept.trim()) return;
    setScriptLoading(true);
    setScriptStatus("Connecting to LLM...");
    setError(null);

    try {
      setScriptStatus("Sending to LLM... This may take 15-30 seconds");

      const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${API}/api/v1/generate/script`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concept, num_scenes: numScenes, genre, mood }),
      });

      if (response.ok) {
        const data = await response.json();
        const apiScenes = data.scenes || [];

        if (apiScenes.length > 0) {
          const newScenes: Scene[] = apiScenes.map((s: Record<string, unknown>, i: number) => ({
            id: crypto.randomUUID(),
            title: (s.title as string) || `Scene ${i + 1}`,
            visual: (s.visual as string) || "",
            narration: (s.narration as string) || "",
            duration: (s.duration as number) || 5,
          }));
          setScenes(newScenes);
          setExpandedScene(0);
          setScriptStatus("");
        } else {
          throw new Error("LLM returned empty scenes");
        }
      } else {
        const errData = await response.json().catch(() => ({ detail: "API error" }));
        throw new Error(errData.detail || `API returned ${response.status}`);
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Unknown error";
      setError(`Script generation failed: ${errMsg}`);

      // Fallback: generate simple scenes locally
      setScriptStatus("Falling back to local scene generation...");
      await new Promise((r) => setTimeout(r, 300));

      const generatedScenes: Scene[] = [];
      for (let i = 0; i < numScenes; i++) {
        generatedScenes.push({
          id: crypto.randomUUID(),
          title: `Scene ${i + 1}`,
          visual: `${concept}, scene ${i + 1} of ${numScenes}, ${genre.toLowerCase()} genre, ${mood.toLowerCase()} atmosphere, cinematic, high quality`,
          narration: "",
          duration: 5,
        });
      }
      setScenes(generatedScenes);
      setExpandedScene(0);
    } finally {
      setScriptLoading(false);
      setScriptStatus("");
    }
  };

  const handleGenerateMovie = async () => {
    if (scenes.length < 2) return;
    setGenerating(true);
    setGenProgress(0);
    setGenStatus("Starting movie generation...");
    setError(null);
    setResultJobId(null);

    try {
      const job = await generateStory({
        concept,
        num_scenes: scenes.length,
        genre,
        mood,
        engine: "wan",
        model: "1.3b",
        enable_narration: narrationEnabled,
        continuity_mode: "Prompt Anchoring",
        transition_type: transition,
      });

      setGenStatus(`Job ${job.job_id} started...`);

      pollRef.current = setInterval(async () => {
        try {
          const status = await getJob(job.job_id);
          const pct = Math.round((status.progress || 0) * 100);
          setGenProgress(pct);
          setGenStatus(status.message || `Processing... ${pct}%`);

          if (status.status === "completed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setResultJobId(job.job_id);
            setGenerating(false);
            setGenStatus("Movie complete!");
          } else if (status.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(status.error || "Generation failed");
            setGenerating(false);
          }
        } catch { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
      setGenerating(false);
    }
  };

  const addScene = () => {
    setScenes([...scenes, {
      id: crypto.randomUUID(),
      title: `Scene ${scenes.length + 1}`,
      visual: "",
      narration: "",
      duration: 5,
    }]);
  };

  const removeScene = (idx: number) => setScenes(scenes.filter((_, i) => i !== idx));

  const updateScene = (idx: number, field: keyof Scene, value: string | number) => {
    const updated = [...scenes];
    updated[idx] = { ...updated[idx], [field]: value };
    setScenes(updated);
  };

  const moveScene = (idx: number, dir: -1 | 1) => {
    const newIdx = idx + dir;
    if (newIdx < 0 || newIdx >= scenes.length) return;
    const updated = [...scenes];
    [updated[idx], updated[newIdx]] = [updated[newIdx], updated[idx]];
    setScenes(updated);
  };

  const totalDuration = scenes.reduce((sum, s) => sum + s.duration, 0);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Story Mode</h1>
        <p className="text-zinc-400 mt-1">Create multi-scene AI movies with narration, music, and transitions</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Planning */}
        <div className="space-y-5">
          <Card>
            <CardTitle className="text-sm mb-3">1. Plan Your Story</CardTitle>

            {/* Story idea pills */}
            <p className="text-xs text-zinc-500 mb-2">Quick ideas:</p>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {STORY_IDEAS.map((idea) => (
                <button
                  key={idea.label}
                  onClick={() => setConcept(idea.concept)}
                  className={`px-2.5 py-1 rounded-full text-[10px] font-medium transition-all ${
                    concept === idea.concept
                      ? "bg-indigo-600 text-white"
                      : "bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                  }`}
                >
                  {idea.label}
                </button>
              ))}
            </div>

            <textarea
              value={concept}
              onChange={(e) => setConcept(e.target.value)}
              placeholder="Describe your movie idea in detail..."
              className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none"
            />
            <div className="grid grid-cols-2 gap-2 mt-3">
              <div>
                <label className="text-[10px] text-zinc-500">Genre</label>
                <select value={genre} onChange={(e) => setGenre(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                  {GENRES.map((g) => <option key={g}>{g}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] text-zinc-500">Mood</label>
                <select value={mood} onChange={(e) => setMood(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                  {MOODS.map((m) => <option key={m}>{m}</option>)}
                </select>
              </div>
            </div>
            <div className="mt-3">
              <label className="text-[10px] text-zinc-500">Scenes: {numScenes}</label>
              <input type="range" min={2} max={8} value={numScenes}
                onChange={(e) => setNumScenes(Number(e.target.value))}
                className="w-full accent-indigo-500" />
            </div>

            <Button className="w-full mt-3" onClick={handleGenerateScript}
              disabled={!concept.trim() || scriptLoading}>
              {scriptLoading ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...</>
              ) : (
                <><Sparkles className="w-4 h-4 mr-2" /> AI Generate Script</>
              )}
            </Button>
            {scriptStatus && (
              <div className="mt-2 flex items-center gap-2 text-xs text-indigo-400">
                <Loader2 className="w-3 h-3 animate-spin flex-shrink-0" />
                <span>{scriptStatus}</span>
              </div>
            )}
          </Card>

          {/* Settings */}
          <Card>
            <CardTitle className="text-sm mb-3">Settings</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Transition</label>
                <select value={transition} onChange={(e) => setTransition(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-1">
                  {TRANSITIONS.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                </select>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={narrationEnabled} onChange={(e) => setNarrationEnabled(e.target.checked)}
                  className="accent-indigo-500 w-3.5 h-3.5" />
                <span className="text-xs text-zinc-300">Enable Narration</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={musicEnabled} onChange={(e) => setMusicEnabled(e.target.checked)}
                  className="accent-indigo-500 w-3.5 h-3.5" />
                <span className="text-xs text-zinc-300">Background Music</span>
              </label>
            </div>
          </Card>
        </div>

        {/* Center + Right: Storyboard */}
        <div className="lg:col-span-2 space-y-4">
          {/* Timeline bar */}
          {scenes.length > 0 && (
            <Card className="py-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-400 font-medium">{scenes.length} scenes</span>
                <span className="text-xs text-zinc-400">Total: {totalDuration}s</span>
              </div>
              <div className="flex gap-0.5 h-2.5 rounded-full overflow-hidden">
                {scenes.map((s, i) => {
                  const pct = (s.duration / Math.max(totalDuration, 1)) * 100;
                  const colors = ["bg-red-400", "bg-teal-400", "bg-blue-400", "bg-green-400", "bg-yellow-400", "bg-purple-400", "bg-pink-400", "bg-orange-400"];
                  return <div key={s.id} className={`${colors[i % colors.length]} transition-all rounded-sm`} style={{ width: `${pct}%` }} title={`${s.title} (${s.duration}s)`} />;
                })}
              </div>
            </Card>
          )}

          {/* Scene Cards */}
          {scenes.length === 0 && !scriptLoading && (
            <Card className="min-h-[200px] flex items-center justify-center">
              <div className="text-center">
                <BookOpen className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-400 text-sm">No scenes yet</p>
                <p className="text-zinc-600 text-xs mt-1">Pick a story idea and click "AI Generate Script", or add scenes manually</p>
              </div>
            </Card>
          )}

          {scriptLoading && scenes.length === 0 && (
            <Card className="min-h-[200px] flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-10 h-10 text-indigo-400 mx-auto mb-3 animate-spin" />
                <p className="text-indigo-400 text-sm font-medium">Generating your story...</p>
                <p className="text-zinc-500 text-xs mt-1">{scriptStatus || "This may take 15-30 seconds"}</p>
              </div>
            </Card>
          )}

          <div className="space-y-3">
            {scenes.map((scene, i) => {
              const isExpanded = expandedScene === i;
              return (
                <Card key={scene.id} className="relative group">
                  <div className="flex items-center gap-3 cursor-pointer select-none" onClick={() => setExpandedScene(isExpanded ? null : i)}>
                    <GripVertical className="w-4 h-4 text-zinc-600" />
                    <Badge variant="info" className="text-[10px]">{i + 1}</Badge>
                    <input
                      value={scene.title}
                      onChange={(e) => updateScene(i, "title", e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      className="flex-1 bg-transparent border-none text-sm font-medium text-zinc-200 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 rounded px-1"
                    />
                    <span className="text-xs text-zinc-500 tabular-nums">{scene.duration}s</span>
                    <div className="flex gap-0.5">
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, -1); }} className="p-1 text-zinc-600 hover:text-zinc-300 transition-colors" title="Move up"><ChevronUp className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, 1); }} className="p-1 text-zinc-600 hover:text-zinc-300 transition-colors" title="Move down"><ChevronDown className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); removeScene(i); }} className="p-1 text-zinc-600 hover:text-red-400 transition-colors" title="Delete"><Trash2 className="w-3.5 h-3.5" /></button>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 pt-3 border-t border-zinc-800">
                      <div>
                        <label className="text-xs text-zinc-400 font-medium">Visual Prompt</label>
                        <textarea
                          value={scene.visual}
                          onChange={(e) => updateScene(i, "visual", e.target.value)}
                          placeholder="Describe the visual scene..."
                          className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-zinc-400 font-medium">Narration</label>
                        <textarea
                          value={scene.narration}
                          onChange={(e) => updateScene(i, "narration", e.target.value)}
                          placeholder="Voiceover text..."
                          className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1"
                        />
                      </div>
                      <div className="md:col-span-2">
                        <label className="text-xs text-zinc-400">Duration: <span className="text-indigo-400 font-mono">{scene.duration}s</span></label>
                        <input type="range" min={2} max={12} value={scene.duration}
                          onChange={(e) => updateScene(i, "duration", Number(e.target.value))}
                          className="w-full mt-1 accent-indigo-500" />
                      </div>
                    </div>
                  )}

                  {i < scenes.length - 1 && transition !== "cut" && (
                    <div className="absolute -bottom-2.5 left-1/2 -translate-x-1/2 z-10">
                      <Badge className="text-[9px] bg-zinc-800 border border-zinc-700">{transition.replace(/_/g, " ")}</Badge>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          {/* Add Scene button */}
          {scenes.length > 0 && (
            <button onClick={addScene}
              className="w-full border-2 border-dashed border-zinc-700 rounded-xl py-3 text-sm text-zinc-500 hover:text-zinc-300 hover:border-zinc-500 transition-all flex items-center justify-center gap-2">
              <Plus className="w-4 h-4" /> Add Scene
            </button>
          )}

          {/* Generate Movie */}
          {scenes.length >= 2 && (
            <>
              <Button size="lg" className="w-full" onClick={handleGenerateMovie} disabled={generating}>
                {generating ? (
                  <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Generating Movie... {genProgress}%</>
                ) : (
                  <><Play className="w-5 h-5 mr-2" /> Generate Movie ({scenes.length} scenes)</>
                )}
              </Button>

              {generating && (
                <Card>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-zinc-300">{genStatus}</span>
                    <span className="text-xs font-mono text-indigo-400">{genProgress}%</span>
                  </div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-indigo-600 via-purple-500 to-pink-500 rounded-full transition-all duration-700"
                      style={{ width: `${Math.max(genProgress, 2)}%` }} />
                  </div>
                </Card>
              )}
            </>
          )}

          {/* Result */}
          {resultJobId && (
            <Card className="border-green-500/20 bg-green-500/5">
              <p className="text-green-400 text-sm font-medium mb-3">Movie Generated!</p>
              <video src={getDownloadUrl(resultJobId)} controls autoPlay loop muted
                className="w-full rounded-lg border border-zinc-700" />
              <a href={getDownloadUrl(resultJobId)} download className="block mt-3">
                <Button variant="secondary" className="w-full">
                  <Download className="w-4 h-4 mr-2" /> Download Movie
                </Button>
              </a>
            </Card>
          )}

          {/* Error */}
          {error && (
            <Card className="border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm">{error}</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
