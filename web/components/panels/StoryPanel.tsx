"use client";

import { useState } from "react";
import { Card, CardTitle, CardDescription } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { BookOpen, Plus, Trash2, GripVertical, Sparkles, Play, Loader2, ChevronDown, ChevronUp } from "lucide-react";

interface Scene {
  id: string;
  title: string;
  visual: string;
  narration: string;
  duration: number;
}

const TRANSITIONS = [
  "cut", "crossfade", "fade_to_black", "fade_to_white",
  "wipe_left", "wipe_right", "zoom", "glitch",
];

const GENRES = ["Cinematic", "Sci-Fi", "Fantasy", "Horror", "Comedy", "Documentary", "Action", "Romance", "Mystery"];
const MOODS = ["Epic", "Calm", "Tense", "Mysterious", "Joyful", "Melancholic", "Energetic", "Dreamy"];

export default function StoryPanel() {
  const [concept, setConcept] = useState("");
  const [genre, setGenre] = useState("Cinematic");
  const [mood, setMood] = useState("Epic");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [transition, setTransition] = useState("crossfade");
  const [generating, setGenerating] = useState(false);
  const [expandedScene, setExpandedScene] = useState<number | null>(0);

  const addScene = () => {
    setScenes([...scenes, {
      id: crypto.randomUUID(),
      title: `Scene ${scenes.length + 1}`,
      visual: "",
      narration: "",
      duration: 5,
    }]);
  };

  const removeScene = (idx: number) => {
    setScenes(scenes.filter((_, i) => i !== idx));
  };

  const updateScene = (idx: number, field: keyof Scene, value: string | number) => {
    const updated = [...scenes];
    (updated[idx] as unknown as Record<string, unknown>)[field] = value;
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
        {/* Left: Story Planning */}
        <div className="space-y-5">
          <Card>
            <CardTitle className="text-sm mb-3">1. Plan Your Story</CardTitle>
            <textarea
              value={concept}
              onChange={(e) => setConcept(e.target.value)}
              placeholder="Describe your movie idea..."
              className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none"
            />
            <div className="grid grid-cols-2 gap-2 mt-3">
              <select value={genre} onChange={(e) => setGenre(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200">
                {GENRES.map((g) => <option key={g}>{g}</option>)}
              </select>
              <select value={mood} onChange={(e) => setMood(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200">
                {MOODS.map((m) => <option key={m}>{m}</option>)}
              </select>
            </div>
            <Button className="w-full mt-3" disabled={!concept.trim()}>
              <Sparkles className="w-4 h-4 mr-2" /> AI Generate Script
            </Button>
          </Card>

          {/* Settings */}
          <Card>
            <CardTitle className="text-sm mb-3">Settings</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Transition</label>
                <select value={transition} onChange={(e) => setTransition(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-1">
                  {TRANSITIONS.map((t) => (
                    <option key={t} value={t}>{t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <input type="checkbox" id="narration" className="accent-indigo-500" defaultChecked />
                <label htmlFor="narration" className="text-xs text-zinc-300">Enable Narration</label>
              </div>
              <div className="flex items-center gap-2">
                <input type="checkbox" id="music" className="accent-indigo-500" />
                <label htmlFor="music" className="text-xs text-zinc-300">Background Music</label>
              </div>
            </div>
          </Card>
        </div>

        {/* Center + Right: Storyboard */}
        <div className="lg:col-span-2 space-y-4">
          {/* Timeline bar */}
          {scenes.length > 0 && (
            <Card className="py-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-400">{scenes.length} scenes</span>
                <span className="text-xs text-zinc-400">Total: {totalDuration}s</span>
              </div>
              <div className="flex gap-0.5 h-2 rounded-full overflow-hidden">
                {scenes.map((s, i) => {
                  const pct = (s.duration / Math.max(totalDuration, 1)) * 100;
                  const colors = ["bg-red-400", "bg-teal-400", "bg-blue-400", "bg-green-400", "bg-yellow-400", "bg-purple-400", "bg-pink-400", "bg-orange-400"];
                  return <div key={s.id} className={`${colors[i % colors.length]} transition-all`} style={{ width: `${pct}%` }} title={`${s.title} (${s.duration}s)`} />;
                })}
              </div>
            </Card>
          )}

          {/* Scene Cards */}
          <div className="space-y-3">
            {scenes.map((scene, i) => {
              const isExpanded = expandedScene === i;
              return (
                <Card key={scene.id} className="relative group">
                  {/* Header */}
                  <div className="flex items-center gap-3 cursor-pointer" onClick={() => setExpandedScene(isExpanded ? null : i)}>
                    <GripVertical className="w-4 h-4 text-zinc-600" />
                    <Badge variant="info">{i + 1}</Badge>
                    <input
                      value={scene.title}
                      onChange={(e) => updateScene(i, "title", e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      className="flex-1 bg-transparent border-none text-sm font-medium text-zinc-200 focus:outline-none"
                    />
                    <span className="text-xs text-zinc-500">{scene.duration}s</span>
                    <div className="flex gap-1">
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, -1); }} className="p-1 text-zinc-500 hover:text-zinc-300"><ChevronUp className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, 1); }} className="p-1 text-zinc-500 hover:text-zinc-300"><ChevronDown className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); removeScene(i); }} className="p-1 text-zinc-500 hover:text-red-400"><Trash2 className="w-3.5 h-3.5" /></button>
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-zinc-400">Visual Prompt</label>
                        <textarea
                          value={scene.visual}
                          onChange={(e) => updateScene(i, "visual", e.target.value)}
                          placeholder="Describe the visual scene for video generation..."
                          className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-zinc-400">Narration</label>
                        <textarea
                          value={scene.narration}
                          onChange={(e) => updateScene(i, "narration", e.target.value)}
                          placeholder="Voiceover text for this scene..."
                          className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-zinc-400">Duration: {scene.duration}s</label>
                        <input type="range" min={2} max={12} value={scene.duration}
                          onChange={(e) => updateScene(i, "duration", Number(e.target.value))}
                          className="w-full mt-1 accent-indigo-500" />
                      </div>
                    </div>
                  )}

                  {/* Transition indicator between scenes */}
                  {i < scenes.length - 1 && (
                    <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 z-10">
                      <Badge variant="default" className="text-[10px]">{transition.replace(/_/g, " ")}</Badge>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          {/* Add Scene */}
          <button
            onClick={addScene}
            className="w-full border-2 border-dashed border-zinc-700 rounded-xl py-4 text-sm text-zinc-500 hover:text-zinc-300 hover:border-zinc-500 transition-all flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" /> Add Scene
          </button>

          {/* Generate */}
          {scenes.length >= 2 && (
            <Button size="lg" className="w-full" disabled={generating}>
              {generating ? (
                <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Generating Movie...</>
              ) : (
                <><Play className="w-5 h-5 mr-2" /> Generate Movie ({scenes.length} scenes)</>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
