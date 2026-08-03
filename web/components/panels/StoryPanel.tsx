"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  BookOpen, Plus, Trash2, GripVertical, Sparkles, Play, Loader2,
  ChevronDown, ChevronUp, Download, Settings2, Users, Link,
  UserCircle, Upload, Shield, ImageIcon, X, Copy, Check, Eye,
  Pause, SkipForward, SkipBack, Volume2, FileText, Film,
} from "lucide-react";
import { generateStory, getJob, getDownloadUrl, getPortraitUrl, downloadBlob } from "@/lib/api";

interface Scene {
  id: string;
  title: string;
  visual: string;
  narration: string;
  duration: number;
  characters: string[];  // Character names assigned to this scene
}

interface StoryCharacter {
  id: string;
  name: string;
  role: string;
  appearance: string;
  refImageName?: string;
  refImageUploaded?: boolean;
}

const TRANSITIONS = [
  { id: "cut", name: "Hard Cut" }, { id: "crossfade", name: "Crossfade" },
  { id: "fade_to_black", name: "Fade to Black" }, { id: "fade_to_white", name: "Fade to White" },
  { id: "wipe_left", name: "Wipe Left" }, { id: "wipe_right", name: "Wipe Right" },
  { id: "zoom", name: "Zoom" }, { id: "glitch", name: "Glitch" },
];

const GENRES = ["Cinematic", "Sci-Fi", "Fantasy", "Horror", "Comedy", "Documentary", "Action", "Romance", "Mystery"];
const MOODS = ["Epic", "Calm", "Tense", "Mysterious", "Joyful", "Melancholic", "Energetic", "Dreamy"];
const ENGINES = [
  { id: "wan", model: "1.3b", name: "Wan 2.1 (1.3B)", desc: "8GB+, best quality/VRAM" },
  { id: "ltx", model: "base", name: "LTX-Video", desc: "Fast generation" },
  { id: "cogvideox", model: "2b", name: "CogVideoX (2B)", desc: "Versatile" },
];
const CONTINUITY_MODES = [
  { id: "none", name: "None", desc: "Scenes generated independently" },
  { id: "prompt", name: "Prompt Anchoring", desc: "Visual identity injected into every scene" },
  { id: "both", name: "Prompt + Shared Seed", desc: "Maximum consistency" },
];

const STORY_IDEAS = [
  { label: "Space Adventure", concept: "A lone astronaut discovers alien life on Mars during a sandstorm, must decide whether to make contact or flee" },
  { label: "Dog Story", concept: "A mischievous husky left home alone destroys the living room, steals food, gets tangled in curtains, then acts innocent when owner returns" },
  { label: "Ocean Mystery", concept: "Deep sea divers discover an ancient underwater temple glowing with bioluminescent light, revealing secrets of a lost civilization" },
  { label: "Time Travel", concept: "A scientist accidentally sends herself 100 years into the future, finding Earth transformed into a lush paradise run by AI" },
  { label: "Mountain Journey", concept: "A golden retriever puppy's first hiking trip through snowy mountains, discovering streams, wildlife, and the summit at sunset" },
  { label: "Cyberpunk Chase", concept: "A hacker on the run through neon-lit streets, chased by drones, ducking through holographic markets and rain-soaked alleyways" },
];

const CHARACTER_TEMPLATES = [
  { name: "Detective Sarah", role: "Lead", appearance: "Woman in her 30s, fiery red curly hair, sharp green eyes, freckles, beige trench coat over dark turtleneck, confident posture" },
  { name: "Robot ARIA", role: "Companion", appearance: "Sleek humanoid robot, polished white and silver chassis, glowing blue LED eyes, smooth rounded features, subtle blue glow from joints" },
  { name: "Captain Wolf", role: "Lead", appearance: "Grizzled man in his 50s, salt-and-pepper beard, one cybernetic red eye, worn leather space captain jacket with mission patches, battle scars" },
  { name: "Luna the Cat", role: "Supporting", appearance: "Sleek black cat, luminous golden eyes, crescent moon marking on forehead, mystical purple aura, elegant and mysterious" },
  { name: "Max the Husky", role: "Lead", appearance: "Young Siberian husky, thick black and white fur, bright blue eyes, playful expression, pink tongue out, bushy curled tail" },
  { name: "Dr. Elara", role: "Lead", appearance: "Asian woman mid-40s, silver-streaked black hair in bun, round glasses, white lab coat, kind warm expression, always carrying a tablet" },
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

  // Video settings
  const [engine, setEngine] = useState("wan");
  const [resolution, setResolution] = useState("480p");
  const [frames, setFrames] = useState(49);
  const [fps, setFps] = useState(16);

  // Character consistency
  const [continuityMode, setContinuityMode] = useState("prompt");
  const [visualAnchor, setVisualAnchor] = useState("");

  // IP-Adapter character consistency
  const [ipAdapterEnabled, setIpAdapterEnabled] = useState(true);
  const [ipAdapterStrength, setIpAdapterStrength] = useState(0.6);
  const [registeredPortraits, setRegisteredPortraits] = useState<{filename: string; url: string; view: string}[]>([]);

  // Fetch available video engines + lip-sync status from backend
  useEffect(() => {
    const fetchEngines = async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${API_BASE}/api/v1/system/engines`);
        if (res.ok) {
          const data = await res.json();
          setAvailableEngines(data.video_engines || []);
          setLipSyncAvailable(!!data.lip_sync?.available);
        }
      } catch {
        // leave defaults — selector will show "auto" only
      }
    };
    fetchEngines();
  }, []);

  // Fetch registered character portraits from backend
  useEffect(() => {
    const fetchPortraits = async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${API_BASE}/api/v1/ip-adapter/status`);
        if (res.ok) {
          const data = await res.json();
          if (data.registered_characters) {
            // Also fetch portrait files from disk
            const pRes = await fetch(`${API_BASE}/api/v1/characters/portraits/_all`);
            if (pRes.ok) {
              const pData = await pRes.json();
              setRegisteredPortraits(pData.portraits || []);
            }
          }
        }
      } catch { /* API offline */ }
    };
    fetchPortraits();
    // Re-check every 10 seconds in case user generates portraits in Character Studio tab
    const interval = setInterval(fetchPortraits, 10000);
    return () => clearInterval(interval);
  }, []);

  // Characters
  const [characters, setCharacters] = useState<StoryCharacter[]>([]);
  const [showCharForm, setShowCharForm] = useState(false);
  const [charName, setCharName] = useState("");
  const [charRole, setCharRole] = useState("Lead");
  const [charAppearance, setCharAppearance] = useState("");
  const [charUploading, setCharUploading] = useState(false);
  const [aiCharLoading, setAiCharLoading] = useState(false);

  // Quality & post-processing
  const [qualityPreset, setQualityPreset] = useState("standard");
  const [enableUpscale, setEnableUpscale] = useState(false);
  const [enableInterpolation, setEnableInterpolation] = useState(false);

  // Video engine selector (auto / wan22 / hunyuan15 / ltx23)
  const [videoEngine, setVideoEngine] = useState<string>("auto");
  const [enableLipSync, setEnableLipSync] = useState(false);
  const [availableEngines, setAvailableEngines] = useState<Array<{ id: string; name: string; available: boolean; tier?: string; description?: string }>>([]);
  const [lipSyncAvailable, setLipSyncAvailable] = useState<boolean>(false);

  // Loading states
  const [scriptLoading, setScriptLoading] = useState(false);
  const [scriptStatus, setScriptStatus] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStatus, setGenStatus] = useState("");
  const [resultJobId, setResultJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedScene, setExpandedScene] = useState<number | null>(0);
  const [activeTab, setActiveTab] = useState<"story" | "characters" | "settings">("story");
  const [copied, setCopied] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Synchronized script player
  const videoRef = useRef<HTMLVideoElement>(null);
  const scriptContainerRef = useRef<HTMLDivElement>(null);
  const [activeSceneIdx, setActiveSceneIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [showScript, setShowScript] = useState(true);
  const [scriptViewMode, setScriptViewMode] = useState<"scenes" | "combined" | "teleprompter">("combined");

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  // Build cumulative time ranges for each scene
  const sceneTimeRanges = scenes.reduce<{ start: number; end: number }[]>((acc, s) => {
    const start = acc.length > 0 ? acc[acc.length - 1].end : 0;
    acc.push({ start, end: start + s.duration });
    return acc;
  }, []);

  // Handle video time updates for script sync
  const handleVideoTimeUpdate = () => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);

    // Find which scene is active based on current time
    const idx = sceneTimeRanges.findIndex((r) => t >= r.start && t < r.end);
    if (idx !== -1 && idx !== activeSceneIdx) {
      setActiveSceneIdx(idx);
      // Auto-scroll the script panel to the active scene
      const el = document.getElementById(`script-scene-${idx}`);
      if (el && scriptContainerRef.current) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  };

  // Seek video to a specific scene
  const seekToScene = (idx: number) => {
    if (!videoRef.current || idx >= sceneTimeRanges.length) return;
    videoRef.current.currentTime = sceneTimeRanges[idx].start;
    setActiveSceneIdx(idx);
  };

  const togglePlayPause = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const skipScene = (dir: -1 | 1) => {
    const next = Math.max(0, Math.min(scenes.length - 1, activeSceneIdx + dir));
    seekToScene(next);
  };

  // Format seconds as MM:SS
  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // Build the visual anchor from characters automatically
  const buildCharacterAnchor = () => {
    if (characters.length === 0) return "";
    const charDescs = characters.map((c) => `${c.name}: ${c.appearance}`).join(". ");
    return `Characters appearing in this film: ${charDescs}. Every scene must show these characters with EXACTLY this appearance, same face, same clothing, same distinguishing features.`;
  };

  // Auto-update visual anchor when characters change
  useEffect(() => {
    if (characters.length > 0) {
      setVisualAnchor(buildCharacterAnchor());
    }
  }, [characters]);

  const addCharacter = () => {
    if (!charName.trim() || !charAppearance.trim()) return;
    const newChar: StoryCharacter = {
      id: crypto.randomUUID(),
      name: charName.trim(),
      role: charRole,
      appearance: charAppearance.trim(),
    };
    setCharacters([...characters, newChar]);
    setCharName("");
    setCharRole("Lead");
    setCharAppearance("");
    setShowCharForm(false);
  };

  // Import characters from Character Studio (backend portraits)
  const [importingChars, setImportingChars] = useState(false);
  const importFromCharacterStudio = async () => {
    setImportingChars(true);
    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${API_BASE}/api/v1/characters/list`);
      if (!res.ok) throw new Error("Failed to fetch characters");
      const data = await res.json();
      const studioChars = (data.characters || []) as {
        name: string; appearance: string; front_url: string | null;
        portraits: { url: string; view: string }[];
      }[];
      if (studioChars.length === 0) {
        alert("No characters found. Go to Character Studio tab to generate character portraits first.");
        return;
      }
      // Add characters that aren't already in the list
      const existingNames = new Set(characters.map((c) => c.name.toLowerCase()));
      const newChars: StoryCharacter[] = [];
      for (const sc of studioChars) {
        if (!existingNames.has(sc.name.toLowerCase())) {
          // Clean appearance: strip image-gen boilerplate, keep the core description
          let cleanAppearance = sc.appearance || "";
          // Remove "Character portrait of Name:" prefix
          cleanAppearance = cleanAppearance.replace(/^Character portrait of [^,]+,\s*/i, "");
          // Remove image-gen suffixes
          cleanAppearance = cleanAppearance.replace(/,\s*(highly detailed|sharp focus|clean neutral background|character reference sheet quality|photorealistic|8K|professional photography|detailed skin texture|professional studio lighting|consistent recognizable features|full body visible|cel shaded|clean lines|vibrant colors|3D rendered|Pixar style|subsurface scattering)[^,]*/gi, "");
          cleanAppearance = cleanAppearance.replace(/,\s*,/g, ",").replace(/,\s*$/, "").trim();
          if (!cleanAppearance) cleanAppearance = sc.name;

          newChars.push({
            id: crypto.randomUUID(),
            name: sc.name,
            role: "Lead",
            appearance: cleanAppearance,
            refImageUploaded: true,
            refImageName: sc.front_url ? `Portrait (${sc.portraits.length} views)` : undefined,
          });
        }
      }
      if (newChars.length > 0) {
        setCharacters((prev) => [...prev, ...newChars]);
        // Auto-build visual anchor from imported characters
        setTimeout(() => setVisualAnchor(buildCharacterAnchor()), 100);
      } else {
        alert(`All ${studioChars.length} character(s) already imported.`);
      }
    } catch (e) {
      console.error("Import failed:", e);
      alert("Import failed — check if API server is running.");
    } finally {
      setImportingChars(false);
    }
  };

  const removeCharacter = (id: string) => {
    setCharacters(characters.filter((c) => c.id !== id));
  };

  const useCharTemplate = (t: typeof CHARACTER_TEMPLATES[0]) => {
    setCharName(t.name);
    setCharRole(t.role);
    setCharAppearance(t.appearance);
    setShowCharForm(true);
  };

  const handleAiGenerateAppearance = async () => {
    if (!charName.trim() || !charRole.trim()) return;
    setAiCharLoading(true);
    try {
      const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${API}/api/v1/generate/script`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          concept: `Generate a detailed visual appearance description for a character named "${charName}" who is the ${charRole} in a ${genre} ${mood} film. The description should be specific enough for AI video generation: exact physical appearance, hair, eyes, skin tone, clothing, accessories, posture, and distinguishing features. Return ONLY the visual description in one paragraph, no JSON.`,
          num_scenes: 1,
          genre,
          mood,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        // The script endpoint returns scenes, but we hacked the prompt to get a description
        const sceneList = data.scenes || [];
        if (sceneList.length > 0 && sceneList[0].visual) {
          setCharAppearance(sceneList[0].visual);
        } else {
          // Fallback: generate a simple description
          setCharAppearance(
            `${charName}, ${charRole.toLowerCase()} character, highly detailed consistent appearance, ` +
            `specific distinguishing features, ${genre.toLowerCase()} style, cinematic quality`
          );
        }
      }
    } catch {
      setCharAppearance(
        `${charName}, ${charRole.toLowerCase()} character, highly detailed consistent appearance, ` +
        `specific distinguishing features, cinematic quality`
      );
    } finally {
      setAiCharLoading(false);
    }
  };

  const uploadCharacterRef = async (charId: string, file: File) => {
    setCharUploading(true);
    try {
      const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API}/api/v1/ip-adapter/upload-reference?char_id=${charId}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.status === "ok") {
        setCharacters((prev) =>
          prev.map((c) => c.id === charId ? { ...c, refImageName: file.name, refImageUploaded: true } : c)
        );
      }
    } catch {
      // silently fail - API might not be running
    } finally {
      setCharUploading(false);
    }
  };

  const copyCharPrompt = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const handleGenerateScript = async () => {
    if (!concept.trim()) return;
    setScriptLoading(true);
    setScriptStatus("Sending to LLM... This may take 15-30 seconds");
    setError(null);

    try {
      const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${API}/api/v1/generate/script`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          concept,
          num_scenes: numScenes,
          genre,
          mood,
          characters: characters.map((c) => ({
            name: c.name,
            role: c.role,
            appearance: c.appearance,
          })),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const apiScenes = data.scenes || [];
        if (apiScenes.length > 0) {
          setScenes(apiScenes.map((s: Record<string, unknown>, i: number) => ({
            id: crypto.randomUUID(),
            title: (s.title as string) || `Scene ${i + 1}`,
            visual: (s.visual as string) || "",
            narration: (s.narration as string) || "",
            duration: (s.duration as number) || 5,
            characters: (s.characters as string[]) || characters.map((c) => c.name),
          })));
          setExpandedScene(0);

          // Auto-generate visual anchor
          if (characters.length > 0) {
            setVisualAnchor(buildCharacterAnchor());
          } else if (!visualAnchor) {
            setVisualAnchor(`${concept}, consistent visual style, same characters and subjects throughout, ${genre.toLowerCase()} genre, ${mood.toLowerCase()} mood, cinematic quality`);
          }
        } else {
          throw new Error("LLM returned no scenes");
        }
      } else {
        const errData = await response.json().catch(() => ({ detail: "API error" }));
        throw new Error(errData.detail || `API returned ${response.status}`);
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Unknown error";
      setError(`Script generation failed: ${errMsg}`);
      setScenes(Array.from({ length: numScenes }, (_, i) => ({
        id: crypto.randomUUID(),
        title: `Scene ${i + 1}`,
        visual: `${concept}, scene ${i + 1} of ${numScenes}, ${genre.toLowerCase()}, ${mood.toLowerCase()}, cinematic`,
        narration: "",
        duration: 5,
        characters: characters.map((c) => c.name),
      })));
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

    const selectedEngine = ENGINES.find((e) => e.id === engine) || ENGINES[0];

    try {
      const scenesPayload = scenes.map((s) => {
        // Inject character appearance into visual prompt for strong consistency
        let visual = s.visual;
        const sceneChars = s.characters || [];
        if (sceneChars.length > 0) {
          const charDescs = sceneChars.map((name) => {
            const char = characters.find((c) => c.name === name);
            return char ? `${char.name} (${char.appearance})` : name;
          }).join(", ");
          // Prepend character descriptions if not already present
          if (!visual.toLowerCase().includes(sceneChars[0].toLowerCase())) {
            visual = `${charDescs} in scene: ${visual}`;
          }
        }
        return { title: s.title, visual, narration: s.narration, duration: s.duration, characters: sceneChars };
      });

      const job = await generateStory({
        concept,
        num_scenes: scenes.length,
        genre,
        mood,
        engine: selectedEngine.id,
        model: selectedEngine.model,
        enable_narration: narrationEnabled,
        continuity_mode: continuityMode === "prompt" || continuityMode === "both" ? "Prompt Anchoring" : "None",
        visual_anchor: visualAnchor,
        transition_type: transition,
        num_frames: frames,
        fps,
        resolution,
        scenes: scenesPayload,
        ip_adapter_enabled: ipAdapterEnabled,
        ip_adapter_strength: ipAdapterStrength,
        quality_preset: qualityPreset,
        enable_upscale: enableUpscale,
        enable_interpolation: enableInterpolation,
        video_engine: videoEngine,
        enable_lip_sync: enableLipSync,
      });

      setGenStatus(`Job ${job.job_id.slice(0, 8)}... started`);

      pollRef.current = setInterval(async () => {
        try {
          const status = await getJob(job.job_id);
          const pct = Math.min(Math.round(status.progress || 0), 100);
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
    setScenes([...scenes, { id: crypto.randomUUID(), title: `Scene ${scenes.length + 1}`, visual: "", narration: "", duration: 5, characters: characters.map((c) => c.name) }]);
  };
  const removeScene = (idx: number) => setScenes(scenes.filter((_, i) => i !== idx));
  const updateScene = (idx: number, field: keyof Scene, value: string | number | string[]) => {
    const updated = [...scenes];
    updated[idx] = { ...updated[idx], [field]: value };
    setScenes(updated);
  };
  const moveScene = (idx: number, dir: -1 | 1) => {
    const ni = idx + dir;
    if (ni < 0 || ni >= scenes.length) return;
    const u = [...scenes];
    [u[idx], u[ni]] = [u[ni], u[idx]];
    setScenes(u);
  };

  // Inject a character's appearance into a scene prompt
  const injectCharacterIntoScene = (sceneIdx: number, char: StoryCharacter) => {
    const currentVisual = scenes[sceneIdx].visual;
    if (currentVisual.includes(char.appearance)) return; // already injected
    const injected = `${char.name}: ${char.appearance}. ${currentVisual}`;
    updateScene(sceneIdx, "visual", injected);
  };

  const totalDuration = scenes.reduce((s, sc) => s + sc.duration, 0);

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text">Story Mode</h1>
        <p className="text-zinc-400 mt-1">Create multi-scene AI movies with characters, narration, and transitions</p>
      </div>

      {/* Workflow steps indicator */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2">
        {[
          { step: 1, label: "Story", tab: "story" as const, done: !!concept.trim() },
          { step: 2, label: `Characters (${characters.length})`, tab: "characters" as const, done: characters.length > 0 },
          { step: 3, label: "Settings", tab: "settings" as const, done: true },
          { step: 4, label: "Scenes", tab: null, done: scenes.length > 0 },
          { step: 5, label: "Generate", tab: null, done: !!resultJobId },
        ].map((s, i) => (
          <div key={i} className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => s.tab && setActiveTab(s.tab)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                s.tab === activeTab
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20"
                  : s.done
                    ? "bg-zinc-800 text-green-400 hover:bg-zinc-700"
                    : "bg-zinc-800/50 text-zinc-500 hover:bg-zinc-800"
              } ${s.tab ? "cursor-pointer" : "cursor-default"}`}
            >
              <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold ${
                s.tab === activeTab ? "bg-white/20" : s.done ? "bg-green-500/20" : "bg-zinc-700"
              }`}>{s.done && s.tab !== activeTab ? <Check className="w-2.5 h-2.5" /> : s.step}</span>
              {s.label}
            </button>
            {i < 4 && <div className="w-4 h-px bg-zinc-700 flex-shrink-0" />}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: Planning, Characters & Settings (tabbed) */}
        <div className="space-y-4">
          {activeTab === "story" && (
            <>
              {/* Story concept */}
              <Card>
                <CardTitle className="text-sm mb-3">1. Plan Your Story</CardTitle>
                <p className="text-xs text-zinc-500 mb-2">Quick ideas:</p>
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {STORY_IDEAS.map((idea) => (
                    <button key={idea.label} onClick={() => setConcept(idea.concept)}
                      className={`px-2.5 py-1 rounded-full text-[10px] font-medium transition-all ${
                        concept === idea.concept ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                      }`}>{idea.label}</button>
                  ))}
                </div>
                <textarea value={concept} onChange={(e) => setConcept(e.target.value)}
                  placeholder="Describe your movie idea..."
                  className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none" />
                <div className="grid grid-cols-3 gap-2 mt-3">
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
                  <div>
                    <label className="text-[10px] text-zinc-500">Scenes: {numScenes}</label>
                    <input type="range" min={2} max={8} value={numScenes}
                      onChange={(e) => setNumScenes(Number(e.target.value))}
                      className="w-full accent-indigo-500 mt-1.5" />
                  </div>
                </div>
                <div className="flex gap-2 mt-3">
                  <Button className="flex-1" onClick={handleGenerateScript} disabled={!concept.trim() || scriptLoading}>
                    {scriptLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> {scriptStatus || "Generating..."}</> : <><Sparkles className="w-4 h-4 mr-2" /> AI Generate Script</>}
                  </Button>
                  <Button variant="secondary" onClick={() => setActiveTab("characters")} className="flex-shrink-0">
                    <Users className="w-4 h-4 mr-1.5" /> Cast
                  </Button>
                </div>
              </Card>

              {/* Character summary (when characters exist) */}
              {characters.length > 0 && (
                <Card className="border-purple-500/20 bg-purple-500/5">
                  <div className="flex items-center justify-between mb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Users className="w-3.5 h-3.5 text-purple-400" /> Cast ({characters.length})
                    </CardTitle>
                    <button onClick={() => setActiveTab("characters")} className="text-[10px] text-purple-400 hover:text-purple-300">Edit Cast</button>
                  </div>
                  <div className="space-y-1.5">
                    {characters.map((c) => {
                      // Find portrait URL from registeredPortraits
                      const charKey = c.name.toLowerCase().replace(/ /g, "_");
                      const portrait = registeredPortraits.find((p) => p.filename.startsWith(charKey) && p.view === "front");
                      return (
                        <div key={c.id} className="flex items-center gap-2 py-1">
                          {portrait ? (
                            <img src={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${portrait.url}`}
                              alt={c.name} className="w-8 h-8 rounded-full object-cover border border-purple-500/30 flex-shrink-0" />
                          ) : (
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                              c.refImageUploaded ? "bg-purple-600/30" : "bg-indigo-600/20"
                            }`}>
                              {c.refImageUploaded ? <Shield className="w-3.5 h-3.5 text-purple-400" /> : <UserCircle className="w-3.5 h-3.5 text-indigo-400" />}
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-medium text-zinc-200 truncate">{c.name}</span>
                              <Badge className="text-[8px] flex-shrink-0">{c.role}</Badge>
                              {(c.refImageUploaded || portrait) && <Badge className="text-[8px] bg-purple-600/20 text-purple-300 border-purple-500/30 flex-shrink-0">Portrait</Badge>}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              )}
            </>
          )}

          {activeTab === "characters" && (
            <>
              {/* Character Builder */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2">
                  <Users className="w-3.5 h-3.5 text-purple-400" /> 2. Cast Your Characters
                </CardTitle>
                <p className="text-xs text-zinc-500 mb-2">
                  Define characters here. Their appearance will be injected into every scene for visual consistency.
                </p>
                <Button variant="secondary" className="w-full mb-3" onClick={importFromCharacterStudio} disabled={importingChars}>
                  {importingChars ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <ImageIcon className="w-3.5 h-3.5 mr-1.5" />}
                  Import from Character Studio
                </Button>

                {/* Template characters */}
                <div className="mb-3">
                  <label className="text-[10px] text-zinc-500 uppercase font-semibold mb-1.5 block">Quick Templates</label>
                  <div className="flex flex-wrap gap-1.5">
                    {CHARACTER_TEMPLATES.map((t) => (
                      <button key={t.name} onClick={() => useCharTemplate(t)}
                        className="px-2 py-1 rounded-full text-[10px] bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-all">
                        {t.name}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Existing characters */}
                {characters.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {characters.map((char) => (
                      <div key={char.id} className="p-2.5 rounded-lg border border-zinc-700 bg-zinc-800/30">
                        <div className="flex items-start gap-2">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${
                            char.refImageUploaded ? "bg-purple-600/30" : "bg-indigo-600/20"
                          }`}>
                            {char.refImageUploaded ? <Shield className="w-4 h-4 text-purple-400" /> : <UserCircle className="w-4 h-4 text-indigo-400" />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5 mb-0.5">
                              <span className="text-xs font-medium text-zinc-200">{char.name}</span>
                              <Badge className="text-[8px]">{char.role}</Badge>
                            </div>
                            <p className="text-[10px] text-zinc-400 line-clamp-2">{char.appearance}</p>
                            {char.refImageName && (
                              <p className="text-[9px] text-purple-400 mt-1 flex items-center gap-1">
                                <ImageIcon className="w-2.5 h-2.5" /> {char.refImageName}
                              </p>
                            )}
                          </div>
                          <div className="flex flex-col gap-1 flex-shrink-0">
                            <button
                              onClick={() => {
                                const input = document.createElement("input");
                                input.type = "file";
                                input.accept = "image/*";
                                input.onchange = (e) => {
                                  const file = (e.target as HTMLInputElement).files?.[0];
                                  if (file) uploadCharacterRef(char.id, file);
                                };
                                input.click();
                              }}
                              disabled={charUploading}
                              className="p-1 rounded text-zinc-500 hover:text-purple-400 hover:bg-purple-500/10 transition-colors"
                              title="Upload reference image"
                            >
                              {charUploading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                            </button>
                            <button onClick={() => copyCharPrompt(char.id, char.appearance)}
                              className="p-1 rounded text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 transition-colors"
                              title="Copy appearance prompt"
                            >
                              {copied === char.id ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                            </button>
                            <button onClick={() => removeCharacter(char.id)}
                              className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                              title="Remove character"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add character form */}
                {showCharForm ? (
                  <div className="space-y-2 p-2.5 rounded-lg border border-indigo-500/30 bg-indigo-500/5">
                    <div className="flex gap-2">
                      <div className="flex-1">
                        <label className="text-[10px] text-zinc-500">Name</label>
                        <input value={charName} onChange={(e) => setCharName(e.target.value)}
                          placeholder="Character name"
                          className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 mt-0.5" />
                      </div>
                      <div className="w-24">
                        <label className="text-[10px] text-zinc-500">Role</label>
                        <select value={charRole} onChange={(e) => setCharRole(e.target.value)}
                          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                          <option>Lead</option>
                          <option>Supporting</option>
                          <option>Companion</option>
                          <option>Villain</option>
                          <option>Extra</option>
                        </select>
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center justify-between">
                        <label className="text-[10px] text-zinc-500">Appearance (detailed)</label>
                        <button onClick={handleAiGenerateAppearance}
                          disabled={aiCharLoading || !charName.trim()}
                          className="text-[9px] text-indigo-400 hover:text-indigo-300 disabled:text-zinc-600 flex items-center gap-1">
                          {aiCharLoading ? <Loader2 className="w-2.5 h-2.5 animate-spin" /> : <Sparkles className="w-2.5 h-2.5" />}
                          AI Generate
                        </button>
                      </div>
                      <textarea value={charAppearance} onChange={(e) => setCharAppearance(e.target.value)}
                        placeholder="Detailed physical appearance: hair, eyes, clothing, build, distinguishing features..."
                        className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-2.5 py-1.5 text-[11px] text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-0.5" />
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" className="flex-1" onClick={addCharacter}
                        disabled={!charName.trim() || !charAppearance.trim()}>
                        <Plus className="w-3 h-3 mr-1" /> Add to Cast
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => { setShowCharForm(false); setCharName(""); setCharAppearance(""); }}>
                        <X className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                ) : (
                  <button onClick={() => setShowCharForm(true)}
                    className="w-full border-2 border-dashed border-zinc-700 rounded-lg py-2.5 text-xs text-zinc-500 hover:text-zinc-300 hover:border-zinc-500 transition-all flex items-center justify-center gap-1.5">
                    <Plus className="w-3.5 h-3.5" /> Add Character
                  </button>
                )}
              </Card>

              {/* IP-Adapter Settings */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2">
                  <Shield className="w-3.5 h-3.5 text-purple-400" /> IP-Adapter Consistency
                </CardTitle>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-zinc-300">Enable face/appearance matching</span>
                    <button onClick={() => setIpAdapterEnabled(!ipAdapterEnabled)}
                      className={`w-9 h-5 rounded-full transition-colors flex items-center ${
                        ipAdapterEnabled ? "bg-purple-600 justify-end" : "bg-zinc-700 justify-start"
                      }`}>
                      <span className="w-3.5 h-3.5 rounded-full bg-white mx-0.5 block" />
                    </button>
                  </div>
                  {ipAdapterEnabled && (
                    <>
                      <div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-zinc-500">Matching Strength</span>
                          <span className="text-purple-400 font-mono">{ipAdapterStrength.toFixed(1)}</span>
                        </div>
                        <input type="range" min={0} max={1} step={0.1} value={ipAdapterStrength}
                          onChange={(e) => setIpAdapterStrength(Number(e.target.value))}
                          className="w-full mt-0.5 accent-purple-500 h-1" />
                        <div className="flex justify-between text-[9px] text-zinc-600">
                          <span>More Creative</span><span>Strict Match</span>
                        </div>
                      </div>
                      {/* Show portraits only for characters in the cast */}
                      {(() => {
                        // Filter portraits to only show cast members
                        const castKeys = characters.map((c) => c.name.toLowerCase().replace(/ /g, "_"));
                        const castPortraits = registeredPortraits.filter((p) =>
                          castKeys.some((key) => p.filename.startsWith(key))
                        );
                        if (castPortraits.length > 0) {
                          return (
                            <div className="bg-purple-900/10 rounded-lg p-2.5 border border-purple-500/10">
                              <p className="text-[10px] text-purple-300 font-medium mb-2">
                                {castPortraits.length} portrait(s) for {characters.length} cast member(s) will be used:
                              </p>
                              <div className="flex gap-2 overflow-x-auto pb-1">
                                {castPortraits.filter((p) => p.view === "front").slice(0, 8).map((p, i) => (
                                  <img key={i} src={getPortraitUrl(p.url)} alt={p.view}
                                    className="w-12 h-12 rounded-md border border-purple-500/20 object-cover flex-shrink-0" />
                                ))}
                              </div>
                              <p className="text-[9px] text-zinc-500 mt-1.5">
                                Front portraits shown. All views (front, side, 3/4) are used during generation.
                              </p>
                            </div>
                          );
                        }
                        if (characters.length > 0) {
                          return (
                            <div className="bg-zinc-800/50 rounded-lg p-2.5">
                              <p className="text-[10px] text-zinc-400 leading-relaxed">
                                <strong className="text-amber-300">No portraits found for cast.</strong> Go to Character Studio to generate portraits for your characters.
                              </p>
                            </div>
                          );
                        }
                        return (
                          <div className="bg-zinc-800/50 rounded-lg p-2.5">
                            <p className="text-[10px] text-zinc-400 leading-relaxed">
                              <strong className="text-purple-300">No characters in cast.</strong> Import from Character Studio or add manually above.
                            </p>
                          </div>
                        );
                      })()}
                    </>
                  )}
                </div>
              </Card>

              <Button variant="secondary" className="w-full" onClick={() => setActiveTab("story")}>
                <ChevronUp className="w-4 h-4 mr-1.5" /> Back to Story
              </Button>
            </>
          )}

          {activeTab === "settings" && (
            <>
              {/* Quality Preset */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2"><Sparkles className="w-3.5 h-3.5 text-amber-400" /> Quality Preset</CardTitle>
                <div className="space-y-2">
                  {([
                    { id: "draft", name: "Draft", desc: "25 steps, fast preview", time: "~3 min/scene", color: "border-zinc-600" },
                    { id: "standard", name: "Standard", desc: "40 steps, good quality", time: "~6 min/scene", color: "border-indigo-500" },
                    { id: "high", name: "High Quality", desc: "50 steps, best detail", time: "~9 min/scene", color: "border-amber-500" },
                  ]).map((q) => (
                    <button key={q.id} onClick={() => setQualityPreset(q.id)}
                      className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                        qualityPreset === q.id ? `${q.color} bg-zinc-800/80` : "border-zinc-700/50 hover:border-zinc-600"
                      }`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <span className={`text-xs font-medium ${qualityPreset === q.id ? "text-zinc-100" : "text-zinc-400"}`}>{q.name}</span>
                          <span className="text-[10px] text-zinc-600 ml-1.5">{q.desc}</span>
                        </div>
                        <span className="text-[9px] text-zinc-600 font-mono">{q.time}</span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Post-processing */}
                <div className="mt-3 pt-3 border-t border-zinc-800 space-y-2">
                  <label className="text-[10px] text-zinc-500 uppercase font-semibold">Post-Processing</label>
                  <label className="flex items-center justify-between cursor-pointer py-1">
                    <div>
                      <span className="text-xs text-zinc-300">Upscale (2x)</span>
                      <span className="text-[10px] text-zinc-600 block">Real-ESRGAN / Lanczos, +30s</span>
                    </div>
                    <button onClick={() => setEnableUpscale(!enableUpscale)}
                      className={`w-8 h-4 rounded-full transition-colors flex items-center ${
                        enableUpscale ? "bg-amber-600 justify-end" : "bg-zinc-700 justify-start"
                      }`}>
                      <span className="w-3 h-3 rounded-full bg-white mx-0.5 block" />
                    </button>
                  </label>
                  <label className="flex items-center justify-between cursor-pointer py-1">
                    <div>
                      <span className="text-xs text-zinc-300">Smooth Motion (RIFE)</span>
                      <span className="text-[10px] text-zinc-600 block">Frame interpolation {fps}→{fps * 2}fps</span>
                    </div>
                    <button onClick={() => setEnableInterpolation(!enableInterpolation)}
                      className={`w-8 h-4 rounded-full transition-colors flex items-center ${
                        enableInterpolation ? "bg-amber-600 justify-end" : "bg-zinc-700 justify-start"
                      }`}>
                      <span className="w-3 h-3 rounded-full bg-white mx-0.5 block" />
                    </button>
                  </label>
                  <label className={`flex items-center justify-between py-1 ${lipSyncAvailable ? "cursor-pointer" : "opacity-50 cursor-not-allowed"}`}>
                    <div>
                      <span className="text-xs text-zinc-300">Lip Sync (LatentSync 1.6)</span>
                      <span className="text-[10px] text-zinc-600 block">
                        {lipSyncAvailable ? "Sync mouth movement to narration, +1–2 min/scene" : "Install: pip install git+https://github.com/bytedance/LatentSync.git"}
                      </span>
                    </div>
                    <button onClick={() => lipSyncAvailable && setEnableLipSync(!enableLipSync)}
                      disabled={!lipSyncAvailable}
                      className={`w-8 h-4 rounded-full transition-colors flex items-center ${
                        enableLipSync ? "bg-amber-600 justify-end" : "bg-zinc-700 justify-start"
                      }`}>
                      <span className="w-3 h-3 rounded-full bg-white mx-0.5 block" />
                    </button>
                  </label>
                </div>
              </Card>

              {/* Video Engine selector */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2"><Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Video Engine</CardTitle>
                <div className="space-y-1.5">
                  {(() => {
                    const autoOpt = { id: "auto", name: "Auto", available: true, description: "Use best available (Wan 2.2 → Hunyuan 1.5 → LTX-2.3)" };
                    const opts = [autoOpt, ...availableEngines];
                    return opts.map((e) => {
                      const disabled = !e.available;
                      const selected = videoEngine === e.id;
                      return (
                        <button key={e.id} onClick={() => !disabled && setVideoEngine(e.id)}
                          disabled={disabled}
                          className={`w-full text-left px-3 py-2 rounded-lg border transition-all ${
                            selected ? "border-indigo-500 bg-zinc-800/80" : disabled ? "border-zinc-800 opacity-40 cursor-not-allowed" : "border-zinc-700/50 hover:border-zinc-600"
                          }`}>
                          <div className="flex items-center justify-between">
                            <span className={`text-xs font-medium ${selected ? "text-zinc-100" : "text-zinc-400"}`}>{e.name}</span>
                            {"tier" in e && e.tier ? (
                              <span className="text-[9px] text-zinc-600 font-mono uppercase">{e.tier}</span>
                            ) : null}
                          </div>
                          {e.description ? (
                            <span className="text-[10px] text-zinc-600 block mt-0.5">{e.description}</span>
                          ) : null}
                          {disabled ? (
                            <span className="text-[10px] text-amber-600/70 block mt-0.5">Weights not downloaded</span>
                          ) : null}
                        </button>
                      );
                    });
                  })()}
                </div>
              </Card>

              {/* Video Output Settings */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2"><Settings2 className="w-3.5 h-3.5 text-indigo-400" /> Video Settings</CardTitle>
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] text-zinc-500">Engine</label>
                    <select value={engine} onChange={(e) => setEngine(e.target.value)}
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                      {ENGINES.map((e) => <option key={e.id} value={e.id}>{e.name} - {e.desc}</option>)}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[10px] text-zinc-500">Resolution</label>
                      <select value={resolution} onChange={(e) => setResolution(e.target.value)}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                        <option value="480p">480p (832x480)</option>
                        <option value="720p">720p (1280x720)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-zinc-500">FPS</label>
                      <select value={fps} onChange={(e) => setFps(Number(e.target.value))}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                        <option value={8}>8 fps</option>
                        <option value={16}>16 fps</option>
                        <option value={24}>24 fps</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500">Frames per scene: {frames} (~{(frames / fps).toFixed(1)}s)</label>
                    <input type="range" min={17} max={81} step={16} value={frames}
                      onChange={(e) => setFrames(Number(e.target.value))} className="w-full accent-indigo-500" />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[10px] text-zinc-500">Transition</label>
                      <select value={transition} onChange={(e) => setTransition(e.target.value)}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-0.5">
                        {TRANSITIONS.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                      </select>
                    </div>
                    <div className="flex flex-col gap-1.5 pt-3">
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={narrationEnabled} onChange={(e) => setNarrationEnabled(e.target.checked)} className="accent-indigo-500 w-3 h-3" />
                        <span className="text-[10px] text-zinc-300">Narration</span>
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={musicEnabled} onChange={(e) => setMusicEnabled(e.target.checked)} className="accent-indigo-500 w-3 h-3" />
                        <span className="text-[10px] text-zinc-300">Music</span>
                      </label>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Scene Continuity */}
              <Card>
                <CardTitle className="text-sm mb-3 flex items-center gap-2"><Link className="w-3.5 h-3.5 text-indigo-400" /> Scene Continuity</CardTitle>
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] text-zinc-500">Mode</label>
                    {CONTINUITY_MODES.map((cm) => (
                      <button key={cm.id} onClick={() => setContinuityMode(cm.id)}
                        className={`w-full text-left px-2.5 py-2 rounded-lg text-xs mt-1 transition-all border ${
                          continuityMode === cm.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                        }`}>
                        <span className="font-medium">{cm.name}</span>
                        <span className="text-zinc-500 block text-[10px]">{cm.desc}</span>
                      </button>
                    ))}
                  </div>
                  {continuityMode !== "none" && (
                    <div>
                      <label className="text-[10px] text-zinc-500">Visual Identity Anchor</label>
                      <textarea value={visualAnchor} onChange={(e) => setVisualAnchor(e.target.value)}
                        placeholder="Describe consistent visual elements..."
                        className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-2.5 py-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-0.5" />
                      {characters.length > 0 && (
                        <button onClick={() => setVisualAnchor(buildCharacterAnchor())}
                          className="text-[9px] text-indigo-400 hover:text-indigo-300 mt-1 flex items-center gap-1">
                          <Sparkles className="w-2.5 h-2.5" /> Rebuild from cast
                        </button>
                      )}
                      <p className="text-[9px] text-zinc-600 mt-0.5">Appended to every scene prompt. Auto-populated from your character cast.</p>
                    </div>
                  )}
                </div>
              </Card>

              <Button variant="secondary" className="w-full" onClick={() => setActiveTab("story")}>
                <ChevronUp className="w-4 h-4 mr-1.5" /> Back to Story
              </Button>
            </>
          )}
        </div>

        {/* CENTER + RIGHT: Storyboard */}
        <div className="lg:col-span-2 space-y-4">
          {/* Timeline bar */}
          {scenes.length > 0 && (
            <Card className="py-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-400 font-medium">
                  {scenes.length} scenes
                  {characters.length > 0 && <span className="text-purple-400 ml-2">{characters.length} character{characters.length > 1 ? "s" : ""}</span>}
                </span>
                <div className="flex items-center gap-3 text-xs text-zinc-400">
                  <span>{resolution} / {fps}fps</span>
                  <span>Total: {totalDuration}s</span>
                </div>
              </div>
              <div className="flex gap-0.5 h-2.5 rounded-full overflow-hidden">
                {scenes.map((s, i) => {
                  const pct = (s.duration / Math.max(totalDuration, 1)) * 100;
                  const colors = ["bg-red-400", "bg-teal-400", "bg-blue-400", "bg-green-400", "bg-yellow-400", "bg-purple-400", "bg-pink-400", "bg-orange-400"];
                  return <div key={s.id} className={`${colors[i % colors.length]} rounded-sm`} style={{ width: `${pct}%` }} title={`${s.title} (${s.duration}s)`} />;
                })}
              </div>
            </Card>
          )}

          {/* Empty state */}
          {scenes.length === 0 && !scriptLoading && (
            <Card className="min-h-[200px] flex items-center justify-center">
              <div className="text-center">
                <BookOpen className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-400 text-sm">No scenes yet</p>
                <p className="text-zinc-600 text-xs mt-1">Pick a story idea, add characters, then click "AI Generate Script"</p>
              </div>
            </Card>
          )}
          {scriptLoading && scenes.length === 0 && (
            <Card className="min-h-[200px] flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-10 h-10 text-indigo-400 mx-auto mb-3 animate-spin" />
                <p className="text-indigo-400 text-sm font-medium">Generating your story...</p>
                <p className="text-zinc-500 text-xs mt-1">{scriptStatus || "This may take 15-30 seconds"}</p>
                {characters.length > 0 && (
                  <p className="text-purple-400 text-xs mt-1">Including {characters.length} character{characters.length > 1 ? "s" : ""} in the script</p>
                )}
              </div>
            </Card>
          )}

          {/* Bulk character assignment */}
          {characters.length > 0 && scenes.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <Button size="sm" variant="secondary" onClick={() => {
                const allNames = characters.map((c) => c.name);
                setScenes((prev) => prev.map((s) => ({ ...s, characters: allNames })));
              }}>
                <Users className="w-3 h-3 mr-1" /> Assign All Characters to Every Scene
              </Button>
              <Button size="sm" variant="ghost" onClick={() => {
                setScenes((prev) => prev.map((s) => ({ ...s, characters: [] })));
              }}>
                <X className="w-3 h-3 mr-1" /> Clear All Assignments
              </Button>
            </div>
          )}

          {/* Scene cards */}
          <div className="space-y-3">
            {scenes.map((scene, i) => {
              const isExpanded = expandedScene === i;
              return (
                <Card key={scene.id} className="relative">
                  <div className="flex items-center gap-3 cursor-pointer select-none" onClick={() => setExpandedScene(isExpanded ? null : i)}>
                    <GripVertical className="w-4 h-4 text-zinc-600" />
                    <Badge variant="info" className="text-[10px]">{i + 1}</Badge>
                    <input value={scene.title} onChange={(e) => updateScene(i, "title", e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      className="flex-1 bg-transparent text-sm font-medium text-zinc-200 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 rounded px-1" />
                    <span className="text-xs text-zinc-500 tabular-nums">{scene.duration}s</span>
                    <div className="flex gap-0.5">
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, -1); }} className="p-1 text-zinc-600 hover:text-zinc-300"><ChevronUp className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); moveScene(i, 1); }} className="p-1 text-zinc-600 hover:text-zinc-300"><ChevronDown className="w-3.5 h-3.5" /></button>
                      <button onClick={(e) => { e.stopPropagation(); removeScene(i); }} className="p-1 text-zinc-600 hover:text-red-400"><Trash2 className="w-3.5 h-3.5" /></button>
                    </div>
                  </div>
                  {isExpanded && (
                    <div className="mt-4 space-y-3 pt-3 border-t border-zinc-800">
                      {/* Character injection buttons */}
                      {characters.length > 0 && (
                        <div>
                          <label className="text-[10px] text-zinc-500 mb-1 block">Add character to scene:</label>
                          <div className="flex flex-wrap gap-1.5">
                            {characters.map((char) => (
                              <button key={char.id}
                                onClick={(e) => { e.stopPropagation(); injectCharacterIntoScene(i, char); }}
                                className="flex items-center gap-1 px-2 py-1 rounded-full text-[10px] bg-purple-600/10 text-purple-300 border border-purple-500/20 hover:bg-purple-600/20 transition-all">
                                <UserCircle className="w-3 h-3" /> {char.name}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                          <label className="text-xs text-zinc-400 font-medium">Visual Prompt</label>
                          <textarea value={scene.visual} onChange={(e) => updateScene(i, "visual", e.target.value)}
                            placeholder="Describe the visual scene..."
                            className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1" />
                        </div>
                        <div>
                          <label className="text-xs text-zinc-400 font-medium">Narration</label>
                          <textarea value={scene.narration} onChange={(e) => updateScene(i, "narration", e.target.value)}
                            placeholder="Voiceover text..."
                            className="w-full h-24 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-1" />
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-zinc-400">Duration: <span className="text-indigo-400 font-mono">{scene.duration}s</span></label>
                        <input type="range" min={2} max={12} value={scene.duration}
                          onChange={(e) => updateScene(i, "duration", Number(e.target.value))} className="w-full accent-indigo-500 mt-1" />
                      </div>
                      {/* Per-scene character assignment */}
                      {characters.length > 0 && (
                        <div>
                          <div className="flex items-center justify-between">
                            <label className="text-xs text-zinc-400 font-medium flex items-center gap-1">
                              <Users className="w-3 h-3" /> Characters in this scene
                            </label>
                            <div className="flex gap-2">
                              <button onClick={() => updateScene(i, "characters", characters.map((c) => c.name))}
                                className="text-[9px] text-indigo-400 hover:text-indigo-300">All</button>
                              <button onClick={() => updateScene(i, "characters", [])}
                                className="text-[9px] text-zinc-500 hover:text-zinc-400">None</button>
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-1.5 mt-1">
                            {characters.map((char) => {
                              const isIn = (scene.characters || []).includes(char.name);
                              return (
                                <button key={char.id} onClick={() => {
                                  const current = scene.characters || [];
                                  const updated = isIn
                                    ? current.filter((n) => n !== char.name)
                                    : [...current, char.name];
                                  updateScene(i, "characters", updated);
                                }}
                                  className={`px-2 py-1 rounded text-[10px] border transition-colors ${
                                    isIn
                                      ? "bg-purple-600/20 border-purple-500/40 text-purple-300"
                                      : "bg-zinc-800/50 border-zinc-700 text-zinc-500 hover:border-zinc-600"
                                  }`}>
                                  {char.name}
                                </button>
                              );
                            })}
                          </div>
                          <p className="text-[9px] text-zinc-600 mt-0.5">
                            Only selected characters&apos; portraits are used as IP-Adapter reference for this scene
                          </p>
                        </div>
                      )}
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
                  <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Generating... {genProgress}%</>
                ) : (
                  <><Play className="w-5 h-5 mr-2" /> Generate Movie ({scenes.length} scenes, {totalDuration}s, {qualityPreset}){enableUpscale ? " + Upscale" : ""}</>
                )}
              </Button>
              {generating && (
                <Card>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-zinc-300 truncate flex-1 mr-2">{genStatus}</span>
                    <span className="text-xs font-mono text-indigo-400 flex-shrink-0">{genProgress}%</span>
                  </div>
                  <div className="h-2.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-indigo-600 via-purple-500 to-pink-500 rounded-full transition-all duration-700"
                      style={{ width: `${Math.max(genProgress, 2)}%` }} />
                  </div>
                  {characters.length > 0 && ipAdapterEnabled && (
                    <p className="text-[10px] text-purple-400 mt-2 flex items-center gap-1">
                      <Shield className="w-3 h-3" /> IP-Adapter active: maintaining {characters.length} character{characters.length > 1 ? "s" : ""} across scenes
                    </p>
                  )}
                </Card>
              )}
            </>
          )}

          {/* Result: Synchronized Script Player */}
          {resultJobId && (
            <div className="space-y-3">
              <Card className="border-green-500/20 bg-green-500/5">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-green-400 text-sm font-medium flex items-center gap-2">
                    <Film className="w-4 h-4" /> Movie Generated!
                  </p>
                  <div className="flex items-center gap-2">
                    <button onClick={() => setShowScript(!showScript)}
                      className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all ${
                        showScript ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                      }`}>
                      <FileText className="w-3 h-3" /> Script View
                    </button>
                    <Button size="sm" variant="secondary" onClick={() => {
                      downloadBlob(getDownloadUrl(resultJobId), `${resultJobId}.mp4`).catch(() => {});
                    }}><Download className="w-3 h-3 mr-1" /> Download</Button>
                  </div>
                </div>

                {/* Video Player */}
                <div className="relative">
                  <video
                    ref={videoRef}
                    src={getDownloadUrl(resultJobId)}
                    controls
                    className="w-full rounded-lg border border-zinc-700"
                    onTimeUpdate={handleVideoTimeUpdate}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onLoadedMetadata={() => {
                      if (videoRef.current) setVideoDuration(videoRef.current.duration);
                    }}
                  />

                  {/* Scene progress overlay at bottom of video */}
                  {scenes.length > 0 && (
                    <div className="absolute bottom-12 left-2 right-2 flex gap-0.5 h-1 rounded-full overflow-hidden opacity-80">
                      {scenes.map((s, i) => {
                        const range = sceneTimeRanges[i];
                        if (!range) return null;
                        const pct = (s.duration / Math.max(totalDuration, 1)) * 100;
                        const isActive = i === activeSceneIdx;
                        const isPast = range.end <= currentTime;
                        const colors = ["bg-red-400", "bg-teal-400", "bg-blue-400", "bg-green-400", "bg-yellow-400", "bg-purple-400", "bg-pink-400", "bg-orange-400"];
                        return (
                          <button key={s.id}
                            onClick={() => seekToScene(i)}
                            className={`h-full rounded-sm transition-all cursor-pointer ${
                              isActive ? `${colors[i % colors.length]} scale-y-150` : isPast ? `${colors[i % colors.length]} opacity-40` : "bg-zinc-600"
                            }`}
                            style={{ width: `${pct}%` }}
                            title={`${s.title} (${fmt(range.start)} - ${fmt(range.end)})`}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Custom playback controls */}
                <div className="flex items-center gap-3 mt-3 px-1">
                  <button onClick={() => skipScene(-1)} className="text-zinc-400 hover:text-white transition-colors">
                    <SkipBack className="w-4 h-4" />
                  </button>
                  <button onClick={togglePlayPause} className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center hover:bg-indigo-500 transition-colors">
                    {isPlaying ? <Pause className="w-4 h-4 text-white" /> : <Play className="w-4 h-4 text-white ml-0.5" />}
                  </button>
                  <button onClick={() => skipScene(1)} className="text-zinc-400 hover:text-white transition-colors">
                    <SkipForward className="w-4 h-4" />
                  </button>
                  <div className="flex-1 flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500 font-mono w-8 text-right">{fmt(currentTime)}</span>
                    <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden relative">
                      <div className="h-full bg-indigo-500 rounded-full transition-all duration-200"
                        style={{ width: `${videoDuration ? (currentTime / videoDuration) * 100 : 0}%` }} />
                    </div>
                    <span className="text-[10px] text-zinc-500 font-mono w-8">{fmt(videoDuration)}</span>
                  </div>
                  <Badge className="text-[9px]">Scene {activeSceneIdx + 1}/{scenes.length}</Badge>
                </div>
              </Card>

              {/* Synchronized Script Panel */}
              {showScript && scenes.length > 0 && (
                <Card>
                  {/* Header with view mode tabs */}
                  <div className="flex items-center justify-between mb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <FileText className="w-3.5 h-3.5 text-indigo-400" /> Script
                    </CardTitle>
                    <div className="flex items-center gap-1">
                      {([
                        { id: "combined" as const, label: "Full Narration" },
                        { id: "scenes" as const, label: "By Scene" },
                        { id: "teleprompter" as const, label: "Teleprompter" },
                      ]).map((v) => (
                        <button key={v.id} onClick={() => setScriptViewMode(v.id)}
                          className={`px-2 py-1 rounded text-[10px] font-medium transition-all ${
                            scriptViewMode === v.id ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                          }`}>{v.label}</button>
                      ))}
                    </div>
                  </div>

                  {/* ===== COMBINED NARRATION VIEW ===== */}
                  {scriptViewMode === "combined" && (
                    <div ref={scriptContainerRef} className="max-h-[420px] overflow-y-auto scroll-smooth">
                      <div className="px-2 py-3 space-y-0">
                        {scenes.map((scene, i) => {
                          const range = sceneTimeRanges[i];
                          const isActive = i === activeSceneIdx;
                          const isPast = range && range.end <= currentTime;
                          const sceneProgress = isActive && range
                            ? Math.min(1, (currentTime - range.start) / (range.end - range.start))
                            : 0;
                          const dotColors = ["bg-red-400", "bg-teal-400", "bg-blue-400", "bg-green-400", "bg-yellow-400", "bg-purple-400", "bg-pink-400", "bg-orange-400"];

                          return (
                            <div key={scene.id} id={`script-scene-${i}`} className="group">
                              {/* Scene divider with title */}
                              <button
                                onClick={() => seekToScene(i)}
                                className={`w-full flex items-center gap-2 py-2 transition-all ${
                                  isActive ? "opacity-100" : "opacity-60 hover:opacity-90"
                                }`}
                              >
                                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColors[i % dotColors.length]} ${
                                  isActive ? "animate-pulse scale-125" : ""
                                }`} />
                                <span className={`text-[10px] font-semibold uppercase tracking-wider ${
                                  isActive ? "text-zinc-300" : "text-zinc-600"
                                }`}>
                                  Scene {i + 1}: {scene.title}
                                </span>
                                <div className="flex-1 h-px bg-zinc-800" />
                                <span className="text-[9px] font-mono text-zinc-600">
                                  {range ? fmt(range.start) : ""}
                                </span>
                                {isActive && (
                                  <span className="flex items-center gap-1 ml-1">
                                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                                    <span className="text-[9px] text-green-400 font-medium">PLAYING</span>
                                  </span>
                                )}
                              </button>

                              {/* Narration text - the main content */}
                              {scene.narration && (
                                <button
                                  onClick={() => seekToScene(i)}
                                  className={`w-full text-left pl-4 pr-2 pb-3 transition-all duration-500 ${
                                    isActive
                                      ? "text-zinc-100"
                                      : isPast
                                        ? "text-zinc-600"
                                        : "text-zinc-500"
                                  }`}
                                >
                                  <p className={`leading-relaxed transition-all duration-500 ${
                                    isActive ? "text-sm" : "text-xs"
                                  }`}>
                                    {isActive ? (
                                      // Word-level highlighting: estimate which words are being spoken
                                      (() => {
                                        const words = scene.narration.split(" ");
                                        const wordsSpoken = Math.floor(sceneProgress * words.length);
                                        return (
                                          <>
                                            {words.map((word, wi) => (
                                              <span key={wi} className={`transition-colors duration-300 ${
                                                wi < wordsSpoken
                                                  ? "text-zinc-300"
                                                  : wi === wordsSpoken
                                                    ? "text-white font-medium bg-indigo-500/20 rounded px-0.5"
                                                    : "text-zinc-500"
                                              }`}>
                                                {word}{" "}
                                              </span>
                                            ))}
                                          </>
                                        );
                                      })()
                                    ) : (
                                      scene.narration
                                    )}
                                  </p>

                                  {/* Scene progress bar */}
                                  {isActive && range && (
                                    <div className="mt-2 h-0.5 bg-zinc-800 rounded-full overflow-hidden">
                                      <div className="h-full bg-indigo-500 rounded-full transition-all duration-200"
                                        style={{ width: `${sceneProgress * 100}%` }}
                                      />
                                    </div>
                                  )}
                                </button>
                              )}
                            </div>
                          );
                        })}

                        {/* End marker */}
                        <div className="flex items-center gap-2 pt-2 opacity-50">
                          <div className="w-2 h-2 rounded-full bg-zinc-600" />
                          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">End</span>
                          <div className="flex-1 h-px bg-zinc-800" />
                          <span className="text-[9px] font-mono text-zinc-600">{fmt(totalDuration)}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ===== SCENE-BY-SCENE VIEW ===== */}
                  {scriptViewMode === "scenes" && (
                    <div ref={scriptContainerRef} className="max-h-[420px] overflow-y-auto space-y-1 pr-1 scroll-smooth">
                      {scenes.map((scene, i) => {
                        const range = sceneTimeRanges[i];
                        const isActive = i === activeSceneIdx;
                        const isPast = range && range.end <= currentTime;
                        const colors = ["border-l-red-400", "border-l-teal-400", "border-l-blue-400", "border-l-green-400", "border-l-yellow-400", "border-l-purple-400", "border-l-pink-400", "border-l-orange-400"];
                        const bgColors = ["bg-red-500/5", "bg-teal-500/5", "bg-blue-500/5", "bg-green-500/5", "bg-yellow-500/5", "bg-purple-500/5", "bg-pink-500/5", "bg-orange-500/5"];

                        return (
                          <button
                            key={scene.id}
                            id={`script-scene-${i}`}
                            onClick={() => seekToScene(i)}
                            className={`w-full text-left px-3 py-2.5 rounded-lg border-l-3 transition-all duration-300 ${colors[i % colors.length]} ${
                              isActive
                                ? `${bgColors[i % bgColors.length]} border border-zinc-600 shadow-lg`
                                : isPast
                                  ? "opacity-50 hover:opacity-80"
                                  : "hover:bg-zinc-800/50"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0 ${
                                isActive ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-500"
                              }`}>{i + 1}</span>
                              <span className={`text-xs font-medium flex-1 ${isActive ? "text-zinc-100" : "text-zinc-400"}`}>
                                {scene.title}
                              </span>
                              <span className="text-[9px] text-zinc-600 font-mono flex-shrink-0">
                                {range ? `${fmt(range.start)} – ${fmt(range.end)}` : ""}
                              </span>
                              {isActive && (
                                <span className="flex items-center gap-1">
                                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                                  <span className="text-[9px] text-green-400 font-medium">NOW</span>
                                </span>
                              )}
                            </div>
                            {scene.narration && (
                              <p className={`text-xs leading-relaxed ml-7 ${
                                isActive ? "text-zinc-200" : isPast ? "text-zinc-600" : "text-zinc-500"
                              }`}>
                                &ldquo;{scene.narration}&rdquo;
                              </p>
                            )}
                            {isActive && scene.visual && (
                              <div className="mt-2 ml-7 px-2 py-1.5 rounded bg-zinc-800/50 border border-zinc-700/50">
                                <p className="text-[10px] text-zinc-500 line-clamp-2">
                                  <Eye className="w-2.5 h-2.5 inline mr-1 -mt-0.5" />
                                  {scene.visual.length > 150 ? scene.visual.slice(0, 150) + "..." : scene.visual}
                                </p>
                              </div>
                            )}
                            {isActive && range && (
                              <div className="mt-2 ml-7">
                                <div className="h-0.5 bg-zinc-800 rounded-full overflow-hidden">
                                  <div className="h-full bg-indigo-500 rounded-full transition-all duration-200"
                                    style={{ width: `${Math.min(100, ((currentTime - range.start) / (range.end - range.start)) * 100)}%` }}
                                  />
                                </div>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* ===== TELEPROMPTER VIEW ===== */}
                  {scriptViewMode === "teleprompter" && (
                    <div ref={scriptContainerRef}
                      className="max-h-[420px] overflow-y-auto scroll-smooth bg-zinc-950 rounded-lg p-6 border border-zinc-800"
                    >
                      <div className="max-w-lg mx-auto space-y-6">
                        {scenes.map((scene, i) => {
                          const range = sceneTimeRanges[i];
                          const isActive = i === activeSceneIdx;
                          const isPast = range && range.end <= currentTime;
                          const sceneProgress = isActive && range
                            ? Math.min(1, (currentTime - range.start) / (range.end - range.start))
                            : 0;

                          return (
                            <div key={scene.id} id={`script-scene-${i}`}
                              className={`transition-all duration-700 ${
                                isActive ? "scale-100 opacity-100" : isPast ? "scale-95 opacity-25" : "scale-95 opacity-40"
                              }`}
                            >
                              {/* Scene marker */}
                              <div className={`text-center mb-2 transition-all ${isActive ? "opacity-100" : "opacity-50"}`}>
                                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-medium ${
                                  isActive ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/30" : "bg-zinc-800 text-zinc-600"
                                }`}>
                                  {isActive && <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />}
                                  Scene {i + 1} &middot; {scene.title}
                                </span>
                              </div>

                              {/* Large narration text */}
                              <button onClick={() => seekToScene(i)} className="w-full text-center">
                                {scene.narration ? (
                                  <p className={`leading-loose transition-all duration-500 ${
                                    isActive ? "text-lg text-white font-medium" : "text-sm text-zinc-500"
                                  }`}>
                                    {isActive ? (
                                      (() => {
                                        const words = scene.narration.split(" ");
                                        const wordsSpoken = Math.floor(sceneProgress * words.length);
                                        return words.map((word, wi) => (
                                          <span key={wi} className={`transition-all duration-200 ${
                                            wi < wordsSpoken
                                              ? "text-zinc-400"
                                              : wi === wordsSpoken
                                                ? "text-white text-xl font-semibold"
                                                : "text-zinc-600"
                                          }`}>
                                            {word}{" "}
                                          </span>
                                        ));
                                      })()
                                    ) : (
                                      scene.narration
                                    )}
                                  </p>
                                ) : (
                                  <p className="text-sm text-zinc-700 italic">(no narration)</p>
                                )}
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Export tools */}
                  <div className="mt-3 pt-3 border-t border-zinc-800 flex gap-2">
                    <Button size="sm" variant="ghost" className="flex-1"
                      onClick={() => {
                        const fullScript = scenes.map((s, i) =>
                          `[Scene ${i + 1}: ${s.title}] (${s.duration}s)\n${s.narration}\n`
                        ).join("\n");
                        navigator.clipboard.writeText(fullScript);
                        setCopied("fullscript");
                        setTimeout(() => setCopied(null), 2000);
                      }}>
                      {copied === "fullscript" ? <><Check className="w-3 h-3 mr-1 text-green-400" /> Copied!</> : <><Copy className="w-3 h-3 mr-1" /> Copy Full Script</>}
                    </Button>
                    <Button size="sm" variant="ghost" className="flex-1"
                      onClick={() => {
                        // Export only narration as plain flowing text
                        const narration = scenes.map((s) => s.narration).filter(Boolean).join(" ");
                        navigator.clipboard.writeText(narration);
                        setCopied("narrationonly");
                        setTimeout(() => setCopied(null), 2000);
                      }}>
                      {copied === "narrationonly" ? <><Check className="w-3 h-3 mr-1 text-green-400" /> Copied!</> : <><FileText className="w-3 h-3 mr-1" /> Copy Narration</>}
                    </Button>
                    <Button size="sm" variant="ghost" className="flex-1"
                      onClick={() => {
                        const srt = scenes.map((s, i) => {
                          const range = sceneTimeRanges[i];
                          if (!range) return "";
                          const fmtSrt = (t: number) => {
                            const h = Math.floor(t / 3600);
                            const m = Math.floor((t % 3600) / 60);
                            const sec = Math.floor(t % 60);
                            const ms = Math.floor((t % 1) * 1000);
                            return `${h.toString().padStart(2,"0")}:${m.toString().padStart(2,"0")}:${sec.toString().padStart(2,"0")},${ms.toString().padStart(3,"0")}`;
                          };
                          return `${i + 1}\n${fmtSrt(range.start)} --> ${fmtSrt(range.end)}\n${s.narration}\n`;
                        }).join("\n");
                        const blob = new Blob([srt], { type: "text/srt" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = "subtitles.srt";
                        a.click();
                        URL.revokeObjectURL(url);
                      }}>
                      <Download className="w-3 h-3 mr-1" /> Export SRT
                    </Button>
                  </div>
                </Card>
              )}
            </div>
          )}

          {error && <Card className="border-red-500/30 bg-red-500/5"><p className="text-red-400 text-sm">{error}</p></Card>}
        </div>
      </div>
    </div>
  );
}
