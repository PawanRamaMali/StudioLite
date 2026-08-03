"use client";
import { useState, useEffect, useRef } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Users, Plus, Sparkles, Trash2, Copy, UserCircle, Loader2, Check,
  Upload, ImageIcon, Shield, RefreshCw, Eye, Camera, Palette, Cpu,
} from "lucide-react";
import { generateCharacterPortrait, getCharacterPortraits, getJob, getPortraitUrl, getSystemStatus, type SystemStatus } from "@/lib/api";

interface Portrait {
  filename: string;
  url: string;
  view: string;
}

interface Character {
  id: string;
  name: string;
  description: string;
  visual: string;
  portraits: Portrait[];
  generating: boolean;
  ipAdapterRegistered: boolean;
  style: string;
}

const TEMPLATE_CHARACTERS = [
  { name: "Detective Sarah", desc: "30s woman, red hair, trench coat", visual: "A woman in her 30s with fiery red curly hair, sharp green eyes, freckles across her nose, wearing a long beige trench coat over a dark turtleneck, confident posture" },
  { name: "Robot ARIA", desc: "Humanoid AI robot, sleek white design", visual: "A sleek humanoid robot with polished white and silver chassis, glowing blue LED eyes, smooth rounded features, subtle panel lines across the body" },
  { name: "Captain Wolf", desc: "Grizzled space captain, cybernetic eye", visual: "A grizzled man in his 50s with a salt-and-pepper beard, one cybernetic red eye, wearing a worn leather space captain jacket with mission patches" },
  { name: "Luna the Cat", desc: "Magical black cat with golden eyes", visual: "A sleek black cat with luminous golden eyes, a crescent moon marking on its forehead, mystical purple aura, fantasy art style" },
];

const VIEW_OPTIONS = [
  { id: "front", label: "Front", icon: "👤" },
  { id: "three_quarter", label: "3/4 View", icon: "👤" },
  { id: "side", label: "Side", icon: "👤" },
  { id: "back", label: "Back", icon: "👤" },
];

const STYLE_OPTIONS = [
  { id: "photorealistic", label: "Realistic", icon: <Camera className="w-3.5 h-3.5" /> },
  { id: "anime", label: "Anime", icon: <Palette className="w-3.5 h-3.5" /> },
  { id: "3d_render", label: "3D / Pixar", icon: <Eye className="w-3.5 h-3.5" /> },
];

export default function CharactersPanel() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [visual, setVisual] = useState("");
  const [style, setStyle] = useState("photorealistic");
  const [selectedViews, setSelectedViews] = useState<string[]>(["front", "three_quarter"]);
  const [copied, setCopied] = useState<string | null>(null);
  const [refUploading, setRefUploading] = useState(false);
  const [aiVisualLoading, setAiVisualLoading] = useState(false);
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);
  const hasGpu = sysStatus?.capabilities?.cuda ?? true;
  const gpuChecked = sysStatus !== null;

  useEffect(() => {
    getSystemStatus().then(setSysStatus).catch(() => setSysStatus(null));
  }, []);

  const toggleView = (viewId: string) => {
    setSelectedViews((prev) =>
      prev.includes(viewId) ? prev.filter((v) => v !== viewId) : [...prev, viewId]
    );
  };

  const generatePortraits = async (charId: string, charName: string, charDesc: string, charVisual: string, charStyle: string, views: string[]) => {
    // Mark character as generating
    setCharacters((prev) => prev.map((c) => c.id === charId ? { ...c, generating: true } : c));

    try {
      const job = await generateCharacterPortrait({
        name: charName,
        description: charDesc,
        visual_prompt: charVisual,
        views: views.length > 0 ? views : ["front"],
        style: charStyle,
        register_ip_adapter: true,
      });

      // Poll for completion
      const pollJob = async () => {
        let attempts = 0;
        while (attempts < 120) { // 2 min max
          await new Promise((r) => setTimeout(r, 2000));
          const status = await getJob(job.job_id);

          if (status.status === "completed" && status.result) {
            const portraits: Portrait[] = [];
            const result = status.result as { portraits?: Record<string, { url: string; filename: string }> };
            if (result.portraits) {
              for (const [view, info] of Object.entries(result.portraits)) {
                portraits.push({ view, url: info.url, filename: info.filename });
              }
            }
            setCharacters((prev) =>
              prev.map((c) =>
                c.id === charId ? { ...c, portraits, generating: false, ipAdapterRegistered: true } : c
              )
            );
            return;
          }
          if (status.status === "failed") {
            setCharacters((prev) => prev.map((c) => c.id === charId ? { ...c, generating: false } : c));
            return;
          }
          attempts++;
        }
        setCharacters((prev) => prev.map((c) => c.id === charId ? { ...c, generating: false } : c));
      };
      pollJob();
    } catch {
      setCharacters((prev) => prev.map((c) => c.id === charId ? { ...c, generating: false } : c));
    }
  };

  const addAndGenerate = () => {
    if (!name.trim() || !desc.trim()) return;
    const id = crypto.randomUUID();
    const newChar: Character = {
      id,
      name,
      description: desc,
      visual: visual || desc,
      portraits: [],
      generating: true,
      ipAdapterRegistered: false,
      style,
    };
    setCharacters((prev) => [...prev, newChar]);
    setName(""); setDesc(""); setVisual("");

    // Generate portraits immediately
    generatePortraits(id, newChar.name, newChar.description, newChar.visual, style, selectedViews);
  };

  const uploadCustomImage = async (charId: string, file: File) => {
    setRefUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/ip-adapter/upload-reference?char_id=${charId}`,
        { method: "POST", body: formData }
      );
      const data = await res.json();
      if (data.status === "ok") {
        setCharacters((prev) =>
          prev.map((c) =>
            c.id === charId
              ? { ...c, ipAdapterRegistered: true, portraits: [...c.portraits, { filename: file.name, url: "", view: "uploaded" }] }
              : c
          )
        );
      }
    } catch (err) { alert(`Upload failed: ${err instanceof Error ? err.message : "Check if API server is running"}`); } finally {
      setRefUploading(false);
    }
  };

  const regeneratePortraits = (char: Character) => {
    generatePortraits(char.id, char.name, char.description, char.visual, char.style, ["front", "three_quarter"]);
  };

  const copyPrompt = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const useTemplate = (t: typeof TEMPLATE_CHARACTERS[0]) => {
    setName(t.name);
    setDesc(t.desc);
    setVisual(t.visual);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Character Studio</h1>
        <p className="text-zinc-400 mt-1">
          Generate real character portraits for consistent appearance across your movie
        </p>
      </div>

      {gpuChecked && !hasGpu ? (
        <div className="mb-4 flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
          <Cpu className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-amber-300">Character portrait generation requires an NVIDIA GPU.</p>
            <p className="text-amber-200/80 mt-1">
              Portraits are rendered locally via SDXL, which needs a CUDA device. You can still draft characters here
              and reuse them once GPU support is available (or configure a cloud image provider in Settings).
            </p>
          </div>
        </div>
      ) : (
        <div className="mb-4 px-4 py-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center gap-2">
          <Camera className="w-4 h-4 text-indigo-400 flex-shrink-0" />
          <p className="text-xs text-indigo-300">
            Characters are generated as actual images (front, side, 3/4 views) using SDXL, then registered with IP-Adapter for face/appearance consistency across all scenes.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Create form */}
        <div className="space-y-4">
          {/* Templates */}
          <Card>
            <CardTitle className="text-sm mb-2">Quick Templates</CardTitle>
            <div className="space-y-1.5">
              {TEMPLATE_CHARACTERS.map((t) => (
                <button key={t.name} onClick={() => useTemplate(t)}
                  className="w-full text-left px-2.5 py-2 rounded-lg text-xs hover:bg-zinc-800 transition-colors border border-transparent hover:border-zinc-700">
                  <span className="font-medium text-zinc-200">{t.name}</span>
                  <span className="text-zinc-500 ml-1.5">{t.desc}</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Create form */}
          <Card>
            <CardTitle className="text-sm mb-3">Create Character</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] text-zinc-500 uppercase">Name</label>
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Character name"
                  className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 mt-0.5" />
              </div>
              <div>
                <label className="text-[10px] text-zinc-500 uppercase">Description / Traits</label>
                <input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="e.g., 30s woman, red curly hair, green eyes, trench coat"
                  className="w-full bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 mt-0.5" />
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-[10px] text-zinc-500 uppercase">Detailed Visual (optional)</label>
                  <button onClick={() => {
                    if (!name.trim() || !desc.trim()) return;
                    setAiVisualLoading(true);
                    // Build a rich visual prompt from the name + traits
                    // This is deterministic and always produces image-generation-ready text
                    const traits = desc.trim();
                    const parts: string[] = [];
                    parts.push(`Character portrait of ${name}`);
                    parts.push(traits);
                    // Expand common shorthand traits into visual detail
                    if (!traits.includes("skin")) parts.push("detailed skin texture");
                    if (!traits.includes("eye")) parts.push("expressive detailed eyes");
                    if (!traits.includes("lighting")) parts.push("professional studio lighting");
                    parts.push("highly detailed");
                    parts.push("sharp focus");
                    parts.push("consistent recognizable features");
                    parts.push("full body visible");
                    parts.push("clean neutral background");
                    parts.push("character reference sheet quality");
                    if (style === "photorealistic") parts.push("photorealistic, 8K, professional photography");
                    else if (style === "anime") parts.push("anime style, cel shaded, clean lines, vibrant colors");
                    else if (style === "3d_render") parts.push("3D rendered, Pixar style, subsurface scattering");
                    setVisual(parts.join(", "));
                    setTimeout(() => setAiVisualLoading(false), 300);
                  }} disabled={aiVisualLoading || !name.trim() || !desc.trim()}
                    className="flex items-center gap-1 text-[10px] text-indigo-400 hover:text-indigo-300 disabled:text-zinc-600 disabled:cursor-not-allowed">
                    {aiVisualLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
                    AI Generate
                  </button>
                </div>
                <textarea value={visual} onChange={(e) => setVisual(e.target.value)}
                  placeholder="Click 'AI Generate' to create a detailed description from name + traits, or write your own..."
                  className="w-full h-20 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:ring-2 focus:ring-indigo-500/50 resize-none mt-0.5" />
              </div>

              {/* Style selector */}
              <div>
                <label className="text-[10px] text-zinc-500 uppercase mb-1.5 block">Art Style</label>
                <div className="flex gap-2">
                  {STYLE_OPTIONS.map((s) => (
                    <button key={s.id} onClick={() => setStyle(s.id)}
                      className={`flex-1 flex items-center justify-center gap-1.5 px-2.5 py-2 rounded-lg text-xs border transition-colors ${
                        style === s.id
                          ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                          : "bg-zinc-800/50 border-zinc-700 text-zinc-400 hover:border-zinc-600"
                      }`}>
                      {s.icon}
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* View selector */}
              <div>
                <label className="text-[10px] text-zinc-500 uppercase mb-1.5 block">Generate Views</label>
                <div className="flex gap-2 flex-wrap">
                  {VIEW_OPTIONS.map((v) => (
                    <button key={v.id} onClick={() => toggleView(v.id)}
                      className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs border transition-colors ${
                        selectedViews.includes(v.id)
                          ? "bg-purple-600/20 border-purple-500/50 text-purple-300"
                          : "bg-zinc-800/50 border-zinc-700 text-zinc-500 hover:border-zinc-600"
                      }`}>
                      {v.label}
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-zinc-600 mt-1">More views = better consistency but longer generation</p>
              </div>

              <Button className="w-full" onClick={addAndGenerate}
                disabled={!name.trim() || !desc.trim() || !hasGpu}>
                {!hasGpu ? (
                  <><Cpu className="w-3.5 h-3.5 mr-1.5" /> Requires NVIDIA GPU</>
                ) : (
                  <><Sparkles className="w-3.5 h-3.5 mr-1.5" /> Generate Character</>
                )}
              </Button>
            </div>
          </Card>
        </div>

        {/* Right column - Character gallery */}
        <div className="lg:col-span-2">
          {characters.length > 0 ? (
            <div className="space-y-4">
              {characters.map((char) => (
                <Card key={char.id} className="group">
                  <div className="flex items-start gap-4">
                    {/* Character info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-sm">{char.name}</h3>
                        <Badge className="text-[9px] bg-zinc-700/50 text-zinc-400 border-zinc-600/30">{char.style}</Badge>
                        {char.ipAdapterRegistered && (
                          <Badge className="text-[9px] bg-purple-600/20 text-purple-300 border-purple-500/20">
                            <Shield className="w-2.5 h-2.5 mr-0.5" /> IP-Adapter
                          </Badge>
                        )}
                        {char.generating && (
                          <Badge className="text-[9px] bg-amber-600/20 text-amber-300 border-amber-500/20 animate-pulse">
                            <Loader2 className="w-2.5 h-2.5 mr-0.5 animate-spin" /> Generating...
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-zinc-500">{char.description}</p>
                      {char.visual && char.visual !== char.description && (
                        <p className="text-xs text-zinc-600 mt-1 line-clamp-2 italic">{char.visual}</p>
                      )}
                    </div>
                  </div>

                  {/* Portrait gallery */}
                  {char.portraits.length > 0 ? (
                    <div className="mt-3 pt-3 border-t border-zinc-800">
                      <div className="flex gap-3 overflow-x-auto pb-2">
                        {char.portraits.map((p, idx) => (
                          <div key={idx} className="flex-shrink-0">
                            {p.url ? (
                              <div className="relative group/img">
                                <img
                                  src={getPortraitUrl(p.url)}
                                  alt={`${char.name} - ${p.view}`}
                                  className="w-32 h-32 object-cover rounded-lg border border-zinc-700 hover:border-indigo-500/50 transition-colors"
                                />
                                <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-[10px] text-zinc-300 text-center py-0.5 rounded-b-lg">
                                  {p.view}
                                </div>
                              </div>
                            ) : (
                              <div className="w-32 h-32 rounded-lg border border-zinc-700 bg-zinc-800/50 flex items-center justify-center">
                                <div className="text-center">
                                  <Upload className="w-5 h-5 text-zinc-600 mx-auto" />
                                  <span className="text-[10px] text-zinc-600 mt-1 block">{p.view}</span>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : char.generating ? (
                    <div className="mt-3 pt-3 border-t border-zinc-800">
                      <div className="flex gap-3">
                        {(selectedViews.length > 0 ? selectedViews : ["front"]).map((v) => (
                          <div key={v} className="w-32 h-32 rounded-lg border border-zinc-700 bg-zinc-800/50 flex items-center justify-center animate-pulse">
                            <div className="text-center">
                              <Loader2 className="w-5 h-5 text-zinc-600 mx-auto animate-spin" />
                              <span className="text-[10px] text-zinc-600 mt-1 block">{v}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {/* Actions */}
                  <div className="flex gap-2 mt-3 pt-3 border-t border-zinc-800 flex-wrap">
                    <Button size="sm" variant="secondary" onClick={() => regeneratePortraits(char)} disabled={char.generating || !hasGpu}>
                      <RefreshCw className={`w-3 h-3 mr-1 ${char.generating ? "animate-spin" : ""}`} /> Regenerate
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => {
                      const input = document.createElement("input");
                      input.type = "file"; input.accept = "image/*";
                      input.onchange = (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0];
                        if (file) uploadCustomImage(char.id, file);
                      };
                      input.click();
                    }} disabled={refUploading}>
                      {refUploading ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Upload className="w-3 h-3 mr-1" />}
                      Upload Own Image
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => copyPrompt(char.id, char.visual)}>
                      {copied === char.id ? <><Check className="w-3 h-3 mr-1 text-green-400" /> Copied!</> : <><Copy className="w-3 h-3 mr-1" /> Copy Prompt</>}
                    </Button>
                    <Button size="sm" variant="danger" onClick={() => setCharacters((prev) => prev.filter((c) => c.id !== char.id))}>
                      <Trash2 className="w-3 h-3 mr-1" /> Delete
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="min-h-[400px] flex items-center justify-center">
              <div className="text-center">
                <Users className="w-16 h-16 text-zinc-700 mx-auto mb-4" />
                <p className="text-zinc-400 text-sm font-medium">No characters yet</p>
                <p className="text-zinc-600 text-xs mt-1 max-w-sm mx-auto">
                  Create a character and we&apos;ll generate actual portrait images (front, side, 3/4 views) using AI.
                  These images are used by IP-Adapter to keep your character&apos;s face and appearance consistent across all movie scenes.
                </p>
                <div className="flex items-center justify-center gap-4 mt-4 text-[10px] text-zinc-600">
                  <span className="flex items-center gap-1"><Camera className="w-3 h-3" /> SDXL generates portraits</span>
                  <span className="flex items-center gap-1"><Shield className="w-3 h-3" /> IP-Adapter locks appearance</span>
                  <span className="flex items-center gap-1"><ImageIcon className="w-3 h-3" /> Or upload your own photos</span>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
