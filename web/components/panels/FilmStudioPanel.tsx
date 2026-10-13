"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Film, Play, Pause, RotateCcw, Trash2, Plus, ArrowLeft, Save, X,
  Loader2, CheckCircle2, Circle, AlertCircle, Clock, ChevronRight,
  Download, Sparkles, Music, Mic, Volume2, Image as ImageIcon,
  Palette, Type as TypeIcon, ArrowUpCircle,
} from "lucide-react";
import {
  FILM_API_BASE, filmCreate, filmCreateFromTemplate, filmDelete,
  filmEditArtifact, filmGet, filmList, filmListStages, filmListTemplates,
  filmPatchConfig, filmPause, filmRewind, filmRun, filmStreamUrl, getLLMBackends,
  listBackends,
  type FilmConfig, type FilmDetail, type FilmListItem, type FilmMeta,
  type FilmStageKey, type FilmStageSpec,
  type FilmStageStatus, type FilmTemplate, type LLMBackendInfo,
} from "@/lib/api";

type ProjectListRes = { projects: FilmListItem[] };

const STATUS_META: Record<FilmStageStatus, { label: string; tone: string }> = {
  pending:       { label: "Pending",       tone: "text-zinc-500" },
  running:       { label: "Running…",      tone: "text-indigo-400" },
  paused:        { label: "Paused",        tone: "text-amber-400" },
  done:          { label: "Done",          tone: "text-green-400" },
  failed:        { label: "Failed",        tone: "text-red-400" },
  stale:         { label: "Needs rerun",   tone: "text-orange-400" },
  needs_review:  { label: "Awaiting you",  tone: "text-sky-400" },
};

function fmtRelTime(ts: number | null | undefined): string {
  if (!ts) return "";
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export default function FilmStudioPanel() {
  const [projects, setProjects] = useState<FilmListItem[] | null>(null);
  const [stages, setStages] = useState<FilmStageSpec[] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Deep-link support: `?film=<id>` opens that film's detail on mount.
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const filmId = params.get("film");
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (filmId) setActiveId(filmId);
    } catch { /* ignore */ }
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      const res: ProjectListRes = await filmList();
      setProjects(res.projects);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to list films");
    }
  }, []);

  useEffect(() => {
    // loadProjects is an async callback that reduces its setState via await,
    // but the lint rule can't see through the returned promise - mark it.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadProjects();
    filmListStages().then((r) => setStages(r.stages)).catch(() => { /* soft */ });
  }, [loadProjects]);

  return (
    <div>
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Film Studio</h1>
          <p className="text-zinc-400 mt-1">
            An orchestrated crew of AI agents - producer to editor - turning your brief into a short film.
          </p>
        </div>
        {activeId && (
          <Button variant="ghost" size="sm" onClick={() => setActiveId(null)}>
            <ArrowLeft className="w-4 h-4 mr-1.5" /> All films
          </Button>
        )}
      </div>

      {error && (
        <Card className="border-red-500/30 bg-red-500/5 mb-4">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5" />
            <p className="text-xs text-red-300">{error}</p>
          </div>
        </Card>
      )}

      {activeId ? (
        <ProjectView
          projectId={activeId}
          stages={stages || []}
          onGone={() => { setActiveId(null); loadProjects(); }}
        />
      ) : (
        <ProjectListView
          projects={projects}
          stages={stages || []}
          onOpen={setActiveId}
          onCreated={(id) => { setActiveId(id); loadProjects(); }}
          onDeleted={loadProjects}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LLM backend catalog - used by the create form's advanced section
// ---------------------------------------------------------------------------

const _DEFAULT_MODELS: Record<"ollama" | "gemini" | "groq" | "hf", string> = {
  ollama: "llama3.2",
  gemini: "gemini-2.5-flash",
  groq:   "llama-3.3-70b-versatile",
  hf:     "meta-llama/Meta-Llama-3-70B-Instruct",
};

const _MODEL_SUGGESTIONS: Record<"ollama" | "gemini" | "groq" | "hf", string[]> = {
  ollama: ["llama3.2", "llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:7b", "phi3.5"],
  gemini: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
  groq:   ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768",
           "qwen/qwen3-32b", "openai/gpt-oss-120b"],
  hf:     ["meta-llama/Meta-Llama-3-70B-Instruct",
           "meta-llama/Meta-Llama-3-8B-Instruct",
           "mistralai/Mistral-Nemo-Instruct-2407"],
};

const _BACKEND_HELP: Record<"ollama" | "gemini" | "groq" | "hf", string> = {
  ollama: "Local. Requires `ollama serve` running.",
  gemini: "Google Gemini free tier. Requires GEMINI_API_KEY.",
  groq:   "Free OpenAI-compat inference. Requires GROQ_API_KEY.",
  hf:     "Hugging Face Router. Requires HF_TOKEN.",
};

// ---------------------------------------------------------------------------
// List view + create form
// ---------------------------------------------------------------------------

function ProjectListView({
  projects, stages, onOpen, onCreated, onDeleted,
}: {
  projects: FilmListItem[] | null;
  stages: FilmStageSpec[];
  onOpen: (id: string) => void;
  onCreated: (id: string) => void;
  onDeleted: () => void;
}) {
  const [brief, setBrief] = useState("");
  const [title, setTitle] = useState("");
  const [style, setStyle] = useState<"stylized" | "photoreal">("stylized");
  const [targetMinutes, setTargetMinutes] = useState(2);
  const [llmBackend, setLlmBackend] = useState<"ollama" | "gemini" | "groq" | "hf">("ollama");
  const [model, setModel] = useState("llama3.2");
  const [llmBackends, setLlmBackends] = useState<Record<string, LLMBackendInfo> | null>(null);
  const [quality, setQuality] = useState<"draft" | "standard" | "high" | "ultra">("standard");
  const [sdxlVariant, setSdxlVariant] = useState("turbo");
  const [motionBackend, setMotionBackend] = useState("auto");
  const [voiceBackend, setVoiceBackend] = useState("piper");
  const [musicBackend, setMusicBackend] = useState("musicgen");
  const [upscaleBackend, setUpscaleBackend] = useState("none");
  const [lipsyncBackend, setLipsyncBackend] = useState("none");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // First-run quickstart: a short catalog of templates so the "new film"
  // screen can prefill brief + config with one click. Fetched once at mount;
  // soft-fails to an empty array so the create form still works offline.
  const [templates, setTemplates] = useState<FilmTemplate[]>([]);
  const [spawningFromTemplate, setSpawningFromTemplate] = useState<string | null>(null);

  useEffect(() => {
    // Probe once so the backend picker can gate cloud options behind
    // "API key configured" without the user having to guess.
    getLLMBackends().then((r) => setLlmBackends(r.backends)).catch(() => { /* soft */ });
    filmListTemplates().then((r) => setTemplates(r.templates)).catch(() => { /* soft */ });
  }, []);

  const spawnFromTemplate = async (t: FilmTemplate) => {
    setErr(null);
    setSpawningFromTemplate(t.id);
    try {
      const res = await filmCreateFromTemplate({ template_id: t.id });
      onCreated(res.project.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : `Could not spawn template ${t.name}`);
    } finally {
      setSpawningFromTemplate(null);
    }
  };

  // When the user switches backend, drop the model back to that backend's
  // default so we don't send `llama3.2` to Groq or `gemini-2.5-flash` to Ollama.
  const changeBackend = (b: "ollama" | "gemini" | "groq" | "hf") => {
    setLlmBackend(b);
    setModel(_DEFAULT_MODELS[b]);
  };

  const create = async () => {
    if (brief.trim().length < 8) {
      setErr("Give the producer at least a sentence to work with.");
      return;
    }
    setErr(null);
    setCreating(true);
    try {
      const res = await filmCreate({
        brief: brief.trim(),
        title: title.trim() || undefined,
        config: {
          llm_backend: llmBackend,
          llm_model: model.trim() || _DEFAULT_MODELS[llmBackend],
          style,
          target_minutes: targetMinutes,
          quality,
          sdxl_variant: sdxlVariant,
          motion_backend: motionBackend,
          voice_backend: voiceBackend,
          music_backend: musicBackend,
          upscale_backend: upscaleBackend,
          lipsync_backend: lipsyncBackend,
        },
      });
      onCreated(res.project.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wide">Your films</h2>
          <span className="text-xs text-zinc-500">{projects?.length ?? "…"} project{projects?.length === 1 ? "" : "s"}</span>
        </div>
        {projects === null && (
          <div className="text-zinc-500 text-sm">Loading…</div>
        )}
        {projects && projects.length === 0 && (
          <Card className="text-zinc-500 text-sm text-center py-8">
            No films yet. Give the crew a brief on the right to start your first one.
          </Card>
        )}
        {projects && projects.map((p) => (
          <Card key={p.id} className="hover:border-zinc-700 transition-colors">
            <div className="flex items-start justify-between gap-3">
              <button className="text-left flex-1 min-w-0" onClick={() => onOpen(p.id)}>
                <CardTitle className="text-base flex items-center gap-2">
                  <Film className="w-4 h-4 text-indigo-400" />
                  <span className="truncate">{p.title || "Untitled Film"}</span>
                </CardTitle>
                <p className="text-xs text-zinc-500 mt-1 line-clamp-2">{p.brief}</p>
                <p className="text-[10px] text-zinc-600 mt-2">Updated {fmtRelTime(p.updated_at)}</p>
              </button>
              <div className="flex items-center gap-1 flex-shrink-0">
                <Button variant="secondary" size="sm" onClick={() => onOpen(p.id)}>
                  Open <ChevronRight className="w-3.5 h-3.5 ml-1" />
                </Button>
                <Button
                  variant="ghost" size="sm"
                  onClick={async () => {
                    if (!window.confirm(`Delete "${p.title}"? Cannot be undone.`)) return;
                    try { await filmDelete(p.id); onDeleted(); }
                    catch (e) { window.alert(e instanceof Error ? e.message : "Delete failed"); }
                  }}
                  title="Delete this film"
                >
                  <Trash2 className="w-3.5 h-3.5 text-zinc-500" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="space-y-3">
        {templates.length > 0 && projects && projects.length === 0 && (
          <Card className="border-indigo-500/20 bg-indigo-500/[0.03]">
            <CardTitle className="text-sm mb-1 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" /> Quickstart
            </CardTitle>
            <p className="text-[11px] text-zinc-500 mb-3">
              Pick a template to spawn a project with a working brief and the config knobs
              already set. Edit it from the project view once it's created.
            </p>
            <div className="space-y-1.5">
              {templates.slice(0, 4).map((t) => (
                <button
                  key={t.id}
                  type="button"
                  disabled={spawningFromTemplate !== null}
                  onClick={() => spawnFromTemplate(t)}
                  className="w-full text-left bg-zinc-900/40 border border-zinc-800 hover:border-indigo-500/40 rounded px-2.5 py-2 transition-colors disabled:opacity-50"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-xs text-zinc-200 font-medium flex items-center gap-1.5">
                        {t.name}
                        <span className="text-[9px] text-zinc-500 font-normal">
                          · {t.target_minutes}m · {t.style}
                        </span>
                      </div>
                      <div className="text-[11px] text-zinc-500 line-clamp-1 mt-0.5">
                        {t.description}
                      </div>
                    </div>
                    {spawningFromTemplate === t.id ? (
                      <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin flex-shrink-0 mt-0.5" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-zinc-600 flex-shrink-0 mt-0.5" />
                    )}
                  </div>
                </button>
              ))}
            </div>
          </Card>
        )}

        <Card>
          <CardTitle className="text-sm mb-3 flex items-center gap-2">
            <Plus className="w-4 h-4 text-indigo-400" /> New film
          </CardTitle>
          <div className="space-y-3">
            <div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">Title (optional)</label>
              <input
                value={title} onChange={(e) => setTitle(e.target.value)}
                placeholder="Working title"
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">Brief</label>
              <textarea
                value={brief} onChange={(e) => setBrief(e.target.value)}
                rows={6}
                placeholder="One paragraph. A ghost story about a barista who serves the same regular for 30 years and slowly realizes he only exists during her shift…"
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:ring-2 focus:ring-indigo-500/50 resize-y"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] uppercase tracking-wide text-zinc-500">Style</label>
                <select
                  value={style}
                  onChange={(e) => setStyle(e.target.value as "stylized" | "photoreal")}
                  className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200"
                >
                  <option value="stylized">Stylized / animated</option>
                  <option value="photoreal">Photoreal</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] uppercase tracking-wide text-zinc-500">Target minutes</label>
                <input
                  type="number" min={1} max={5} step={0.5}
                  value={targetMinutes}
                  onChange={(e) => setTargetMinutes(parseFloat(e.target.value) || 2)}
                  className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200"
                />
              </div>
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">LLM backend</label>
              <div className="grid grid-cols-4 gap-1 mt-1">
                {(["ollama", "gemini", "groq", "hf"] as const).map((b) => {
                  const info = llmBackends?.[b];
                  const ready = b === "ollama"
                    ? (info?.reachable ?? true)
                    : (info?.configured ?? false);
                  return (
                    <button
                      type="button"
                      key={b}
                      onClick={() => changeBackend(b)}
                      title={_BACKEND_HELP[b] + (info && !ready
                        ? ` - not ${b === "ollama" ? "reachable" : "configured"}`
                        : "")}
                      className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                        llmBackend === b
                          ? "bg-indigo-500/15 border-indigo-500/40 text-indigo-300"
                          : "bg-zinc-800/40 border-zinc-700 text-zinc-300 hover:border-zinc-600"
                      } ${!ready && "opacity-60"}`}
                    >
                      <span className="capitalize">{b === "hf" ? "HF" : b}</span>
                      <span className={`ml-1 inline-block w-1.5 h-1.5 rounded-full ${
                        ready ? "bg-green-500" : "bg-zinc-600"
                      }`} />
                    </button>
                  );
                })}
              </div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500 mt-2 block">Model</label>
              <input
                value={model} onChange={(e) => setModel(e.target.value)}
                placeholder={_DEFAULT_MODELS[llmBackend]}
                list={`llm-models-${llmBackend}`}
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 font-mono"
              />
              <datalist id={`llm-models-${llmBackend}`}>
                {_MODEL_SUGGESTIONS[llmBackend].map((m) => <option key={m} value={m} />)}
              </datalist>
              <p className="text-[10px] text-zinc-500 mt-1">
                {_BACKEND_HELP[llmBackend]}
                {llmBackend !== "ollama" && llmBackends && !llmBackends[llmBackend]?.configured && (
                  <>
                    {" "}Set{" "}
                    <code className="text-amber-300">{llmBackends[llmBackend]?.env_var}</code>
                    {" "}in <code className="text-zinc-300">.env</code> next to <code className="text-zinc-300">api_server.py</code>.
                  </>
                )}
              </p>
            </div>

            <button type="button"
                    onClick={() => setShowAdvanced((v) => !v)}
                    className="w-full text-left text-[10px] uppercase tracking-wide text-zinc-500 hover:text-zinc-300 flex items-center gap-1">
              <ChevronRight className={`w-3 h-3 transition-transform ${showAdvanced ? "rotate-90" : ""}`} />
              Advanced backends
            </button>
            {showAdvanced && (
              <div className="border-l-2 border-zinc-800 pl-3 space-y-2">
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-zinc-500">Image quality preset</label>
                  <select value={quality} onChange={(e) => setQuality(e.target.value as typeof quality)}
                          className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                    <option value="draft">draft - fast iteration</option>
                    <option value="standard">standard - default</option>
                    <option value="high">high - finishing look</option>
                    <option value="ultra">ultra - final render</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-zinc-500">SDXL variant</label>
                  <select value={sdxlVariant} onChange={(e) => setSdxlVariant(e.target.value)}
                          className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                    <option value="turbo">SDXL Turbo - 1–4 steps (default)</option>
                    <option value="base">SDXL 1.0 - sharper, 30–50 steps</option>
                    <option value="z-image-turbo">Z-Image Turbo - Flux-tier at Turbo cost (if cached)</option>
                    <option value="flux-schnell">Flux Schnell - 4 steps (if cached)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-zinc-500">Motion backend</label>
                  <select value={motionBackend} onChange={(e) => setMotionBackend(e.target.value)}
                          className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                    <option value="auto">auto (SVD → AnimateDiff → Wan → Ken Burns)</option>
                    <option value="svd">SVD 1.1</option>
                    <option value="animatediff">AnimateDiff</option>
                    <option value="wan22">Wan 2.2 TI2V-5B (12 GB VRAM)</option>
                    <option value="framepack">FramePack (~6 GB VRAM, any length)</option>
                    <option value="ltx">LTX-Video (fastest)</option>
                    <option value="kenburns">Ken Burns only (no ML motion)</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-zinc-500">Voice backend</label>
                    <select value={voiceBackend} onChange={(e) => setVoiceBackend(e.target.value)}
                            className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                      <option value="piper">Piper (fast, flat)</option>
                      <option value="xtts">XTTS v2 (24 kHz)</option>
                      <option value="indextts2">IndexTTS-2 (emotion)</option>
                      <option value="qwen3tts">Qwen3-TTS (voice-design)</option>
                      <option value="chatterbox">Chatterbox (expressive)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] uppercase tracking-wide text-zinc-500">Music backend</label>
                    <select value={musicBackend} onChange={(e) => setMusicBackend(e.target.value)}
                            className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                      <option value="musicgen">MusicGen (CC-BY-NC)</option>
                      <option value="acestep">ACE-Step (MIT)</option>
                      <option value="diffrhythm2">DiffRhythm 2 (roadmap - needs install or cloud)</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label htmlFor="upscale-backend" className="text-[10px] uppercase tracking-wide text-zinc-500">Final upscale</label>
                    <select id="upscale-backend" value={upscaleBackend} onChange={(e) => setUpscaleBackend(e.target.value)}
                            className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                      <option value="none">Off (finish at Editor / Titles)</option>
                      <option value="realesrgan">Real-ESRGAN to ~1440p</option>
                      <option value="seedvr2">SeedVR2 (roadmap - needs install or cloud)</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="lipsync-backend" className="text-[10px] uppercase tracking-wide text-zinc-500">Lip sync backend</label>
                    <select id="lipsync-backend" value={lipsyncBackend} onChange={(e) => setLipsyncBackend(e.target.value)}
                            className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200">
                      <option value="none">Off (mouths stay as rendered)</option>
                      <option value="musetalk">MuseTalk 1.5 (isolated venv)</option>
                      <option value="latentsync">LatentSync 1.5 (isolated venv)</option>
                      <option value="echomimicv3">EchoMimicV3 (holistic, Apache-2.0)</option>
                    </select>
                  </div>
                </div>
                <p className="text-[10px] text-zinc-500">
                  Unavailable backends fall through to defaults automatically - nothing you pick here breaks the run.
                </p>
              </div>
            )}
            {err && <p className="text-xs text-red-300">{err}</p>}
            <Button className="w-full" onClick={create} disabled={creating}>
              {creating ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Creating…</> :
                          <><Sparkles className="w-4 h-4 mr-2" /> Create project</>}
            </Button>
          </div>
        </Card>

        {stages.length > 0 && (
          <Card>
            <CardTitle className="text-sm mb-2">The crew</CardTitle>
            <ol className="space-y-1.5">
              {stages.map((s, i) => (
                <li key={s.key} className="text-xs text-zinc-400 flex items-start gap-2">
                  <span className="text-zinc-600 font-mono w-5">{i + 1}.</span>
                  <span>
                    <span className="text-zinc-200 font-medium">{s.label}</span>
                    <span className="text-zinc-500"> - {s.description}</span>
                    {s.gated_by_default && (
                      <Badge className="ml-1.5 text-[9px]">gate</Badge>
                    )}
                  </span>
                </li>
              ))}
            </ol>
          </Card>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail view for one project
// ---------------------------------------------------------------------------

function ProjectView({
  projectId, stages, onGone,
}: {
  projectId: string;
  stages: FilmStageSpec[];
  onGone: () => void;
}) {
  const [detail, setDetail] = useState<FilmDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);   // action name in flight
  const [selectedStage, setSelectedStage] = useState<FilmStageKey | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [wsAlive, setWsAlive] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setDetail(await filmGet(projectId));
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load project");
    }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Live event stream. Any state-changing event triggers a refetch so the
  // full detail (artifacts, final_url) stays in sync without us duplicating
  // the merge logic on the client.
  useEffect(() => {
    const ws = new WebSocket(filmStreamUrl(projectId));
    wsRef.current = ws;
    ws.onopen = () => setWsAlive(true);
    ws.onclose = () => setWsAlive(false);
    ws.onerror = () => setWsAlive(false);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "event" || msg.type === "snapshot") {
          refresh();
        }
      } catch { /* ignore */ }
    };
    return () => {
      try { ws.close(); } catch { /* ignore */ }
      wsRef.current = null;
    };
  }, [projectId, refresh]);

  // Polling fallback. The WebSocket above rides through Next.js's dev-server
  // rewrite of /api/*, which does NOT proxy WS upgrades, so live events
  // silently never arrive and the panel stays frozen on whatever the initial
  // fetch showed. Polling every 3 seconds whenever a stage is active keeps
  // the UI honest even when WS is dead - the refresh is a cheap ~200ms
  // fetch. When nothing is in flight we don't poll, so idle films are quiet.
  const anyActive = useMemo(() => {
    if (!detail) return false;
    return Object.values(detail.state.stage_status).some(
      (s) => s === "running" || s === "paused" || s === "needs_review"
    );
  }, [detail]);
  useEffect(() => {
    if (!anyActive) return;
    const id = setInterval(() => { refresh(); }, 3000);
    return () => clearInterval(id);
  }, [anyActive, refresh]);

  const runAction = async (name: string, fn: () => Promise<unknown>) => {
    setBusy(name);
    try { await fn(); await refresh(); }
    catch (e) { setErr(e instanceof Error ? e.message : `${name} failed`); }
    finally { setBusy(null); }
  };

  const currentStageStatus = useMemo(() => {
    if (!detail) return null;
    const running = (Object.entries(detail.state.stage_status) as [FilmStageKey, FilmStageStatus][])
      .find(([, s]) => s === "running" || s === "needs_review" || s === "failed" || s === "paused");
    return running?.[0] || null;
  }, [detail]);

  if (!detail) {
    return <Card className="text-zinc-500 text-sm">Loading project…</Card>;
  }

  const { project, state, artifacts, final_url, final_mixed_url } = detail;
  const playbackUrl = final_mixed_url || final_url;
  const hasAudio = Boolean(final_mixed_url);
  const isRunning = Object.values(state.stage_status).some((s) => s === "running");
  const isAwaiting = Object.values(state.stage_status).some((s) => s === "needs_review");
  const hasStale = Object.values(state.stage_status).some((s) => s === "stale");
  const allDone = Object.values(state.stage_status).every((s) => s === "done");

  const doneCount = Object.values(state.stage_status).filter((s) => s === "done").length;
  const currentStageSpec = state.current_stage
    ? stages.find((s) => s.key === state.current_stage) ?? null
    : null;

  return (
    <div className="space-y-4">
      {/* Compact meta strip: brief + config badges. No more separate card taking a whole column. */}
      <div className="flex flex-col md:flex-row md:items-center gap-3 md:gap-6">
        <p className="text-xs text-zinc-400 leading-relaxed md:flex-1 line-clamp-2">{project.brief}</p>
        <div className="flex items-center gap-1.5 flex-wrap text-[10px]">
          <Badge>{project.config.style}</Badge>
          <Badge>{project.config.target_minutes} min</Badge>
          <Badge>{project.config.llm_model}</Badge>
          <span className="text-zinc-600">· Updated {fmtRelTime(project.updated_at)}</span>
        </div>
      </div>

      <BackendSettings project={project} isRunning={isRunning} onSaved={refresh} />

      {err && (
        <Card className="border-red-500/30 bg-red-500/5">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 mt-0.5" />
              <p className="text-xs text-red-300">{err}</p>
            </div>
            <button onClick={() => setErr(null)} className="text-zinc-500 hover:text-zinc-300">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </Card>
      )}

      {/* HERO - the star of the page. Video when there is one; live status when running; call to
          action when idle. Only one of these renders at a time. */}
      {playbackUrl ? (
        <Card className="border-green-500/20 bg-green-500/5 p-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 pt-3 pb-2">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-400" />
              <span className="text-sm font-semibold text-zinc-100">Final cut</span>
              <Badge className={
                hasAudio
                  ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                  : "bg-zinc-500/15 text-zinc-400 border-zinc-500/30"
              }>
                {hasAudio ? "with audio" : "silent · mixer pending"}
              </Badge>
            </div>
            <div className="flex items-center gap-3">
              <a
                href={`${FILM_API_BASE}${playbackUrl}`}
                download
                className="inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300"
              >
                <Download className="w-3.5 h-3.5" /> Download
              </a>
              {hasAudio && final_url && (
                <a
                  href={`${FILM_API_BASE}${final_url}`}
                  download
                  className="text-[11px] text-zinc-500 hover:text-zinc-300"
                >
                  silent version
                </a>
              )}
            </div>
          </div>
          <video
            key={playbackUrl}
            src={`${FILM_API_BASE}${playbackUrl}`}
            controls
            className="w-full bg-black block"
          />
        </Card>
      ) : isRunning || isAwaiting ? (
        <Card className="border-indigo-500/30 bg-indigo-500/5">
          <div className="flex items-center gap-3">
            {isRunning
              ? <Loader2 className="w-5 h-5 text-indigo-300 animate-spin" />
              : <Clock className="w-5 h-5 text-sky-300" />}
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold text-zinc-100">
                {currentStageSpec?.label || (isAwaiting ? "Awaiting your input" : "Working…")}
              </div>
              <div className="text-xs text-zinc-400 mt-0.5">
                {currentStageSpec?.description || "Pipeline in progress."}
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono text-zinc-100 leading-none">{doneCount}<span className="text-zinc-600">/{stages.length}</span></div>
              <div className="text-[10px] text-zinc-500 uppercase tracking-wide mt-1">stages done</div>
            </div>
          </div>
        </Card>
      ) : (
        <Card>
          <div className="flex items-center gap-3">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <div className="flex-1">
              <div className="text-sm font-semibold text-zinc-100">Ready when you are</div>
              <div className="text-xs text-zinc-400 mt-0.5">Start the pipeline to produce your film. All {stages.length} stages will run end to end.</div>
            </div>
            {state.last_error && (
              <Badge className="bg-red-500/15 text-red-300 border-red-500/30">error on last run</Badge>
            )}
          </div>
        </Card>
      )}

      {/* CONTROLS - compact row, primary action prominent, secondary actions as icons. */}
      <div className="flex items-center flex-wrap gap-2">
        {!isRunning && !allDone && (
          <Button
            size="md"
            onClick={() => runAction("run", () => filmRun(projectId))}
            disabled={busy !== null}
          >
            {busy === "run" ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Starting…</> :
                              <><Play className="w-4 h-4 mr-2" />
                                {isAwaiting ? "Continue" : hasStale ? "Re-run stale" : "Start pipeline"}</>}
          </Button>
        )}
        {isRunning && (
          <Button
            size="md" variant="danger"
            onClick={() => runAction("pause", () => filmPause(projectId))}
            disabled={busy !== null || state.pause_requested}
          >
            {state.pause_requested
              ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Pausing after this stage…</>
              : <><Pause className="w-4 h-4 mr-2" />Pause</>}
          </Button>
        )}
        <div className="flex-1" />
        <Button
          size="sm" variant="ghost"
          disabled={isRunning || busy !== null}
          onClick={async () => {
            if (!window.confirm("Delete this film and all its artifacts? Cannot be undone.")) return;
            setBusy("delete");
            try { await filmDelete(projectId); onGone(); }
            catch (e) { setErr(e instanceof Error ? e.message : "Delete failed"); setBusy(null); }
          }}
        >
          <Trash2 className="w-3.5 h-3.5 mr-1.5" /> Delete
        </Button>
      </div>

      {state.last_error && (
        <Card className="border-red-500/20 bg-red-500/5">
          <div className="flex items-start gap-2 text-[11px] text-red-300">
            <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-semibold mb-1">Last error</div>
              <div className="font-mono whitespace-pre-wrap break-words">{state.last_error}</div>
            </div>
          </div>
        </Card>
      )}

      {/* TIMELINE - horizontal 12-stage grid. Each stage is clickable to open its artifact. */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <CardTitle className="text-sm">Pipeline</CardTitle>
          <span className="text-[10px] text-zinc-500">click a stage to view or edit its artifact · rewind icon on hover</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2">
          {stages.map((spec, i) => {
            const status = state.stage_status[spec.key];
            const meta = STATUS_META[status] || STATUS_META.pending;
            const isCur = state.current_stage === spec.key;
            const isSel = selectedStage === spec.key;
            const canRewind = status === "done" || status === "stale" || status === "needs_review" || status === "failed";
            const border =
              isSel ? "border-indigo-400/60 bg-indigo-500/10" :
              isCur ? "border-indigo-500/40 bg-indigo-500/5" :
              status === "done" ? "border-green-500/20 bg-green-500/5 hover:bg-green-500/10" :
              status === "failed" ? "border-red-500/30 bg-red-500/5" :
              status === "stale" ? "border-orange-500/30 bg-orange-500/5" :
              status === "needs_review" ? "border-sky-500/30 bg-sky-500/5" :
              status === "paused" ? "border-amber-500/30 bg-amber-500/5" :
              "border-zinc-800 bg-zinc-900/40 hover:bg-zinc-800/60";
            return (
              <button
                key={spec.key}
                onClick={() => setSelectedStage(spec.key)}
                className={`group relative text-left rounded-lg border p-2.5 transition-colors ${border}`}
              >
                <div className="flex items-start gap-2">
                  <StageStatusIcon status={status} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-zinc-500 font-mono">{String(i + 1).padStart(2, "0")}</span>
                      <span className={`text-xs font-medium truncate ${isCur ? "text-indigo-200" : "text-zinc-100"}`}>
                        {spec.label}
                      </span>
                    </div>
                    <div className={`text-[10px] mt-0.5 ${meta.tone}`}>
                      {meta.label}
                      {spec.gated_by_default && <span className="text-zinc-600"> · gated</span>}
                    </div>
                  </div>
                </div>
                {canRewind && (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!window.confirm(`Rewind from "${spec.label}" onward?`)) return;
                      runAction("rewind", () => filmRewind(projectId, spec.key));
                    }}
                    className="absolute top-1.5 right-1.5 text-zinc-600 hover:text-orange-400 p-1 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                    title="Rewind pipeline to this stage"
                  >
                    <RotateCcw className="w-3 h-3" />
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </Card>

      {/* ARTIFACT DRAWER - modal overlay so it doesn't push the hero out of the way. */}
      {selectedStage && (
        <div className="fixed inset-0 z-40 flex" onClick={() => setSelectedStage(null)}>
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <div
            className="ml-auto relative z-50 w-full max-w-3xl h-full overflow-y-auto bg-zinc-950 border-l border-zinc-800 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <ArtifactView
              projectId={projectId}
              stage={stages.find((s) => s.key === selectedStage)!}
              status={state.stage_status[selectedStage]}
              artifact={artifacts[selectedStage] || null}
              onSaved={refresh}
              onClose={() => setSelectedStage(null)}
              disabled={isRunning}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Artifact viewer / editor
// ---------------------------------------------------------------------------

function ArtifactView({
  projectId, stage, status, artifact, onSaved, onClose, disabled,
}: {
  projectId: string;
  stage: FilmStageSpec;
  status: FilmStageStatus;
  artifact: Record<string, unknown> | null;
  onSaved: () => void;
  onClose: () => void;
  disabled: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setText(artifact ? JSON.stringify(artifact, null, 2) : "");
    setEditing(false);
    setErr(null);
  }, [artifact, stage.key]);

  const save = async () => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      setErr(e instanceof Error ? `Invalid JSON: ${e.message}` : "Invalid JSON");
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      await filmEditArtifact(projectId, stage.key, parsed);
      onSaved();
      setEditing(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const meta = STATUS_META[status] || STATUS_META.pending;
  return (
    <Card>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <CardTitle className="text-sm flex items-center gap-2">
            <StageStatusIcon status={status} /> {stage.label}
          </CardTitle>
          <p className="text-[11px] text-zinc-500 mt-0.5">{stage.description}</p>
          <p className={`text-[10px] mt-1 ${meta.tone}`}>{meta.label}</p>
        </div>
        <div className="flex items-center gap-1">
          {artifact && !editing && (
            <Button variant="secondary" size="sm" onClick={() => setEditing(true)} disabled={disabled}
                    title={disabled ? "Pause the run to edit" : "Edit this artifact"}>
              Edit
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onClose} title="Close viewer">
            <X className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {stage.key === "producer" && artifact && !editing && (
        <ProducerPreview artifact={artifact} projectId={projectId} onChanged={onSaved} disabled={disabled} />
      )}

      {stage.key === "screenwriter" && artifact && !editing && (
        <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-950/50 border border-zinc-800 rounded p-3 max-h-[60vh] overflow-auto">
          {(artifact as { fountain?: string }).fountain || ""}
        </pre>
      )}

      {stage.key === "story_editor" && artifact && !editing && (
        <div className="space-y-2">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Notes</div>
            <pre className="text-xs text-zinc-300 whitespace-pre-wrap bg-zinc-950/50 border border-zinc-800 rounded p-3">
              {(() => {
                const n = (artifact as { notes?: unknown }).notes;
                if (typeof n === "string") return n;
                if (Array.isArray(n)) {
                  return n.map((entry) => {
                    if (typeof entry === "string") return `• ${entry}`;
                    if (entry && typeof entry === "object") {
                      const e = entry as { fail?: string; quote?: string; why?: string };
                      return `• ${e.fail || ""}${e.quote ? `\n  "${e.quote}"` : ""}${e.why ? `\n  ${e.why}` : ""}`;
                    }
                    return String(entry);
                  }).join("\n");
                }
                return "";
              })()}
            </pre>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Revised draft</div>
            <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-950/50 border border-zinc-800 rounded p-3 max-h-[50vh] overflow-auto">
              {(artifact as { revised_fountain?: string }).revised_fountain || ""}
            </pre>
          </div>
        </div>
      )}

      {stage.key === "shots" && artifact && !editing && (
        <ShotsGallery projectId={projectId} artifact={artifact} />
      )}

      {stage.key === "breakdown" && artifact && !editing && (
        <BreakdownView artifact={artifact} />
      )}
      {stage.key === "storyboard" && artifact && !editing && (
        <StoryboardView artifact={artifact} />
      )}
      {stage.key === "cinematographer" && artifact && !editing && (
        <CinematographyView artifact={artifact} />
      )}

      {stage.key === "voice_cast" && artifact && !editing && (
        <VoiceCastView artifact={artifact} />
      )}
      {stage.key === "character_portraits" && artifact && !editing && (
        <PortraitsGallery projectId={projectId} artifact={artifact} />
      )}
      {stage.key === "voice_actor" && artifact && !editing && (
        <VoiceActorView projectId={projectId} artifact={artifact} />
      )}
      {stage.key === "motion_shots" && artifact && !editing && (
        <MotionShotsGallery projectId={projectId} artifact={artifact} />
      )}
      {(stage.key === "ambient" || stage.key === "composer") && artifact && !editing && (
        <SingleTrackView projectId={projectId} artifact={artifact} kind={stage.key as "ambient" | "composer"} />
      )}
      {(stage.key === "mixer" || stage.key === "colorist" || stage.key === "titles" || stage.key === "upscale" || stage.key === "editor") && artifact && !editing && (
        <SingleVideoView projectId={projectId} artifact={artifact}
                         stageKey={stage.key as "mixer" | "colorist" | "titles" | "upscale" | "editor"} />
      )}
      {stage.key === "lipsync" && artifact && !editing && (
        <LipSyncView projectId={projectId} artifact={artifact} />
      )}
      {stage.key === "continuity" && artifact && !editing && (
        <ContinuityView artifact={artifact} />
      )}

      {/* Fallback: raw JSON viewer for everything else */}
      {editing || (artifact && ![
        "producer","screenwriter","story_editor","shots","editor",
        "breakdown","storyboard","cinematographer",
        "voice_cast","character_portraits","voice_actor","motion_shots",
        "ambient","composer","mixer","colorist","titles","upscale",
        "lipsync","continuity",
      ].includes(stage.key)) ? (
        <>
          {editing && (
            <div className="mb-2 flex items-center gap-2">
              <Button size="sm" onClick={save} disabled={saving}>
                {saving ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />Saving…</> :
                          <><Save className="w-3.5 h-3.5 mr-1.5" />Save & mark downstream stale</>}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => { setEditing(false); setText(artifact ? JSON.stringify(artifact, null, 2) : ""); }}>
                Cancel
              </Button>
            </div>
          )}
          {err && <div className="text-xs text-red-300 mb-2">{err}</div>}
          <textarea
            value={text}
            readOnly={!editing}
            onChange={(e) => setText(e.target.value)}
            className="w-full font-mono text-xs bg-zinc-950/50 border border-zinc-800 rounded p-3 text-zinc-300 focus:ring-2 focus:ring-indigo-500/50 h-[55vh]"
            spellCheck={false}
          />
        </>
      ) : null}

      {!artifact && (
        <div className="text-xs text-zinc-500">
          {"No artifact yet - this stage hasn't run."}
        </div>
      )}
    </Card>
  );
}

function ProducerPreview({ artifact, projectId, onChanged, disabled }: {
  artifact: Record<string, unknown>;
  projectId: string;
  onChanged: () => void;
  disabled: boolean;
}) {
  const loglines = (artifact.loglines as Array<{ title: string; tone: string; logline: string }>) || [];
  const chosen = Number(artifact.chosen_index ?? 0);
  const [busy, setBusy] = useState<number | null>(null);
  const pick = async (i: number) => {
    setBusy(i);
    try {
      await filmEditArtifact(projectId, "producer", { ...artifact, chosen_index: i });
      onChanged();
    } finally { setBusy(null); }
  };
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-zinc-500">Producer proposed {loglines.length} loglines. Pick one, then Continue.</p>
      {loglines.map((l, i) => (
        <button
          key={i}
          onClick={() => pick(i)}
          disabled={disabled || busy !== null}
          className={`w-full text-left border rounded-lg p-3 transition-colors ${
            i === chosen ? "border-indigo-500/40 bg-indigo-600/10" : "border-zinc-800 hover:border-zinc-700"
          } ${disabled ? "opacity-70 cursor-not-allowed" : ""}`}
        >
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-zinc-100">{l.title}</span>
            <Badge>{l.tone}</Badge>
            {i === chosen && <span className="text-[10px] text-indigo-300">Chosen</span>}
            {busy === i && <Loader2 className="w-3 h-3 animate-spin text-indigo-400" />}
          </div>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">{l.logline}</p>
        </button>
      ))}
    </div>
  );
}

function ShotsGallery({ projectId, artifact }: {
  projectId: string;
  artifact: Record<string, unknown>;
}) {
  const shots = (artifact.shots as Array<{ scene_id: string; shot_id: string; path: string; prompt: string; rendered: boolean; duration_sec: number }>) || [];
  return (
    <div>
      <p className="text-[11px] text-zinc-500 mb-3">{shots.length} shots rendered.</p>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {shots.map((s) => (
          <div key={`${s.scene_id}-${s.shot_id}`} className="bg-zinc-950/50 border border-zinc-800 rounded-lg overflow-hidden">
            <img
              src={`${FILM_API_BASE}/static/films/${projectId}/${s.path}`}
              alt={`${s.scene_id} ${s.shot_id}`}
              className="w-full aspect-video object-cover bg-zinc-950"
              loading="lazy"
            />
            <div className="p-2">
              <div className="text-[10px] font-mono text-zinc-500 flex items-center justify-between">
                <span>{s.scene_id} · {s.shot_id}</span>
                <span>{s.duration_sec}s{s.rendered ? "" : " · placeholder"}</span>
              </div>
              <p className="text-[10px] text-zinc-500 mt-1 line-clamp-2" title={s.prompt}>{s.prompt}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Viewers for T2/T3 stages
// ---------------------------------------------------------------------------

function filmStatic(projectId: string, path: string): string {
  return `${FILM_API_BASE}/static/films/${projectId}/${path.replace(/^\.?\/+/, "")}`;
}

function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`
    : `${m}:${String(r).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Breakdown / Storyboard / Cinematographer viewers
// ---------------------------------------------------------------------------

type BreakdownScene = {
  id: string;
  heading?: string;
  summary?: string;
  location?: string;
  interior?: boolean;
  time_of_day?: string;
  characters?: string[];
  mood?: string;
  estimated_seconds?: number;
};

function BreakdownView({ artifact }: { artifact: Record<string, unknown> }) {
  const scenes = (artifact.scenes as BreakdownScene[]) || [];
  if (scenes.length === 0) {
    return <p className="text-xs text-zinc-500">No scenes parsed yet.</p>;
  }
  const totalSec = scenes.reduce((s, c) => s + (c.estimated_seconds || 0), 0);
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-zinc-500">
        {scenes.length} scene{scenes.length === 1 ? "" : "s"} · estimated {fmtClock(totalSec)}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-zinc-500 text-left">
              <th className="py-1 pr-2">#</th>
              <th className="py-1 pr-2">Scene</th>
              <th className="py-1 pr-2">Location</th>
              <th className="py-1 pr-2">Time</th>
              <th className="py-1 pr-2">Characters</th>
              <th className="py-1 pr-2">Mood</th>
              <th className="py-1 pr-2 text-right">Est.</th>
            </tr>
          </thead>
          <tbody>
            {scenes.map((s, i) => (
              <tr key={s.id} className="border-t border-zinc-800 align-top">
                <td className="py-1.5 pr-2 font-mono text-zinc-500">{s.id || `s${i + 1}`}</td>
                <td className="py-1.5 pr-2">
                  <div className="font-mono text-[10px] text-zinc-500">{s.heading || "—"}</div>
                  <div className="text-zinc-200">{s.summary || ""}</div>
                </td>
                <td className="py-1.5 pr-2 text-zinc-300">
                  <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 align-middle ${s.interior ? "bg-amber-400" : "bg-sky-400"}`}
                        title={s.interior ? "Interior" : "Exterior"} />
                  {s.location || "—"}
                </td>
                <td className="py-1.5 pr-2 text-zinc-400 lowercase">{s.time_of_day || "—"}</td>
                <td className="py-1.5 pr-2">
                  <div className="flex flex-wrap gap-1">
                    {(s.characters || []).length === 0
                      ? <span className="text-zinc-600">—</span>
                      : (s.characters || []).map((c) =>
                          <Badge key={c} className="text-[9px]">{c}</Badge>)}
                  </div>
                </td>
                <td className="py-1.5 pr-2 text-zinc-400 italic">{s.mood || ""}</td>
                <td className="py-1.5 pr-2 text-right font-mono text-zinc-400">
                  {s.estimated_seconds ? fmtClock(s.estimated_seconds) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type StoryboardShot = {
  id: string;
  description?: string;
  subject?: string;
  action?: string;
  dialogue?: string;
};

function StoryboardView({ artifact }: { artifact: Record<string, unknown> }) {
  const sceneShots = (artifact.scenes as Record<string, StoryboardShot[]>) || {};
  const sceneIds = Object.keys(sceneShots);
  if (sceneIds.length === 0) {
    return <p className="text-xs text-zinc-500">No shots laid out yet.</p>;
  }
  const totalShots = sceneIds.reduce((s, k) => s + (sceneShots[k]?.length || 0), 0);
  const dialogueShots = sceneIds.reduce((s, k) =>
    s + (sceneShots[k] || []).filter((sh) => sh.dialogue && sh.dialogue.trim()).length, 0);
  return (
    <div className="space-y-3">
      <p className="text-[11px] text-zinc-500">
        {totalShots} shot{totalShots === 1 ? "" : "s"} across {sceneIds.length} scene{sceneIds.length === 1 ? "" : "s"} · {dialogueShots} with dialogue
      </p>
      {sceneIds.map((sid) => {
        const shots = sceneShots[sid] || [];
        return (
          <div key={sid} className="border border-zinc-800 rounded-lg overflow-hidden">
            <div className="bg-zinc-950/40 px-3 py-1.5 flex items-center gap-2">
              <span className="text-xs font-mono text-zinc-500">{sid}</span>
              <span className="text-[10px] text-zinc-500">· {shots.length} shot{shots.length === 1 ? "" : "s"}</span>
            </div>
            <div className="divide-y divide-zinc-800">
              {shots.map((sh) => (
                <div key={sh.id} className="px-3 py-2 text-xs">
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-[10px] text-zinc-500 w-10 flex-shrink-0 pt-0.5">{sh.id}</span>
                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="text-zinc-100">{sh.description || sh.action || ""}</div>
                      {sh.subject && (
                        <div className="text-[10px] text-zinc-500">
                          <span className="uppercase mr-1">subject:</span>{sh.subject}
                        </div>
                      )}
                      {sh.action && sh.action !== sh.description && (
                        <div className="text-[10px] text-zinc-500">
                          <span className="uppercase mr-1">action:</span>{sh.action}
                        </div>
                      )}
                      {sh.dialogue && sh.dialogue.trim() && (
                        <div className="text-[11px] text-indigo-300 border-l-2 border-indigo-500/40 pl-2 mt-1 italic">
                          &quot;{sh.dialogue}&quot;
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

type CinemaShot = {
  framing?: string;
  lens_mm?: number;
  camera_move?: string;
  lighting?: string;
  palette?: string;
  duration_sec?: number;
};

function CinematographyView({ artifact }: { artifact: Record<string, unknown> }) {
  const sceneSpecs = (artifact.scenes as Record<string, Record<string, CinemaShot>>) || {};
  const sceneIds = Object.keys(sceneSpecs);
  if (sceneIds.length === 0) {
    return <p className="text-xs text-zinc-500">No shot specs assigned yet.</p>;
  }
  // Aggregate durations for a filmwide total.
  let totalShots = 0, totalDur = 0;
  for (const sid of sceneIds) {
    const shots = sceneSpecs[sid] || {};
    for (const key in shots) {
      totalShots += 1;
      totalDur += shots[key]?.duration_sec || 0;
    }
  }
  return (
    <div className="space-y-3">
      <p className="text-[11px] text-zinc-500">
        {totalShots} shot spec{totalShots === 1 ? "" : "s"} · total runtime {fmtClock(totalDur)}
      </p>
      {sceneIds.map((sid) => {
        const shots = sceneSpecs[sid] || {};
        const shotKeys = Object.keys(shots);
        return (
          <div key={sid} className="border border-zinc-800 rounded-lg overflow-hidden">
            <div className="bg-zinc-950/40 px-3 py-1.5 flex items-center gap-2">
              <span className="text-xs font-mono text-zinc-500">{sid}</span>
              <span className="text-[10px] text-zinc-500">· {shotKeys.length} shot{shotKeys.length === 1 ? "" : "s"}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wide text-zinc-500 text-left">
                    <th className="py-1 pl-3 pr-2">Shot</th>
                    <th className="py-1 pr-2">Framing</th>
                    <th className="py-1 pr-2">Lens</th>
                    <th className="py-1 pr-2">Move</th>
                    <th className="py-1 pr-2">Lighting</th>
                    <th className="py-1 pr-2">Palette</th>
                    <th className="py-1 pr-3 text-right">Dur</th>
                  </tr>
                </thead>
                <tbody>
                  {shotKeys.map((sk) => {
                    const sh = shots[sk] || {};
                    return (
                      <tr key={sk} className="border-t border-zinc-800">
                        <td className="py-1.5 pl-3 pr-2 font-mono text-zinc-500">{sk}</td>
                        <td className="py-1.5 pr-2">
                          <Badge className="text-[9px] font-mono">{sh.framing || "—"}</Badge>
                        </td>
                        <td className="py-1.5 pr-2 font-mono text-zinc-300">{sh.lens_mm ? `${sh.lens_mm}mm` : "—"}</td>
                        <td className="py-1.5 pr-2 text-zinc-400 lowercase">{sh.camera_move || "static"}</td>
                        <td className="py-1.5 pr-2 text-zinc-400">{sh.lighting || "—"}</td>
                        <td className="py-1.5 pr-2 text-zinc-400">{sh.palette || "—"}</td>
                        <td className="py-1.5 pr-3 text-right font-mono text-zinc-400">
                          {sh.duration_sec ? `${sh.duration_sec}s` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

type CastVoice = { voice?: string; gender?: string; why?: string };

function VoiceCastView({ artifact }: { artifact: Record<string, unknown> }) {
  const cast = (artifact.cast as Record<string, CastVoice>) || {};
  const rows = Object.entries(cast);
  if (rows.length === 0) {
    return <p className="text-xs text-zinc-500">No cast assigned yet.</p>;
  }
  return (
    <div className="space-y-1.5">
      <p className="text-[11px] text-zinc-500 mb-2">
        <Mic className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
        {rows.length} character{rows.length === 1 ? "" : "s"} cast.
      </p>
      {rows.map(([name, v]) => (
        <div key={name} className="border border-zinc-800 rounded-lg p-2 flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-sm text-zinc-100 font-semibold">{name}</div>
            {v.why && <p className="text-[10px] text-zinc-500 line-clamp-2">{v.why}</p>}
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {v.gender && <Badge className="text-[9px] uppercase">{v.gender}</Badge>}
            <code className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 border border-indigo-500/25 rounded px-1.5 py-0.5">
              {v.voice || " - "}
            </code>
          </div>
        </div>
      ))}
    </div>
  );
}

function PortraitsGallery({ projectId, artifact }: {
  projectId: string;
  artifact: Record<string, unknown>;
}) {
  const portraits = (artifact.portraits as Record<string, { path: string; prompt: string; rendered: boolean; error?: string | null }>) || {};
  const rows = Object.entries(portraits);
  if (rows.length === 0) {
    return <p className="text-xs text-zinc-500">No portraits rendered yet.</p>;
  }
  return (
    <div>
      <p className="text-[11px] text-zinc-500 mb-3">
        <ImageIcon className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
        {rows.length} portrait{rows.length === 1 ? "" : "s"}.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {rows.map(([name, p]) => (
          <div key={name} className="bg-zinc-950/50 border border-zinc-800 rounded-lg overflow-hidden">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={filmStatic(projectId, p.path)} alt={name}
                 loading="lazy" className="w-full aspect-square object-cover bg-black" />
            <div className="p-2">
              <div className="text-xs text-zinc-100 font-semibold truncate">{name}</div>
              <p className="text-[10px] text-zinc-500 mt-0.5 line-clamp-2" title={p.prompt}>{p.prompt}</p>
              {!p.rendered && (
                <div className="text-[10px] text-amber-400 mt-1">placeholder</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

type VoiceLine = {
  scene_id: string; shot_id: string;
  speaker: string; voice: string; backend: string;
  text: string; wav: string;
  duration_sec: number; synthesized: boolean; error?: string | null;
};

function VoiceActorView({ projectId, artifact }: {
  projectId: string; artifact: Record<string, unknown>;
}) {
  const lines = (artifact.lines as VoiceLine[]) || [];
  if (lines.length === 0) {
    return <p className="text-xs text-zinc-500">No dialogue lines synthesized yet.</p>;
  }
  const speakers = new Set(lines.map((l) => l.speaker));
  return (
    <div className="space-y-1.5">
      <p className="text-[11px] text-zinc-500 mb-2">
        <Volume2 className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
        {lines.length} line{lines.length === 1 ? "" : "s"} across {speakers.size} speaker{speakers.size === 1 ? "" : "s"}.
      </p>
      <div className="space-y-1.5 max-h-[70vh] overflow-y-auto pr-1">
        {lines.map((l, i) => (
          <div key={`${l.scene_id}-${l.shot_id}-${i}`}
               className="border border-zinc-800 rounded-lg p-2 bg-zinc-950/40">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-mono text-zinc-500">
                {l.scene_id} · {l.shot_id}
              </span>
              <Badge className="text-[9px]">{l.speaker}</Badge>
              <code className="text-[9px] text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded px-1">
                {l.voice}
              </code>
              <span className="text-[9px] text-zinc-500 ml-auto">{l.duration_sec.toFixed(1)}s</span>
            </div>
            <p className="text-xs text-zinc-200 leading-relaxed mb-1.5">{l.text}</p>
            {l.wav && (
              <audio src={filmStatic(projectId, l.wav)} controls preload="none"
                     className="w-full h-8" />
            )}
            {l.error && <p className="text-[10px] text-red-300 mt-1">{l.error}</p>}
            {!l.synthesized && !l.error && (
              <p className="text-[10px] text-amber-400 mt-1">
                silent placeholder - TTS backend unavailable
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

type MotionShot = {
  scene_id: string; shot_id: string; path: string;
  duration_sec: number; backend: string; rendered: boolean; error?: string | null;
};

function MotionShotsGallery({ projectId, artifact }: {
  projectId: string; artifact: Record<string, unknown>;
}) {
  const shots = (artifact.motion_shots as MotionShot[]) || [];
  const counts = (artifact.backend_counts as Record<string, number> | undefined) || {};
  if (shots.length === 0) {
    return <p className="text-xs text-zinc-500">No motion clips rendered yet.</p>;
  }
  return (
    <div>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <p className="text-[11px] text-zinc-500">
          <Film className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
          {shots.length} motion clip{shots.length === 1 ? "" : "s"}.
        </p>
        {Object.entries(counts).filter(([, n]) => n).map(([k, n]) => (
          <Badge key={k} className="text-[9px] uppercase">
            {k}: {n as number}
          </Badge>
        ))}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
        {shots.map((s) => (
          <div key={`${s.scene_id}-${s.shot_id}`}
               className="bg-zinc-950/50 border border-zinc-800 rounded-lg overflow-hidden">
            <video src={filmStatic(projectId, s.path)} controls preload="metadata"
                   className="w-full aspect-video bg-black" />
            <div className="p-2">
              <div className="text-[10px] font-mono text-zinc-500 flex items-center justify-between">
                <span>{s.scene_id} · {s.shot_id}</span>
                <span>{s.duration_sec}s · {s.backend}</span>
              </div>
              {!s.rendered && (
                <div className="text-[10px] text-amber-400 mt-1">
                  Ken Burns fallback (motion model OOM)
                </div>
              )}
              {s.error && (
                <p className="text-[10px] text-red-300 mt-1 line-clamp-2">{s.error}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SingleTrackView({ projectId, artifact, kind }: {
  projectId: string;
  artifact: Record<string, unknown>;
  kind: "ambient" | "composer";
}) {
  const a = artifact as { path?: string; prompt?: string; duration_sec?: number; rendered?: boolean; error?: string | null };
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        {kind === "composer"
          ? <Music className="w-4 h-4 text-indigo-400" />
          : <Volume2 className="w-4 h-4 text-indigo-400" />}
        <span className="text-xs text-zinc-100 font-medium">
          {kind === "composer" ? "Score" : "Ambient bed"}
        </span>
        {a.duration_sec && (
          <span className="text-[10px] text-zinc-500 ml-auto">
            {fmtClock(a.duration_sec)}
          </span>
        )}
      </div>
      {a.path && (
        <audio src={filmStatic(projectId, a.path)} controls preload="metadata"
               className="w-full" />
      )}
      {a.prompt && (
        <div className="text-[10px] text-zinc-500 border border-zinc-800 rounded p-2">
          <div className="uppercase tracking-wide text-zinc-600 mb-0.5">Prompt</div>
          {a.prompt}
        </div>
      )}
      {a.rendered === false && (
        <p className="text-[10px] text-amber-400">
          Silent placeholder - backend model unavailable.
        </p>
      )}
      {a.error && <p className="text-[10px] text-red-300">{a.error}</p>}
    </div>
  );
}

function SingleVideoView({ projectId, artifact, stageKey }: {
  projectId: string;
  artifact: Record<string, unknown>;
  stageKey: "mixer" | "colorist" | "titles" | "upscale" | "editor";
}) {
  const a = artifact as {
    output_path?: string; title?: string; grade_name?: string;
    grade_vf?: string; backend?: string; scale?: number;
    cast_line_count?: number; has_dialogue?: boolean; has_score?: boolean;
    has_ambient?: boolean; dialogue_lines?: number; source?: string;
    duration_sec?: number; shot_count?: number;
  };
  const icon = stageKey === "mixer" ? <Volume2 className="w-4 h-4 text-indigo-400" />
             : stageKey === "colorist" ? <Palette className="w-4 h-4 text-indigo-400" />
             : stageKey === "titles" ? <TypeIcon className="w-4 h-4 text-indigo-400" />
             : stageKey === "editor" ? <Film className="w-4 h-4 text-indigo-400" />
             : <ArrowUpCircle className="w-4 h-4 text-indigo-400" />;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-zinc-100 font-medium capitalize">{stageKey}</span>
        {stageKey === "titles" && a.title &&
          <Badge className="text-[10px]">{a.title}</Badge>}
        {stageKey === "colorist" && a.grade_name &&
          <Badge className="text-[10px] uppercase">{a.grade_name}</Badge>}
        {stageKey === "upscale" && a.backend &&
          <Badge className="text-[10px] uppercase">{a.backend}{a.scale ? ` · ${a.scale}×` : ""}</Badge>}
      </div>
      {a.output_path && (
        <video src={filmStatic(projectId, a.output_path)} controls preload="metadata"
               className="w-full max-h-[55vh] bg-black rounded-lg" />
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
        {stageKey === "mixer" && (
          <>
            <Meta label="Dialogue" value={a.has_dialogue ? "yes" : "no"} tone={a.has_dialogue ? "ok" : "dim"} />
            <Meta label="Score" value={a.has_score ? "yes" : "no"} tone={a.has_score ? "ok" : "dim"} />
            <Meta label="Ambient" value={a.has_ambient ? "yes" : "no"} tone={a.has_ambient ? "ok" : "dim"} />
            <Meta label="Lines" value={a.dialogue_lines ?? 0} />
          </>
        )}
        {stageKey === "titles" && (
          <Meta label="Cast lines" value={a.cast_line_count ?? 0} />
        )}
        {stageKey === "editor" && (
          <>
            <Meta label="Duration" value={a.duration_sec ? fmtClock(a.duration_sec) : " - "} />
            <Meta label="Shots" value={a.shot_count ?? 0} />
          </>
        )}
      </div>
      {stageKey === "colorist" && a.grade_vf && (
        <div className="text-[10px] font-mono text-zinc-500 border border-zinc-800 rounded p-2 whitespace-pre-wrap">
          {a.grade_vf}
        </div>
      )}
      {a.source && (
        <div className="text-[10px] text-zinc-600 font-mono truncate">
          source: {a.source}
        </div>
      )}
      {a.output_path && (
        <a href={filmStatic(projectId, a.output_path)} target="_blank" rel="noreferrer"
           className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300">
          Open MP4 <Download className="w-3 h-3" />
        </a>
      )}
    </div>
  );
}

function LipSyncView({ projectId, artifact }: {
  projectId: string;
  artifact: Record<string, unknown>;
}) {
  const a = artifact as {
    backend?: string;
    rendered?: boolean;
    skipped?: string;
    install_hint?: string;
    input_path?: string;
    output_path?: string;
    line_count?: number;
  };
  const backend = a.backend || "none";
  const rendered = Boolean(a.rendered);
  // When the backend is a passthrough (none, not installed, or soft-fail),
  // output_path equals input_path - show the untouched video so the user
  // still sees the final cut in context.
  const videoPath = a.output_path || a.input_path;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Mic className="w-4 h-4 text-indigo-400" />
        <span className="text-xs text-zinc-100 font-medium">Lip sync</span>
        <Badge aria-label={backend}>{backend}</Badge>
        {rendered ? (
          <Badge variant="success">rendered</Badge>
        ) : (
          <Badge variant="warning">passthrough</Badge>
        )}
      </div>
      {!rendered && a.skipped && (
        <div className="text-xs text-amber-200/90 border border-amber-500/30 bg-amber-500/5 rounded px-3 py-2">
          <div className="font-medium">Skipped: {a.skipped}</div>
          {a.install_hint && (
            <div className="text-[11px] text-amber-200/70 mt-1 italic">{a.install_hint}</div>
          )}
        </div>
      )}
      {videoPath && (
        <video src={filmStatic(projectId, videoPath)} controls preload="metadata"
               className="w-full max-h-[55vh] bg-black rounded-lg" />
      )}
      {a.line_count !== undefined && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
          <Meta label="Lines synced" value={String(a.line_count)} />
        </div>
      )}
      {a.input_path && a.output_path && a.input_path !== a.output_path && (
        <div className="text-[10px] text-zinc-600 font-mono truncate">
          {a.input_path} → {a.output_path}
        </div>
      )}
      {rendered && a.output_path && (
        <a href={filmStatic(projectId, a.output_path)} target="_blank" rel="noreferrer"
           className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300">
          Open MP4 <Download className="w-3 h-3" />
        </a>
      )}
    </div>
  );
}

function ContinuityView({ artifact }: { artifact: Record<string, unknown> }) {
  const ok = Boolean(artifact.ok);
  const issues = (artifact.issues as Array<{
    severity: "error" | "warning";
    stage: string;
    message: string;
    fix_hint?: string;
    scene_id?: string;
    shot_id?: string;
    character?: string;
  }>) || [];
  const stats = (artifact.stats as Record<string, unknown>) || {};
  const errorCount = Number(artifact.error_count ?? 0);
  const warningCount = Number(artifact.warning_count ?? 0);

  // Sort errors first, then warnings, then stable by stage key.
  const ordered = [...issues].sort((a, b) => {
    const sev = (s: string) => (s === "error" ? 0 : 1);
    return sev(a.severity) - sev(b.severity) || a.stage.localeCompare(b.stage);
  });

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded border text-xs ${
          ok
            ? "bg-green-500/10 text-green-300 border-green-500/30"
            : "bg-red-500/10 text-red-300 border-red-500/30"
        }`}>
          {ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
          {ok ? "No blocking issues" : `${errorCount} blocker${errorCount === 1 ? "" : "s"}`}
        </div>
        {warningCount > 0 && (
          <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded border text-xs bg-amber-500/10 text-amber-300 border-amber-500/30">
            <Clock className="w-3.5 h-3.5" />
            {warningCount} warning{warningCount === 1 ? "" : "s"}
          </div>
        )}
      </div>

      {Object.keys(stats).length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5">
          {stats.scene_count !== undefined && (
            <Meta label="Scenes" value={String(stats.scene_count)} />
          )}
          {stats.shot_count !== undefined && (
            <Meta label="Shots" value={String(stats.shot_count)} />
          )}
          {stats.character_count !== undefined && (
            <Meta label="Characters" value={String(stats.character_count)} />
          )}
          {stats.dialogue_line_count !== undefined && (
            <Meta label="Dialogue lines" value={String(stats.dialogue_line_count)} />
          )}
          {stats.target_duration_sec !== undefined && (
            <Meta label="Target" value={`${stats.target_duration_sec}s`} />
          )}
          {stats.actual_duration_sec !== undefined && (
            <Meta
              label="Actual"
              value={`${stats.actual_duration_sec}s`}
              tone={
                Math.abs(
                  Number(stats.actual_duration_sec) -
                  Number(stats.target_duration_sec ?? stats.actual_duration_sec)
                ) < 5
                  ? "ok"
                  : undefined
              }
            />
          )}
          {stats.final_source_stage !== undefined && (
            <Meta label="Final from" value={String(stats.final_source_stage)} />
          )}
        </div>
      )}

      {ordered.length > 0 ? (
        <div className="space-y-1.5">
          {ordered.map((iss, i) => (
            <div
              key={i}
              className={`border rounded px-3 py-2 ${
                iss.severity === "error"
                  ? "bg-red-500/5 border-red-500/30"
                  : "bg-amber-500/5 border-amber-500/20"
              }`}
            >
              <div className="flex items-start gap-2">
                {iss.severity === "error" ? (
                  <AlertCircle className="w-3.5 h-3.5 text-red-400 mt-0.5 flex-shrink-0" />
                ) : (
                  <Clock className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-0.5">
                    <Badge variant="default" className="text-[10px] uppercase tracking-wide">
                      {iss.stage.replace("_", " ")}
                    </Badge>
                    {iss.scene_id && (
                      <span className="text-[10px] text-zinc-500">{iss.scene_id}</span>
                    )}
                    {iss.shot_id && (
                      <span className="text-[10px] text-zinc-500">·{iss.shot_id}</span>
                    )}
                    {iss.character && (
                      <span className="text-[10px] text-zinc-500">·{iss.character}</span>
                    )}
                  </div>
                  <div className="text-xs text-zinc-200">{iss.message}</div>
                  {iss.fix_hint && (
                    <div className="text-[11px] text-zinc-500 mt-1 italic">{iss.fix_hint}</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs text-zinc-500 italic px-3 py-6 text-center border border-zinc-800 rounded">
          Clean sweep — every upstream artifact is accounted for.
        </div>
      )}
    </div>
  );
}

function Meta({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "ok" | "dim" }) {
  return (
    <div className="bg-zinc-950/40 border border-zinc-800 rounded px-2 py-1">
      <div className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`text-xs mt-0.5 ${tone === "ok" ? "text-green-400" : tone === "dim" ? "text-zinc-500" : "text-zinc-100"}`}>
        {value}
      </div>
    </div>
  );
}

// Inline editor for the four swappable backends. Collapsed by default
// so it does not fight the meta strip for attention. Disabled while a
// run is in flight — the API rejects config edits mid-run and we would
// otherwise just surface the 409.
function BackendSettings({
  project, isRunning, onSaved,
}: {
  project: FilmMeta;
  isRunning: boolean;
  onSaved: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [voice, setVoice] = useState<string>(project.config.voice_backend || "piper");
  const [music, setMusic] = useState<string>(project.config.music_backend || "musicgen");
  const [lipsync, setLipsync] = useState<string>(project.config.lipsync_backend || "none");
  const [upscale, setUpscale] = useState<string>(project.config.upscale_backend || "none");
  const [cloudReady, setCloudReady] = useState<boolean | null>(null);

  // Fetch cloud-provider status so the "needs install" hint on roadmap
  // backends (seedvr2, diffrhythm2) can be accurate. One shot; status
  // only changes when the launcher restarts.
  useEffect(() => {
    listBackends()
      .then((r) => {
        const cp = r.cloud_providers || {};
        setCloudReady(!!(cp.replicate || cp.runpod));
      })
      .catch(() => setCloudReady(false));
  }, []);
  const roadmapHint = cloudReady
    ? "Roadmap backend. Will route through configured cloud provider."
    : "Roadmap backend. Needs a local install (config.json) or REPLICATE_API_TOKEN / RUNPOD_API_KEY.";

  // When the project refreshes (e.g. after save or external change), reseat
  // the local form values. Keyed on each backend field so a fresh server
  // state is reflected without needing to close the editor.
  useEffect(() => {
    setVoice(project.config.voice_backend || "piper");
    setMusic(project.config.music_backend || "musicgen");
    setLipsync(project.config.lipsync_backend || "none");
    setUpscale(project.config.upscale_backend || "none");
  }, [
    project.config.voice_backend, project.config.music_backend,
    project.config.lipsync_backend, project.config.upscale_backend,
  ]);

  if (!editing) {
    return (
      <div className="flex items-center gap-2 text-[10px] text-zinc-500">
        <span>
          voice <span className="text-zinc-300">{project.config.voice_backend || "piper"}</span>
          {" · "}music <span className="text-zinc-300">{project.config.music_backend || "musicgen"}</span>
          {" · "}lip sync <span className="text-zinc-300">{project.config.lipsync_backend || "none"}</span>
          {" · "}upscale <span className="text-zinc-300">{project.config.upscale_backend || "none"}</span>
        </span>
        <button
          onClick={() => { setErr(null); setEditing(true); }}
          disabled={isRunning}
          className="text-indigo-400 hover:text-indigo-300 disabled:opacity-40 disabled:cursor-not-allowed"
          title={isRunning ? "Pause the run to edit backends" : "Change backends"}
        >
          edit
        </button>
      </div>
    );
  }

  const save = async () => {
    const patch: Partial<FilmConfig> = {};
    if (voice !== (project.config.voice_backend || "piper")) patch.voice_backend = voice;
    if (music !== (project.config.music_backend || "musicgen")) patch.music_backend = music;
    if (lipsync !== (project.config.lipsync_backend || "none")) patch.lipsync_backend = lipsync;
    if (upscale !== (project.config.upscale_backend || "none")) patch.upscale_backend = upscale;
    if (Object.keys(patch).length === 0) {
      setEditing(false);
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      await filmPatchConfig(project.id, patch);
      setEditing(false);
      onSaved();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const selectClass = "w-full mt-0.5 bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200";
  const labelClass = "text-[10px] uppercase tracking-wide text-zinc-500";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 space-y-2">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <div>
          <label htmlFor="cfg-voice" className={labelClass}>Voice</label>
          <select id="cfg-voice" value={voice} onChange={(e) => setVoice(e.target.value)}
                  disabled={saving} className={selectClass}>
            <option value="piper">Piper</option>
            <option value="xtts">XTTS v2</option>
            <option value="indextts2">IndexTTS-2</option>
            <option value="qwen3tts">Qwen3-TTS</option>
            <option value="chatterbox">Chatterbox</option>
          </select>
        </div>
        <div>
          <label htmlFor="cfg-music" className={labelClass}>Music</label>
          <select id="cfg-music" value={music} onChange={(e) => setMusic(e.target.value)}
                  disabled={saving} className={selectClass}>
            <option value="musicgen">MusicGen</option>
            <option value="acestep">ACE-Step</option>
            <option value="diffrhythm2" title={roadmapHint}>
              DiffRhythm 2 {cloudReady ? "" : "(roadmap)"}
            </option>
          </select>
        </div>
        <div>
          <label htmlFor="cfg-lipsync" className={labelClass}>Lip sync</label>
          <select id="cfg-lipsync" value={lipsync} onChange={(e) => setLipsync(e.target.value)}
                  disabled={saving} className={selectClass}>
            <option value="none">Off</option>
            <option value="musetalk">MuseTalk 1.5</option>
            <option value="latentsync">LatentSync 1.5</option>
            <option value="echomimicv3">EchoMimicV3</option>
          </select>
        </div>
        <div>
          <label htmlFor="cfg-upscale" className={labelClass}>Upscale</label>
          <select id="cfg-upscale" value={upscale} onChange={(e) => setUpscale(e.target.value)}
                  disabled={saving} className={selectClass}>
            <option value="none">Off</option>
            <option value="realesrgan">Real-ESRGAN</option>
            <option value="seedvr2" title={roadmapHint}>
              SeedVR2 {cloudReady ? "" : "(roadmap)"}
            </option>
          </select>
        </div>
      </div>
      {err && <p className="text-xs text-red-300">{err}</p>}
      <div className="flex items-center justify-between gap-2">
        <p className="text-[10px] text-zinc-500">
          Changing a backend rewinds its stage so the next run picks up the swap.
        </p>
        <div className="flex items-center gap-2">
          <button onClick={() => { setErr(null); setEditing(false); }} disabled={saving}
                  className="text-[11px] text-zinc-500 hover:text-zinc-300">
            Cancel
          </button>
          <Button size="sm" onClick={save} disabled={saving}>
            {saving ? <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Saving</> : "Save"}
          </Button>
        </div>
      </div>
    </div>
  );
}

function StageStatusIcon({ status }: { status: FilmStageStatus }) {
  switch (status) {
    case "running":       return <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin mt-0.5" />;
    case "done":          return <CheckCircle2 className="w-3.5 h-3.5 text-green-400 mt-0.5" />;
    case "failed":        return <AlertCircle className="w-3.5 h-3.5 text-red-400 mt-0.5" />;
    case "needs_review":  return <Clock className="w-3.5 h-3.5 text-sky-400 mt-0.5" />;
    case "paused":        return <Pause className="w-3.5 h-3.5 text-amber-400 mt-0.5" />;
    case "stale":         return <RotateCcw className="w-3.5 h-3.5 text-orange-400 mt-0.5" />;
    default:              return <Circle className="w-3.5 h-3.5 text-zinc-600 mt-0.5" />;
  }
}
