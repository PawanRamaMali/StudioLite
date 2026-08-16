"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Film, Play, Pause, RotateCcw, Trash2, Plus, ArrowLeft, Save, X,
  Loader2, CheckCircle2, Circle, AlertCircle, Clock, ChevronRight,
  Download, Sparkles,
} from "lucide-react";
import {
  FILM_API_BASE, filmCreate, filmDelete, filmEditArtifact, filmGet,
  filmList, filmListStages, filmPause, filmRewind, filmRun,
  filmStreamUrl,
  type FilmDetail, type FilmListItem, type FilmStageKey, type FilmStageSpec,
  type FilmStageStatus,
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
    // but the lint rule can't see through the returned promise — mark it.
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
            An orchestrated crew of AI agents — producer to editor — turning your brief into a short film.
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
  const [model, setModel] = useState("llama3.2");
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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
          llm_backend: "ollama",
          llm_model: model.trim() || "llama3.2",
          style,
          target_minutes: targetMinutes,
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
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">Ollama model</label>
              <input
                value={model} onChange={(e) => setModel(e.target.value)}
                placeholder="llama3.2"
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 font-mono"
              />
              <p className="text-[10px] text-zinc-500 mt-1">
                Must be pulled locally (`ollama pull llama3.2`). Gemini / Groq / HF backends land in T2.
              </p>
            </div>
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
                    <span className="text-zinc-500"> — {s.description}</span>
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

  return (
    <div className="space-y-4">
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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* left column: metadata + controls + timeline */}
      <div className="lg:col-span-1 space-y-3">
        <Card>
          <CardTitle className="text-base flex items-center gap-2">
            <Film className="w-4 h-4 text-indigo-400" /> {project.title}
          </CardTitle>
          <p className="text-xs text-zinc-400 mt-2 leading-relaxed line-clamp-4">{project.brief}</p>
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            <Badge>{project.config.style}</Badge>
            <Badge>{project.config.target_minutes} min</Badge>
            <Badge>{project.config.llm_model}</Badge>
            <span className="text-[10px] text-zinc-600">Updated {fmtRelTime(project.updated_at)}</span>
          </div>
          <div className="mt-3 flex items-center gap-2 text-[10px]">
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${wsAlive ? "bg-green-500" : "bg-zinc-600"}`} />
            <span className="text-zinc-500">{wsAlive ? "Live events connected" : "Reconnecting…"}</span>
          </div>
        </Card>

        <Card>
          <CardTitle className="text-sm mb-3">Controls</CardTitle>
          <div className="space-y-2">
            {!isRunning && !allDone && (
              <Button
                className="w-full" size="md"
                onClick={() => runAction("run", () => filmRun(projectId))}
                disabled={busy !== null}
              >
                {busy === "run" ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Starting…</> :
                                  <><Play className="w-4 h-4 mr-2" />
                                    {isAwaiting ? "Continue" : hasStale ? "Re-run stale stages" : "Start pipeline"}</>}
              </Button>
            )}
            {isRunning && (
              <Button
                className="w-full" size="md" variant="danger"
                onClick={() => runAction("pause", () => filmPause(projectId))}
                disabled={busy !== null || state.pause_requested}
              >
                {state.pause_requested
                  ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Pausing after this stage…</>
                  : <><Pause className="w-4 h-4 mr-2" />Pause</>}
              </Button>
            )}
            <Button
              className="w-full" size="sm" variant="ghost"
              disabled={isRunning || busy !== null}
              onClick={async () => {
                if (!window.confirm("Delete this film and all its artifacts? Cannot be undone.")) return;
                setBusy("delete");
                try { await filmDelete(projectId); onGone(); }
                catch (e) { setErr(e instanceof Error ? e.message : "Delete failed"); setBusy(null); }
              }}
            >
              <Trash2 className="w-3.5 h-3.5 mr-1.5" /> Delete film
            </Button>
            {state.last_error && (
              <div className="text-[11px] text-red-300 bg-red-500/5 border border-red-500/20 rounded p-2">
                <div className="font-semibold mb-1">Last error</div>
                <div className="font-mono whitespace-pre-wrap break-words">{state.last_error}</div>
              </div>
            )}
          </div>
        </Card>

        <Card>
          <CardTitle className="text-sm mb-3">Pipeline</CardTitle>
          <ol className="space-y-1">
            {stages.map((spec, i) => {
              const status = state.stage_status[spec.key];
              const meta = STATUS_META[status] || STATUS_META.pending;
              const isCur = state.current_stage === spec.key;
              const isSel = selectedStage === spec.key;
              return (
                <li key={spec.key}>
                  <button
                    onClick={() => setSelectedStage(spec.key)}
                    className={`w-full flex items-start gap-2 px-2 py-1.5 rounded-md text-left transition-colors ${
                      isSel ? "bg-indigo-600/10 border border-indigo-500/30" : "hover:bg-zinc-800/40 border border-transparent"
                    }`}
                  >
                    <StageStatusIcon status={status} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-zinc-500 font-mono">{i + 1}.</span>
                        <span className={`text-xs font-medium ${isCur ? "text-indigo-300" : "text-zinc-200"} truncate`}>
                          {spec.label}
                        </span>
                        {spec.gated_by_default && (
                          <Badge className="text-[9px]">gate</Badge>
                        )}
                      </div>
                      <div className={`text-[10px] ${meta.tone}`}>{meta.label}</div>
                    </div>
                    {(status === "done" || status === "stale" || status === "needs_review" || status === "failed") && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!window.confirm(`Rewind from "${spec.label}" onward?`)) return;
                          runAction("rewind", () => filmRewind(projectId, spec.key));
                        }}
                        className="text-zinc-600 hover:text-orange-400 p-0.5"
                        title="Rewind pipeline to this stage"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </button>
                    )}
                  </button>
                </li>
              );
            })}
          </ol>
          {currentStageStatus && !selectedStage && (
            <p className="text-[10px] text-zinc-500 mt-2">Tip: click a stage to view or edit its artifact.</p>
          )}
        </Card>
      </div>

      {/* middle + right: artifact viewer + final video */}
      <div className="lg:col-span-2 space-y-3">
        {playbackUrl && (
          <Card className="border-green-500/20 bg-green-500/5">
            <CardTitle className="text-sm mb-2 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-400" /> Final cut
              <Badge className={
                hasAudio
                  ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                  : "bg-zinc-500/15 text-zinc-400 border-zinc-500/30"
              }>
                {hasAudio ? "with audio" : "silent"}
              </Badge>
            </CardTitle>
            <video
              key={playbackUrl}
              src={`${FILM_API_BASE}${playbackUrl}`}
              controls
              className="w-full rounded-lg bg-black"
            />
            <div className="mt-2 flex gap-3 items-center">
              <a
                href={`${FILM_API_BASE}${playbackUrl}`}
                download
                className="inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300"
              >
                <Download className="w-3.5 h-3.5" /> Download MP4
              </a>
              {hasAudio && final_url && (
                <a
                  href={`${FILM_API_BASE}${final_url}`}
                  download
                  className="inline-flex items-center gap-1.5 text-[11px] text-zinc-500 hover:text-zinc-300"
                >
                  silent version
                </a>
              )}
            </div>
          </Card>
        )}

        {selectedStage ? (
          <ArtifactView
            projectId={projectId}
            stage={stages.find((s) => s.key === selectedStage)!}
            status={state.stage_status[selectedStage]}
            artifact={artifacts[selectedStage] || null}
            onSaved={refresh}
            onClose={() => setSelectedStage(null)}
            disabled={isRunning}
          />
        ) : (
          <Card>
            <CardTitle className="text-sm mb-2">What now</CardTitle>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Click <span className="text-zinc-200 font-medium">Start pipeline</span> to run every stage.
              The Producer will pause at a review gate — pick a logline in its artifact viewer,
              then Continue. The Cinematographer also pauses by default so you can tweak the shot
              plan before rendering. Any stage can be rewound to redo it plus everything downstream.
            </p>
            <p className="text-[10px] text-zinc-600 mt-3">
              T1 renders SDXL keyframes and assembles a silent slideshow. T2 adds motion, voice,
              music, and lipsync.
            </p>
          </Card>
        )}
      </div>
    </div>
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
              {(artifact as { notes?: string }).notes || ""}
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

      {/* Fallback: raw JSON viewer for everything else */}
      {editing || (artifact && !["producer","screenwriter","story_editor","shots"].includes(stage.key)) ? (
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
          {"No artifact yet — this stage hasn't run."}
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
