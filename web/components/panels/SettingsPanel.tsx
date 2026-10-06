"use client";
import { useState, useEffect, useCallback } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Settings, Cpu, HardDrive, Wifi, Check, X, Activity,
  Key, Plus, Trash2, Save, RefreshCw, FileText, AlertTriangle,
  Loader2, Download, XCircle, FolderOpen, ExternalLink, Package,
} from "lucide-react";
import {
  getSystemStatus, SystemStatus, getModelInventory, type ModelInventoryItem,
  downloadModel, deleteModel, getJob, cancelJob, type Job, authFetch,
  getModelRegistry, openModelFolder,
  type ModelRegistryRow, type ModelRegistryKind, type ModelRegistrySummary,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface EnvVar { key: string; value: string; }
interface JobLog { job_id: string; kind: string; status: string; progress: number; message: string; error: string | null; elapsed: number; }

export default function SettingsPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"system" | "models" | "env" | "logs">("system");
  const [models, setModels] = useState<ModelInventoryItem[]>([]);
  // Cross-backend registry inventory (CLIP, Whisper, YuNet/SFace, SDXL,
  // Wan, MusicGen, AudioLDM, Real-ESRGAN, Piper, …). Loaded lazily when
  // the user first opens the Models tab.
  const [registry, setRegistry] = useState<ModelRegistryRow[]>([]);
  const [registrySummary, setRegistrySummary] = useState<ModelRegistrySummary | null>(null);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [registryLoaded, setRegistryLoaded] = useState(false);
  // Folder-path reveal: shows the on-disk directory the user should open.
  const [revealedFolder, setRevealedFolder] = useState<{ id: string; folder: string; exists: boolean } | null>(null);
  // Per-model download state, keyed by model.key.
  // job: the live Job (progress/message/status). error: last error message.
  const [downloadJobs, setDownloadJobs] = useState<Record<string, Job>>({});
  const [downloadErrors, setDownloadErrors] = useState<Record<string, string>>({});
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  // Keys the user has clicked cancel on but whose worker hasn't reported
  // status="cancelled" yet. Used to disable the X button and show "Cancelling…".
  const [cancelling, setCancelling] = useState<Set<string>>(new Set());

  // Env vars
  const [envVars, setEnvVars] = useState<EnvVar[]>([]);
  const [managedKeys, setManagedKeys] = useState<string[]>([]);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [envSaving, setEnvSaving] = useState<string | null>(null);

  // HF Token
  const [hfToken, setHfToken] = useState("");
  const [hfTestResult, setHfTestResult] = useState<string | null>(null);
  const [hfTesting, setHfTesting] = useState(false);

  // Logs
  const [logLines, setLogLines] = useState<string[]>([]);
  const [jobLogs, setJobLogs] = useState<JobLog[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);

  const refreshModels = useCallback(async () => {
    try {
      const d = await getModelInventory();
      setModels(d.models);
      // If the server tells us there's an active_job for a model, hydrate our
      // local job map so a page refresh mid-download resumes progress display.
      setDownloadJobs((prev) => {
        const next = { ...prev };
        for (const m of d.models) {
          if (m.active_job && !next[m.key]) {
            next[m.key] = {
              job_id: m.active_job, status: "running", progress: 0,
              message: "Reconnecting…", created_at: Date.now() / 1000, elapsed: 0,
            };
          }
        }
        return next;
      });
    } catch {
      setModels([]);
    }
  }, []);

  // Fetch system status + model inventory
  useEffect(() => {
    setLoading(true);
    getSystemStatus()
      .then((s) => { setStatus(s); setApiConnected(true); })
      .catch(() => setApiConnected(false))
      .finally(() => setLoading(false));
    refreshModels();
  }, [refreshModels]);

  // Poll every in-flight download job every 2s.
  useEffect(() => {
    const active = Object.entries(downloadJobs).filter(
      ([, j]) => j.status === "queued" || j.status === "running"
    );
    if (active.length === 0) return;
    const t = setInterval(async () => {
      for (const [key, job] of active) {
        try {
          const updated = await getJob(job.job_id);
          setDownloadJobs((prev) => ({ ...prev, [key]: updated }));
          if (
            updated.status === "completed" ||
            updated.status === "failed" ||
            updated.status === "cancelled"
          ) {
            if (updated.status === "failed" && updated.error) {
              setDownloadErrors((prev) => ({ ...prev, [key]: updated.error! }));
            }
            // Clear cancelling flag once the worker acknowledges terminal state.
            setCancelling((prev) => {
              if (!prev.has(key)) return prev;
              const next = new Set(prev);
              next.delete(key);
              return next;
            });
            refreshModels();
          }
        } catch {
          // network blip - try next tick
        }
      }
    }, 2000);
    return () => clearInterval(t);
  }, [downloadJobs, refreshModels]);

  const handleDownload = async (key: string) => {
    setDownloadErrors((prev) => { const n = { ...prev }; delete n[key]; return n; });
    try {
      const job = await downloadModel(key);
      setDownloadJobs((prev) => ({ ...prev, [key]: job }));
    } catch (e) {
      setDownloadErrors((prev) => ({ ...prev, [key]: (e as Error).message || "Failed to start download" }));
    }
  };

  const handleCancel = async (key: string) => {
    const job = downloadJobs[key];
    if (!job) return;
    setCancelling((prev) => new Set(prev).add(key));
    try {
      const updated = await cancelJob(job.job_id);
      setDownloadJobs((prev) => ({ ...prev, [key]: updated }));
    } catch (e) {
      setDownloadErrors((prev) => ({ ...prev, [key]: (e as Error).message || "Cancel failed" }));
      setCancelling((prev) => { const next = new Set(prev); next.delete(key); return next; });
    }
  };

  const handleDelete = async (key: string, name: string) => {
    if (!window.confirm(`Delete ${name} from disk? Weights will need to be re-downloaded to use this model again.`)) return;
    setDeletingKey(key);
    try {
      await deleteModel(key);
      await refreshModels();
    } catch (e) {
      setDownloadErrors((prev) => ({ ...prev, [key]: (e as Error).message || "Delete failed" }));
    } finally {
      setDeletingKey(null);
    }
  };

  // Fetch env vars
  const fetchEnv = useCallback(async () => {
    try {
      const res = await authFetch(`${API_BASE}/api/v1/system/env`);
      if (res.ok) {
        const data = await res.json();
        const vars: EnvVar[] = Object.entries(data.env || {}).map(([k, v]) => ({ key: k, value: v as string }));
        setEnvVars(vars);
        setManagedKeys(data.managed_keys || []);
      }
    } catch { /* offline */ }
  }, []);

  const fetchLogs = useCallback(async () => {
    setLogsLoading(true);
    try {
      const res = await authFetch(`${API_BASE}/api/v1/system/logs?lines=200`);
      if (res.ok) {
        const data = await res.json();
        setLogLines(data.log_lines || []);
        setJobLogs(data.recent_jobs || []);
      }
    } catch { /* offline */ } finally { setLogsLoading(false); }
  }, []);

  useEffect(() => { if (activeTab === "env") fetchEnv(); }, [activeTab, fetchEnv]);
  useEffect(() => { if (activeTab === "logs") fetchLogs(); }, [activeTab, fetchLogs]);

  // Lazy-load the cross-backend registry inventory the first time the
  // Models tab is opened, then re-probe when the user hits Refresh.
  const fetchRegistry = useCallback(async () => {
    setRegistryLoading(true);
    setRegistryError(null);
    try {
      const d = await getModelRegistry();
      setRegistry(d.models);
      setRegistrySummary(d.summary);
      setRegistryLoaded(true);
    } catch (e) {
      setRegistryError((e as Error).message || "Failed to load registry");
    } finally {
      setRegistryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "models" && !registryLoaded && !registryLoading) {
      fetchRegistry();
    }
  }, [activeTab, registryLoaded, registryLoading, fetchRegistry]);

  const handleOpenFolder = async (id: string) => {
    try {
      const r = await openModelFolder(id);
      setRevealedFolder({ id, folder: r.folder, exists: r.exists });
    } catch (e) {
      setRevealedFolder({ id, folder: (e as Error).message, exists: false });
    }
  };

  // Auto-refresh logs
  useEffect(() => {
    if (activeTab !== "logs") return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [activeTab, fetchLogs]);

  const saveEnvVar = async (key: string, value: string) => {
    setEnvSaving(key);
    try {
      await authFetch(`${API_BASE}/api/v1/system/env?key=${encodeURIComponent(key)}&value=${encodeURIComponent(value)}`, { method: "POST" });
      await fetchEnv();
    } catch { /* */ } finally { setEnvSaving(null); }
  };

  const deleteEnvVar = async (key: string) => {
    try {
      await authFetch(`${API_BASE}/api/v1/system/env?key=${encodeURIComponent(key)}`, { method: "DELETE" });
      await fetchEnv();
    } catch { /* */ }
  };

  const addEnvVar = async () => {
    if (!newKey.trim()) return;
    await saveEnvVar(newKey.trim(), newValue);
    setNewKey(""); setNewValue("");
  };

  const testHfToken = async () => {
    if (!hfToken.trim()) return;
    setHfTesting(true); setHfTestResult(null);
    try {
      const res = await authFetch(`${API_BASE}/api/v1/system/hf-token-test?token=${encodeURIComponent(hfToken.trim())}`, { method: "POST" });
      const data = await res.json();
      setHfTestResult(data.message);
      if (data.status === "ok") {
        await fetchEnv();
        setHfToken("");
      }
    } catch { setHfTestResult("Failed to connect to API"); } finally { setHfTesting(false); }
  };

  const gpu = status?.gpu;
  const tabs = [
    { id: "system" as const, label: "System" },
    { id: "models" as const, label: "Models" },
    { id: "env" as const, label: "Environment" },
    { id: "logs" as const, label: "Logs" },
  ];

  // Pretty-print bytes (null-safe). Shows the expected minimum when the
  // model is missing, actual on-disk bytes when it's present.
  const fmtBytes = (n: number): string => {
    if (!n || n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let v = n;
    let u = 0;
    while (v >= 1024 && u < units.length - 1) { v /= 1024; u += 1; }
    return `${v.toFixed(v >= 10 || u === 0 ? 0 : 1)} ${units[u]}`;
  };

  const KIND_LABEL: Record<ModelRegistryKind, string> = {
    text: "Text", image: "Image", video: "Video",
    audio: "Audio", face: "Face", other: "Other",
  };
  // Stable group ordering - matches the kind column in the registry.
  const KIND_ORDER: ModelRegistryKind[] = ["text", "image", "video", "audio", "face", "other"];
  const groupedRegistry: Record<ModelRegistryKind, ModelRegistryRow[]> = {
    text: [], image: [], video: [], audio: [], face: [], other: [],
  };
  for (const row of registry) {
    const bucket = (groupedRegistry[row.kind as ModelRegistryKind] ??= []);
    bucket.push(row);
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text">Settings</h1>
        <p className="text-zinc-400 mt-1">System status, environment variables, and logs</p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 bg-zinc-900 rounded-lg p-1 w-fit">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === t.id ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-zinc-200"
            }`}>{t.label}</button>
        ))}
      </div>

      {/* SYSTEM TAB */}
      {activeTab === "system" && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <Card className="text-center py-4">
              <Wifi className={`w-6 h-6 mx-auto mb-2 ${apiConnected ? "text-green-400" : "text-red-400"}`} />
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">API Server</p>
              <Badge variant={apiConnected ? "success" : "error"} className="mt-1.5">
                {loading ? "Checking..." : apiConnected ? "Connected" : "Offline"}
              </Badge>
            </Card>
            <Card className="text-center py-4">
              <Cpu className="w-6 h-6 mx-auto mb-2 text-indigo-400" />
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">GPU</p>
              <p className="text-xs font-medium mt-1.5 px-2 truncate">
                {gpu?.gpu_name?.replace("NVIDIA GeForce ", "") || (loading ? "..." : "N/A")}
              </p>
              {gpu?.cuda_version && <p className="text-[10px] text-zinc-600 mt-0.5">CUDA {gpu.cuda_version}</p>}
            </Card>
            <Card className="text-center py-4">
              <HardDrive className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">VRAM Free</p>
              <p className="text-lg font-bold text-emerald-400 mt-1">
                {gpu?.free_vram_gb !== undefined ? `${gpu.free_vram_gb.toFixed(1)}` : "..."}<span className="text-xs font-normal text-zinc-500"> GB</span>
              </p>
            </Card>
            <Card className="text-center py-4">
              <Activity className="w-6 h-6 mx-auto mb-2 text-amber-400" />
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Active Jobs</p>
              <p className="text-lg font-bold text-amber-400 mt-1">{status?.active_jobs ?? "..."}</p>
            </Card>
          </div>

          <Card className="mb-6">
            <CardTitle className="mb-4">Model Hub</CardTitle>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-left">Model</th>
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-left">VRAM</th>
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-left">Quality</th>
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-left">Speed</th>
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-left">Status</th>
                    <th className="pb-2 text-xs text-zinc-500 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {models.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-4 text-center text-xs text-zinc-500">
                        Loading model inventory…
                      </td>
                    </tr>
                  ) : (
                    models.map((m) => {
                      const job = downloadJobs[m.key];
                      const inFlight = job && (job.status === "queued" || job.status === "running");
                      const wasCancelled = job && job.status === "cancelled";
                      const isCancelling = cancelling.has(m.key);
                      const err = downloadErrors[m.key];
                      return (
                        <tr key={m.key} className="border-b border-zinc-800/30 hover:bg-zinc-800/20 align-top">
                          <td className="py-2.5 font-medium text-zinc-200 text-xs">
                            {m.name}
                            {err && (
                              <div className="text-[10px] text-red-400 mt-1 flex items-start gap-1">
                                <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" /> {err}
                              </div>
                            )}
                          </td>
                          <td className="py-2.5 text-zinc-400 text-xs">{m.vram_min}GB+</td>
                          <td className="py-2.5"><Badge className="text-[10px]">{m.quality || " - "}</Badge></td>
                          <td className="py-2.5">
                            <Badge variant={m.speed === "fast" ? "success" : "default"} className="text-[10px]">
                              {m.speed || " - "}
                            </Badge>
                          </td>
                          <td className="py-2.5">
                            {inFlight ? (
                              <span className="flex items-center gap-1 text-indigo-400 text-xs">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                {isCancelling ? "Cancelling…" : (job.message || "Downloading…")}
                              </span>
                            ) : m.installed ? (
                              <span className="flex items-center gap-1 text-green-400 text-xs"><Check className="w-3 h-3" /> Ready</span>
                            ) : wasCancelled ? (
                              <span className="flex items-center gap-1 text-amber-400 text-xs" title={job.message}>
                                <XCircle className="w-3 h-3" /> Cancelled - click Download to resume
                              </span>
                            ) : (
                              <span className="flex items-center gap-1 text-zinc-600 text-xs"><X className="w-3 h-3" /> Not downloaded</span>
                            )}
                          </td>
                          <td className="py-2.5 text-right">
                            {inFlight ? (
                              <Button
                                size="sm" variant="ghost"
                                onClick={() => handleCancel(m.key)}
                                disabled={isCancelling}
                                title="Cancel download (partial files stay on disk, safe to resume)"
                              >
                                {isCancelling ? (
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                  <X className="w-3 h-3" />
                                )}
                              </Button>
                            ) : m.installed ? (
                              <Button
                                size="sm" variant="ghost"
                                onClick={() => handleDelete(m.key, m.name)}
                                disabled={deletingKey === m.key}
                                title="Delete cached weights"
                              >
                                {deletingKey === m.key ? (
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                  <Trash2 className="w-3 h-3" />
                                )}
                              </Button>
                            ) : (
                              <Button size="sm" variant="secondary" onClick={() => handleDownload(m.key)}>
                                <Download className="w-3 h-3 mr-1" /> {wasCancelled ? "Resume" : "Download"}
                              </Button>
                            )}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* MODELS TAB — unified cross-backend inventory */}
      {activeTab === "models" && (
        <>
          <Card className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <CardTitle className="flex items-center gap-2">
                <Package className="w-4 h-4 text-indigo-400" /> Model Inventory
              </CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchRegistry} disabled={registryLoading}>
                {registryLoading ? (
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                ) : (
                  <RefreshCw className="w-3 h-3 mr-1" />
                )}
                Refresh
              </Button>
            </div>
            <p className="text-xs text-zinc-500 mb-4">
              Every weight StudioLite can load, probed live from disk. Rows marked grey are
              missing — click <span className="text-indigo-400">Get</span> to open the official
              source, then drop the file at the expected path shown below.
            </p>

            {/* Summary banner */}
            {registrySummary && (
              <div className="flex items-center gap-2 mb-4 text-xs text-zinc-400">
                <Badge variant="success" className="text-[10px]">
                  {registrySummary.present}/{registrySummary.total} models present
                </Badge>
                <span className="text-zinc-600">·</span>
                <span className="tabular-nums">{fmtBytes(registrySummary.bytes_on_disk)} on disk</span>
                {registrySummary.missing > 0 && (
                  <>
                    <span className="text-zinc-600">·</span>
                    <span className="text-amber-400">{registrySummary.missing} missing</span>
                  </>
                )}
              </div>
            )}

            {registryError && (
              <div className="mb-4 flex items-start gap-2 text-xs text-red-400 bg-red-500/5 border border-red-500/20 rounded px-3 py-2">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span>{registryError}</span>
              </div>
            )}

            {registryLoading && registry.length === 0 ? (
              <div className="py-6 text-center text-xs text-zinc-500 flex items-center justify-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" /> Probing disk…
              </div>
            ) : (
              KIND_ORDER.filter((k) => groupedRegistry[k].length > 0).map((kind) => (
                <div key={kind} className="mb-5 last:mb-0">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500 mb-2">
                    {KIND_LABEL[kind]}
                  </div>
                  <div className="space-y-1.5">
                    {groupedRegistry[kind].map((m) => {
                      const sizeLabel = m.present
                        ? fmtBytes(m.size_bytes)
                        : `~${fmtBytes(m.min_size_bytes)} expected`;
                      const isRevealed = revealedFolder?.id === m.id;
                      return (
                        <div
                          key={m.id}
                          className="flex flex-col gap-1 border border-zinc-800 bg-zinc-900/40 rounded-lg px-3 py-2"
                        >
                          <div className="flex items-center gap-3 flex-wrap">
                            {m.present ? (
                              <Check className="w-3.5 h-3.5 text-green-400 flex-shrink-0" aria-label="present" />
                            ) : (
                              <X className="w-3.5 h-3.5 text-zinc-600 flex-shrink-0" aria-label="missing" />
                            )}
                            <code className="text-xs font-mono text-zinc-300 flex-shrink-0" title={m.description}>
                              {m.id}
                            </code>
                            <span className="text-xs text-zinc-400 truncate flex-1 min-w-[200px]">
                              {m.name}
                            </span>
                            <Badge
                              variant={m.present ? "success" : "default"}
                              className="text-[10px] tabular-nums flex-shrink-0"
                              title={m.present ? "Actual on-disk size" : "Approximate size of a healthy install"}
                            >
                              {sizeLabel}
                            </Badge>
                            <a
                              href={m.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[10px] text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1 flex-shrink-0"
                              title={`Open ${m.source_url}`}
                            >
                              <ExternalLink className="w-3 h-3" /> Get
                            </a>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleOpenFolder(m.id)}
                              title="Show the parent directory you should drop the weight into"
                            >
                              <FolderOpen className="w-3 h-3 mr-1" /> Open folder
                            </Button>
                          </div>
                          <div
                            className="font-mono text-[10px] text-zinc-500 truncate pl-6"
                            title={m.expected_path}
                          >
                            {m.expected_path}
                          </div>
                          {isRevealed && revealedFolder && (
                            <div className="pl-6 mt-1 text-[10px]">
                              <span className="text-zinc-500">Folder: </span>
                              <code
                                className={`font-mono select-all ${revealedFolder.exists ? "text-emerald-400" : "text-amber-400"}`}
                              >
                                {revealedFolder.folder}
                              </code>
                              <span className="text-zinc-600 ml-2">
                                {revealedFolder.exists ? "(exists)" : "(create it)"}
                              </span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))
            )}
          </Card>
        </>
      )}

      {/* ENVIRONMENT TAB */}
      {activeTab === "env" && (
        <>
          {/* HF Token */}
          <Card className="mb-6 border-indigo-500/20">
            <CardTitle className="text-sm mb-3 flex items-center gap-2">
              <Key className="w-4 h-4 text-indigo-400" /> HuggingFace Token
            </CardTitle>
            <p className="text-xs text-zinc-500 mb-3">
              Required for downloading models (Wan VACE, SDXL, etc). Get one from{" "}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener" className="text-indigo-400 hover:text-indigo-300">huggingface.co/settings/tokens</a>
            </p>
            <div className="flex gap-2">
              <input value={hfToken} onChange={(e) => setHfToken(e.target.value)}
                type="password" placeholder="hf_..."
                className="flex-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:ring-2 focus:ring-indigo-500/50 font-mono" />
              <Button onClick={testHfToken} disabled={hfTesting || !hfToken.trim()}>
                {hfTesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4 mr-1" />}
                Test & Save
              </Button>
            </div>
            {hfTestResult && (
              <p className={`text-xs mt-2 ${hfTestResult.includes("valid") || hfTestResult.includes("ok") ? "text-green-400" : "text-red-400"}`}>
                {hfTestResult}
              </p>
            )}
          </Card>

          {/* Env Vars */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Settings className="w-4 h-4 text-zinc-400" /> Environment Variables
              </CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchEnv}>
                <RefreshCw className="w-3 h-3 mr-1" /> Refresh
              </Button>
            </div>
            <p className="text-xs text-zinc-500 mb-4">
              Changes are applied immediately and persisted to <code className="text-zinc-400">.env</code> file.
            </p>

            <div className="space-y-2 mb-4">
              {envVars.map((v) => (
                <div key={v.key} className="flex items-center gap-2">
                  <span className="text-xs font-mono text-indigo-400 w-48 flex-shrink-0 truncate" title={v.key}>{v.key}</span>
                  <input
                    defaultValue={v.value}
                    onBlur={(e) => { if (e.target.value !== v.value) saveEnvVar(v.key, e.target.value); }}
                    className="flex-1 bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono focus:ring-1 focus:ring-indigo-500/50" />
                  {envSaving === v.key ? (
                    <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin flex-shrink-0" />
                  ) : (
                    <button onClick={() => deleteEnvVar(v.key)} className="text-zinc-600 hover:text-red-400 flex-shrink-0">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))}
              {envVars.length === 0 && (
                <p className="text-xs text-zinc-600 py-4 text-center">No environment variables set</p>
              )}
            </div>

            {/* Add new */}
            <div className="flex items-center gap-2 pt-3 border-t border-zinc-800">
              <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="KEY"
                className="w-48 bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono placeholder-zinc-600 focus:ring-1 focus:ring-indigo-500/50" />
              <input value={newValue} onChange={(e) => setNewValue(e.target.value)} placeholder="value"
                className="flex-1 bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono placeholder-zinc-600 focus:ring-1 focus:ring-indigo-500/50" />
              <Button size="sm" onClick={addEnvVar} disabled={!newKey.trim()}>
                <Plus className="w-3 h-3 mr-1" /> Add
              </Button>
            </div>

            {/* Common vars hint */}
            <div className="mt-4 pt-3 border-t border-zinc-800">
              <p className="text-[10px] text-zinc-600 mb-1.5">Common variables:</p>
              <div className="flex flex-wrap gap-1.5">
                {["HF_TOKEN", "HF_HOME", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"].map((k) => (
                  <button key={k} onClick={() => setNewKey(k)}
                    className="px-2 py-0.5 rounded text-[10px] bg-zinc-800 text-zinc-500 hover:text-zinc-300 font-mono">{k}</button>
                ))}
              </div>
            </div>
          </Card>
        </>
      )}

      {/* LOGS TAB */}
      {activeTab === "logs" && (
        <>
          {/* Recent Jobs */}
          <Card className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="w-4 h-4 text-amber-400" /> Recent Jobs
              </CardTitle>
              <Button size="sm" variant="ghost" onClick={fetchLogs}>
                {logsLoading ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
                Refresh
              </Button>
            </div>
            <div className="space-y-1.5 max-h-[300px] overflow-y-auto">
              {jobLogs.map((j) => (
                <div key={j.job_id} className={`flex items-center gap-3 px-2.5 py-1.5 rounded text-xs ${
                  j.status === "failed" ? "bg-red-500/5 border border-red-500/20" :
                  j.status === "completed" ? "bg-green-500/5 border border-green-500/10" :
                  j.status === "running" ? "bg-amber-500/5 border border-amber-500/10" :
                  "bg-zinc-800/30 border border-zinc-800"
                }`}>
                  <span className="font-mono text-zinc-500 w-16 flex-shrink-0">{j.job_id}</span>
                  <Badge variant={
                    j.status === "completed" ? "success" :
                    j.status === "failed" ? "error" :
                    j.status === "running" ? "info" : "default"
                  } className="text-[9px] w-16 justify-center flex-shrink-0">{j.status}</Badge>
                  <span className="text-zinc-400 flex-shrink-0 w-14">{j.kind}</span>
                  <span className="text-zinc-300 flex-1 truncate">{j.message}</span>
                  {j.error && (
                    <span className="text-red-400 flex-shrink-0 flex items-center gap-1" title={j.error}>
                      <AlertTriangle className="w-3 h-3" /> Error
                    </span>
                  )}
                  <span className="text-zinc-600 flex-shrink-0 tabular-nums">{j.elapsed > 60 ? `${(j.elapsed/60).toFixed(1)}m` : `${j.elapsed.toFixed(0)}s`}</span>
                </div>
              ))}
              {jobLogs.length === 0 && (
                <p className="text-xs text-zinc-600 py-4 text-center">No jobs recorded yet</p>
              )}
            </div>
          </Card>

          {/* Application Logs */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileText className="w-4 h-4 text-zinc-400" /> Application Logs
              </CardTitle>
              <span className="text-[10px] text-zinc-600">{logLines.length} lines</span>
            </div>
            <div className="bg-zinc-950 rounded-lg p-3 max-h-[500px] overflow-y-auto font-mono text-[11px] leading-relaxed">
              {logLines.length > 0 ? logLines.map((line, i) => {
                const isError = /error|fail|exception|traceback/i.test(line);
                const isWarning = /warn/i.test(line);
                return (
                  <div key={i} className={`${
                    isError ? "text-red-400" : isWarning ? "text-amber-400" : "text-zinc-500"
                  } hover:bg-zinc-800/30 px-1 rounded`}>
                    {line}
                  </div>
                );
              }) : (
                <p className="text-zinc-600 text-center py-8">No log entries yet. Logs appear during video/audio generation.</p>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
