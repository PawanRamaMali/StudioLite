"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  FolderPlus, RefreshCw, ScanSearch, Copy, Trash2,
  AlertCircle, Loader2, HardDrive, Folder, Film, Sparkles,
  Search, X, CheckCircle2, Layers, ArrowRight, Wand2, Zap, Grid3x3, Tag,
  Image as ImageIcon, FolderX,
} from "lucide-react";
import {
  libraryAddRoot, libraryAddRootsBatch, libraryBatchDelete, libraryDeleteRoot,
  libraryDeleteVideo, libraryDeletionPlan, libraryDuplicates, libraryListRoots,
  libraryListVideos, libraryStartScan, libraryStats,
  libraryStreamUrl, libraryThumbUrl,
  libraryIndexStats, libraryStartEmbed, librarySearch, libraryBuildClusters,
  libraryClusterMembers,
  libraryEnhanceRecommend, libraryStartEnhance,
  libraryCleanupEmptyFolders,
  getJob,
  type KeeperStrategy, type LibraryBatchDeleteResult, type LibraryDeletionPlan,
  type LibraryDuplicates, type LibraryRoot, type LibraryStats, type LibraryVideo,
  type LibrarySearchHit, type LibraryClusterSummary,
  type LibraryEnhanceRecommendResponse, type LibraryEnhancePreset,
  type LibraryMediaKind,
  type Job,
} from "@/lib/api";

function fmtBytes(n: number): string {
  if (!n || n < 1024) return `${n | 0} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

function fmtDuration(sec: number | null | undefined): string {
  if (!sec || sec < 0) return "—";
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`
    : `${m}:${String(r).padStart(2, "0")}`;
}

function baseName(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

type ViewMode = "browse" | "duplicates" | "search" | "clusters";

export default function LibraryPanel() {
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [roots, setRoots] = useState<LibraryRoot[] | null>(null);
  const [view, setView] = useState<ViewMode>("browse");
  const [err, setErr] = useState<string | null>(null);
  const [scanJobId, setScanJobId] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState<Job | null>(null);

  const refreshStatsAndRoots = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([libraryStats(), libraryListRoots()]);
      setStats(s); setRoots(r.roots);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load library");
    }
  }, []);

  useEffect(() => {
    // refreshStatsAndRoots resolves setState via an await; the lint rule
    // can't see through the returned promise — mark it explicitly.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshStatsAndRoots();
  }, [refreshStatsAndRoots]);

  // Poll the scan job while it's running so the progress bar moves.
  useEffect(() => {
    if (!scanJobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const j = await getJob(scanJobId);
        if (cancelled) return;
        setScanProgress(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") {
          refreshStatsAndRoots();
          window.setTimeout(() => setScanJobId(null), 1500);
          return;
        }
      } catch { /* soft — try again */ }
      if (!cancelled) window.setTimeout(tick, 1200);
    };
    tick();
    return () => { cancelled = true; };
  }, [scanJobId, refreshStatsAndRoots]);

  const startScan = useCallback(async (rootIds?: number[]) => {
    setErr(null);
    try {
      const res = await libraryStartScan(rootIds);
      setScanJobId(res.job_id);
      setScanProgress(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to start scan");
    }
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Library</h1>
          <p className="text-zinc-400 mt-1">
            Organize your video files locally — scan folders, find duplicates, browse everything in one place.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant={view === "browse" ? "primary" : "secondary"} size="sm" onClick={() => setView("browse")}>
            <Film className="w-3.5 h-3.5 mr-1.5" /> Browse
          </Button>
          <Button variant={view === "search" ? "primary" : "secondary"} size="sm" onClick={() => setView("search")}>
            <Search className="w-3.5 h-3.5 mr-1.5" /> Search
          </Button>
          <Button variant={view === "clusters" ? "primary" : "secondary"} size="sm" onClick={() => setView("clusters")}>
            <Grid3x3 className="w-3.5 h-3.5 mr-1.5" /> Clusters
          </Button>
          <Button variant={view === "duplicates" ? "primary" : "secondary"} size="sm" onClick={() => setView("duplicates")}>
            <Copy className="w-3.5 h-3.5 mr-1.5" /> Duplicates
          </Button>
        </div>
      </div>

      <StatsBar stats={stats} />

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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1 space-y-3">
          <RootsPanel
            roots={roots}
            onChange={refreshStatsAndRoots}
            onScanAll={() => startScan()}
            onScanOne={(id) => startScan([id])}
            disabled={!!scanJobId && scanProgress?.status === "running"}
            setError={setErr}
          />
          {scanJobId && <ScanProgressCard job={scanProgress} />}
        </div>

        <div className="lg:col-span-2">
          {view === "browse" && <BrowseView onError={setErr} />}
          {view === "search" && <SearchView onError={setErr} />}
          {view === "clusters" && <ClustersView onError={setErr} />}
          {view === "duplicates" && <DuplicatesView onError={setErr} onChange={refreshStatsAndRoots} />}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats bar
// ---------------------------------------------------------------------------

function StatsBar({ stats }: { stats: LibraryStats | null }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <StatTile icon={<Folder className="w-4 h-4" />} label="Roots" value={stats?.roots ?? "…"} />
      <StatTile icon={<Film className="w-4 h-4" />} label="Videos" value={stats?.videos ?? "…"} />
      <StatTile icon={<HardDrive className="w-4 h-4" />}
                label="Total size" value={stats ? fmtBytes(stats.total_bytes) : "…"} />
      <StatTile icon={<Sparkles className="w-4 h-4" />}
                label="Indexed"
                value={stats
                  ? `${stats.hashed}/${stats.videos} hash · ${stats.phashed}/${stats.videos} phash`
                  : "…"} />
    </div>
  );
}

function StatTile({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <Card className="!py-3">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-600/15 text-indigo-400 flex items-center justify-center flex-shrink-0">
          {icon}
        </div>
        <div className="min-w-0">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
          <div className="text-sm font-medium text-zinc-100 truncate">{value}</div>
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Roots + Add root
// ---------------------------------------------------------------------------

function RootsPanel({
  roots, onChange, onScanAll, onScanOne, disabled, setError,
}: {
  roots: LibraryRoot[] | null;
  onChange: () => void;
  onScanAll: () => void;
  onScanOne: (id: number) => void;
  disabled: boolean;
  setError: (s: string) => void;
}) {
  const [newPath, setNewPath] = useState("");
  const [include, setInclude] = useState("");
  const [exclude, setExclude] = useState("");
  const [adding, setAdding] = useState(false);

  const add = useCallback(async () => {
    const lines = newPath.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    if (lines.length === 0) return;
    setAdding(true);
    try {
      if (lines.length === 1) {
        await libraryAddRoot(lines[0], {
          include_glob: include.trim() || undefined,
          exclude_glob: exclude.trim() || undefined,
        });
      } else {
        const res = await libraryAddRootsBatch(lines, {
          include_glob: include.trim() || undefined,
          exclude_glob: exclude.trim() || undefined,
        });
        if (res.skipped.length) {
          setError(
            `Skipped ${res.skipped.length} path${res.skipped.length === 1 ? "" : "s"}: ` +
            res.skipped.slice(0, 3).map((s) => `${s.path} (${s.reason})`).join("; ") +
            (res.skipped.length > 3 ? "…" : "")
          );
        }
      }
      setNewPath(""); setInclude(""); setExclude("");
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add root");
    } finally {
      setAdding(false);
    }
  }, [newPath, include, exclude, onChange, setError]);

  const remove = useCallback(async (id: number, path: string) => {
    if (!window.confirm(
      `Remove "${path}" from the library index?\n\nFiles on disk are NOT touched.`
    )) return;
    try {
      await libraryDeleteRoot(id);
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to remove root");
    }
  }, [onChange, setError]);

  return (
    <>
      <Card>
        <div className="flex items-center justify-between mb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Folder className="w-4 h-4 text-indigo-400" /> Watched folders
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button variant="secondary" size="sm"
                    onClick={async () => {
                      if (!roots || roots.length === 0) return;
                      try {
                        const dry = await libraryCleanupEmptyFolders({ dry_run: true });
                        const n = dry.count_removed;
                        const preview = dry.removed.slice(0, 5).join("\n") +
                          (dry.removed.length > 5 ? `\n… +${dry.removed.length - 5} more` : "");
                        if (n === 0) {
                          window.alert("No empty folders found.");
                          return;
                        }
                        const ok = window.confirm(
                          `Remove ${n} empty folder${n === 1 ? "" : "s"}?\n\n${preview}\n\n` +
                          "Watched-root folders themselves are never removed."
                        );
                        if (!ok) return;
                        const res = await libraryCleanupEmptyFolders({ dry_run: false });
                        setError(`Removed ${res.count_removed} empty folder${res.count_removed === 1 ? "" : "s"}.`);
                      } catch (e) {
                        setError(e instanceof Error ? e.message : "Cleanup failed");
                      }
                    }}
                    disabled={!roots || roots.length === 0 || disabled}
                    title="Bottom-up sweep of every watched folder — removes any subfolder whose only contents are junk (Thumbs.db, .DS_Store, desktop.ini)">
              <FolderX className="w-3.5 h-3.5 mr-1.5" /> Clean empty
            </Button>
            <Button variant="secondary" size="sm" onClick={onScanAll}
                    disabled={!roots || roots.length === 0 || disabled}
                    title={roots?.length ? "Scan every watched folder" : "Add a folder first"}>
              <ScanSearch className="w-3.5 h-3.5 mr-1.5" /> Scan all
            </Button>
          </div>
        </div>
        {!roots && <div className="text-xs text-zinc-500">Loading…</div>}
        {roots && roots.length === 0 && (
          <p className="text-xs text-zinc-500">No folders yet. Add one below to start scanning.</p>
        )}
        <ul className="space-y-1.5">
          {roots?.map((r) => (
            <li key={r.id} className="group border border-zinc-800 rounded-lg p-2 hover:border-zinc-700 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-zinc-200 font-mono truncate" title={r.path}>{r.path}</div>
                  <div className="text-[10px] text-zinc-500 flex flex-wrap gap-2 mt-0.5">
                    <span>{r.video_count} video{r.video_count === 1 ? "" : "s"}</span>
                    {r.include_glob && <span>include: {r.include_glob}</span>}
                    {r.exclude_glob && <span>exclude: {r.exclude_glob}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  <Button variant="ghost" size="sm" onClick={() => onScanOne(r.id)} disabled={disabled}
                          title="Scan just this folder">
                    <ScanSearch className="w-3.5 h-3.5" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => remove(r.id, r.path)}
                          title="Remove from index (files untouched)">
                    <Trash2 className="w-3.5 h-3.5 text-zinc-500" />
                  </Button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Card>

      <Card>
        <CardTitle className="text-sm mb-3 flex items-center gap-2">
          <FolderPlus className="w-4 h-4 text-indigo-400" /> Add a folder
        </CardTitle>
        <div className="space-y-2">
          <div>
            <label className="text-[10px] uppercase tracking-wide text-zinc-500">
              Path{"  "}<span className="text-zinc-600 lowercase">(one per line to add several at once)</span>
            </label>
            <textarea
              value={newPath} onChange={(e) => setNewPath(e.target.value)}
              placeholder={"C:\\Users\\you\\Videos\nD:\\Recordings\nE:\\Backups\\Family"}
              rows={3}
              className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 font-mono resize-y"
              spellCheck={false}
            />
            <p className="text-[10px] text-zinc-500 mt-1">
              Subfolders are searched automatically — no need to add each one.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">Include glob</label>
              <input
                value={include} onChange={(e) => setInclude(e.target.value)}
                placeholder="*.mp4,*.mkv"
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 font-mono"
                spellCheck={false}
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wide text-zinc-500">Exclude glob</label>
              <input
                value={exclude} onChange={(e) => setExclude(e.target.value)}
                placeholder="*.tmp,*.part"
                className="w-full mt-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-zinc-200 font-mono"
                spellCheck={false}
              />
            </div>
          </div>
          <Button className="w-full" size="sm" onClick={add} disabled={adding || !newPath.trim()}>
            {adding ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> Adding…</> :
                      <><FolderPlus className="w-3.5 h-3.5 mr-1.5" /> Add folder(s)</>}
          </Button>
          <p className="text-[10px] text-zinc-500">
            The path must exist on the server running the API. Nothing on disk is modified until you explicitly ask.
          </p>
        </div>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------
// Scan progress
// ---------------------------------------------------------------------------

function ScanProgressCard({ job }: { job: Job | null }) {
  const counts = (job?.result as { counts?: Record<string, number> } | undefined)?.counts;
  const pct = Math.round(((job?.progress ?? 0) as number) * 100);
  const done = job?.status === "completed" || job?.status === "failed" || job?.status === "cancelled";
  return (
    <Card>
      <CardTitle className="text-sm mb-2 flex items-center gap-2">
        {done
          ? <CheckCircle2 className="w-4 h-4 text-green-400" />
          : <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />}
        Scan
      </CardTitle>
      <div className="h-1.5 bg-zinc-800 rounded overflow-hidden mb-2">
        <div
          className={`h-full transition-all ${
            job?.status === "failed" ? "bg-red-500"
              : job?.status === "cancelled" ? "bg-zinc-500"
              : "bg-indigo-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-[11px] text-zinc-400">{job?.message || "…"}</div>
      {counts && (
        <div className="grid grid-cols-4 gap-2 mt-2">
          <MiniCount label="Files"    value={counts.discovered} />
          <MiniCount label="Hashed"   value={counts.hashed} />
          <MiniCount label="pHashed"  value={counts.phashed} />
          <MiniCount label="Skipped"  value={counts.skipped} tone="dim" />
        </div>
      )}
      {job?.error && <p className="text-[10px] text-red-300 mt-2">{job.error}</p>}
    </Card>
  );
}

function MiniCount({ label, value, tone }: { label: string; value: number; tone?: "dim" }) {
  return (
    <div className="text-center bg-zinc-900/60 rounded px-1 py-1.5 border border-zinc-800">
      <div className={`text-sm font-mono ${tone === "dim" ? "text-zinc-500" : "text-zinc-100"}`}>{value ?? 0}</div>
      <div className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Browse
// ---------------------------------------------------------------------------

function BrowseView({ onError }: { onError: (s: string) => void }) {
  const [videos, setVideos] = useState<LibraryVideo[] | null>(null);
  const [total, setTotal] = useState(0);
  const [q, setQ] = useState("");
  const [kind, setKind] = useState<LibraryMediaKind | "all">("all");
  const [pending, setPending] = useState(false);
  const [selected, setSelected] = useState<LibraryVideo | null>(null);
  const debounceRef = useRef<number | null>(null);

  const load = useCallback(async (query: string, filterKind: LibraryMediaKind | "all") => {
    setPending(true);
    try {
      const res = await libraryListVideos({
        limit: 60,
        q: query || undefined,
        kind: filterKind === "all" ? undefined : filterKind,
      });
      setVideos(res.videos); setTotal(res.total);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to list media");
    } finally {
      setPending(false);
    }
  }, [onError]);

  useEffect(() => { load("", kind); }, [load, kind]);

  useEffect(() => {
    if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => load(q, kind), 250);
    return () => { if (debounceRef.current !== null) window.clearTimeout(debounceRef.current); };
  }, [q, kind, load]);

  const del = useCallback(async (v: LibraryVideo, deleteFile: boolean) => {
    const label = deleteFile ? "DELETE FROM DISK" : "remove from index";
    if (!window.confirm(
      deleteFile
        ? `PERMANENTLY DELETE this file from disk?\n\n${v.abs_path}\n\nThis cannot be undone.`
        : `Remove from library index?\n\n${v.abs_path}\n\nThe file on disk is not touched.`
    )) return;
    try {
      await libraryDeleteVideo(v.id, deleteFile);
      setSelected(null);
      load(q, kind);
    } catch (e) {
      onError(e instanceof Error ? e.message : `Failed to ${label}`);
    }
  }, [load, q, kind, onError]);

  return (
    <Card className="min-h-[500px]">
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <CardTitle className="text-sm flex items-center gap-2">
          <Film className="w-4 h-4 text-indigo-400" /> All media
          <span className="text-[10px] text-zinc-500 font-mono">
            {videos ? `${videos.length} of ${total}` : "…"}
          </span>
        </CardTitle>
        <div className="flex items-center rounded-lg border border-zinc-700 overflow-hidden">
          {(["all", "video", "image"] as const).map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className={`px-2.5 py-1 text-[11px] transition-colors ${
                kind === k
                  ? "bg-indigo-500/20 text-indigo-300"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
              title={k === "all" ? "Show everything" : `Show only ${k}s`}
            >
              {k === "all" ? "All" : k === "video" ? "Videos" : "Images"}
            </button>
          ))}
        </div>
        <div className="ml-auto relative">
          <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-2 top-1/2 -translate-y-1/2" />
          <input
            value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Filter by path…"
            className="pl-7 pr-3 py-1.5 bg-zinc-800/50 border border-zinc-700 rounded-lg text-xs text-zinc-200 w-56"
          />
          {pending && <Loader2 className="w-3 h-3 text-zinc-500 absolute right-2 top-1/2 -translate-y-1/2 animate-spin" />}
        </div>
      </div>

      {videos && videos.length === 0 && (
        <p className="text-xs text-zinc-500 text-center py-16">
          {q ? "No matches for that filter." : "No videos indexed yet. Add a folder and run a scan."}
        </p>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {videos?.map((v) => (
          <button
            key={v.id}
            onClick={() => setSelected(v)}
            className="text-left group bg-zinc-950/40 border border-zinc-800 rounded-lg overflow-hidden hover:border-indigo-500/40 transition-colors"
          >
            <div className="relative bg-black aspect-video">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={libraryThumbUrl(v.id)}
                alt={baseName(v.abs_path)}
                loading="lazy"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-1 left-1 text-[9px] font-mono bg-black/70 text-zinc-300 rounded px-1 flex items-center gap-1">
                {v.kind === "image"
                  ? <><ImageIcon className="w-2.5 h-2.5" /> IMG</>
                  : <><Film className="w-2.5 h-2.5" /> VID</>}
              </div>
              <div className="absolute bottom-1 right-1 text-[9px] font-mono bg-black/70 text-zinc-200 rounded px-1">
                {v.kind === "image" ? (v.width && v.height ? `${v.width}×${v.height}` : "IMG") : fmtDuration(v.duration_sec)}
              </div>
            </div>
            <div className="p-2">
              <div className="text-[11px] text-zinc-200 truncate" title={v.abs_path}>{baseName(v.abs_path)}</div>
              <div className="text-[10px] text-zinc-500 flex items-center gap-1.5">
                {v.width && v.height && <span>{v.width}×{v.height}</span>}
                {v.codec && <span>· {v.codec}</span>}
                <span>· {fmtBytes(v.size_bytes)}</span>
              </div>
            </div>
          </button>
        ))}
      </div>

      {selected && (
        <VideoDetailModal
          video={selected}
          onClose={() => setSelected(null)}
          onRemoveIndex={() => del(selected, false)}
          onDeleteFile={() => del(selected, true)}
        />
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Duplicates
// ---------------------------------------------------------------------------

function DuplicatesView({
  onError, onChange,
}: { onError: (s: string) => void; onChange: () => void }) {
  const [data, setData] = useState<LibraryDuplicates | null>(null);
  const [threshold, setThreshold] = useState(8);
  const [pending, setPending] = useState(false);
  const [planStrategy, setPlanStrategy] = useState<KeeperStrategy>("largest");
  const [showReview, setShowReview] = useState(false);

  const load = useCallback(async () => {
    setPending(true);
    try {
      setData(await libraryDuplicates(threshold));
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to load duplicates");
    } finally {
      setPending(false);
    }
  }, [threshold, onError]);

  useEffect(() => { load(); }, [load]);

  const del = useCallback(async (v: LibraryVideo, deleteFile: boolean) => {
    if (!window.confirm(deleteFile
      ? `PERMANENTLY DELETE from disk?\n\n${v.abs_path}\n\nThis cannot be undone.`
      : `Remove just this row from the library index?\n\n${v.abs_path}\n\nFile is not touched.`
    )) return;
    try {
      await libraryDeleteVideo(v.id, deleteFile);
      load(); onChange();
    } catch (e) {
      onError(e instanceof Error ? e.message : "Delete failed");
    }
  }, [load, onChange, onError]);

  const totalSaveable = (data?.exact_saveable_bytes ?? 0) + (data?.near_saveable_bytes ?? 0);
  const hasClusters = !!data && (data.exact_clusters.length + data.near_clusters.length > 0);

  return (
    <div className="space-y-3">
      <Card>
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="text-sm flex items-center gap-2">
              <Copy className="w-4 h-4 text-indigo-400" /> Duplicate clusters
            </CardTitle>
            {data && (
              <p className="text-[11px] text-zinc-500 mt-1">
                {data.exact_clusters.length} exact · {data.near_clusters.length} near ·
                {" "}~{fmtBytes(totalSaveable)} recoverable if you delete the extras
              </p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <label className="text-[10px] uppercase tracking-wide text-zinc-500">Near threshold</label>
            <input
              type="range" min={0} max={20} value={threshold}
              onChange={(e) => setThreshold(parseInt(e.target.value, 10))}
              className="accent-indigo-500"
            />
            <span className="text-xs text-zinc-300 font-mono w-6 text-right">{threshold}</span>
            <Button variant="secondary" size="sm" onClick={load} disabled={pending}
                    title="Recompute clusters">
              <RefreshCw className={`w-3.5 h-3.5 ${pending ? "animate-spin" : ""}`} />
            </Button>
            <div className="w-px h-6 bg-zinc-800 mx-1" />
            <label className="text-[10px] uppercase tracking-wide text-zinc-500">Keep</label>
            <select
              value={planStrategy}
              onChange={(e) => setPlanStrategy(e.target.value as KeeperStrategy)}
              className="bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200"
            >
              <option value="largest">Largest file</option>
              <option value="smallest">Smallest file</option>
              <option value="oldest">Oldest</option>
              <option value="newest">Newest</option>
              <option value="shortest_path">Shortest path</option>
            </select>
            <Button variant="danger" size="sm"
                    disabled={!hasClusters}
                    onClick={() => setShowReview(true)}
                    title="Review a full 'delete all but one' plan before anything runs">
              <Layers className="w-3.5 h-3.5 mr-1.5" /> Delete all but 1
            </Button>
          </div>
        </div>
      </Card>

      {showReview && (
        <DeleteAllButOneReview
          strategy={planStrategy}
          threshold={threshold}
          onClose={() => setShowReview(false)}
          onDone={() => { setShowReview(false); load(); onChange(); }}
          onError={onError}
        />
      )}

      {data && data.exact_clusters.length === 0 && data.near_clusters.length === 0 && (
        <Card className="text-center py-10">
          <CheckCircle2 className="w-8 h-8 text-green-400 mx-auto mb-2" />
          <p className="text-sm text-zinc-200">No duplicates found.</p>
          <p className="text-[11px] text-zinc-500 mt-1">
            Try lowering the near-threshold to catch fuzzier matches — or run a scan first to populate hashes.
          </p>
        </Card>
      )}

      {data?.exact_clusters.map((c) => (
        <ClusterCard key={c.key} cluster={c} onDelete={del} />
      ))}
      {data?.near_clusters.map((c) => (
        <ClusterCard key={c.key} cluster={c} onDelete={del} />
      ))}
    </div>
  );
}

function ClusterCard({
  cluster, onDelete,
}: {
  cluster: LibraryDuplicates["exact_clusters"][number];
  onDelete: (v: LibraryVideo, deleteFile: boolean) => void;
}) {
  const totalWaste = useMemo(
    () => cluster.members.slice(1).reduce((s, m) => s + m.size_bytes, 0),
    [cluster.members],
  );
  return (
    <Card>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        {cluster.kind === "exact"
          ? <Badge className="text-[10px] bg-red-500/15 text-red-300 border border-red-500/25">EXACT</Badge>
          : <Badge className="text-[10px] bg-orange-500/15 text-orange-300 border border-orange-500/25">NEAR</Badge>}
        <span className="text-xs text-zinc-300 font-medium">
          {cluster.members.length} files
        </span>
        <span className="text-[10px] text-zinc-500">· ~{fmtBytes(totalWaste)} freeable</span>
        {cluster.kind === "exact" && (
          <span className="text-[9px] font-mono text-zinc-600 ml-auto">{cluster.key.slice(0, 12)}…</span>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {cluster.members.map((v, i) => (
          <div key={v.id}
               className={`border rounded-lg overflow-hidden ${
                 i === 0 ? "border-green-500/30 bg-green-500/5" : "border-zinc-800"
               }`}>
            <div className="relative bg-black aspect-video">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={libraryThumbUrl(v.id)}
                alt={baseName(v.abs_path)}
                loading="lazy"
                className="w-full h-full object-cover"
              />
              {i === 0 && (
                <div className="absolute top-1 left-1 text-[9px] font-mono bg-green-500/80 text-black rounded px-1">
                  KEEP
                </div>
              )}
              <div className="absolute bottom-1 right-1 text-[9px] font-mono bg-black/70 text-zinc-200 rounded px-1">
                {fmtDuration(v.duration_sec)}
              </div>
            </div>
            <div className="p-2 text-[11px]">
              <div className="text-zinc-200 truncate" title={v.abs_path}>{baseName(v.abs_path)}</div>
              <div className="text-[10px] text-zinc-500 truncate" title={v.abs_path}>{v.abs_path}</div>
              <div className="text-[10px] text-zinc-500 mt-0.5">
                {v.width && v.height && <span>{v.width}×{v.height} · </span>}
                {fmtBytes(v.size_bytes)}
              </div>
              {i > 0 && (
                <div className="flex items-center gap-1 mt-1.5">
                  <Button variant="secondary" size="sm" className="flex-1 !text-[10px] !py-1"
                          onClick={() => onDelete(v, false)}
                          title="Remove from library index only">
                    Unindex
                  </Button>
                  <Button variant="danger" size="sm" className="flex-1 !text-[10px] !py-1"
                          onClick={() => onDelete(v, true)}
                          title="Delete from disk permanently">
                    <Trash2 className="w-3 h-3 mr-1" /> Delete
                  </Button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Video detail modal
// ---------------------------------------------------------------------------

function VideoDetailModal({
  video, onClose, onRemoveIndex, onDeleteFile,
}: {
  video: LibraryVideo;
  onClose: () => void;
  onRemoveIndex: () => void;
  onDeleteFile: () => void;
}) {
  const [showEnhance, setShowEnhance] = useState(false);
  const tags = (video.tags ?? []) as { tag: string; score: number }[];
  const isImage = video.kind === "image";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="min-w-0">
            <h2 className="text-base font-semibold text-zinc-100 truncate flex items-center gap-2">
              {isImage
                ? <ImageIcon className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                : <Film className="w-4 h-4 text-indigo-400 flex-shrink-0" />}
              {baseName(video.abs_path)}
            </h2>
            <p className="text-[11px] text-zinc-500 font-mono truncate">{video.abs_path}</p>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        {isImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={libraryStreamUrl(video.id)}
            alt={baseName(video.abs_path)}
            className="w-full max-h-[55vh] bg-black rounded-lg object-contain"
          />
        ) : (
          <video
            src={libraryStreamUrl(video.id)}
            controls preload="metadata"
            className="w-full max-h-[55vh] bg-black rounded-lg"
          />
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          {isImage
            ? <DetailField label="Kind" value="Image" />
            : <DetailField label="Duration" value={fmtDuration(video.duration_sec)} />}
          <DetailField label="Resolution" value={video.width && video.height ? `${video.width}×${video.height}` : "—"} />
          <DetailField label={isImage ? "Format" : "Codec"} value={video.codec ?? "—"} />
          <DetailField label="Size" value={fmtBytes(video.size_bytes)} />
        </div>

        {tags.length > 0 && (
          <div className="mt-3">
            <div className="text-[9px] uppercase tracking-wide text-zinc-500 mb-1 flex items-center gap-1">
              <Tag className="w-3 h-3" /> Content tags
            </div>
            <div className="flex flex-wrap gap-1.5">
              {tags.map((t) => (
                <span key={t.tag} className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/25 text-indigo-300">
                  {t.tag} <span className="text-indigo-400/60">{Math.round(t.score * 100)}%</span>
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <a href={libraryStreamUrl(video.id)} target="_blank" rel="noreferrer"
             className="text-xs text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1">
            Open in new tab <ArrowRight className="w-3 h-3" />
          </a>
          <div className="ml-auto flex items-center gap-2">
            <Button variant="primary" size="sm" onClick={() => setShowEnhance((s) => !s)}
                    title="Upscale + optional face restoration">
              <Zap className="w-3.5 h-3.5 mr-1.5" /> Enhance
            </Button>
            <Button variant="secondary" size="sm" onClick={onRemoveIndex}
                    title="Drop from the library index; file on disk is untouched">
              <Layers className="w-3.5 h-3.5 mr-1.5" /> Remove from index
            </Button>
            <Button variant="danger" size="sm" onClick={onDeleteFile}
                    title="Permanently delete the file from disk">
              <Trash2 className="w-3.5 h-3.5 mr-1.5" /> Delete from disk
            </Button>
          </div>
        </div>

        {showEnhance && <EnhancePanel videoId={video.id} onClose={() => setShowEnhance(false)} />}

        <div className="mt-3 text-[10px] text-zinc-500 grid grid-cols-2 gap-x-4">
          {video.sha256 && <div>SHA-256: <span className="font-mono">{video.sha256.slice(0, 20)}…</span></div>}
          {video.phash_hex && <div>pHash: <span className="font-mono">{video.phash_hex}</span></div>}
        </div>
      </div>
    </div>
  );
}

function EnhancePanel({ videoId, onClose }: { videoId: number; onClose: () => void }) {
  const [rec, setRec] = useState<LibraryEnhanceRecommendResponse | null>(null);
  const [preset, setPreset] = useState<LibraryEnhancePreset>("quality_2x");
  const [faceRestore, setFaceRestore] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    libraryEnhanceRecommend(videoId).then((r) => {
      setRec(r);
      if (r.recommendations[0]) {
        setPreset(r.recommendations[0].preset);
        setFaceRestore(r.recommendations[0].face_restore && r.face_restore_available);
      }
    }).catch((e) => setErr(e instanceof Error ? e.message : "Recommend failed"));
  }, [videoId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (cancelled) return;
        setJob(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") return;
      } catch { /* soft */ }
      if (!cancelled) window.setTimeout(tick, 1500);
    };
    tick();
    return () => { cancelled = true; };
  }, [jobId]);

  const start = useCallback(async () => {
    setErr(null);
    try {
      const res = await libraryStartEnhance(videoId, preset, faceRestore);
      setJobId(res.job_id);
      setJob(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Start failed");
    }
  }, [videoId, preset, faceRestore]);

  const result = job?.result as { output_path?: string } | undefined;
  const outputPath = result?.output_path;
  const outputFileName = outputPath ? outputPath.split(/[\\/]/).pop() : null;
  const outputUrl = outputFileName ? `/static/library/enhanced/${outputFileName}` : null;

  return (
    <div className="mt-3 border border-zinc-800 rounded-lg bg-zinc-950/40 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-zinc-100 flex items-center gap-2">
          <Wand2 className="w-4 h-4 text-indigo-400" /> Enhance video
        </div>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300"><X className="w-3.5 h-3.5" /></button>
      </div>

      {rec && rec.recommendations.length > 0 && !jobId && (
        <div className="mb-2 space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Recommendations</div>
          {rec.recommendations.map((r) => (
            <button key={r.preset + r.reason}
                    onClick={() => { setPreset(r.preset); setFaceRestore(r.face_restore && rec.face_restore_available); }}
                    className={`w-full text-left p-2 border rounded transition-colors ${
                      preset === r.preset ? "border-indigo-500/40 bg-indigo-500/10" : "border-zinc-800 hover:border-zinc-700"
                    }`}>
              <div className="text-xs text-zinc-100">
                {r.preset}
                {r.face_restore && rec.face_restore_available && <span className="text-[10px] text-indigo-300 ml-1.5">+ face restore</span>}
              </div>
              <div className="text-[10px] text-zinc-400">{r.reason}</div>
            </button>
          ))}
        </div>
      )}

      {!jobId && (
        <div className="flex items-end gap-2 flex-wrap">
          <div>
            <label className="text-[10px] uppercase tracking-wide text-zinc-500">Preset</label>
            <select value={preset} onChange={(e) => setPreset(e.target.value as LibraryEnhancePreset)}
                    className="block bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 mt-1">
              {(rec?.presets ?? ["fast_2x", "quality_2x", "ultra_4x", "anime_4x"]).map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          {rec?.face_restore_available && (
            <label className="flex items-center gap-1.5 text-xs text-zinc-300">
              <input type="checkbox" checked={faceRestore} onChange={(e) => setFaceRestore(e.target.checked)}
                     className="accent-indigo-500" />
              Face restore (GFPGAN)
            </label>
          )}
          <Button size="sm" onClick={start} className="ml-auto">
            <Zap className="w-3.5 h-3.5 mr-1.5" /> Start
          </Button>
        </div>
      )}

      {jobId && job && (
        <div className="space-y-1.5">
          <div className="text-[11px] text-zinc-400 flex items-center gap-2">
            {job.status === "completed"
              ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
              : job.status === "failed"
              ? <AlertCircle className="w-3.5 h-3.5 text-red-400" />
              : <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />}
            <span>{job.message || "…"}</span>
            <span className="ml-auto font-mono">{Math.round(((job.progress ?? 0) as number) * 100)}%</span>
          </div>
          <div className="h-1 bg-zinc-800 rounded overflow-hidden">
            <div className={`h-full transition-all ${job.status === "failed" ? "bg-red-500" : "bg-indigo-500"}`}
                 style={{ width: `${Math.round(((job.progress ?? 0) as number) * 100)}%` }} />
          </div>
          {job.error && <p className="text-[10px] text-red-300">{job.error}</p>}
          {job.status === "completed" && outputUrl && (
            <a href={outputUrl} target="_blank" rel="noreferrer"
               className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300">
              Open enhanced video <ArrowRight className="w-3 h-3" />
            </a>
          )}
        </div>
      )}

      {err && <p className="text-[10px] text-red-300 mt-1">{err}</p>}
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-2">
      <div className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="text-xs text-zinc-100 mt-0.5">{value}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// "Delete all but 1" review + execute
// ---------------------------------------------------------------------------

function DeleteAllButOneReview({
  strategy, threshold, onClose, onDone, onError,
}: {
  strategy: KeeperStrategy;
  threshold: number;
  onClose: () => void;
  onDone: () => void;
  onError: (s: string) => void;
}) {
  const [plan, setPlan] = useState<LibraryDeletionPlan | null>(null);
  const [loading, setLoading] = useState(true);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [deleteFile, setDeleteFile] = useState(true);
  const [removeEmptyFolders, setRemoveEmptyFolders] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<LibraryBatchDeleteResult | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const p = await libraryDeletionPlan({
        include_exact: true,
        include_near: true,
        near_threshold: threshold,
        keeper_strategy: strategy,
        cluster_keeper_overrides: overrides,
      });
      setPlan(p);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to build plan");
    } finally {
      setLoading(false);
    }
  }, [threshold, strategy, overrides, onError]);

  useEffect(() => { load(); }, [load]);

  const execute = useCallback(async () => {
    if (!plan) return;
    const label = deleteFile ? "PERMANENTLY DELETE FROM DISK" : "remove from the library index";
    const ok = window.confirm(
      `About to ${label} ${plan.total_delete_files} file${plan.total_delete_files === 1 ? "" : "s"}` +
      (deleteFile ? ` (~${humanBytes(plan.total_delete_bytes)}).\n\nThis cannot be undone.` : ".") +
      "\n\nContinue?"
    );
    if (!ok) return;
    setExecuting(true);
    try {
      const res = await libraryBatchDelete(plan.delete_ids, deleteFile, deleteFile && removeEmptyFolders);
      setResult(res);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Batch delete failed");
    } finally {
      setExecuting(false);
    }
  }, [plan, deleteFile, removeEmptyFolders, onError]);

  const swapKeeper = (clusterKey: string, newKeeperId: number) => {
    setOverrides((o) => ({ ...o, [clusterKey]: newKeeperId }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-5xl w-full max-h-[92vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-4 border-b border-zinc-800 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-zinc-100 flex items-center gap-2">
              <Layers className="w-4 h-4 text-orange-400" /> Delete all but one — review
            </h2>
            <p className="text-[11px] text-zinc-500 mt-0.5">
              Nothing is deleted until you click Execute. Click a file within a cluster to make it the keeper instead.
            </p>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        {loading && (
          <div className="p-10 text-center">
            <Loader2 className="w-6 h-6 text-indigo-400 mx-auto mb-2 animate-spin" />
            <p className="text-xs text-zinc-500">Building plan…</p>
          </div>
        )}

        {!loading && plan && plan.clusters.length === 0 && (
          <div className="p-10 text-center">
            <CheckCircle2 className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <p className="text-sm text-zinc-200">Nothing to delete — no duplicates matched.</p>
          </div>
        )}

        {!loading && plan && plan.clusters.length > 0 && (
          <>
            <div className="p-4 bg-orange-500/5 border-b border-orange-500/20 grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
              <SummaryTile label="Clusters affected" value={plan.clusters.length} />
              <SummaryTile label="Files to delete" value={plan.total_delete_files} tone="warn" />
              <SummaryTile label="Space to reclaim" value={humanBytes(plan.total_delete_bytes)} tone="warn" />
              <SummaryTile label="Keeper policy"
                value={<span className="capitalize">{plan.strategy.replace("_", " ")}</span>} />
            </div>

            <div className="overflow-y-auto flex-1 p-4 space-y-3">
              {plan.clusters.map((c) => (
                <div key={c.key} className="border border-zinc-800 rounded-lg overflow-hidden">
                  <div className="bg-zinc-950/40 px-3 py-1.5 flex items-center gap-2 border-b border-zinc-800">
                    {c.kind === "exact"
                      ? <Badge className="text-[10px] bg-red-500/15 text-red-300 border border-red-500/25">EXACT</Badge>
                      : <Badge className="text-[10px] bg-orange-500/15 text-orange-300 border border-orange-500/25">NEAR</Badge>}
                    <span className="text-xs text-zinc-200">{c.delete.length + 1} files</span>
                    <span className="text-[10px] text-zinc-500 ml-auto">-{humanBytes(c.delete_bytes)}</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 p-2">
                    <PlanRow video={c.keeper} isKeeper onClick={() => { /* already keeper */ }} />
                    {c.delete.map((v) => (
                      <PlanRow key={v.id} video={v} isKeeper={false}
                               onClick={() => swapKeeper(c.key, v.id)} />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="p-3 border-t border-zinc-800 flex items-center gap-3 flex-wrap bg-zinc-950/60">
              <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                <input type="checkbox" checked={deleteFile}
                       onChange={(e) => setDeleteFile(e.target.checked)}
                       className="accent-red-500" />
                Also delete from disk (uncheck to only remove from the library index)
              </label>
              <label className={`flex items-center gap-2 text-xs cursor-pointer ${
                deleteFile ? "text-zinc-300" : "text-zinc-600"
              }`}>
                <input type="checkbox" checked={removeEmptyFolders}
                       disabled={!deleteFile}
                       onChange={(e) => setRemoveEmptyFolders(e.target.checked)}
                       className="accent-indigo-500" />
                Remove folders left empty
              </label>
              <div className="flex-1" />
              <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
              <Button variant={deleteFile ? "danger" : "primary"} size="md"
                      disabled={executing || plan.total_delete_files === 0}
                      onClick={execute}>
                {executing
                  ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> Deleting…</>
                  : <>
                      <Trash2 className="w-3.5 h-3.5 mr-1.5" />
                      Execute — {plan.total_delete_files} file{plan.total_delete_files === 1 ? "" : "s"}
                    </>}
              </Button>
            </div>
          </>
        )}

        {result && (
          <ResultOverlay result={result} onClose={onDone} />
        )}
      </div>
    </div>
  );
}

function PlanRow({ video, isKeeper, onClick }: {
  video: LibraryVideo; isKeeper: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`text-left flex items-center gap-2 p-2 rounded-md border transition-colors ${
        isKeeper
          ? "border-green-500/40 bg-green-500/5"
          : "border-zinc-800 hover:border-zinc-700 bg-zinc-950/40"
      }`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={libraryThumbUrl(video.id)}
        alt=""
        loading="lazy"
        className="w-24 aspect-video object-cover rounded bg-black flex-shrink-0"
      />
      <div className="min-w-0 flex-1">
        <div className="text-[11px] text-zinc-100 truncate" title={video.abs_path}>
          {baseName(video.abs_path)}
        </div>
        <div className="text-[10px] text-zinc-500 truncate" title={video.abs_path}>
          {video.abs_path}
        </div>
        <div className="text-[10px] mt-0.5 flex items-center gap-1.5">
          {isKeeper
            ? <span className="text-green-400 font-semibold">KEEP</span>
            : <span className="text-red-400 font-semibold">DELETE</span>}
          <span className="text-zinc-500">
            · {video.width && video.height && `${video.width}×${video.height} · `}
            {fmtBytes(video.size_bytes)}
          </span>
        </div>
      </div>
    </button>
  );
}

function SummaryTile({ label, value, tone }: {
  label: string; value: React.ReactNode; tone?: "warn";
}) {
  return (
    <div className="bg-zinc-950/40 rounded-lg py-2 px-3 border border-zinc-800">
      <div className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`text-sm font-medium mt-0.5 ${tone === "warn" ? "text-orange-300" : "text-zinc-100"}`}>
        {value}
      </div>
    </div>
  );
}

function ResultOverlay({ result, onClose }: {
  result: LibraryBatchDeleteResult; onClose: () => void;
}) {
  const failures = result.results.filter((r) => r.status === "failed" || r.file_error);
  return (
    <div className="absolute inset-0 bg-zinc-950/95 flex items-center justify-center p-4">
      <div className="max-w-lg w-full text-center">
        <CheckCircle2 className={`w-10 h-10 mx-auto mb-2 ${failures.length ? "text-orange-400" : "text-green-400"}`} />
        <h3 className="text-lg font-semibold text-zinc-100">
          Deleted {result.files_deleted} file{result.files_deleted === 1 ? "" : "s"}
        </h3>
        <p className="text-sm text-zinc-400 mt-1">
          Freed {humanBytes(result.bytes_freed)} of disk space
          {failures.length > 0 && `, ${failures.length} skipped`}
          {result.folders_removed && result.folders_removed.length > 0 &&
            ` — plus ${result.folders_removed.length} emptied folder${result.folders_removed.length === 1 ? "" : "s"}`}.
        </p>
        {failures.length > 0 && (
          <div className="text-left mt-3 max-h-40 overflow-y-auto border border-zinc-800 rounded-lg p-2 text-[10px] font-mono text-red-300">
            {failures.slice(0, 12).map((f) => (
              <div key={f.id} className="truncate">
                {f.abs_path || f.id}: {f.file_error || f.reason || f.status}
              </div>
            ))}
            {failures.length > 12 && <div className="text-zinc-500">…{failures.length - 12} more</div>}
          </div>
        )}
        <Button className="mt-4" onClick={onClose}>Done</Button>
      </div>
    </div>
  );
}

function humanBytes(n: number): string { return fmtBytes(n); }

// ---------------------------------------------------------------------------
// T2: Semantic search view
// ---------------------------------------------------------------------------

function SearchView({ onError }: { onError: (s: string) => void }) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<LibrarySearchHit[] | null>(null);
  const [pending, setPending] = useState(false);
  const [selected, setSelected] = useState<LibraryVideo | null>(null);
  const [indexStats, setIndexStats] = useState<{ embedded: number; videos: number; embed_model: string } | null>(null);
  const [embedJobId, setEmbedJobId] = useState<string | null>(null);
  const [embedProgress, setEmbedProgress] = useState<Job | null>(null);

  const loadStats = useCallback(async () => {
    try {
      const s = await libraryIndexStats();
      setIndexStats({ embedded: s.embedded, videos: s.videos, embed_model: s.embed_model });
    } catch { /* soft */ }
  }, []);

  useEffect(() => { loadStats(); }, [loadStats]);

  // Poll the embed job while running.
  useEffect(() => {
    if (!embedJobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const j = await getJob(embedJobId);
        if (cancelled) return;
        setEmbedProgress(j);
        if (j.status === "completed" || j.status === "failed" || j.status === "cancelled") {
          loadStats();
          window.setTimeout(() => setEmbedJobId(null), 1500);
          return;
        }
      } catch { /* soft */ }
      if (!cancelled) window.setTimeout(tick, 1500);
    };
    tick();
    return () => { cancelled = true; };
  }, [embedJobId, loadStats]);

  const search = useCallback(async () => {
    if (!query.trim()) return;
    setPending(true);
    try {
      const res = await librarySearch(query.trim());
      setHits(res.hits);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Search failed");
    } finally { setPending(false); }
  }, [query, onError]);

  const startEmbed = useCallback(async () => {
    try {
      const res = await libraryStartEmbed(true);
      setEmbedJobId(res.job_id);
      setEmbedProgress(null);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Could not start indexing");
    }
  }, [onError]);

  const canSearch = !indexStats || indexStats.embedded > 0;
  return (
    <Card className="min-h-[500px]">
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <CardTitle className="text-sm flex items-center gap-2">
          <Search className="w-4 h-4 text-indigo-400" /> Semantic search
        </CardTitle>
        {indexStats && (
          <span className="text-[10px] text-zinc-500">
            {indexStats.embedded}/{indexStats.videos} indexed{indexStats.embed_model && ` · ${indexStats.embed_model}`}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button variant="secondary" size="sm" onClick={startEmbed}
                  disabled={!!embedJobId && embedProgress?.status === "running"}
                  title={embedJobId ? "Indexing in progress" : "Encode CLIP embeddings for every video (uses the ~600 MB openai/clip-vit-base-patch32 model — downloaded on first run)"}>
            <Sparkles className="w-3.5 h-3.5 mr-1.5" /> Index content
          </Button>
        </div>
      </div>

      {embedJobId && embedProgress && (
        <div className="mb-3 p-2 border border-zinc-800 rounded-lg bg-zinc-950/40">
          <div className="text-[11px] text-zinc-400 flex items-center gap-2 mb-1">
            {embedProgress.status === "completed"
              ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
              : embedProgress.status === "failed"
              ? <AlertCircle className="w-3.5 h-3.5 text-red-400" />
              : <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />}
            <span>{embedProgress.message || "…"}</span>
            <span className="ml-auto font-mono">{Math.round(((embedProgress.progress ?? 0) as number) * 100)}%</span>
          </div>
          <div className="h-1 bg-zinc-800 rounded overflow-hidden">
            <div className="h-full bg-indigo-500 transition-all"
                 style={{ width: `${Math.round(((embedProgress.progress ?? 0) as number) * 100)}%` }} />
          </div>
          {embedProgress.error && <p className="text-[10px] text-red-300 mt-1">{embedProgress.error}</p>}
        </div>
      )}

      <div className="flex items-center gap-2 mb-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") search(); }}
          placeholder="e.g. sunset beach, kids birthday, my dog running, city night traffic…"
          className="flex-1 bg-zinc-800/50 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200"
        />
        <Button onClick={search} disabled={pending || !query.trim() || !canSearch} size="md">
          {pending
            ? <><Loader2 className="w-4 h-4 mr-1.5 animate-spin" /> Searching…</>
            : <><Search className="w-4 h-4 mr-1.5" /> Search</>}
        </Button>
      </div>

      {!canSearch && (
        <div className="text-[11px] text-zinc-500 text-center py-8 border border-dashed border-zinc-800 rounded-lg">
          The library has no content embeddings yet. Click <span className="text-zinc-200">Index content</span> above to encode every video.
          The CLIP model downloads once (~600 MB); subsequent runs are offline.
        </div>
      )}

      {hits && hits.length === 0 && canSearch && (
        <p className="text-xs text-zinc-500 text-center py-10">
          No videos matched. Try a broader description (drop specifics like names/dates — CLIP is best at general scenes).
        </p>
      )}

      {hits && hits.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
          {hits.map((h) => (
            <button
              key={h.video.id}
              onClick={() => setSelected(h.video)}
              className="text-left group bg-zinc-950/40 border border-zinc-800 rounded-lg overflow-hidden hover:border-indigo-500/40 transition-colors"
            >
              <div className="relative bg-black aspect-video">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={libraryThumbUrl(h.video.id)} alt="" loading="lazy" className="w-full h-full object-cover" />
                <div className="absolute top-1 right-1 text-[9px] font-mono bg-indigo-500/80 text-white rounded px-1">
                  {(h.score * 100).toFixed(0)}%
                </div>
                <div className="absolute bottom-1 right-1 text-[9px] font-mono bg-black/70 text-zinc-200 rounded px-1">
                  {fmtDuration(h.video.duration_sec)}
                </div>
              </div>
              <div className="p-2">
                <div className="text-[11px] text-zinc-200 truncate" title={h.video.abs_path}>{baseName(h.video.abs_path)}</div>
                {h.video.tags && h.video.tags.length > 0 && (
                  <div className="text-[10px] text-zinc-500 truncate" title={h.video.tags.map((t) => (t as { tag: string }).tag).join(", ")}>
                    {(h.video.tags as { tag: string }[]).slice(0, 3).map((t) => t.tag).join(" · ")}
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
      )}

      {selected && (
        <VideoDetailModal
          video={selected}
          onClose={() => setSelected(null)}
          onRemoveIndex={async () => { setSelected(null); }}
          onDeleteFile={async () => { setSelected(null); }}
        />
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// T2: Clusters view
// ---------------------------------------------------------------------------

function ClustersView({ onError }: { onError: (s: string) => void }) {
  const [clusters, setClusters] = useState<LibraryClusterSummary[] | null>(null);
  const [k, setK] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [selectedMembers, setSelectedMembers] = useState<LibraryVideo[] | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<LibraryVideo | null>(null);

  const buildClusters = useCallback(async () => {
    setPending(true);
    try {
      const res = await libraryBuildClusters(k ?? undefined);
      setClusters(res.clusters);
      setSelectedClusterId(null);
      setSelectedMembers(null);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Clustering failed");
    } finally { setPending(false); }
  }, [k, onError]);

  const openCluster = useCallback(async (cid: number) => {
    setSelectedClusterId(cid);
    setSelectedMembers(null);
    try {
      const res = await libraryClusterMembers(cid);
      setSelectedMembers(res.videos);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to load cluster");
    }
  }, [onError]);

  return (
    <Card className="min-h-[500px]">
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <CardTitle className="text-sm flex items-center gap-2">
          <Grid3x3 className="w-4 h-4 text-indigo-400" /> Content clusters
        </CardTitle>
        <div className="ml-auto flex items-center gap-2">
          <label className="text-[10px] uppercase tracking-wide text-zinc-500">k</label>
          <input
            type="number" min={1} max={64} placeholder="auto"
            value={k ?? ""}
            onChange={(e) => setK(e.target.value === "" ? null : Math.max(1, Math.min(64, parseInt(e.target.value, 10))))}
            className="w-16 bg-zinc-800/50 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200"
          />
          <Button size="sm" onClick={buildClusters} disabled={pending}>
            {pending
              ? <><Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> Clustering…</>
              : <><Sparkles className="w-3.5 h-3.5 mr-1.5" /> Compute clusters</>}
          </Button>
        </div>
      </div>

      {clusters === null && (
        <p className="text-xs text-zinc-500 text-center py-10">
          Click <span className="text-zinc-200">Compute clusters</span> to group videos by what they show. Requires content embeddings (Search tab → Index content).
        </p>
      )}

      {clusters && clusters.length === 0 && (
        <p className="text-xs text-zinc-500 text-center py-10">
          No embeddings yet. Index the library first (Search tab → Index content).
        </p>
      )}

      {clusters && clusters.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
          {clusters.map((c) => (
            <button
              key={c.id}
              onClick={() => openCluster(c.id)}
              className={`text-left bg-zinc-950/40 border rounded-lg overflow-hidden transition-colors ${
                selectedClusterId === c.id ? "border-indigo-500/50" : "border-zinc-800 hover:border-zinc-700"
              }`}
            >
              {c.preview && (
                <div className="relative bg-black aspect-video">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={libraryThumbUrl(c.preview.id)} alt="" loading="lazy" className="w-full h-full object-cover" />
                  <div className="absolute bottom-1 right-1 text-[9px] font-mono bg-black/70 text-zinc-200 rounded px-1">
                    {c.size} video{c.size === 1 ? "" : "s"}
                  </div>
                </div>
              )}
              <div className="p-2">
                <div className="text-xs text-zinc-100 truncate">{c.label}</div>
              </div>
            </button>
          ))}
        </div>
      )}

      {selectedClusterId !== null && (
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-2">
            Members of cluster {selectedClusterId}
          </div>
          {selectedMembers === null && <div className="text-xs text-zinc-500">Loading…</div>}
          {selectedMembers && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {selectedMembers.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setSelectedVideo(v)}
                  className="text-left bg-zinc-950/40 border border-zinc-800 rounded-lg overflow-hidden hover:border-indigo-500/40 transition-colors"
                >
                  <div className="relative bg-black aspect-video">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={libraryThumbUrl(v.id)} alt="" loading="lazy" className="w-full h-full object-cover" />
                  </div>
                  <div className="p-1.5">
                    <div className="text-[10px] text-zinc-200 truncate">{baseName(v.abs_path)}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {selectedVideo && (
        <VideoDetailModal
          video={selectedVideo}
          onClose={() => setSelectedVideo(null)}
          onRemoveIndex={async () => { setSelectedVideo(null); }}
          onDeleteFile={async () => { setSelectedVideo(null); }}
        />
      )}
    </Card>
  );
}
