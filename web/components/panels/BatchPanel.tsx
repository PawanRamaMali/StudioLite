"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Upload, Zap, Layers, X, CheckCircle2, AlertCircle, Loader2, Clock } from "lucide-react";
import {
  uploadEditVideo,
  thumbnailVideo,
  extractVideoAudio,
  removeVideoAudio,
  compressVideo,
  loopVideo,
  reverseVideo,
  getJob,
  jobDownloadUrl,
  type EditUploadResponse,
  type Job,
} from "@/lib/api";

type Operation =
  | "thumbnail"
  | "extract-audio"
  | "remove-audio"
  | "compress"
  | "loop"
  | "reverse";

type OperationMeta = {
  id: Operation;
  label: string;
  description: string;
};

const OPERATIONS: OperationMeta[] = [
  { id: "thumbnail",     label: "Thumbnail (grid)", description: "3x3 grid of frames from each video." },
  { id: "extract-audio", label: "Extract audio",    description: "Save each video's audio as .wav." },
  { id: "remove-audio",  label: "Remove audio",     description: "Produce a silent copy of each video." },
  { id: "compress",      label: "Compress",         description: "Re-encode each video with x264 at a preset bitrate." },
  { id: "loop",          label: "Loop",             description: "Repeat each video N times." },
  { id: "reverse",       label: "Reverse",          description: "Play each video backwards." },
];

type Row = {
  key: string;
  file: EditUploadResponse;
  jobId: string | null;
  status: Job["status"] | "pending" | "uploading";
  progress: number;
  message: string;
  downloadUrl: string | null;
  error: string | null;
};

const TERMINAL: Job["status"][] = ["completed", "failed", "cancelled"];

export default function BatchPanel() {
  const [op, setOp] = useState<Operation>("thumbnail");
  const [rows, setRows] = useState<Row[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  // Operation-specific params
  const [compressPreset, setCompressPreset] = useState<"high" | "medium" | "low">("medium");
  const [loopCount, setLoopCount] = useState(2);
  const [extractFormat, setExtractFormat] = useState<"wav" | "mp3">("wav");
  const [reverseKeepAudio, setReverseKeepAudio] = useState(true);
  const [thumbWidth, setThumbWidth] = useState(1280);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clearPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };
  useEffect(() => clearPoll, []);

  const onAddFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    const newRows: Row[] = [];
    try {
      for (const f of Array.from(files)) {
        try {
          const r = await uploadEditVideo(f);
          newRows.push({
            key: `${r.video_path}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            file: r,
            jobId: null,
            status: "pending",
            progress: 0,
            message: "",
            downloadUrl: null,
            error: null,
          });
        } catch (e) {
          setError((e as Error).message);
          break;
        }
      }
      setRows((prev) => [...prev, ...newRows]);
    } finally {
      setUploading(false);
    }
  }, []);

  const removeRow = (key: string) => {
    setRows((prev) => prev.filter((r) => r.key !== key));
  };

  const clearRows = () => {
    setRows([]);
    setError(null);
    clearPoll();
    setRunning(false);
  };

  const startJobForRow = async (row: Row): Promise<Job> => {
    switch (op) {
      case "thumbnail":
        return thumbnailVideo({
          video_path: row.file.video_path,
          mode: "grid",
          cols: 3, rows: 3,
          width: thumbWidth,
        });
      case "extract-audio":
        return extractVideoAudio({ video_path: row.file.video_path, format: extractFormat });
      case "remove-audio":
        return removeVideoAudio({ video_path: row.file.video_path });
      case "compress":
        return compressVideo({ video_path: row.file.video_path, preset: compressPreset });
      case "loop":
        return loopVideo({ video_path: row.file.video_path, count: loopCount });
      case "reverse":
        return reverseVideo({ video_path: row.file.video_path, include_audio: reverseKeepAudio });
    }
  };

  const runBatch = async () => {
    if (rows.length === 0 || running) return;
    setError(null);
    setRunning(true);
    // Reset existing rows
    setRows((prev) => prev.map((r) => ({
      ...r,
      jobId: null, status: "pending", progress: 0, message: "", downloadUrl: null, error: null,
    })));

    // Dispatch all jobs - they run server-side in parallel
    const started: Record<string, string> = {}; // rowKey -> jobId
    for (const row of rows) {
      try {
        setRows((prev) => prev.map((r) => r.key === row.key ? { ...r, status: "queued" } : r));
        const j = await startJobForRow(row);
        started[row.key] = j.job_id;
        setRows((prev) => prev.map((r) => r.key === row.key
          ? { ...r, jobId: j.job_id, status: j.status, progress: j.progress, message: j.message }
          : r));
      } catch (e) {
        setRows((prev) => prev.map((r) => r.key === row.key
          ? { ...r, status: "failed", error: (e as Error).message }
          : r));
      }
    }

    // Poll until all terminal
    clearPoll();
    pollRef.current = setInterval(async () => {
      const current = Object.entries(started);
      if (current.length === 0) { clearPoll(); setRunning(false); return; }
      try {
        const updates = await Promise.all(current.map(async ([key, jid]) => {
          try {
            const latest = await getJob(jid);
            return { key, jid, latest };
          } catch (e) {
            return { key, jid, error: (e as Error).message };
          }
        }));
        let activeCount = 0;
        setRows((prev) => prev.map((r) => {
          const u = updates.find((x) => x.key === r.key);
          if (!u) return r;
          if (u.error) {
            return { ...r, status: "failed", error: u.error };
          }
          const latest = u.latest!;
          if (!TERMINAL.includes(latest.status)) activeCount++;
          const dl = latest.status === "completed" ? jobDownloadUrl(latest.job_id) : r.downloadUrl;
          return {
            ...r,
            jobId: latest.job_id,
            status: latest.status,
            progress: latest.progress,
            message: latest.message,
            downloadUrl: dl,
            error: latest.error || null,
          };
        }));
        if (activeCount === 0) { clearPoll(); setRunning(false); }
      } catch {
        // Keep polling; a transient read-failure shouldn't kill the whole watch.
      }
    }, 1500);
  };

  const statusCount = (s: Row["status"]) => rows.filter((r) => r.status === s).length;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Batch Processing</h1>
        <p className="text-zinc-400 mt-1">
          Apply the same operation to many files at once. Each file becomes
          its own job, dispatched together and tracked live.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <CardTitle className="text-sm mb-3">Operation</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {OPERATIONS.map((o) => (
                <button
                  key={o.id}
                  onClick={() => setOp(o.id)}
                  disabled={running}
                  className={`px-3 py-2 rounded-md text-xs border text-left transition-colors ${
                    op === o.id
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                      : "bg-zinc-900/50 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <div className="font-medium">{o.label}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">{o.description}</div>
                </button>
              ))}
            </div>
          </Card>

          {op === "thumbnail" && (
            <Card>
              <CardTitle className="text-sm mb-3">Options</CardTitle>
              <label className="text-xs text-zinc-400">Grid width: {thumbWidth}px</label>
              <input
                type="range" min={640} max={3840} step={20}
                value={thumbWidth} disabled={running}
                onChange={(e) => setThumbWidth(Number(e.target.value))}
                className="w-full mt-1 accent-indigo-500"
              />
            </Card>
          )}

          {op === "extract-audio" && (
            <Card>
              <CardTitle className="text-sm mb-3">Options</CardTitle>
              <div className="grid grid-cols-2 gap-2">
                {(["wav", "mp3"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setExtractFormat(f)}
                    disabled={running}
                    className={`px-3 py-2 rounded-md text-xs uppercase border ${
                      extractFormat === f
                        ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                        : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                    }`}
                  >{f}</button>
                ))}
              </div>
            </Card>
          )}

          {op === "compress" && (
            <Card>
              <CardTitle className="text-sm mb-3">Quality</CardTitle>
              <div className="grid grid-cols-3 gap-2">
                {(["high", "medium", "low"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setCompressPreset(p)}
                    disabled={running}
                    className={`px-3 py-2 rounded-md text-xs capitalize border ${
                      compressPreset === p
                        ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                        : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
                    }`}
                  >{p}</button>
                ))}
              </div>
            </Card>
          )}

          {op === "loop" && (
            <Card>
              <CardTitle className="text-sm mb-3">Options</CardTitle>
              <label className="text-xs text-zinc-400">Repeat count: {loopCount}x</label>
              <input
                type="range" min={2} max={10} step={1}
                value={loopCount} disabled={running}
                onChange={(e) => setLoopCount(Number(e.target.value))}
                className="w-full mt-1 accent-indigo-500"
              />
            </Card>
          )}

          {op === "reverse" && (
            <Card>
              <CardTitle className="text-sm mb-3">Options</CardTitle>
              <label className="flex items-center gap-2 text-sm text-zinc-300">
                <input
                  type="checkbox" checked={reverseKeepAudio}
                  onChange={(e) => setReverseKeepAudio(e.target.checked)}
                  disabled={running}
                />
                Reverse audio too
              </label>
            </Card>
          )}

          <Card>
            <CardTitle className="text-sm mb-3">Files ({rows.length})</CardTitle>
            <label className="block cursor-pointer">
              <input
                type="file" accept="video/*" multiple
                className="hidden"
                disabled={uploading || running}
                onChange={(e) => onAddFiles(e.target.files)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                <p className="text-sm text-zinc-400">
                  {uploading ? "Uploading..." : "Add videos"}
                </p>
              </div>
            </label>
            {rows.length > 0 && !running && (
              <Button size="sm" variant="ghost" className="mt-3" onClick={clearRows}>
                Clear all
              </Button>
            )}
          </Card>

          <Button
            size="lg" className="w-full"
            disabled={rows.length === 0 || running || uploading}
            onClick={runBatch}
          >
            <Zap className="w-5 h-5 mr-2" />
            {running
              ? `Running - ${statusCount("completed")}/${rows.length} done`
              : `Run on ${rows.length || "..."} ${rows.length === 1 ? "file" : "files"}`}
          </Button>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <Card className="min-h-[400px]">
          <CardTitle className="text-sm mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4" /> Queue
            <span className="text-xs text-zinc-500 font-normal">
              {statusCount("completed")} done &middot; {statusCount("failed")} failed
            </span>
          </CardTitle>
          {rows.length === 0 ? (
            <p className="text-sm text-zinc-500 text-center py-8">
              Add videos to see them queued here.
            </p>
          ) : (
            <div className="space-y-2 max-h-[560px] overflow-y-auto">
              {rows.map((r) => {
                const isActive = r.status === "running" || r.status === "queued";
                const pct = Math.round((r.progress || 0) * 100);
                return (
                  <div
                    key={r.key}
                    className="flex items-center gap-3 bg-zinc-900/60 border border-zinc-800 rounded-md px-3 py-2"
                  >
                    <div className="flex-shrink-0">
                      {r.status === "completed" ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      ) : r.status === "failed" ? (
                        <AlertCircle className="w-4 h-4 text-red-500" />
                      ) : isActive ? (
                        <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                      ) : (
                        <Clock className="w-4 h-4 text-zinc-600" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-zinc-200 truncate">{r.file.filename}</p>
                      <p className="text-[11px] text-zinc-500 truncate">
                        {r.error ? r.error : (r.message || r.status)}
                        {isActive && ` - ${pct}%`}
                      </p>
                      {isActive && (
                        <div className="w-full bg-zinc-800 rounded-full h-1 mt-1">
                          <div
                            className="bg-indigo-500 h-1 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      )}
                    </div>
                    {r.downloadUrl && (
                      <Button
                        size="sm" variant="ghost"
                        onClick={() => window.open(r.downloadUrl!, "_blank", "noopener,noreferrer")}
                      >
                        Download
                      </Button>
                    )}
                    {!running && (
                      <button
                        onClick={() => removeRow(r.key)}
                        className="text-zinc-600 hover:text-red-400"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
