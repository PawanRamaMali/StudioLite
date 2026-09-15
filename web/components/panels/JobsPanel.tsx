"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Briefcase, Clock, Download, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatDuration } from "@/lib/utils";
import { listJobs, cancelJob, jobDownloadUrl, type JobListEntry } from "@/lib/api";

type StatusFilter = "all" | "active" | "completed" | "failed";

export default function JobsPanel() {
  const [jobs, setJobs] = useState<JobListEntry[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const data = await listJobs(100);
      setJobs(data.jobs || []);
      setTotal(data.total || 0);
    } catch (e) {
      setError((e as Error).message || "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll every 2 s while any job is running or queued so the progress
  // bar stays live without user interaction. Slows to every 10 s when
  // nothing is in flight so we don't churn if the UI is left open.
  useEffect(() => {
    const anyActive = jobs.some(
      (j) => j.status === "running" || j.status === "queued"
    );
    const interval = setInterval(refresh, anyActive ? 2000 : 10000);
    return () => clearInterval(interval);
  }, [jobs, refresh]);

  const filtered = useMemo(() => {
    if (filter === "all") return jobs;
    if (filter === "active")
      return jobs.filter(
        (j) => j.status === "running" || j.status === "queued"
      );
    if (filter === "completed") return jobs.filter((j) => j.status === "completed");
    if (filter === "failed")
      return jobs.filter(
        (j) => j.status === "failed" || j.status === "cancelled"
      );
    return jobs;
  }, [jobs, filter]);

  const onCancel = async (jobId: string) => {
    setCancelling((s) => new Set(s).add(jobId));
    try {
      await cancelJob(jobId);
      refresh();
    } catch (e) {
      setError((e as Error).message || "Cancel failed");
    } finally {
      setCancelling((s) => {
        const n = new Set(s);
        n.delete(jobId);
        return n;
      });
    }
  };

  const counts = useMemo(() => {
    const active = jobs.filter(
      (j) => j.status === "running" || j.status === "queued"
    ).length;
    const done = jobs.filter((j) => j.status === "completed").length;
    const bad = jobs.filter(
      (j) => j.status === "failed" || j.status === "cancelled"
    ).length;
    return { all: jobs.length, active, completed: done, failed: bad };
  }, [jobs]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Jobs</h1>
          <p className="text-zinc-400 mt-1">
            {total > 0
              ? `${total} jobs tracked · ${counts.active} active`
              : "Track generation and processing jobs"}
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={refresh}>
          <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} /> Refresh
        </Button>
      </div>

      <div className="flex gap-2 mb-4 text-xs">
        {(["all", "active", "completed", "failed"] as StatusFilter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md border ${
              filter === f
                ? "bg-indigo-500/20 border-indigo-500/50 text-indigo-200"
                : "border-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {f} <span className="opacity-60">({counts[f]})</span>
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-md border border-red-800/50 bg-red-950/40 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {filtered.map((job) => {
          const showProgress = job.status === "running" || job.status === "queued";
          const canCancel = showProgress;
          const canDownload = job.status === "completed";
          const isCancelling = cancelling.has(job.job_id);
          return (
            <Card key={job.job_id} className="flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <Badge
                    variant={
                      job.status === "completed"
                        ? "success"
                        : job.status === "running"
                        ? "warning"
                        : job.status === "failed"
                        ? "error"
                        : "default"
                    }
                  >
                    {job.status}
                  </Badge>
                  <span className="text-xs text-zinc-500 font-mono">
                    {job.job_id.slice(0, 8)}
                  </span>
                  <Badge>{job.kind}</Badge>
                </div>
                <p className="text-sm text-zinc-300 truncate" title={job.message}>
                  {job.message || " "}
                </p>
                {showProgress && (
                  <div className="mt-2">
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all"
                        style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-zinc-500 mt-1">
                      {Math.round((job.progress || 0) * 100)}%
                    </p>
                  </div>
                )}
              </div>
              <div className="text-right flex-shrink-0 space-y-1">
                <div className="flex items-center gap-1 text-xs text-zinc-500 justify-end">
                  <Clock className="w-3 h-3" /> {formatDuration(job.elapsed)}
                </div>
                {canDownload && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      const url = jobDownloadUrl(job.job_id);
                      window.open(url, "_blank", "noopener,noreferrer");
                    }}
                  >
                    <Download className="w-3 h-3 mr-1" /> Download
                  </Button>
                )}
                {canCancel && (
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={isCancelling}
                    onClick={() => onCancel(job.job_id)}
                  >
                    <X className="w-3 h-3 mr-1" />
                    {isCancelling ? "Cancelling…" : "Cancel"}
                  </Button>
                )}
              </div>
            </Card>
          );
        })}

        {!loading && filtered.length === 0 && (
          <Card className="min-h-[200px] flex items-center justify-center">
            <div className="text-center">
              <Briefcase className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-500 text-sm">
                {filter === "all" ? "No jobs yet" : `No ${filter} jobs`}
              </p>
              <p className="text-zinc-600 text-xs mt-1">
                {filter === "all"
                  ? "Generate a video to see jobs here"
                  : "Try a different filter"}
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
