"use client";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Briefcase, Clock, Download, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatDuration } from "@/lib/utils";

const MOCK_JOBS = [
  { id: "abc123", type: "text2video", status: "completed", progress: 1.0, created: Date.now() / 1000 - 3600, elapsed: 342, prompt: "A majestic eagle soaring..." },
  { id: "def456", type: "story", status: "running", progress: 0.45, created: Date.now() / 1000 - 120, elapsed: 120, prompt: "Mischievous husky story..." },
  { id: "ghi789", type: "upscale", status: "queued", progress: 0, created: Date.now() / 1000 - 30, elapsed: 0, prompt: "Upscale 480p to 4K" },
];

export default function JobsPanel() {
  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Jobs</h1>
          <p className="text-zinc-400 mt-1">Track generation and processing jobs</p>
        </div>
        <Button variant="secondary" size="sm">
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" /> Refresh
        </Button>
      </div>

      <div className="space-y-3">
        {MOCK_JOBS.map((job) => (
          <Card key={job.id} className="flex items-center gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <Badge variant={
                  job.status === "completed" ? "success" :
                  job.status === "running" ? "warning" :
                  job.status === "failed" ? "error" : "default"
                }>
                  {job.status}
                </Badge>
                <span className="text-xs text-zinc-500 font-mono">{job.id}</span>
                <Badge>{job.type}</Badge>
              </div>
              <p className="text-sm text-zinc-300 truncate">{job.prompt}</p>
              {job.status === "running" && (
                <div className="mt-2">
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${job.progress * 100}%` }} />
                  </div>
                  <p className="text-[10px] text-zinc-500 mt-1">{(job.progress * 100).toFixed(0)}%</p>
                </div>
              )}
            </div>
            <div className="text-right flex-shrink-0">
              <div className="flex items-center gap-1 text-xs text-zinc-500">
                <Clock className="w-3 h-3" /> {formatDuration(job.elapsed)}
              </div>
              {job.status === "completed" && (
                <Button size="sm" variant="ghost" className="mt-1">
                  <Download className="w-3 h-3 mr-1" /> Download
                </Button>
              )}
            </div>
          </Card>
        ))}

        {MOCK_JOBS.length === 0 && (
          <Card className="min-h-[200px] flex items-center justify-center">
            <div className="text-center">
              <Briefcase className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-500 text-sm">No jobs yet</p>
              <p className="text-zinc-600 text-xs mt-1">Generate a video to see jobs here</p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
