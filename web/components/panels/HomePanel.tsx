"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import {
  Cpu, HardDrive, CircleCheck, CircleAlert, CircleMinus, Download,
  Video, BookOpen, Image as ImageIcon, Users, Music, Scissors,
  Radio, FileVideo, Sparkles, ExternalLink, Key,
} from "lucide-react";
import {
  getSystemStatus, type SystemStatus,
  getModelInventory, type ModelInventoryItem,
} from "@/lib/api";

interface HomePanelProps {
  onNavigate?: (tab: string) => void;
}

type Ready = "ready" | "slow" | "needs-download" | "needs-gpu" | "needs-config";

interface FeatureRow {
  id: string;
  name: string;
  icon: typeof Video;
  tab: string;
  status: Ready;
  detail: string;
  action?: { label: string; goto: string };
}

function statusBadge(status: Ready) {
  const map: Record<Ready, { label: string; className: string; Icon: typeof CircleCheck }> = {
    "ready":          { label: "Ready",            className: "text-green-400 bg-green-500/10 border-green-500/30",    Icon: CircleCheck },
    "slow":           { label: "Slow on CPU",      className: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30", Icon: CircleAlert },
    "needs-download": { label: "Needs model",      className: "text-indigo-400 bg-indigo-500/10 border-indigo-500/30", Icon: Download },
    "needs-gpu":      { label: "Requires GPU",     className: "text-zinc-500 bg-zinc-800/40 border-zinc-700",          Icon: CircleMinus },
    "needs-config":   { label: "Needs setup",      className: "text-orange-400 bg-orange-500/10 border-orange-500/30", Icon: Key },
  };
  const { label, className, Icon } = map[status];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${className}`}>
      <Icon className="w-3 h-3" /> {label}
    </span>
  );
}

export default function HomePanel({ onNavigate }: HomePanelProps) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelInventoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getSystemStatus().catch(() => null),
      getModelInventory().then(d => d.models).catch(() => []),
    ]).then(([s, m]) => {
      setStatus(s);
      setModels(m);
      setLoading(false);
    });
  }, []);

  const hasGpu = status?.capabilities?.cuda ?? false;
  const gpuName = status?.gpu?.gpu_name;
  const vram = status?.gpu?.total_vram_gb;
  const freeVram = status?.gpu?.free_vram_gb;
  const platform = status?.capabilities?.platform ?? "unknown";
  const features = status?.capabilities?.features;

  const anyVideoModel = models.some(m => ["wan", "hunyuan", "ltx", "cogvideox"].includes(m.engine) && m.installed);
  const anySdxlModel = models.some(m => m.engine === "sdxl" && m.installed);

  const rows: FeatureRow[] = useMemo(() => {
    if (!features) return [];
    return [
      {
        id: "video",
        name: "Video Generator",
        icon: Video,
        tab: "generate",
        status: !hasGpu ? "needs-gpu" : features.video_generation ? "ready" : "needs-download",
        detail: !hasGpu
          ? "LTX / HunyuanVideo / Wan need a CUDA GPU."
          : features.video_generation
            ? "At least one engine is downloaded and ready."
            : "GPU detected, but no video engine weights are on disk yet.",
        action: hasGpu && !features.video_generation
          ? { label: "Download models", goto: "settings" }
          : undefined,
      },
      {
        id: "story",
        name: "Story Mode",
        icon: BookOpen,
        tab: "story",
        status: !hasGpu ? "needs-gpu" : features.video_generation ? "ready" : "needs-download",
        detail: !hasGpu
          ? "Multi-scene rendering requires a CUDA GPU."
          : features.video_generation
            ? "Ready — script writer, storyboard, and per-scene rendering all work."
            : "Downloads a video engine first (see Model Hub in Settings).",
      },
      {
        id: "images",
        name: "Images Studio",
        icon: ImageIcon,
        tab: "images",
        status: hasGpu ? "ready" : "slow",
        detail: hasGpu
          ? "SDXL runs on the GPU — a few seconds per image."
          : `SDXL runs on the CPU here — expect 5-15 min per image. First-run downloads happen automatically.${anySdxlModel ? "" : " No SDXL models cached yet."}`,
      },
      {
        id: "characters",
        name: "Character Portraits",
        icon: Users,
        tab: "characters",
        status: hasGpu ? "ready" : "slow",
        detail: hasGpu
          ? "Portraits render in ~30s per view via SDXL."
          : "SDXL on CPU means 5-15 min per view. Two views ≈ 15-30 min per character.",
      },
      {
        id: "audio",
        name: "Audio Studio",
        icon: Music,
        tab: "audio",
        status: "ready",
        detail: "TTS, SFX, voice isolation, normalization — all CPU-friendly.",
      },
      {
        id: "transcribe",
        name: "Transcription",
        icon: Radio,
        tab: "video-transcribe",
        status: "ready",
        detail: "faster-whisper picks the best compute type for your hardware (int8 on CPU, float16 on GPU).",
      },
      {
        id: "editor",
        name: "Video Editor",
        icon: Scissors,
        tab: "edit",
        status: "ready",
        detail: "FFmpeg-based — trim, cut, merge, overlay, speed, filter. No GPU required.",
      },
    ];
  }, [features, hasGpu, anySdxlModel]);

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-3xl font-bold gradient-text flex items-center gap-2">
          <Sparkles className="w-7 h-7 text-indigo-400" /> Welcome to StudioLite
        </h1>
        <p className="text-zinc-400 mt-1">
          A local-first AI media studio. Here&apos;s what your machine can do right now.
        </p>
      </div>

      {/* System card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Cpu className={`w-8 h-8 ${hasGpu ? "text-indigo-400" : "text-zinc-500"}`} />
            <div className="min-w-0">
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Compute</p>
              <p className="text-sm font-semibold text-zinc-200 truncate">
                {loading ? "Detecting…" : hasGpu ? (gpuName || "NVIDIA GPU") : "CPU only"}
              </p>
              {hasGpu && vram && <p className="text-[10px] text-zinc-500">{vram.toFixed(1)} GB VRAM</p>}
              {!hasGpu && !loading && <p className="text-[10px] text-zinc-500">No CUDA device detected</p>}
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <HardDrive className="w-8 h-8 text-emerald-400" />
            <div>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Free VRAM</p>
              <p className="text-sm font-semibold text-zinc-200">
                {hasGpu && freeVram !== undefined ? `${freeVram.toFixed(1)} GB` : "—"}
              </p>
              <p className="text-[10px] text-zinc-500">Platform: {platform}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Download className="w-8 h-8 text-purple-400" />
            <div>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Models installed</p>
              <p className="text-sm font-semibold text-zinc-200">
                {loading ? "…" : `${models.filter(m => m.installed).length} / ${models.length}`}
              </p>
              <p className="text-[10px] text-zinc-500">
                {anyVideoModel ? "Video engine ready" : "No video engine downloaded"}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Feature grid */}
      <Card className="mb-6">
        <CardTitle className="mb-4 flex items-center justify-between">
          <span>What you can do today</span>
          <Badge className="text-[10px]">{rows.filter(r => r.status === "ready").length} ready</Badge>
        </CardTitle>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {rows.map((r) => {
            const Icon = r.icon;
            const disabled = r.status === "needs-gpu";
            return (
              <div
                key={r.id}
                className={`flex items-start gap-3 rounded-lg border p-3 transition-colors ${
                  disabled
                    ? "border-zinc-800 bg-zinc-900/40"
                    : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                }`}
              >
                <div className={`p-2 rounded-lg ${disabled ? "bg-zinc-800/50" : "bg-indigo-500/10"}`}>
                  <Icon className={`w-5 h-5 ${disabled ? "text-zinc-600" : "text-indigo-400"}`} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <p className="text-sm font-semibold text-zinc-200">{r.name}</p>
                    {statusBadge(r.status)}
                  </div>
                  <p className="text-xs text-zinc-500">{r.detail}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <Button
                      size="sm"
                      variant={disabled ? "ghost" : "secondary"}
                      onClick={() => onNavigate?.(r.tab)}
                      disabled={disabled}
                    >
                      Open
                    </Button>
                    {r.action && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => onNavigate?.(r.action!.goto)}
                      >
                        {r.action.label} →
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Quick links */}
      <Card>
        <CardTitle className="mb-3">Quick links</CardTitle>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="secondary" onClick={() => onNavigate?.("settings")}>
            <Download className="w-3.5 h-3.5 mr-1.5" /> Manage models
          </Button>
          <Button size="sm" variant="secondary" onClick={() => onNavigate?.("settings")}>
            <Key className="w-3.5 h-3.5 mr-1.5" /> API keys &amp; env
          </Button>
          <Button size="sm" variant="ghost" onClick={() => onNavigate?.("jobs")}>
            <FileVideo className="w-3.5 h-3.5 mr-1.5" /> Job history
          </Button>
          <a
            href="https://github.com/PawanRamaMali/StudioLite"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-600 transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" /> Docs on GitHub
          </a>
          <a
            href="https://ollama.com/download"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-600 transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" /> Install Ollama (for LLM)
          </a>
        </div>
      </Card>

      {!loading && !hasGpu && (
        <p className="text-xs text-zinc-600 mt-6 text-center">
          Some tiles are dimmed because they need an NVIDIA GPU. Everything else works today.
        </p>
      )}
      {!loading && hasGpu && !anyVideoModel && (
        <p className="text-xs text-zinc-600 mt-6 text-center">
          Head to <button onClick={() => onNavigate?.("settings")} className="underline hover:text-zinc-300">Settings → Model Hub</button> to download a video engine when you&apos;re ready.
        </p>
      )}
    </div>
  );
}
