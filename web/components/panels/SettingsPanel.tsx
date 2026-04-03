"use client";
import { useState, useEffect } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Settings, Cpu, HardDrive, Wifi, Check, X, Activity } from "lucide-react";
import { getSystemStatus, SystemStatus } from "@/lib/api";

const MODELS = [
  { name: "Wan 2.1 (1.3B)", vram: "8GB+", quality: "Good", speed: "Medium", installed: true },
  { name: "Wan 2.1 (14B)", vram: "16GB+", quality: "Excellent", speed: "Slow", installed: false },
  { name: "Wan 2.2 (14B)", vram: "16GB+", quality: "Excellent", speed: "Slow", installed: false },
  { name: "LTX-Video", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "CogVideoX (2B)", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "CogVideoX (5B)", vram: "16GB+", quality: "Very Good", speed: "Medium", installed: false },
  { name: "HunyuanVideo", vram: "24GB+", quality: "Excellent", speed: "Slow", installed: false },
  { name: "AnimateDiff", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "Stable Video Diffusion", vram: "8GB+", quality: "Very Good", speed: "Medium", installed: false },
  { name: "Mochi 1", vram: "12GB+", quality: "Very Good", speed: "Medium", installed: false },
];

export default function SettingsPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getSystemStatus()
      .then((s) => { setStatus(s); setApiConnected(true); })
      .catch(() => setApiConnected(false))
      .finally(() => setLoading(false));
  }, []);

  const gpu = status?.gpu;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Settings</h1>
        <p className="text-zinc-400 mt-1">System status, model hub, and API configuration</p>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <Card className="text-center py-4">
          <Wifi className={`w-6 h-6 mx-auto mb-2 ${apiConnected ? "text-green-400" : "text-red-400"}`} />
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider">API Server</p>
          <Badge variant={apiConnected ? "success" : "error"} className="mt-1.5">
            {loading ? "Checking..." : apiConnected ? "Connected" : "Offline"}
          </Badge>
          {apiConnected && <p className="text-[10px] text-zinc-600 mt-1">Port 8000</p>}
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
            {gpu ? `${gpu.free_vram_gb.toFixed(1)}` : "..."}
            <span className="text-xs font-normal text-zinc-500"> GB</span>
          </p>
        </Card>
        <Card className="text-center py-4">
          <Activity className="w-6 h-6 mx-auto mb-2 text-amber-400" />
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Active Jobs</p>
          <p className="text-lg font-bold text-amber-400 mt-1">{status?.active_jobs ?? "..."}</p>
        </Card>
      </div>

      {!apiConnected && !loading && (
        <Card className="border-amber-500/30 bg-amber-500/5 mb-6">
          <p className="text-amber-400 text-sm font-medium">API Server Not Running</p>
          <p className="text-amber-400/70 text-xs mt-1">Start the backend with:</p>
          <code className="block bg-zinc-800 rounded-lg px-3 py-2 text-xs text-indigo-300 font-mono mt-2">
            cd C:\Github\StudioLite && uvicorn api_server:app --port 8000
          </code>
        </Card>
      )}

      {/* Model Hub */}
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
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.name} className="border-b border-zinc-800/30 hover:bg-zinc-800/20 transition-colors">
                  <td className="py-2.5 font-medium text-zinc-200 text-xs">{m.name}</td>
                  <td className="py-2.5 text-zinc-400 text-xs">{m.vram}</td>
                  <td className="py-2.5"><Badge variant={m.quality === "Excellent" ? "info" : "default"} className="text-[10px]">{m.quality}</Badge></td>
                  <td className="py-2.5"><Badge variant={m.speed === "Fast" ? "success" : "default"} className="text-[10px]">{m.speed}</Badge></td>
                  <td className="py-2.5">
                    {m.installed ? (
                      <span className="flex items-center gap-1 text-green-400 text-xs"><Check className="w-3 h-3" /> Installed</span>
                    ) : (
                      <span className="flex items-center gap-1 text-zinc-600 text-xs"><X className="w-3 h-3" /> Not downloaded</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* API Docs */}
      <Card>
        <CardTitle className="mb-3">REST API Endpoints</CardTitle>
        <div className="space-y-1.5 font-mono text-xs">
          {[
            ["POST", "/api/v1/generate/text2video", "Text to Video"],
            ["POST", "/api/v1/generate/image2video", "Image to Video"],
            ["POST", "/api/v1/generate/story", "Story Mode"],
            ["POST", "/api/v1/audio/tts", "Text to Speech"],
            ["POST", "/api/v1/edit/trim", "Trim Video"],
            ["POST", "/api/v1/edit/merge", "Merge Videos"],
            ["POST", "/api/v1/edit/upscale", "Upscale Video"],
            ["GET", "/api/v1/jobs/{id}", "Job Status"],
            ["GET", "/api/v1/system/status", "System Status"],
          ].map(([method, path, desc]) => (
            <div key={path} className="flex items-center gap-3 py-1 px-2 rounded hover:bg-zinc-800/50">
              <Badge variant={method === "POST" ? "info" : "success"} className="text-[10px] w-12 justify-center">{method}</Badge>
              <span className="text-zinc-300 flex-1">{path}</span>
              <span className="text-zinc-600">{desc}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-zinc-500 mt-3">
          Swagger UI: <a href="http://localhost:8000/docs" target="_blank" rel="noopener" className="text-indigo-400 hover:text-indigo-300">http://localhost:8000/docs</a>
        </p>
      </Card>
    </div>
  );
}
