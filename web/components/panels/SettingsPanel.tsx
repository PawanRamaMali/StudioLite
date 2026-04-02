"use client";
import { useState, useEffect } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Settings, Cpu, HardDrive, Wifi, Check, X, ExternalLink } from "lucide-react";
import { getSystemStatus, SystemStatus } from "@/lib/api";

const MODELS = [
  { name: "Wan 2.1 (1.3B)", vram: "8GB+", quality: "Good", speed: "Medium", installed: true },
  { name: "Wan 2.1 (14B)", vram: "16GB+", quality: "Excellent", speed: "Slow", installed: false },
  { name: "LTX-Video", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "CogVideoX (2B)", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "CogVideoX (5B)", vram: "16GB+", quality: "Very Good", speed: "Medium", installed: false },
  { name: "HunyuanVideo", vram: "24GB+", quality: "Excellent", speed: "Slow", installed: false },
  { name: "AnimateDiff", vram: "8GB+", quality: "Good", speed: "Fast", installed: false },
  { name: "Stable Video Diffusion", vram: "8GB+", quality: "Very Good", speed: "Medium", installed: false },
];

export default function SettingsPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [apiConnected, setApiConnected] = useState(false);

  useEffect(() => {
    getSystemStatus()
      .then((s) => { setStatus(s); setApiConnected(true); })
      .catch(() => setApiConnected(false));
  }, []);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Settings</h1>
        <p className="text-zinc-400 mt-1">System status, model hub, and API configuration</p>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card className="text-center">
          <Wifi className={`w-6 h-6 mx-auto mb-2 ${apiConnected ? "text-green-400" : "text-red-400"}`} />
          <p className="text-xs text-zinc-400">API Server</p>
          <Badge variant={apiConnected ? "success" : "error"} className="mt-1">
            {apiConnected ? "Connected" : "Offline"}
          </Badge>
        </Card>
        <Card className="text-center">
          <Cpu className="w-6 h-6 mx-auto mb-2 text-indigo-400" />
          <p className="text-xs text-zinc-400">GPU</p>
          <p className="text-sm font-medium mt-1">{status?.gpu_name?.slice(0, 20) || "Checking..."}</p>
        </Card>
        <Card className="text-center">
          <HardDrive className="w-6 h-6 mx-auto mb-2 text-purple-400" />
          <p className="text-xs text-zinc-400">VRAM Free</p>
          <p className="text-sm font-medium mt-1">{status ? `${status.vram_free_gb.toFixed(1)}GB` : "..."}</p>
        </Card>
        <Card className="text-center">
          <HardDrive className="w-6 h-6 mx-auto mb-2 text-cyan-400" />
          <p className="text-xs text-zinc-400">VRAM Total</p>
          <p className="text-sm font-medium mt-1">{status ? `${status.vram_total_gb.toFixed(1)}GB` : "..."}</p>
        </Card>
      </div>

      {/* Model Hub */}
      <Card className="mb-6">
        <CardTitle className="mb-4">Model Hub</CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left">
                <th className="pb-2 text-xs text-zinc-400 font-medium">Model</th>
                <th className="pb-2 text-xs text-zinc-400 font-medium">VRAM</th>
                <th className="pb-2 text-xs text-zinc-400 font-medium">Quality</th>
                <th className="pb-2 text-xs text-zinc-400 font-medium">Speed</th>
                <th className="pb-2 text-xs text-zinc-400 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.name} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                  <td className="py-2.5 font-medium text-zinc-200">{m.name}</td>
                  <td className="py-2.5 text-zinc-400">{m.vram}</td>
                  <td className="py-2.5"><Badge variant={m.quality === "Excellent" ? "info" : "default"}>{m.quality}</Badge></td>
                  <td className="py-2.5"><Badge variant={m.speed === "Fast" ? "success" : "default"}>{m.speed}</Badge></td>
                  <td className="py-2.5">
                    {m.installed ? (
                      <span className="flex items-center gap-1 text-green-400 text-xs"><Check className="w-3 h-3" /> Installed</span>
                    ) : (
                      <span className="flex items-center gap-1 text-zinc-500 text-xs"><X className="w-3 h-3" /> Not downloaded</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* API Info */}
      <Card>
        <CardTitle className="mb-3">REST API</CardTitle>
        <p className="text-xs text-zinc-400 mb-3">Launch the API server for programmatic access to all features</p>
        <code className="block bg-zinc-800 rounded-lg px-3 py-2 text-xs text-indigo-300 font-mono">
          uvicorn api_server:app --host 0.0.0.0 --port 8000
        </code>
        <p className="text-xs text-zinc-500 mt-2">
          Swagger docs at <span className="text-indigo-400">http://localhost:8000/docs</span>
        </p>
      </Card>
    </div>
  );
}
