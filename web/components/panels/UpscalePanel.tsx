"use client";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { ArrowUpCircle, Upload, Zap } from "lucide-react";
import { useState } from "react";

const PRESETS = [
  { id: "fast_2x", name: "Fast 2x", desc: "Lanczos interpolation", badge: "Fast" },
  { id: "quality_2x", name: "Quality 2x", desc: "Real-ESRGAN", badge: "Quality" },
  { id: "ultra_4x", name: "Ultra 4x", desc: "Real-ESRGAN 4x", badge: "Ultra" },
  { id: "anime_4x", name: "Anime 4x", desc: "Anime-optimized", badge: "Anime" },
];

export default function UpscalePanel() {
  const [preset, setPreset] = useState("quality_2x");
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Upscale Video</h1>
        <p className="text-zinc-400 mt-1">Enhance video resolution up to 4K with AI super-resolution</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <div className="border-2 border-dashed border-zinc-700 rounded-lg p-12 text-center hover:border-indigo-500/50 transition-colors cursor-pointer">
              <Upload className="w-10 h-10 text-zinc-500 mx-auto mb-3" />
              <p className="text-sm text-zinc-400">Drop video here or click to upload</p>
            </div>
          </Card>
          <Card>
            <CardTitle className="text-sm mb-3">Upscale Preset</CardTitle>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button key={p.id} onClick={() => setPreset(p.id)}
                  className={`p-3 rounded-lg border text-left transition-all ${preset === p.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{p.name}</span>
                    <Badge variant={p.badge === "Ultra" ? "warning" : p.badge === "Quality" ? "info" : "default"}>{p.badge}</Badge>
                  </div>
                  <p className="text-[10px] text-zinc-500 mt-1">{p.desc}</p>
                </button>
              ))}
            </div>
          </Card>
          <Button size="lg" className="w-full">
            <Zap className="w-5 h-5 mr-2" /> Upscale Video
          </Button>
        </div>
        <Card className="min-h-[400px] flex items-center justify-center">
          <div className="text-center">
            <ArrowUpCircle className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
            <p className="text-zinc-500 text-sm">Upscaled video will appear here</p>
            <p className="text-zinc-600 text-xs mt-1">With before/after comparison</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
