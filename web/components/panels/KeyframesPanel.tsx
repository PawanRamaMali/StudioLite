"use client";
import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { KeyRound, Upload, Play, ArrowRight } from "lucide-react";

const EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out", "bounce"];
const METHODS = [
  { id: "blend", name: "Alpha Blend", desc: "Fast, no AI required" },
  { id: "i2v", name: "AI Image-to-Video", desc: "Uses GPU, best quality" },
  { id: "interpolate", name: "Frame Interpolation", desc: "RIFE-based, smooth" },
];

export default function KeyframesPanel() {
  const [easing, setEasing] = useState("ease_in_out");
  const [method, setMethod] = useState("blend");
  const [frames, setFrames] = useState(30);
  const [fps, setFps] = useState(24);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Keyframe Animation</h1>
        <p className="text-zinc-400 mt-1">Create smooth transitions between keyframe images</p>
      </div>

      {/* Keyframe upload area */}
      <div className="grid grid-cols-5 gap-4 mb-6 items-center">
        <Card className="col-span-2 min-h-[200px] flex items-center justify-center">
          <div className="text-center cursor-pointer hover:opacity-80 transition-opacity">
            <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
            <p className="text-sm text-zinc-400 font-medium">Start Keyframe</p>
            <p className="text-xs text-zinc-600">Upload image</p>
          </div>
        </Card>
        <div className="flex justify-center">
          <div className="w-10 h-10 rounded-full bg-indigo-600/20 flex items-center justify-center">
            <ArrowRight className="w-5 h-5 text-indigo-400" />
          </div>
        </div>
        <Card className="col-span-2 min-h-[200px] flex items-center justify-center">
          <div className="text-center cursor-pointer hover:opacity-80 transition-opacity">
            <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
            <p className="text-sm text-zinc-400 font-medium">End Keyframe</p>
            <p className="text-xs text-zinc-600">Upload image</p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          {/* Method */}
          <Card>
            <CardTitle className="text-sm mb-3">Method</CardTitle>
            <div className="space-y-2">
              {METHODS.map((m) => (
                <button key={m.id} onClick={() => setMethod(m.id)}
                  className={`w-full p-2.5 rounded-lg border text-left transition-all ${method === m.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"}`}>
                  <div className="text-xs font-medium">{m.name}</div>
                  <div className="text-[10px] text-zinc-500">{m.desc}</div>
                </button>
              ))}
            </div>
          </Card>

          {/* Settings */}
          <Card>
            <CardTitle className="text-sm mb-3">Settings</CardTitle>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Easing</label>
                <select value={easing} onChange={(e) => setEasing(e.target.value)}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1.5 text-xs text-zinc-200 mt-1">
                  {EASINGS.map((e) => <option key={e} value={e}>{e.replace(/_/g, " ")}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Frames: {frames}</label>
                <input type="range" min={10} max={120} value={frames}
                  onChange={(e) => setFrames(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500" />
              </div>
              <div>
                <label className="text-xs text-zinc-400">FPS: {fps}</label>
                <input type="range" min={8} max={30} value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500" />
              </div>
              <p className="text-xs text-zinc-500">Duration: {(frames / fps).toFixed(1)}s</p>
            </div>
          </Card>

          <Button size="lg" className="w-full">
            <Play className="w-5 h-5 mr-2" /> Generate Animation
          </Button>
        </div>

        <div className="lg:col-span-2">
          <Card className="min-h-[300px] flex items-center justify-center">
            <div className="text-center">
              <KeyRound className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-500 text-sm">Generated animation will appear here</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
