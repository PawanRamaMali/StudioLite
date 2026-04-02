"use client";

import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Music, Mic, Headphones, Volume2, Waves } from "lucide-react";

const SFX_CATEGORIES = {
  "Nature": ["rain", "thunder", "wind", "ocean_waves"],
  "Impact": ["explosion", "glass_break"],
  "Ambient": ["fire_crackling", "bird_chirp", "clock_tick"],
  "Human": ["heartbeat", "footsteps", "typing"],
  "Motion": ["whoosh", "car_engine", "door_close"],
};

export default function AudioPanel() {
  const [mode, setMode] = useState<"sfx" | "isolate" | "extract" | "normalize">("sfx");
  const [selectedSfx, setSelectedSfx] = useState("rain");
  const [sfxDuration, setSfxDuration] = useState(2);

  const modes = [
    { id: "sfx", label: "Sound Effects", icon: Music },
    { id: "isolate", label: "Voice Isolation", icon: Headphones },
    { id: "extract", label: "Extract Audio", icon: Waves },
    { id: "normalize", label: "Normalize", icon: Volume2 },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Audio Studio</h1>
        <p className="text-zinc-400 mt-1">Generate sound effects, isolate vocals, and process audio</p>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {modes.map((m) => {
          const Icon = m.icon;
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id as typeof mode)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === m.id ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Icon className="w-4 h-4" /> {m.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          {mode === "sfx" && (
            <>
              <Card>
                <CardTitle className="text-sm mb-3">Sound Effects Library</CardTitle>
                {Object.entries(SFX_CATEGORIES).map(([cat, effects]) => (
                  <div key={cat} className="mb-3">
                    <p className="text-[10px] text-zinc-500 uppercase font-semibold mb-1">{cat}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {effects.map((sfx) => (
                        <button
                          key={sfx}
                          onClick={() => setSelectedSfx(sfx)}
                          className={`px-2.5 py-1 rounded-full text-xs transition-all ${
                            selectedSfx === sfx
                              ? "bg-indigo-600 text-white"
                              : "bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                          }`}
                        >
                          {sfx.replace(/_/g, " ")}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </Card>
              <Card>
                <label className="text-xs text-zinc-400">Duration: {sfxDuration}s</label>
                <input type="range" min={0.5} max={10} step={0.5} value={sfxDuration}
                  onChange={(e) => setSfxDuration(Number(e.target.value))}
                  className="w-full mt-1 accent-indigo-500" />
              </Card>
              <Button className="w-full">
                <Music className="w-4 h-4 mr-2" /> Generate SFX
              </Button>
            </>
          )}

          {mode === "isolate" && (
            <Card>
              <CardTitle className="text-sm mb-3">Voice Isolation</CardTitle>
              <p className="text-xs text-zinc-400 mb-3">Upload audio or video to separate vocals from background</p>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-zinc-500 transition-colors cursor-pointer">
                <Mic className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Upload audio/video file</p>
              </div>
              <Button className="w-full mt-3">
                <Headphones className="w-4 h-4 mr-2" /> Isolate Voice
              </Button>
            </Card>
          )}

          {mode === "extract" && (
            <Card>
              <CardTitle className="text-sm mb-3">Extract Audio</CardTitle>
              <p className="text-xs text-zinc-400 mb-3">Pull the audio track from any video file</p>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-zinc-500 transition-colors cursor-pointer">
                <Waves className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Upload video file</p>
              </div>
              <Button className="w-full mt-3">
                <Waves className="w-4 h-4 mr-2" /> Extract Audio
              </Button>
            </Card>
          )}

          {mode === "normalize" && (
            <Card>
              <CardTitle className="text-sm mb-3">Normalize Volume</CardTitle>
              <p className="text-xs text-zinc-400 mb-3">Adjust audio to consistent volume level</p>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-6 text-center hover:border-zinc-500 transition-colors cursor-pointer">
                <Volume2 className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">Upload audio file</p>
              </div>
              <label className="text-xs text-zinc-400 mt-3 block">Target dB: -3.0</label>
              <input type="range" min={-10} max={0} step={0.5} defaultValue={-3}
                className="w-full mt-1 accent-indigo-500" />
              <Button className="w-full mt-3">
                <Volume2 className="w-4 h-4 mr-2" /> Normalize
              </Button>
            </Card>
          )}
        </div>

        <div className="lg:col-span-2">
          <Card className="min-h-[300px] flex items-center justify-center">
            <div className="text-center">
              <Music className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-500 text-sm">Generated audio will appear here</p>
              <p className="text-zinc-600 text-xs mt-1">With waveform preview and download</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
