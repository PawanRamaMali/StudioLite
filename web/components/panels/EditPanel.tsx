"use client";

import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Scissors, Upload, Wand2, Loader2, Download, AlertCircle } from "lucide-react";

const STYLE_FILTERS = [
  { id: "grayscale", name: "Grayscale", desc: "Black & white", preview: "from-gray-400 to-gray-600" },
  { id: "sepia", name: "Sepia", desc: "Warm vintage", preview: "from-amber-700 to-yellow-600" },
  { id: "invert", name: "Invert", desc: "Negative colors", preview: "from-white to-gray-800" },
  { id: "warm", name: "Warm", desc: "Orange tones", preview: "from-orange-400 to-red-400" },
  { id: "cool", name: "Cool", desc: "Blue tones", preview: "from-blue-400 to-cyan-400" },
  { id: "vintage", name: "Vintage", desc: "Retro faded", preview: "from-amber-600 to-orange-800" },
  { id: "high_contrast", name: "High Contrast", desc: "Bold contrast", preview: "from-black to-white" },
];

const EDIT_OPS = [
  { id: "remove", name: "Remove Object", desc: "Inpaint and remove using AI", icon: "x" },
  { id: "blur", name: "Blur Region", desc: "Censor or soften an area", icon: "blur" },
  { id: "brighten", name: "Brighten", desc: "Increase brightness of area", icon: "sun" },
  { id: "darken", name: "Darken", desc: "Decrease brightness of area", icon: "moon" },
  { id: "color_shift", name: "Color Shift", desc: "Change color tint", icon: "palette" },
  { id: "grayscale", name: "Grayscale Region", desc: "Desaturate specific area", icon: "circle" },
];

export default function EditPanel() {
  const [mode, setMode] = useState<"filters" | "region">("filters");
  const [selectedFilter, setSelectedFilter] = useState("grayscale");
  const [selectedOp, setSelectedOp] = useState("blur");
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");

  const handleApply = () => {
    setLoading(true);
    setLoadingMsg(mode === "filters" ? `Applying ${selectedFilter} filter...` : `Applying ${selectedOp} to region...`);
    // Show feedback then inform about Streamlit requirement
    setTimeout(() => {
      setLoadingMsg("Video file operations require the Streamlit UI at http://localhost:8501 > Video Editor");
      setTimeout(() => { setLoading(false); setLoadingMsg(""); }, 3000);
    }, 1000);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Editor</h1>
        <p className="text-zinc-400 mt-1">Apply style filters, remove objects, and edit video regions</p>
      </div>

      <div className="flex gap-2 mb-6">
        {[{ id: "filters", label: "Style Filters", count: STYLE_FILTERS.length },
          { id: "region", label: "Region Edit", count: EDIT_OPS.length }].map((tab) => (
          <button key={tab.id} onClick={() => setMode(tab.id as "filters" | "region")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === tab.id ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/20" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}>
            {tab.label} <Badge className="ml-1.5">{tab.count}</Badge>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <Card>
            <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-indigo-500/30 transition-colors cursor-pointer">
              <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
              <p className="text-sm text-zinc-400">Drop video here or click to upload</p>
              <p className="text-xs text-zinc-600 mt-1">MP4, MOV, AVI, MKV</p>
            </div>
          </Card>

          {mode === "filters" ? (
            <Card>
              <CardTitle className="text-sm mb-3">Select Filter</CardTitle>
              <div className="space-y-2">
                {STYLE_FILTERS.map((f) => (
                  <button key={f.id} onClick={() => setSelectedFilter(f.id)}
                    className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-left transition-all ${
                      selectedFilter === f.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                    }`}>
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${f.preview} flex-shrink-0`} />
                    <div>
                      <div className="text-xs font-medium">{f.name}</div>
                      <div className="text-[10px] text-zinc-500">{f.desc}</div>
                    </div>
                  </button>
                ))}
              </div>
            </Card>
          ) : (
            <>
              <Card>
                <CardTitle className="text-sm mb-3">Edit Operation</CardTitle>
                <div className="space-y-2">
                  {EDIT_OPS.map((op) => (
                    <button key={op.id} onClick={() => setSelectedOp(op.id)}
                      className={`w-full p-2.5 rounded-lg border text-left transition-all ${
                        selectedOp === op.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                      }`}>
                      <div className="text-xs font-medium">{op.name}</div>
                      <div className="text-[10px] text-zinc-500">{op.desc}</div>
                    </button>
                  ))}
                </div>
              </Card>
              <Card>
                <CardTitle className="text-sm mb-3">Region (pixels)</CardTitle>
                <div className="grid grid-cols-2 gap-2">
                  {["X", "Y", "Width", "Height"].map((label, i) => (
                    <div key={label}>
                      <label className="text-[10px] text-zinc-500">{label}</label>
                      <input type="number" defaultValue={[50, 50, 200, 200][i]}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 mt-0.5" />
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          <Button className="w-full" onClick={handleApply} disabled={loading}>
            {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</> : <><Wand2 className="w-4 h-4 mr-2" /> Apply {mode === "filters" ? "Filter" : "Edit"}</>}
          </Button>
        </div>

        <div className="lg:col-span-2">
          {loading ? (
            <Card className="min-h-[400px] flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-10 h-10 text-indigo-400 mx-auto mb-3 animate-spin" />
                <p className="text-indigo-400 text-sm font-medium">{loadingMsg}</p>
              </div>
            </Card>
          ) : (
            <Card className="min-h-[400px] flex items-center justify-center">
              <div className="text-center">
                <Scissors className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Upload a video to start editing</p>
                <p className="text-zinc-600 text-xs mt-1">Select a filter or operation, then click Apply</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
