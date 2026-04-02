"use client";

import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Scissors, Upload, Wand2 } from "lucide-react";

const STYLE_FILTERS = [
  { id: "grayscale", name: "Grayscale", preview: "bg-gradient-to-r from-gray-400 to-gray-600" },
  { id: "sepia", name: "Sepia", preview: "bg-gradient-to-r from-amber-700 to-yellow-600" },
  { id: "invert", name: "Invert", preview: "bg-gradient-to-r from-white to-gray-800" },
  { id: "warm", name: "Warm", preview: "bg-gradient-to-r from-orange-400 to-red-400" },
  { id: "cool", name: "Cool", preview: "bg-gradient-to-r from-blue-400 to-cyan-400" },
  { id: "vintage", name: "Vintage", preview: "bg-gradient-to-r from-amber-600 to-orange-800" },
  { id: "high_contrast", name: "High Contrast", preview: "bg-gradient-to-r from-black to-white" },
];

const EDIT_OPS = [
  { id: "remove", name: "Remove Object", desc: "Inpaint and remove" },
  { id: "blur", name: "Blur Region", desc: "Censor or soften" },
  { id: "brighten", name: "Brighten", desc: "Increase brightness" },
  { id: "darken", name: "Darken", desc: "Decrease brightness" },
  { id: "color_shift", name: "Color Shift", desc: "Change color tint" },
  { id: "grayscale", name: "Grayscale Region", desc: "Desaturate area" },
];

export default function EditPanel() {
  const [mode, setMode] = useState<"filters" | "region">("filters");
  const [selectedFilter, setSelectedFilter] = useState("grayscale");
  const [selectedOp, setSelectedOp] = useState("blur");

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Editor</h1>
        <p className="text-zinc-400 mt-1">Apply style filters, remove objects, and edit video regions</p>
      </div>

      {/* Mode Tabs */}
      <div className="flex gap-2 mb-6">
        {[{ id: "filters", label: "Style Filters" }, { id: "region", label: "Region Edit" }].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setMode(tab.id as "filters" | "region")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === tab.id ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          {/* Upload */}
          <Card>
            <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-zinc-500 transition-colors cursor-pointer">
              <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
              <p className="text-sm text-zinc-400">Drop video here or click to upload</p>
              <p className="text-xs text-zinc-600 mt-1">MP4, MOV, AVI, MKV</p>
            </div>
          </Card>

          {mode === "filters" ? (
            <Card>
              <CardTitle className="text-sm mb-3">Select Filter</CardTitle>
              <div className="grid grid-cols-2 gap-2">
                {STYLE_FILTERS.map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setSelectedFilter(f.id)}
                    className={`p-2 rounded-lg border text-left transition-all ${
                      selectedFilter === f.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                    }`}
                  >
                    <div className={`h-3 rounded-full mb-1.5 ${f.preview}`} />
                    <span className="text-xs">{f.name}</span>
                  </button>
                ))}
              </div>
            </Card>
          ) : (
            <Card>
              <CardTitle className="text-sm mb-3">Edit Operation</CardTitle>
              <div className="space-y-2">
                {EDIT_OPS.map((op) => (
                  <button
                    key={op.id}
                    onClick={() => setSelectedOp(op.id)}
                    className={`w-full p-2.5 rounded-lg border text-left transition-all ${
                      selectedOp === op.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                    }`}
                  >
                    <div className="text-xs font-medium">{op.name}</div>
                    <div className="text-[10px] text-zinc-500">{op.desc}</div>
                  </button>
                ))}
              </div>
            </Card>
          )}

          <Button className="w-full">
            <Wand2 className="w-4 h-4 mr-2" /> Apply
          </Button>
        </div>

        <div className="lg:col-span-2">
          <Card className="min-h-[400px] flex items-center justify-center">
            <div className="text-center">
              <Scissors className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-500 text-sm">Upload a video to start editing</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
