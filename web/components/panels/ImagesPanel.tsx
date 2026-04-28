"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  Image as ImageIcon, Wand2, Sparkles, Upload, Layers, ArrowUpCircle,
  Eraser, Copy, Download, Loader2, RefreshCw, Dice5,
} from "lucide-react";
import {
  getImagesCatalog, enhancePrompt, generateImage, editImage, variationImage,
  upscaleImage, removeBgImage, getJob, getImagesHistory, uploadImage, getImageUrl,
  downloadBlob,
} from "@/lib/api";

type Mode = "generate" | "edit" | "variation" | "upscale" | "removebg";

interface CatalogProvider { id: string; name: string; description: string }
interface CatalogModel { id: string; name: string; path: string | null }
interface CatalogSize { id: string; name: string; width: number; height: number }
interface Catalog {
  providers: CatalogProvider[];
  models: CatalogModel[];
  sizes: CatalogSize[];
  styles: string[];
  fooocus_styles: string[];
  edit_techniques: { id: string; name: string; description: string }[];
  qwen_edit_available: boolean;
  default_negatives: Record<string, string>;
  rembg_available: boolean;
  realesrgan_available: boolean;
}

interface HistoryItem { filename: string; path: string; url: string; size_bytes: number; mtime: number }

const MODES: { id: Mode; label: string; icon: typeof Wand2; desc: string }[] = [
  { id: "generate", label: "Generate", icon: Wand2,        desc: "Text → Image" },
  { id: "edit",     label: "Edit",     icon: Layers,       desc: "Image + prompt → New image" },
  { id: "variation",label: "Variation",icon: Copy,         desc: "More like this" },
  { id: "upscale",  label: "Upscale",  icon: ArrowUpCircle,desc: "2x or 4x super-resolution" },
  { id: "removebg", label: "Remove BG",icon: Eraser,       desc: "Cut out subject" },
];

export default function ImagesPanel() {
  const [mode, setMode] = useState<Mode>("generate");
  const [catalog, setCatalog] = useState<Catalog | null>(null);

  // Common state
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [provider, setProvider] = useState("sdxl");
  const [model, setModel] = useState<string>("sdxl_turbo");
  const [style, setStyle] = useState<string>("photorealistic");
  const [size, setSize] = useState<string>("portrait_9x16");
  const [seed, setSeed] = useState<number | null>(null);
  const [steps, setSteps] = useState(8);
  const [guidance, setGuidance] = useState(2.0);
  const [strength, setStrength] = useState(0.75);
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleMethod, setUpscaleMethod] = useState<"lanczos" | "realesrgan">("lanczos");
  // Edit technique: auto/qwen_edit/instruct/redraw — see imagegen.image_to_image docstring.
  const [editTechnique, setEditTechnique] = useState<"auto" | "qwen_edit" | "instruct" | "redraw">("auto");
  const [imageGuidance, setImageGuidance] = useState(1.5);

  // Upload (for edit/variation/upscale/removebg)
  const [uploadedPath, setUploadedPath] = useState<string | null>(null);
  const [uploadedUrl, setUploadedUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Job + result
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultPath, setResultPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enhancing, setEnhancing] = useState(false);

  // Gallery
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Load catalog + history on mount
  useEffect(() => {
    getImagesCatalog().then(c => {
      setCatalog(c);
      // Pre-fill negative prompt with the style-aware default the FIRST time
      // we see the catalog — only if user hasn't typed anything yet.
      if (c?.default_negatives && !negativePrompt) {
        setNegativePrompt(c.default_negatives[style] || c.default_negatives._base || "");
      }
    }).catch(() => setCatalog(null));
    refreshHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When style changes, swap the negative — but only if it currently matches
  // one of our defaults. If the user has customized it, leave it alone.
  useEffect(() => {
    if (!catalog?.default_negatives) return;
    const allDefaults = Object.values(catalog.default_negatives);
    if (negativePrompt === "" || allDefaults.includes(negativePrompt)) {
      setNegativePrompt(catalog.default_negatives[style] || catalog.default_negatives._base || "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [style, catalog]);

  // When provider changes, default model accordingly
  useEffect(() => {
    if (provider === "fooocus" && catalog?.fooocus_styles?.length) {
      setStyle(catalog.fooocus_styles[0]);
    }
  }, [provider, catalog]);

  // SDXL Turbo settings (8 steps / guidance 0) only apply to plain Generate.
  // Edit / variation / inpaint REQUIRE full 30-step + 7.5-guidance regardless
  // of the chosen base model — InstructPix2Pix and SDXL Img2Img produce
  // unchanged output with Turbo's settings.
  useEffect(() => {
    if (provider !== "sdxl") return;
    const isTurbo = !model || model === "sdxl_turbo" || model.toLowerCase().includes("turbo");
    if (mode === "generate" && isTurbo) {
      setSteps(8); setGuidance(2.0);
    } else if (mode === "edit") {
      // InstructPix2Pix likes ~30 steps + guidance 7.5; redraw img2img too.
      setSteps(30); setGuidance(7.5);
    } else if (mode === "variation") {
      setSteps(25); setGuidance(7.0);
    } else {
      // Generate with non-Turbo
      setSteps(30); setGuidance(7.5);
    }
  }, [provider, model, mode]);

  function refreshHistory() {
    getImagesHistory(40).then(d => setHistory(d.images || [])).catch(() => {});
  }

  async function handleEnhancePrompt() {
    if (!prompt.trim()) return;
    setEnhancing(true);
    setError(null);
    try {
      // For modes that act on a source image, pass image_path so the LLM is
      // grounded in the actual scene (BLIP captions it on the backend).
      const useImage = (mode === "edit" || mode === "variation" || mode === "removebg")
        && !!uploadedPath;
      const out = await enhancePrompt({
        prompt,
        style,
        mode: "expand",
        image_path: useImage ? uploadedPath : null,
        negative_prompt: negativePrompt || null,
      });
      if (out.enhanced && out.enhanced !== prompt) {
        setPrompt(out.enhanced);
      } else if (out.enhanced === prompt) {
        setError("LLM returned the same text (likely backend not responding). Check Ollama/llama.cpp.");
      }
      if (out.negative) {
        setNegativePrompt(out.negative);
      }
    } catch (e: unknown) {
      setError((e as Error)?.message || "Enhance failed — is the LLM backend running?");
    } finally {
      setEnhancing(false);
    }
  }

  async function handleEnhanceNegative() {
    if (!prompt.trim() && !negativePrompt.trim()) return;
    setEnhancing(true);
    setError(null);
    try {
      const out = await enhancePrompt({
        prompt: prompt || "(no positive prompt yet)",
        style,
        mode: "negative",
        negative_prompt: negativePrompt || null,
      });
      // mode=negative leaves the positive untouched; only negative is rewritten.
      if (out.negative) setNegativePrompt(out.negative);
    } catch (e: unknown) {
      setError((e as Error)?.message || "Enhance failed — is the LLM backend running?");
    } finally {
      setEnhancing(false);
    }
  }

  function resetNegativeToDefault() {
    if (!catalog?.default_negatives) return;
    setNegativePrompt(catalog.default_negatives[style] || catalog.default_negatives._base || "");
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploading(true);
    try {
      const out = await uploadImage(f);
      setUploadedPath(out.image_path);
      setUploadedUrl(getImageUrl(out.url));
    } catch (err: unknown) {
      setError((err as Error)?.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function pollJob(jobId: string) {
    while (true) {
      const j = await getJob(jobId);
      setProgress(Math.round(j.progress));
      setStatusMsg(j.message || "");
      if (j.status === "completed") {
        const url = (j.result?.url as string) || "";
        setResultUrl(getImageUrl(url));
        setResultPath((j.result?.image_path as string) || "");
        return j;
      }
      if (j.status === "failed") {
        throw new Error(j.error || j.message || "Job failed");
      }
      await new Promise(r => setTimeout(r, 800));
    }
  }

  async function handleGenerate() {
    setBusy(true); setError(null); setResultUrl(null); setProgress(0); setStatusMsg("Submitting...");
    try {
      let job;
      if (mode === "generate") {
        job = await generateImage({
          prompt, negative_prompt: negativePrompt, size, style,
          seed, steps, guidance, provider, model,
        });
      } else if (mode === "edit") {
        if (!uploadedPath) throw new Error("Upload an image first");
        job = await editImage({
          image_path: uploadedPath, prompt, negative_prompt: negativePrompt,
          strength, style, seed, steps, guidance, provider,
          technique: editTechnique, image_guidance: imageGuidance,
        });
      } else if (mode === "variation") {
        if (!uploadedPath) throw new Error("Upload an image first");
        job = await variationImage({
          image_path: uploadedPath, prompt: prompt || null,
          strength, style, seed,
        });
      } else if (mode === "upscale") {
        if (!uploadedPath) throw new Error("Upload an image first");
        job = await upscaleImage({
          image_path: uploadedPath, scale: upscaleFactor, method: upscaleMethod,
        });
      } else if (mode === "removebg") {
        if (!uploadedPath) throw new Error("Upload an image first");
        job = await removeBgImage({ image_path: uploadedPath });
      } else {
        throw new Error("Unknown mode");
      }
      await pollJob(job.job_id);
      refreshHistory();
    } catch (e: unknown) {
      setError((e as Error)?.message || "Failed");
    } finally {
      setBusy(false);
    }
  }

  function randomSeed() {
    setSeed(Math.floor(Math.random() * 2 ** 31));
  }

  async function downloadResult() {
    if (!resultUrl) return;
    try {
      const fname = resultPath ? resultPath.split(/[\\/]/).pop() || "image.png" : "image.png";
      await downloadBlob(resultUrl, fname);
    } catch (e: unknown) {
      setError((e as Error)?.message || "Download failed");
    }
  }

  const needsUpload = mode !== "generate";
  const needsPrompt = mode === "generate" || mode === "edit";
  const needsStrength = mode === "edit" || mode === "variation";

  const fooocusStyleOptions = catalog?.fooocus_styles ?? [];
  const styleOptions = provider === "fooocus" && fooocusStyleOptions.length > 0
    ? fooocusStyleOptions
    : (catalog?.styles ?? []);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Images Studio</h1>
        <p className="text-zinc-400 mt-1">
          Generate, edit, vary, upscale and clean up images — local SDXL{catalog?.providers?.find(p => p.id === "fooocus") ? ", Fooocus" : ""}{catalog?.providers?.find(p => p.id === "nanobanana2") ? ", Gemini" : ""}.
        </p>
      </div>

      {/* Mode tabs */}
      <div className="grid grid-cols-5 gap-2 mb-6">
        {MODES.map(m => {
          const Icon = m.icon;
          const active = mode === m.id;
          const disabled = m.id === "removebg" && catalog && !catalog.rembg_available;
          return (
            <button key={m.id}
              onClick={() => !disabled && setMode(m.id)}
              disabled={!!disabled}
              className={`p-3 rounded-lg border text-left transition-all ${
                active ? "border-indigo-500 bg-indigo-500/10" :
                disabled ? "border-zinc-800 opacity-40 cursor-not-allowed" :
                "border-zinc-700 hover:border-zinc-600"
              }`}>
              <div className="flex items-center gap-2 mb-1">
                <Icon className={`w-4 h-4 ${active ? "text-indigo-400" : "text-zinc-400"}`} />
                <span className="text-sm font-medium">{m.label}</span>
              </div>
              <p className="text-[10px] text-zinc-500">{m.desc}</p>
              {disabled ? <p className="text-[9px] text-amber-600/70 mt-1">Install rembg</p> : null}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT: Controls */}
        <div className="space-y-4">
          {needsUpload && (
            <Card>
              <CardTitle className="text-sm mb-3 flex items-center gap-2"><Upload className="w-3.5 h-3.5 text-indigo-400" /> Source Image</CardTitle>
              <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp"
                className="hidden" onChange={handleUpload} />
              {uploadedUrl ? (
                <div className="space-y-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={uploadedUrl} alt="source" className="w-full max-h-64 object-contain rounded border border-zinc-800" />
                  <div className="flex gap-2">
                    <Button variant="secondary" className="flex-1" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                      {uploading ? <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" /> : <Upload className="w-3.5 h-3.5 mr-1" />}
                      Replace
                    </Button>
                    <Button variant="secondary" onClick={() => { setUploadedPath(null); setUploadedUrl(null); }}>Clear</Button>
                  </div>
                </div>
              ) : (
                <button onClick={() => fileInputRef.current?.click()}
                  className="w-full border-2 border-dashed border-zinc-700 rounded-lg p-8 hover:border-indigo-500/50 transition-colors text-center">
                  {uploading ? <Loader2 className="w-7 h-7 text-indigo-400 mx-auto mb-2 animate-spin" /> : <Upload className="w-7 h-7 text-zinc-500 mx-auto mb-2" />}
                  <p className="text-sm text-zinc-400">{uploading ? "Uploading..." : "Drop or click to upload"}</p>
                </button>
              )}
            </Card>
          )}

          {needsPrompt && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <CardTitle className="text-sm flex items-center gap-2"><Wand2 className="w-3.5 h-3.5 text-indigo-400" /> Prompt</CardTitle>
                <Button size="sm" variant="secondary" onClick={handleEnhancePrompt} disabled={!prompt.trim() || enhancing}>
                  {enhancing ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Sparkles className="w-3 h-3 mr-1" />}
                  Enhance with AI
                </Button>
              </div>
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
                placeholder={mode === "edit" ? "What to change (e.g. 'raise her hand', 'make it sunset')..." : "Describe the image you want..."}
                className="w-full h-24 bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-sm text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none focus:border-indigo-500"
              />

              <div className="flex items-center justify-between mt-4 mb-2">
                <label className="text-[11px] uppercase font-semibold text-zinc-500">Negative Prompt</label>
                <div className="flex gap-1">
                  <Button size="sm" variant="secondary" onClick={resetNegativeToDefault} title="Reset to style default">
                    <RefreshCw className="w-3 h-3" />
                  </Button>
                  <Button size="sm" variant="secondary" onClick={handleEnhanceNegative} disabled={enhancing}>
                    {enhancing ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Sparkles className="w-3 h-3 mr-1" />}
                    Enhance
                  </Button>
                </div>
              </div>
              <textarea value={negativePrompt} onChange={e => setNegativePrompt(e.target.value)}
                placeholder="What to avoid (artifacts, anatomy issues, style mismatches)..."
                className="w-full h-20 bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-xs text-zinc-100 placeholder-zinc-500 resize-none focus:outline-none focus:border-indigo-500"
              />
            </Card>
          )}

          {mode === "edit" && (
            <Card>
              <CardTitle className="text-sm mb-3 flex items-center gap-2"><Layers className="w-3.5 h-3.5 text-amber-400" /> Edit Technique</CardTitle>
              <div className="space-y-1.5">
                {(catalog?.edit_techniques ?? [
                  { id: "auto", name: "Auto", description: "" },
                  { id: "qwen_edit", name: "Qwen-Image-Edit", description: "" },
                  { id: "instruct", name: "Instruction (legacy)", description: "" },
                  { id: "redraw", name: "Redraw", description: "" },
                ]).map(t => {
                  const disabled = t.id === "qwen_edit" && catalog && !catalog.qwen_edit_available;
                  return (
                    <button key={t.id}
                      onClick={() => !disabled && setEditTechnique(t.id as "auto" | "qwen_edit" | "instruct" | "redraw")}
                      disabled={!!disabled}
                      className={`w-full text-left p-2 rounded border ${
                        editTechnique === t.id ? "border-amber-500 bg-amber-500/10" :
                        disabled ? "border-zinc-800 opacity-40 cursor-not-allowed" :
                        "border-zinc-700 hover:border-zinc-600"
                      }`}>
                      <div className="text-xs font-medium">{t.name}</div>
                      {t.description ? <div className="text-[10px] text-zinc-500 mt-0.5">{t.description}</div> : null}
                      {disabled ? <div className="text-[10px] text-amber-600/70 mt-0.5">Weights downloading...</div> : null}
                    </button>
                  );
                })}
              </div>
              {editTechnique === "instruct" && (
                <div className="mt-3">
                  <div className="flex justify-between text-[11px] text-zinc-400 mb-1">
                    <span>Image preservation</span><span className="font-mono">{imageGuidance.toFixed(1)}</span>
                  </div>
                  <input type="range" min="1.0" max="2.5" step="0.1"
                    value={imageGuidance} onChange={e => setImageGuidance(parseFloat(e.target.value))}
                    className="w-full" />
                  <p className="text-[10px] text-zinc-600">Higher = preserve source more, lower = change more</p>
                </div>
              )}
            </Card>
          )}

          {(mode === "generate" || mode === "edit") && (
            <Card>
              <CardTitle className="text-sm mb-3">Provider & Model</CardTitle>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] text-zinc-500 uppercase block mb-1">Provider</label>
                  <select value={provider} onChange={e => setProvider(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs">
                    {(catalog?.providers ?? []).map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </div>
                {provider === "sdxl" && (
                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase block mb-1">Model</label>
                    <select value={model} onChange={e => setModel(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-xs">
                      {(catalog?.models ?? []).map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
              <p className="text-[10px] text-zinc-600 mt-2">
                {catalog?.providers?.find(p => p.id === provider)?.description}
              </p>
            </Card>
          )}

          {mode === "generate" && (
            <Card>
              <CardTitle className="text-sm mb-3">Size</CardTitle>
              <div className="grid grid-cols-3 gap-2">
                {(catalog?.sizes ?? []).slice(0, 9).map(s => (
                  <button key={s.id} onClick={() => setSize(s.id)}
                    className={`p-2 rounded border text-left transition-all ${
                      size === s.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"
                    }`}>
                    <div className="text-[11px] font-medium">{s.name}</div>
                    <div className="text-[9px] text-zinc-500">{s.width}×{s.height}</div>
                  </button>
                ))}
              </div>
            </Card>
          )}

          {(mode === "generate" || mode === "edit" || mode === "variation") && (
            <Card>
              <CardTitle className="text-sm mb-3">Style</CardTitle>
              <div className="flex flex-wrap gap-1.5">
                <button onClick={() => setStyle("")}
                  className={`text-[11px] px-2 py-1 rounded border ${!style ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700"}`}>(none)</button>
                {styleOptions.map(s => (
                  <button key={s} onClick={() => setStyle(s)}
                    className={`text-[11px] px-2 py-1 rounded border ${style === s ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700 hover:border-zinc-600"}`}>{s}</button>
                ))}
              </div>
            </Card>
          )}

          {mode === "upscale" && (
            <Card>
              <CardTitle className="text-sm mb-3">Upscale</CardTitle>
              <div className="grid grid-cols-2 gap-2 mb-3">
                {[2, 4].map(f => (
                  <button key={f} onClick={() => setUpscaleFactor(f)}
                    className={`p-3 rounded border text-center ${upscaleFactor === f ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-700"}`}>
                    <div className="text-lg font-bold">{f}x</div>
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {(["lanczos", "realesrgan"] as const).map(m => {
                  const disabled = m === "realesrgan" && catalog && !catalog.realesrgan_available;
                  return (
                    <button key={m} onClick={() => !disabled && setUpscaleMethod(m)} disabled={!!disabled}
                      className={`p-2 rounded border text-center text-xs ${
                        upscaleMethod === m ? "border-indigo-500 bg-indigo-500/10" :
                        disabled ? "border-zinc-800 opacity-40 cursor-not-allowed" : "border-zinc-700"
                      }`}>
                      {m === "lanczos" ? "Lanczos (fast)" : "Real-ESRGAN"}
                      {disabled ? <div className="text-[9px] text-amber-600/70 mt-0.5">install required</div> : null}
                    </button>
                  );
                })}
              </div>
            </Card>
          )}

          {(mode === "generate" || mode === "edit" || mode === "variation") && (
            <Card>
              <CardTitle className="text-sm mb-3">Advanced</CardTitle>
              <div className="space-y-3">
                {needsStrength && (
                  <div>
                    <div className="flex justify-between text-[11px] text-zinc-400 mb-1">
                      <span>Strength</span>
                      <span className="font-mono">{strength.toFixed(2)}</span>
                    </div>
                    <input type="range" min="0.05" max="0.95" step="0.05"
                      value={strength} onChange={e => setStrength(parseFloat(e.target.value))}
                      className="w-full" />
                    <p className="text-[10px] text-zinc-600">0 = identical to source, 1 = ignore source</p>
                  </div>
                )}
                {(mode === "generate" || mode === "edit") && (
                  <>
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400 mb-1">
                        <span>Steps</span><span className="font-mono">{steps}</span>
                      </div>
                      <input type="range" min="1" max="80" step="1" value={steps}
                        onChange={e => setSteps(parseInt(e.target.value))} className="w-full" />
                    </div>
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400 mb-1">
                        <span>Guidance</span><span className="font-mono">{guidance.toFixed(1)}</span>
                      </div>
                      <input type="range" min="0" max="15" step="0.5" value={guidance}
                        onChange={e => setGuidance(parseFloat(e.target.value))} className="w-full" />
                    </div>
                  </>
                )}
                <div>
                  <label className="text-[11px] text-zinc-400 block mb-1">Seed</label>
                  <div className="flex gap-2">
                    <input type="number" value={seed ?? ""} placeholder="Random"
                      onChange={e => setSeed(e.target.value === "" ? null : parseInt(e.target.value))}
                      className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs font-mono" />
                    <Button size="sm" variant="secondary" onClick={randomSeed} title="Random seed">
                      <Dice5 className="w-3.5 h-3.5" />
                    </Button>
                    <Button size="sm" variant="secondary" onClick={() => setSeed(null)}>Clear</Button>
                  </div>
                </div>
              </div>
            </Card>
          )}

          <Button size="lg" className="w-full" onClick={handleGenerate}
            disabled={busy || (needsPrompt && !prompt.trim()) || (needsUpload && !uploadedPath)}>
            {busy ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : <Sparkles className="w-5 h-5 mr-2" />}
            {busy ? `${progress}% — ${statusMsg.slice(0, 40) || "Working"}` : `Run ${MODES.find(m => m.id === mode)?.label}`}
          </Button>

          {error ? <div className="text-xs text-red-400 bg-red-900/20 border border-red-800 rounded p-2">{error}</div> : null}
        </div>

        {/* RIGHT: Result + history */}
        <div className="space-y-4">
          <Card className="min-h-[400px] flex items-center justify-center">
            {resultUrl ? (
              <div className="w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={resultUrl} alt="result" className="w-full rounded border border-zinc-800" />
                <div className="flex gap-2 mt-3">
                  <Button variant="secondary" className="flex-1" onClick={downloadResult}>
                    <Download className="w-3.5 h-3.5 mr-1" /> Download
                  </Button>
                  <Button variant="secondary" className="flex-1" onClick={() => {
                    if (resultPath) { setUploadedPath(resultPath); setUploadedUrl(resultUrl); setMode("edit"); }
                  }}>
                    <Layers className="w-3.5 h-3.5 mr-1" /> Edit further
                  </Button>
                </div>
                {statusMsg ? <p className="text-[10px] text-zinc-500 mt-2 truncate">{statusMsg}</p> : null}
              </div>
            ) : busy ? (
              <div className="text-center w-full">
                <Loader2 className="w-12 h-12 text-indigo-400 mx-auto mb-3 animate-spin" />
                <div className="w-full bg-zinc-800 rounded-full h-1.5 mb-2 max-w-xs mx-auto">
                  <div className="bg-indigo-500 h-1.5 rounded-full transition-all" style={{ width: `${progress}%` }}></div>
                </div>
                <p className="text-zinc-400 text-sm">{progress}%</p>
                <p className="text-zinc-600 text-xs mt-1">{statusMsg}</p>
              </div>
            ) : (
              <div className="text-center">
                <ImageIcon className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                <p className="text-zinc-500 text-sm">Result will appear here</p>
              </div>
            )}
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm">Recent</CardTitle>
              <Button size="sm" variant="secondary" onClick={refreshHistory}>
                <RefreshCw className="w-3 h-3 mr-1" /> Refresh
              </Button>
            </div>
            {history.length === 0 ? (
              <p className="text-xs text-zinc-600">No images yet.</p>
            ) : (
              <div className="grid grid-cols-4 gap-2">
                {history.slice(0, 16).map(h => (
                  <button key={h.filename} className="aspect-square rounded overflow-hidden border border-zinc-800 hover:border-indigo-500 transition-colors"
                    onClick={() => { setUploadedPath(h.path); setUploadedUrl(getImageUrl(h.url)); }}
                    title={h.filename}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={getImageUrl(h.url)} alt={h.filename} className="w-full h-full object-cover" />
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
