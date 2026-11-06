"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Sparkles, Plus, Save, Trash2, Copy, Search, X, Tag, Upload, FileDown, Wand2 } from "lucide-react";
import {
  listPresets,
  savePreset,
  deletePreset,
  type PromptPreset,
  type PresetLibrary,
} from "@/lib/api";

const CATEGORY_LABELS: Record<string, string> = {
  style: "Visual styles",
  genre: "Genres",
  shot: "Shot types",
  lighting: "Lighting",
  custom: "Your presets",
};

interface PresetsPanelProps {
  onNavigate?: (tab: string) => void;
}

export default function PresetsPanel({ onNavigate }: PresetsPanelProps = {}) {
  const [library, setLibrary] = useState<PresetLibrary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [selected, setSelected] = useState<PromptPreset | null>(null);
  const [showEditor, setShowEditor] = useState(false);
  const [editorDraft, setEditorDraft] = useState<Partial<PromptPreset>>({});
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setLibrary(await listPresets());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const allPresets = useMemo(() => {
    if (!library) return [];
    return [...library.builtin, ...library.user];
  }, [library]);

  const filtered = useMemo(() => {
    let items = allPresets;
    if (activeCategory !== "all") {
      items = items.filter((p) => p.category === activeCategory);
    }
    if (query.trim()) {
      const q = query.toLowerCase();
      items = items.filter((p) =>
        p.name.toLowerCase().includes(q) ||
        p.description?.toLowerCase().includes(q) ||
        p.positive_prompt?.toLowerCase().includes(q) ||
        (p.tags || []).some((t) => t.toLowerCase().includes(q))
      );
    }
    return items;
  }, [allPresets, activeCategory, query]);

  const categories = useMemo(() => {
    if (!library) return [];
    const list = [...library.categories];
    if (library.user.length > 0 && !list.includes("custom")) list.push("custom");
    return list;
  }, [library]);

  const beginNew = () => {
    setEditorDraft({ name: "", category: "custom", positive_prompt: "", negative_prompt: "", description: "" });
    setShowEditor(true);
  };

  const beginDuplicate = (p: PromptPreset) => {
    setEditorDraft({
      name: `${p.name} (copy)`,
      category: "custom",
      description: p.description,
      positive_prompt: p.positive_prompt,
      negative_prompt: p.negative_prompt,
      tags: p.tags,
    });
    setShowEditor(true);
  };

  const beginEdit = (p: PromptPreset) => {
    if (p.builtin) return beginDuplicate(p);
    setEditorDraft({ ...p });
    setShowEditor(true);
  };

  const onSave = async () => {
    if (!editorDraft.name || !editorDraft.name.trim()) {
      setError("Preset needs a name");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await savePreset({
        name: editorDraft.name,
        id: editorDraft.id,
        category: editorDraft.category || "custom",
        description: editorDraft.description || "",
        positive_prompt: editorDraft.positive_prompt || "",
        negative_prompt: editorDraft.negative_prompt || "",
        recommended_settings: editorDraft.recommended_settings || {},
        tags: editorDraft.tags || [],
      });
      setShowEditor(false);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const onDelete = async (p: PromptPreset) => {
    if (p.builtin) return;
    try {
      await deletePreset(p.id);
      if (selected?.id === p.id) setSelected(null);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const copyPrompt = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const exportAll = () => {
    const user = library?.user ?? [];
    if (user.length === 0) {
      setError("No custom presets to export. Save at least one to enable export.");
      return;
    }
    const data = JSON.stringify({
      version: 1,
      exported_at: new Date().toISOString(),
      presets: user,
    }, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `studiolite_presets_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const importJson = async (file: File | null) => {
    if (!file) return;
    setError(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      // Accept either a single preset object, an array, or the exported
      // wrapper shape { presets: [...] }.
      let items: unknown[] = [];
      if (Array.isArray(parsed)) items = parsed;
      else if (parsed && Array.isArray((parsed as { presets?: unknown[] }).presets)) {
        items = (parsed as { presets: unknown[] }).presets;
      } else if (parsed && typeof parsed === "object") items = [parsed];

      let imported = 0;
      for (const raw of items) {
        if (!raw || typeof raw !== "object") continue;
        const p = raw as PromptPreset;
        if (!p.name) continue;
        try {
          // Drop id so the server regenerates a unique one and avoids
          // overwriting existing presets with the same id on re-import.
          const { id: _ignoredId, builtin: _ignoredBuiltin, ...rest } = p;
          void _ignoredId; void _ignoredBuiltin;
          await savePreset({ ...rest, name: p.name });
          imported++;
        } catch { /* skip malformed, keep going */ }
      }
      if (imported === 0) {
        setError("No importable presets found in that file.");
      } else {
        await refresh();
      }
    } catch (e) {
      setError(`Import failed: ${(e as Error).message}`);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Prompt Templates</h1>
        <p className="text-zinc-400 mt-1">
          A library of style presets and genre templates. Pick a preset,
          copy its prompts into any generator, or save your own combinations
          for reuse.
        </p>
      </div>

      <div className="flex flex-col md:flex-row gap-3 mb-6">
        <div className="flex items-center gap-2 flex-1 bg-zinc-900/60 border border-zinc-800 rounded-md px-3 py-1.5">
          <Search className="w-4 h-4 text-zinc-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search presets by name, description or tag"
            className="flex-1 bg-transparent text-sm text-zinc-200 outline-none"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-zinc-500 hover:text-zinc-300">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <Button onClick={beginNew}>
          <Plus className="w-4 h-4 mr-1" /> New preset
        </Button>
        <Button variant="secondary" onClick={() => exportAll()}>
          <FileDown className="w-4 h-4 mr-1" /> Export
        </Button>
        <label className="inline-flex">
          <input type="file" accept="application/json" className="hidden"
                 onChange={(e) => importJson(e.target.files?.[0] ?? null)} />
          <span className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-50 bg-zinc-800 text-zinc-100 hover:bg-zinc-700 h-10 px-4 py-2 cursor-pointer">
            <Upload className="w-4 h-4 mr-1" /> Import
          </span>
        </label>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => setActiveCategory("all")}
          className={`px-3 py-1.5 rounded-full text-xs border ${
            activeCategory === "all"
              ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
              : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
          }`}
        >All ({allPresets.length})</button>
        {categories.map((c) => {
          const count = allPresets.filter((p) => p.category === c).length;
          return (
            <button
              key={c}
              onClick={() => setActiveCategory(c)}
              className={`px-3 py-1.5 rounded-full text-xs border capitalize ${
                activeCategory === c
                  ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300"
                  : "bg-zinc-900/50 border-zinc-800 text-zinc-400"
              }`}
            >{CATEGORY_LABELS[c] || c} ({count})</button>
          );
        })}
      </div>

      {error && (
        <div className="mb-4 text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={`lg:col-span-${selected ? 2 : 3}`}>
          {loading ? (
            <p className="text-sm text-zinc-500">Loading presets...</p>
          ) : filtered.length === 0 ? (
            <p className="text-sm text-zinc-500">
              No presets match. Try widening the search or <button className="text-indigo-400 hover:underline" onClick={beginNew}>create a new one</button>.
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {filtered.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setSelected(p)}
                  className={`text-left rounded-lg border p-4 transition-colors ${
                    selected?.id === p.id
                      ? "border-indigo-500/50 bg-indigo-600/10"
                      : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <h3 className="font-semibold text-sm text-zinc-100">{p.name}</h3>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 capitalize">
                      {p.category}
                    </span>
                  </div>
                  {p.description && (
                    <p className="text-xs text-zinc-400 line-clamp-2">{p.description}</p>
                  )}
                  {p.tags && p.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {p.tags.slice(0, 4).map((t) => (
                        <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800/70 text-zinc-500">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                  {!p.builtin && (
                    <p className="text-[10px] text-amber-400 mt-2">Custom</p>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {selected && (
          <Card className="lg:col-span-1 h-fit sticky top-4">
            <div className="flex items-center justify-between mb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Sparkles className="w-4 h-4" /> {selected.name}
              </CardTitle>
              <button onClick={() => setSelected(null)} className="text-zinc-500 hover:text-zinc-300">
                <X className="w-4 h-4" />
              </button>
            </div>

            {selected.description && (
              <p className="text-xs text-zinc-400 mb-3">{selected.description}</p>
            )}

            {selected.positive_prompt && (
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] uppercase tracking-wide text-zinc-500">Positive</span>
                  <button
                    onClick={() => copyPrompt(selected.positive_prompt || "", "positive")}
                    className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                  >
                    <Copy className="w-3 h-3" />
                    {copied === "positive" ? "Copied!" : "Copy"}
                  </button>
                </div>
                <div className="bg-zinc-900/80 border border-zinc-800 rounded-md p-2 text-xs text-zinc-200 whitespace-pre-wrap">
                  {selected.positive_prompt}
                </div>
              </div>
            )}

            {selected.negative_prompt && (
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] uppercase tracking-wide text-zinc-500">Negative</span>
                  <button
                    onClick={() => copyPrompt(selected.negative_prompt || "", "negative")}
                    className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                  >
                    <Copy className="w-3 h-3" />
                    {copied === "negative" ? "Copied!" : "Copy"}
                  </button>
                </div>
                <div className="bg-zinc-900/80 border border-zinc-800 rounded-md p-2 text-xs text-zinc-200 whitespace-pre-wrap">
                  {selected.negative_prompt}
                </div>
              </div>
            )}

            {selected.recommended_settings && Object.keys(selected.recommended_settings).length > 0 && (
              <div className="mb-3">
                <span className="text-[11px] uppercase tracking-wide text-zinc-500">Recommended</span>
                <div className="bg-zinc-900/80 border border-zinc-800 rounded-md p-2 text-xs text-zinc-300 mt-1">
                  {Object.entries(selected.recommended_settings).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-zinc-500">{k}</span>
                      <span>{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selected.tags && selected.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {selected.tags.map((t) => (
                  <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 flex items-center gap-1">
                    <Tag className="w-2.5 h-2.5" />{t}
                  </span>
                ))}
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {onNavigate && (() => {
                const stash = () => {
                  try {
                    sessionStorage.setItem(
                      "studiolite:apply-preset",
                      JSON.stringify({
                        id: selected.id,
                        positive_prompt: selected.positive_prompt,
                        negative_prompt: selected.negative_prompt,
                      }),
                    );
                  } catch { /* storage denied - target panel falls back to empty */ }
                };
                return (
                  <>
                    <Button size="sm" onClick={() => { stash(); onNavigate("generate"); }}>
                      <Wand2 className="w-3.5 h-3.5 mr-1" /> Apply in Generator
                    </Button>
                    <Button size="sm" variant="secondary" onClick={() => { stash(); onNavigate("story"); }}>
                      Apply in Story
                    </Button>
                  </>
                );
              })()}
              <Button size="sm" variant="secondary" onClick={() => beginDuplicate(selected)}>
                <Copy className="w-3.5 h-3.5 mr-1" /> Duplicate
              </Button>
              {!selected.builtin && (
                <>
                  <Button size="sm" variant="secondary" onClick={() => beginEdit(selected)}>
                    Edit
                  </Button>
                  <Button size="sm" variant="danger" onClick={() => onDelete(selected)}>
                    <Trash2 className="w-3.5 h-3.5 mr-1" /> Delete
                  </Button>
                </>
              )}
            </div>
          </Card>
        )}
      </div>

      {showEditor && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-zinc-100">
                {editorDraft.id ? "Edit preset" : "New preset"}
              </h2>
              <button
                onClick={() => setShowEditor(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400">Name</label>
                <input
                  type="text" value={editorDraft.name || ""}
                  onChange={(e) => setEditorDraft({ ...editorDraft, name: e.target.value })}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                  placeholder="My cinematic look"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Category</label>
                <select
                  value={editorDraft.category || "custom"}
                  onChange={(e) => setEditorDraft({ ...editorDraft, category: e.target.value })}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                >
                  <option value="custom">Custom</option>
                  <option value="style">Style</option>
                  <option value="genre">Genre</option>
                  <option value="shot">Shot type</option>
                  <option value="lighting">Lighting</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Description</label>
                <input
                  type="text" value={editorDraft.description || ""}
                  onChange={(e) => setEditorDraft({ ...editorDraft, description: e.target.value })}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                  placeholder="Short summary of the look"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Positive prompt</label>
                <textarea
                  value={editorDraft.positive_prompt || ""}
                  onChange={(e) => setEditorDraft({ ...editorDraft, positive_prompt: e.target.value })}
                  rows={4}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                  placeholder="Cinematic, warm color grading, 35mm grain"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Negative prompt</label>
                <textarea
                  value={editorDraft.negative_prompt || ""}
                  onChange={(e) => setEditorDraft({ ...editorDraft, negative_prompt: e.target.value })}
                  rows={3}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                  placeholder="blurry, watermark, text, ..."
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400">Tags (comma separated)</label>
                <input
                  type="text"
                  value={(editorDraft.tags || []).join(", ")}
                  onChange={(e) => setEditorDraft({
                    ...editorDraft,
                    tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean),
                  })}
                  className="w-full mt-1 bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-sm text-zinc-200 outline-none focus:border-indigo-500"
                  placeholder="film, warm, soft"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="ghost" onClick={() => setShowEditor(false)}>Cancel</Button>
              <Button onClick={onSave} disabled={saving || !editorDraft.name?.trim()}>
                <Save className="w-4 h-4 mr-1" />
                {saving ? "Saving..." : "Save preset"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
