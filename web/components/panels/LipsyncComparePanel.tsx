"use client";

import { useEffect, useState } from "react";
import { Card, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Loader2, Play, Trash2, Plus } from "lucide-react";
import {
  filmList, lipsyncEval,
  type FilmListItem, type LipsyncMatrixScorecard,
} from "@/lib/api";

const ALL_BACKENDS = ["musetalk", "latentsync", "echomimicv3"] as const;

type Pair = { reference: string; audio: string };

function emptyPair(): Pair {
  return { reference: "", audio: "" };
}

export default function LipsyncComparePanel() {
  const [films, setFilms] = useState<FilmListItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [backends, setBackends] = useState<string[]>([...ALL_BACKENDS]);
  const [corpus, setCorpus] = useState<Pair[]>([emptyPair()]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<LipsyncMatrixScorecard | null>(null);

  useEffect(() => {
    filmList()
      .then((r) => {
        setFilms(r.projects);
        if (r.projects[0]) setSelected(r.projects[0].id);
      })
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : "Load failed"));
  }, []);

  const toggleBackend = (b: string) => {
    setBackends((prev) =>
      prev.includes(b) ? prev.filter((x) => x !== b) : [...prev, b],
    );
  };

  const setPair = (i: number, patch: Partial<Pair>) => {
    setCorpus((prev) => prev.map((p, idx) => (idx === i ? { ...p, ...patch } : p)));
  };

  const addPair = () => setCorpus((prev) => [...prev, emptyPair()]);
  const dropPair = (i: number) =>
    setCorpus((prev) => (prev.length <= 1 ? prev : prev.filter((_, idx) => idx !== i)));

  const run = async () => {
    if (!selected) return;
    setErr(null);
    setResult(null);
    setRunning(true);
    try {
      const r = await lipsyncEval({
        project_id: selected,
        corpus: corpus
          .filter((p) => p.reference.trim())
          .map((p) => ({
            reference: p.reference.trim(),
            audio: p.audio.trim() || null,
          })),
        backends,
      });
      setResult(r);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Eval failed");
    } finally {
      setRunning(false);
    }
  };

  const canRun =
    !!selected &&
    !running &&
    backends.length > 0 &&
    corpus.some((p) => p.reference.trim());

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-zinc-100">Lip sync comparison</h2>
        <p className="text-sm text-zinc-400 mt-1">
          Head-to-head matrix over MuseTalk, LatentSync, and EchoMimicV3 on your
          own corpus. Light metrics (duration drift, frame count delta) run with
          only ffprobe; heavy metrics (SyncNet, LMD, FID, FVD) light up when the
          <code> LIPSYNC_EVAL_*_BIN</code> env vars point at user-owned CLIs.
        </p>
      </div>

      <Card>
        <CardTitle>Project</CardTitle>
        <CardDescription>
          The matrix resolves each backend&apos;s venv config relative to the chosen project.
        </CardDescription>
        <select
          value={selected ?? ""}
          onChange={(e) => setSelected(e.target.value || null)}
          className="mt-3 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 w-full md:w-[20rem]"
        >
          {films.length === 0 && <option value="">(no films)</option>}
          {films.map((f) => (
            <option key={f.id} value={f.id}>{f.title}</option>
          ))}
        </select>
      </Card>

      <Card>
        <CardTitle>Backends</CardTitle>
        <CardDescription>
          Pick which of the three lipsync backends to score. Unchecked backends are skipped.
        </CardDescription>
        <div className="mt-3 flex flex-wrap gap-2">
          {ALL_BACKENDS.map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => toggleBackend(b)}
              className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
                backends.includes(b)
                  ? "border-indigo-500/40 bg-indigo-500/10 text-indigo-300"
                  : "border-zinc-700 bg-zinc-900/50 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {b}
            </button>
          ))}
        </div>
      </Card>

      <Card>
        <CardTitle>Corpus</CardTitle>
        <CardDescription>
          Each row is one (reference video, audio) pair. Paths are relative to
          the project directory. Leave audio blank to re-use the reference&apos;s own track.
        </CardDescription>
        <div className="mt-3 space-y-2">
          {corpus.map((pair, i) => (
            <div key={i} className="flex items-center gap-2">
              <input
                type="text"
                placeholder="reference.mp4"
                value={pair.reference}
                onChange={(e) => setPair(i, { reference: e.target.value })}
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 font-mono"
              />
              <input
                type="text"
                placeholder="speech.wav (optional)"
                value={pair.audio}
                onChange={(e) => setPair(i, { audio: e.target.value })}
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 font-mono"
              />
              <button
                type="button"
                onClick={() => dropPair(i)}
                disabled={corpus.length <= 1}
                className="p-1.5 text-zinc-500 hover:text-red-400 disabled:opacity-30"
                title="Remove row"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addPair}
            className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300"
          >
            <Plus className="w-3.5 h-3.5" /> Add pair
          </button>
        </div>
      </Card>

      <div className="flex items-center gap-3">
        <Button onClick={run} disabled={!canRun}>
          {running ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Play className="w-4 h-4 mr-1" />}
          Run matrix
        </Button>
        {err && <span className="text-sm text-red-300">{err}</span>}
      </div>

      {result && <ResultView card={result} />}
    </div>
  );
}

function ResultView({ card }: { card: LipsyncMatrixScorecard }) {
  const metricNames = Array.from(
    new Set(card.backends.flatMap((b) => Object.keys(b.metrics))),
  );

  return (
    <>
      <Card>
        <CardTitle>Ranking</CardTitle>
        <CardDescription>
          Best-first per metric. Backends with no verdict on a metric are omitted,
          not sorted to the bottom — so an uninstalled SyncNet does not look like a loss.
        </CardDescription>
        <div className="mt-4 space-y-2">
          {Object.entries(card.ranking).length === 0 && (
            <p className="text-sm text-zinc-500">No metric produced a ranking.</p>
          )}
          {Object.entries(card.ranking).map(([metric, order]) => (
            <div key={metric} className="flex items-center gap-3 text-sm">
              <span className="font-mono text-xs text-zinc-400 w-48 flex-shrink-0">{metric}</span>
              <div className="flex items-center gap-1 flex-wrap">
                {order.map((b, i) => (
                  <Badge
                    key={b}
                    variant={i === 0 ? "success" : i === order.length - 1 ? "warning" : "default"}
                  >
                    {i + 1}. {b}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <CardTitle>Backends ({card.corpus_count} clips)</CardTitle>
        <CardDescription>
          Each column is one metric&apos;s aggregate across rendered clips. &ldquo;—&rdquo; means the
          backend had no opinion on that metric.
        </CardDescription>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-zinc-500 border-b border-zinc-800">
                <th className="py-2 pr-4">Backend</th>
                <th className="py-2 pr-4">Rendered</th>
                {metricNames.map((m) => (
                  <th key={m} className="py-2 pr-4 font-mono">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {card.backends.map((row) => (
                <tr key={row.backend} className="border-b border-zinc-800/50">
                  <td className="py-2 pr-4 font-medium text-zinc-200">{row.backend}</td>
                  <td className="py-2 pr-4 text-zinc-400">
                    {row.rendered_count}/{row.clip_count}
                  </td>
                  {metricNames.map((m) => (
                    <td key={m} className="py-2 pr-4 font-mono text-zinc-300">
                      {row.metrics[m] == null ? "—" : row.metrics[m]!.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
