"use client";

import { useEffect, useState } from "react";
import { Card, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Loader2, RefreshCw, Film as FilmIcon } from "lucide-react";
import {
  filmList, filmVbench, type FilmListItem, type VBenchScorecard,
  type VBenchDimension,
} from "@/lib/api";

const DIMENSION_LABELS: Record<string, string> = {
  physics: "Physical realism",
  creativity: "Creativity",
  commonsense: "Commonsense",
  controllability: "Controllability",
  human: "Human fidelity",
};

export default function QualityPanel() {
  const [films, setFilms] = useState<FilmListItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [card, setCard] = useState<VBenchScorecard | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    filmList()
      .then((r) => {
        setFilms(r.projects);
        if (r.projects[0]) setSelected(r.projects[0].id);
      })
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : "Load failed"));
  }, []);

  const load = (id: string, refresh = false) => {
    setErr(null);
    setNote(null);
    if (refresh) setRunning(true);
    else setLoading(true);
    filmVbench(id, refresh)
      .then((c) => setCard(c))
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : "Load failed";
        if (!refresh && /not found|404/i.test(msg)) {
          setCard(null);
          setNote('No scorecard yet. Run "Rescore" to generate one.');
        } else {
          setErr(msg);
        }
      })
      .finally(() => {
        setLoading(false);
        setRunning(false);
      });
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (selected) load(selected);
  }, [selected]);

  const overall = card?.overall ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-bold text-zinc-100">Quality — VBench 2.0</h2>
          <p className="text-sm text-zinc-400 mt-1">
            Regression gating on five dimensions (physics, creativity, commonsense,
            controllability, human fidelity) across 18 sub-capabilities.
            Scores read <code>artifacts/vbench2.json</code>; Rescore re-runs the harness.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={selected ?? ""}
            onChange={(e) => setSelected(e.target.value || null)}
            className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 min-w-[12rem]"
          >
            {films.length === 0 && <option value="">(no films)</option>}
            {films.map((f) => (
              <option key={f.id} value={f.id}>{f.title}</option>
            ))}
          </select>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => selected && load(selected, true)}
            disabled={!selected || running}
          >
            {running ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
            Rescore
          </Button>
        </div>
      </div>

      {err && (
        <Card className="border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-300">{err}</p>
        </Card>
      )}
      {note && !err && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <p className="text-sm text-amber-300">{note}</p>
        </Card>
      )}

      {loading && !card && (
        <Card>
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading scorecard…
          </div>
        </Card>
      )}

      {card && (
        <>
          <Card>
            <CardTitle>Overall</CardTitle>
            <CardDescription>
              Mean across every dimension that produced a value. {card.clip_count}{" "}
              clip{card.clip_count === 1 ? "" : "s"} scored.
            </CardDescription>
            <div className="mt-4 flex items-baseline gap-3">
              <span className="text-4xl font-bold text-zinc-100">
                {overall == null ? "—" : overall.toFixed(3)}
              </span>
              <span className="text-xs text-zinc-500">on 0 - 1 scale (higher is better)</span>
            </div>
          </Card>

          <Card>
            <CardTitle>Dimensions</CardTitle>
            <CardDescription>
              Each row is a VBench 2.0 dimension aggregated across clips.
            </CardDescription>
            <div className="mt-4 space-y-3">
              {card.aggregate.map((d) => (
                <DimensionRow key={d.dimension} dim={d} />
              ))}
              {card.aggregate.length === 0 && (
                <p className="text-sm text-zinc-500">No dimensions scored yet.</p>
              )}
            </div>
          </Card>

          {card.per_clip.length > 0 && (
            <Card>
              <CardTitle>Per clip</CardTitle>
              <CardDescription>
                Only clips flagged by <code>motion_shots</code>.
              </CardDescription>
              <div className="mt-4 space-y-2">
                {card.per_clip.map((c) => (
                  <div key={c.clip_path} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 min-w-0">
                        <FilmIcon className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" />
                        <span className="text-xs text-zinc-300 truncate font-mono">
                          {c.clip_path}
                        </span>
                      </div>
                      <span className="text-sm font-mono text-zinc-200 flex-shrink-0">
                        {c.overall == null ? "—" : c.overall.toFixed(3)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function scoreColor(score: number | null): string {
  if (score == null) return "text-zinc-500";
  if (score >= 0.7) return "text-green-400";
  if (score >= 0.45) return "text-amber-400";
  return "text-red-400";
}

function DimensionRow({ dim }: { dim: VBenchDimension }) {
  const label = DIMENSION_LABELS[dim.dimension] || dim.dimension;
  const pct = dim.score == null ? 0 : Math.max(0, Math.min(1, dim.score)) * 100;
  const covered = dim.capabilities.filter((c) => c.score != null).length;
  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-200">{label}</span>
          <Badge>
            {covered}/{dim.capabilities.length} capabilities
          </Badge>
        </div>
        <span className={`text-sm font-mono ${scoreColor(dim.score)}`}>
          {dim.score == null ? "—" : dim.score.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className="h-full bg-indigo-500/60"
          style={{ width: `${pct}%` }}
        />
      </div>
      {dim.capabilities.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {dim.capabilities.map((cap) => (
            <span
              key={cap.capability}
              className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border border-zinc-800 bg-zinc-900/60 text-zinc-400"
              title={cap.scorer_names.join(", ")}
            >
              <span>{cap.capability}</span>
              <span className={`font-mono ${scoreColor(cap.score)}`}>
                {cap.score == null ? "—" : cap.score.toFixed(2)}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
