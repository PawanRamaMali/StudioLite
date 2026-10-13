"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  Cpu, Cloud, CloudOff, ExternalLink, Loader2, RefreshCw,
} from "lucide-react";
import {
  listBackends, type BackendSpec, type BackendsResponse,
} from "@/lib/api";

const STAGE_LABELS: Record<string, string> = {
  upscale: "Upscale",
  lipsync: "Lip sync",
  motion_shots: "Motion shots",
  composer: "Music composer",
  voice_actor: "Voice actor",
  shots: "Shots",
};

function stageOrder(stage: string): number {
  const order = ["motion_shots", "shots", "voice_actor", "composer", "lipsync", "upscale"];
  const i = order.indexOf(stage);
  return i === -1 ? 99 : i;
}

function availabilityBadge(available: string) {
  if (available === "shipping") return <Badge variant="success">shipping</Badge>;
  if (available === "roadmap") return <Badge variant="info">roadmap</Badge>;
  return <Badge>{available}</Badge>;
}

export default function BackendsPanel() {
  const [data, setData] = useState<BackendsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const refresh = () => {
    setLoading(true);
    setErr(null);
    listBackends()
      .then((d) => setData(d))
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : "Load failed"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { refresh(); }, []);

  const grouped = useMemo(() => {
    if (!data) return [] as [string, BackendSpec[]][];
    const by: Record<string, BackendSpec[]> = {};
    for (const b of data.backends) {
      (by[b.stage] ||= []).push(b);
    }
    return Object.entries(by).sort(
      ([a], [b]) => stageOrder(a) - stageOrder(b),
    );
  }, [data]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-zinc-100">Backend registry</h2>
          <p className="text-sm text-zinc-400 mt-1">
            Every model the pipeline can call, grouped by stage. Numbers come
            from <code className="text-zinc-300">filmmaker/backend_registry.py</code>.
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={refresh} disabled={loading}>
          {loading ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
          Refresh
        </Button>
      </div>

      {err && (
        <Card className="border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-300">{err}</p>
        </Card>
      )}

      {data && (
        <Card>
          <CardTitle>Cloud providers</CardTitle>
          <CardDescription>
            Roadmap-only backends still work when a cloud provider is wired
            up. Set <code>REPLICATE_API_TOKEN</code> or <code>RUNPOD_API_KEY</code> to light these.
          </CardDescription>
          <div className="mt-4 flex flex-wrap gap-3">
            {Object.entries(data.cloud_providers).map(([name, on]) => (
              <div
                key={name}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs ${
                  on
                    ? "border-green-500/30 bg-green-500/10 text-green-300"
                    : "border-zinc-700 bg-zinc-900/50 text-zinc-500"
                }`}
              >
                {on ? <Cloud className="w-3.5 h-3.5" /> : <CloudOff className="w-3.5 h-3.5" />}
                <span className="font-medium">{name}</span>
                <span className="text-[10px] opacity-70">{on ? "configured" : "off"}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {grouped.map(([stage, specs]) => (
        <Card key={stage}>
          <CardTitle>{STAGE_LABELS[stage] || stage}</CardTitle>
          <CardDescription>
            {specs.length} backend{specs.length === 1 ? "" : "s"} registered.
          </CardDescription>
          <div className="mt-4 space-y-2">
            {specs.map((b) => (
              <BackendRow key={b.name} spec={b} />
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

function BackendRow({ spec }: { spec: BackendSpec }) {
  const vram = spec.min_vram_gb_fp16;
  const vramLabel =
    vram == null ? "unknown" : vram === 0 ? "CPU-fine" : `${vram.toFixed(1)} GB fp16`;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="font-semibold text-zinc-100 text-sm">{spec.name}</h4>
            {availabilityBadge(spec.available)}
            <Badge>{spec.license}</Badge>
            {spec.cloud_burst && <Badge variant="info">cloud-burst</Badge>}
          </div>
          {spec.base_model && (
            <p className="text-[11px] text-zinc-500 mt-0.5 truncate">
              base: <span className="text-zinc-400">{spec.base_model}</span>
            </p>
          )}
          {spec.notes && (
            <p className="text-xs text-zinc-400 mt-1.5 leading-relaxed">{spec.notes}</p>
          )}
          {spec.quant_variants.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {spec.quant_variants.map((q) => (
                <span
                  key={q.format}
                  className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border border-zinc-700 bg-zinc-800/40 text-zinc-300"
                  title={q.notes || q.source_url || ""}
                >
                  <span className="font-mono">{q.format}</span>
                  <span className="text-zinc-500">·</span>
                  <span>{q.vram_gb.toFixed(2)} GB</span>
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="flex flex-col items-end gap-2 flex-shrink-0 text-xs">
          <div className="flex items-center gap-1 text-zinc-400">
            <Cpu className="w-3 h-3" />
            <span>{vramLabel}</span>
          </div>
          {spec.source_url && (
            <a
              href={spec.source_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-indigo-400 hover:text-indigo-300 text-[11px]"
            >
              source <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
