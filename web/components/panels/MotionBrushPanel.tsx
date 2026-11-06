"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { MousePointer2, Upload, Trash2, Copy, Check } from "lucide-react";
import {
  uploadImage,
  applyMotionBrush,
  type ImageUploadResponse,
  type MotionStrokeInput,
} from "@/lib/api";

interface Stroke extends MotionStrokeInput {
  id: string;
}

function drawArrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number,
                   color: string, label: string) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  // Arrowhead
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const headLen = 14;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(
    x1 - headLen * Math.cos(angle - Math.PI / 6),
    y1 - headLen * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    x1 - headLen * Math.cos(angle + Math.PI / 6),
    y1 - headLen * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "bold 13px sans-serif";
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(0,0,0,0.75)";
    ctx.lineWidth = 3;
    const midx = (x0 + x1) / 2;
    const midy = (y0 + y1) / 2 - 8;
    ctx.strokeText(label, midx, midy);
    ctx.fillText(label, midx, midy);
  }
}

const COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185"];

export default function MotionBrushPanel() {
  const [image, setImage] = useState<ImageUploadResponse | null>(null);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [drawingLabel, setDrawingLabel] = useState("subject");
  const [basePrompt, setBasePrompt] = useState("");
  const [finalPrompt, setFinalPrompt] = useState("");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<{ x0: number; y0: number } | null>(null);

  const onImage = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setStrokes([]);
    setFinalPrompt("");
    try {
      setImage(await uploadImage(file));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    strokes.forEach((s, i) => {
      drawArrow(
        ctx,
        s.x0 * canvas.width, s.y0 * canvas.height,
        s.x1 * canvas.width, s.y1 * canvas.height,
        COLORS[i % COLORS.length],
        s.label ?? "",
      );
    });
  }, [strokes]);

  useEffect(() => { redraw(); }, [redraw, image]);
  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redraw]);

  const canvasPos = (e: React.MouseEvent | React.TouchEvent): { nx: number; ny: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = "touches" in e ? e.touches[0].clientX : e.clientX;
    const cy = "touches" in e ? e.touches[0].clientY : e.clientY;
    return {
      nx: Math.max(0, Math.min(1, (cx - rect.left) / rect.width)),
      ny: Math.max(0, Math.min(1, (cy - rect.top) / rect.height)),
    };
  };

  const onDown = (e: React.MouseEvent) => {
    const p = canvasPos(e);
    if (!p) return;
    dragRef.current = { x0: p.nx, y0: p.ny };
  };

  const onMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const p = canvasPos(e);
    if (!p) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    redraw();
    drawArrow(
      ctx,
      dragRef.current.x0 * canvas.width, dragRef.current.y0 * canvas.height,
      p.nx * canvas.width, p.ny * canvas.height,
      "#ffffff80",
      drawingLabel,
    );
  };

  const onUp = (e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const p = canvasPos(e);
    if (!p) { dragRef.current = null; return; }
    const { x0, y0 } = dragRef.current;
    dragRef.current = null;
    const dx = p.nx - x0;
    const dy = p.ny - y0;
    if (dx * dx + dy * dy < 0.0004) return;  // under 2% — treat as click, not stroke
    setStrokes((prev) => [
      ...prev,
      { id: crypto.randomUUID(), x0, y0, x1: p.nx, y1: p.ny, label: drawingLabel },
    ]);
  };

  const removeStroke = (id: string) => {
    setStrokes((prev) => prev.filter((s) => s.id !== id));
  };

  const clearAll = () => setStrokes([]);

  const applyPrompt = async () => {
    setError(null);
    setFinalPrompt("");
    try {
      const r = await applyMotionBrush({
        strokes: strokes.map(({ x0, y0, x1, y1, label }) => ({ x0, y0, x1, y1, label })),
        base_prompt: basePrompt,
      });
      setFinalPrompt(r.final_prompt);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const copyPrompt = async () => {
    if (!finalPrompt) return;
    try {
      await navigator.clipboard.writeText(finalPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard denied */ }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <MousePointer2 className="w-6 h-6 text-indigo-400" />
        <h1 className="text-2xl font-bold">Motion Brush</h1>
      </div>
      <p className="text-sm text-zinc-400 -mt-4">
        Draw arrows on a reference image to describe subject motion. Each arrow is
        translated into directional language and appended to the prompt. This is a
        <strong> prompt-level </strong>motion brush — works with every I2V engine
        here. Trajectory conditioning (DragNUWA / Motion-I2V) is a separate
        integration.
      </p>

      <Card>
        <CardTitle>1. Reference image</CardTitle>
        <label className="flex items-center gap-3 px-4 py-6 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-indigo-500 transition">
          <Upload className="w-5 h-5 text-zinc-400" />
          <input type="file" accept="image/*" className="hidden"
                 onChange={(e) => onImage(e.target.files?.[0] || null)} />
          <div className="flex-1">
            <div className="font-medium">
              {image ? image.image_path.split(/[\\/]/).pop() : "A scene image to annotate"}
            </div>
            {image && <div className="text-xs text-zinc-500 mt-1 truncate">{image.image_path}</div>}
          </div>
        </label>
      </Card>

      {image && (
        <Card>
          <CardTitle>2. Draw arrows (click and drag)</CardTitle>
          <div className="relative inline-block max-w-full"
               onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => dragRef.current = null}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img ref={imgRef} src={image.url} alt="reference" onLoad={redraw}
                 className="max-w-full rounded-lg border border-zinc-700 select-none"
                 draggable={false} />
            <canvas ref={canvasRef}
                    className="absolute inset-0 cursor-crosshair" />
          </div>
          <div className="flex items-center gap-3 mt-3">
            <label className="text-xs text-zinc-400">Subject label for next stroke:</label>
            <input type="text" value={drawingLabel}
                   onChange={(e) => setDrawingLabel(e.target.value)}
                   placeholder="car, person, camera..."
                   className="px-2 py-1 bg-zinc-900 border border-zinc-800 rounded text-sm w-48" />
            <Button variant="secondary" onClick={clearAll} disabled={!strokes.length}>
              <Trash2 className="w-4 h-4 mr-1" /> Clear all
            </Button>
          </div>
        </Card>
      )}

      {strokes.length > 0 && (
        <Card>
          <CardTitle>3. Strokes ({strokes.length})</CardTitle>
          <ul className="space-y-1.5">
            {strokes.map((s, i) => (
              <li key={s.id} className="flex items-center gap-3 text-sm">
                <span className="w-3 h-3 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                <span className="flex-1 font-mono text-xs">
                  {(s.label || "subject")} — ({s.x0.toFixed(2)}, {s.y0.toFixed(2)}) → ({s.x1.toFixed(2)}, {s.y1.toFixed(2)})
                </span>
                <button onClick={() => removeStroke(s.id)}
                        className="text-zinc-500 hover:text-red-400">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card>
        <CardTitle>4. Base prompt + apply</CardTitle>
        <textarea rows={2} value={basePrompt} onChange={(e) => setBasePrompt(e.target.value)}
                  placeholder="The base prompt you&apos;d send to the generator..."
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm" />
        <Button onClick={applyPrompt} disabled={!strokes.length} variant="primary" className="mt-3">
          Apply motion
        </Button>
        {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
        {finalPrompt && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-zinc-400">Enriched prompt</label>
              <button onClick={copyPrompt}
                      className="text-xs text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1">
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="whitespace-pre-wrap p-3 bg-zinc-950 border border-zinc-800 rounded text-xs text-zinc-200 font-mono">
              {finalPrompt}
            </pre>
          </div>
        )}
      </Card>
    </div>
  );
}
