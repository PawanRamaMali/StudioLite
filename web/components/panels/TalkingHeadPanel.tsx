"use client";
import { useCallback, useEffect, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Mic, Upload, Download, AlertTriangle } from "lucide-react";
import {
  uploadImage,
  uploadEditAudio,
  talkingHeadStatus,
  generateTalkingHead,
  getJob,
  getDownloadUrl,
  type TalkingHeadStatus,
  type ImageUploadResponse,
  type EditAudioUploadResponse,
} from "@/lib/api";

export default function TalkingHeadPanel() {
  const [portrait, setPortrait] = useState<ImageUploadResponse | null>(null);
  const [audio, setAudio] = useState<EditAudioUploadResponse | null>(null);
  const [status, setStatus] = useState<TalkingHeadStatus | null>(null);
  const [backend, setBackend] = useState<"auto" | "sadtalker" | "hallo3">("auto");
  const [expressionScale, setExpressionScale] = useState(1.0);
  const [guidanceScale, setGuidanceScale] = useState(2.0);
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    talkingHeadStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const onPortrait = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    try {
      setPortrait(await uploadImage(file));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const onAudio = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    try {
      setAudio(await uploadEditAudio(file));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const start = async () => {
    if (!portrait || !audio) {
      setError("Upload a portrait and a driving audio clip first.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Queueing...");
    try {
      const job = await generateTalkingHead({
        image_path: portrait.image_path,
        audio_path: audio.audio_path,
        backend,
        expression_scale: expressionScale,
        guidance_scale: guidanceScale,
        inference_steps: inferenceSteps,
      });
      const poll = setInterval(async () => {
        try {
          const j = await getJob(job.job_id);
          setProgress(Math.min(Math.round(j.progress || 0), 100));
          setProgressMsg(j.message || "Working...");
          if (j.status === "completed") {
            clearInterval(poll);
            const out = (j.result as { output_path?: string } | undefined)?.output_path;
            if (out) setResult(out);
            setRunning(false);
          } else if (j.status === "failed") {
            clearInterval(poll);
            setError(j.error || "Talking head failed");
            setRunning(false);
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch (e) {
      setError((e as Error).message);
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Mic className="w-6 h-6 text-indigo-400" />
        <h1 className="text-2xl font-bold">Talking Head (Lip Sync at Generation)</h1>
      </div>
      <p className="text-sm text-zinc-400 -mt-4">
        Audio-driven talking-head video: upload a portrait + an audio clip, get back
        a video where the person speaks the audio with correct mouth shapes, head
        motion, and expression. SadTalker (moderate quality, 12GB VRAM) or Hallo3
        (higher quality, 24GB VRAM) behind one interface.
      </p>

      {status && !status.available && (
        <Card className="border-yellow-500/40 bg-yellow-500/5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-200">
              <div className="font-semibold mb-1">No backend installed yet.</div>
              <pre className="whitespace-pre-wrap text-xs text-yellow-200/80 font-mono">
                {status.install_hint}
              </pre>
            </div>
          </div>
        </Card>
      )}
      {status && status.available && (
        <div className="text-xs text-green-400">
          Ready: {status.backends.join(", ")}
        </div>
      )}

      <Card>
        <CardTitle>1. Portrait</CardTitle>
        <label className="flex items-center gap-3 px-4 py-6 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-indigo-500 transition">
          <Upload className="w-5 h-5 text-zinc-400" />
          <input type="file" accept="image/png,image/jpeg,image/jpg" className="hidden"
                 onChange={(e) => onPortrait(e.target.files?.[0] || null)} />
          <div className="flex-1">
            <div className="font-medium">
              {portrait ? portrait.image_path.split(/[\\/]/).pop() : "JPG or PNG of a face (no webp/heic)"}
            </div>
            {portrait && <div className="text-xs text-zinc-500 mt-1 truncate">{portrait.image_path}</div>}
          </div>
        </label>
      </Card>

      <Card>
        <CardTitle>2. Driving audio</CardTitle>
        <label className="flex items-center gap-3 px-4 py-6 border-2 border-dashed border-zinc-700 rounded-lg cursor-pointer hover:border-indigo-500 transition">
          <Upload className="w-5 h-5 text-zinc-400" />
          <input type="file" accept="audio/*" className="hidden"
                 onChange={(e) => onAudio(e.target.files?.[0] || null)} />
          <div className="flex-1">
            <div className="font-medium">
              {audio ? audio.audio_path.split(/[\\/]/).pop() : "Voice recording or TTS output"}
            </div>
            {audio && <div className="text-xs text-zinc-500 mt-1 truncate">{audio.audio_path}</div>}
          </div>
        </label>
      </Card>

      <Card>
        <CardTitle>3. Options</CardTitle>
        <div className="space-y-4">
          <div>
            <label className="text-xs text-zinc-400">Backend</label>
            <select value={backend}
                    onChange={(e) => setBackend(e.target.value as "auto" | "sadtalker" | "hallo3")}
                    className="w-full mt-1 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded text-sm">
              <option value="auto">Auto (prefers Hallo3 when installed)</option>
              <option value="sadtalker">SadTalker (12GB VRAM, moderate)</option>
              <option value="hallo3">Hallo3 (24GB VRAM, higher quality)</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-zinc-400">
              Expression scale: {expressionScale.toFixed(2)} (SadTalker: 0.5 subtle, 1.5 expressive)
            </label>
            <input type="range" min={0.3} max={2.0} step={0.05}
                   value={expressionScale}
                   onChange={(e) => setExpressionScale(Number(e.target.value))}
                   className="w-full mt-1" />
          </div>
          {backend === "hallo3" && (
            <>
              <div>
                <label className="text-xs text-zinc-400">
                  Guidance scale: {guidanceScale.toFixed(1)} (Hallo3)
                </label>
                <input type="range" min={0.5} max={8.0} step={0.1}
                       value={guidanceScale}
                       onChange={(e) => setGuidanceScale(Number(e.target.value))}
                       className="w-full mt-1" />
              </div>
              <div>
                <label className="text-xs text-zinc-400">
                  Inference steps: {inferenceSteps} (Hallo3)
                </label>
                <input type="range" min={4} max={60} step={1}
                       value={inferenceSteps}
                       onChange={(e) => setInferenceSteps(Number(e.target.value))}
                       className="w-full mt-1" />
              </div>
            </>
          )}
        </div>
      </Card>

      <Card>
        <CardTitle>4. Generate</CardTitle>
        <Button onClick={start}
                disabled={!portrait || !audio || running || (status !== null && !status.available)}
                variant="primary">
          {running ? "Generating..." : "Generate talking head"}
        </Button>
        {running && (
          <div className="mt-4">
            <div className="h-2 bg-zinc-900 rounded overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="text-xs text-zinc-400 mt-2">{progress}% — {progressMsg}</div>
          </div>
        )}
        {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
        {result && (
          <div className="mt-4 space-y-3">
            <video src={getDownloadUrl(result)} controls autoPlay loop
                   className="w-full rounded-lg border border-zinc-700" />
            <a href={getDownloadUrl(result)} target="_blank" rel="noreferrer"
               className="inline-flex items-center gap-2 text-sm text-green-400 hover:underline">
              <Download className="w-4 h-4" /> Download MP4
            </a>
          </div>
        )}
      </Card>
    </div>
  );
}
