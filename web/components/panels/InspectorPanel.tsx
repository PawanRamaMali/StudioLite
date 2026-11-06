"use client";
import { useCallback, useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Info, Upload, Film, Volume2 } from "lucide-react";
import {
  uploadEditVideo,
  inspectVideo,
  type EditUploadResponse,
  type VideoInfoResult,
} from "@/lib/api";

function fmtSeconds(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "--";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${ss.toFixed(2).padStart(5, "0")}`;
  return `${m}:${ss.toFixed(2).padStart(5, "0")}`;
}

function fmtBytes(b: number | null | undefined): string {
  if (b == null) return "--";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(2)} MB`;
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtBitrate(b: number | null | undefined): string {
  if (b == null) return "--";
  if (b < 1000) return `${b} bps`;
  if (b < 1_000_000) return `${(b / 1000).toFixed(0)} kbps`;
  return `${(b / 1_000_000).toFixed(2)} Mbps`;
}

export default function InspectorPanel() {
  const [upload, setUpload] = useState<EditUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [info, setInfo] = useState<VideoInfoResult | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setError(null);
    setInfo(null);
    setUploading(true);
    try {
      const r = await uploadEditVideo(file);
      setUpload(r);
      // Auto-inspect right away: the whole value of this panel is "drop
      // a file, see its stats".
      setInspecting(true);
      try {
        setInfo(await inspectVideo(r.video_path));
      } finally {
        setInspecting(false);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }, []);

  const reinspect = async () => {
    if (!upload) return;
    setError(null);
    setInspecting(true);
    try {
      setInfo(await inspectVideo(upload.video_path));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setInspecting(false);
    }
  };

  const data = info?.info;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">Video Inspector</h1>
        <p className="text-zinc-400 mt-1">
          Drop a video to see codec, dimensions, frame rate, duration, and
          per-stream details via ffprobe.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Card>
            <label className="block cursor-pointer">
              <input
                type="file" accept="video/*,audio/*" className="hidden"
                disabled={uploading || inspecting}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              />
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-10 text-center hover:border-indigo-500/50 transition-colors">
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                {upload ? (
                  <>
                    <p className="text-sm text-zinc-300 truncate">{upload.filename}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {fmtBytes(upload.size_bytes)} - click to replace
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-zinc-400">
                    {uploading ? "Uploading..." : "Drop a video here or click to upload"}
                  </p>
                )}
              </div>
            </label>
            {upload && (
              <Button size="sm" variant="secondary" className="mt-3" onClick={reinspect} disabled={inspecting}>
                {inspecting ? "Reading..." : "Re-inspect"}
              </Button>
            )}
          </Card>

          {error && (
            <div className="text-sm text-red-400 border border-red-800/60 bg-red-950/40 rounded-md p-2">
              {error}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card className="min-h-[200px]">
            <CardTitle className="text-sm mb-3 flex items-center gap-2">
              <Info className="w-4 h-4" /> Container
            </CardTitle>
            {!data ? (
              <p className="text-sm text-zinc-500">
                {info && info.ok === false ? (info.error || "Could not read file") :
                  inspecting ? "Reading metadata..." : "Upload a file to see stats."}
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <span className="text-zinc-500">Format</span>
                <span className="text-zinc-200 truncate" title={data.format_name}>{data.format_name || "--"}</span>
                <span className="text-zinc-500">Duration</span>
                <span className="text-zinc-200">{fmtSeconds(data.duration)}</span>
                <span className="text-zinc-500">File size</span>
                <span className="text-zinc-200">{fmtBytes(data.size_bytes)}</span>
                <span className="text-zinc-500">Overall bitrate</span>
                <span className="text-zinc-200">{fmtBitrate(data.bitrate)}</span>
                <span className="text-zinc-500">Stream count</span>
                <span className="text-zinc-200">{data.streams.length}</span>
              </div>
            )}
          </Card>

          {data?.video && (
            <Card>
              <CardTitle className="text-sm mb-3 flex items-center gap-2">
                <Film className="w-4 h-4" /> Video stream
              </CardTitle>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <span className="text-zinc-500">Codec</span>
                <span className="text-zinc-200">
                  {data.video.codec_name}
                  {data.video.codec_long_name && (
                    <span className="text-zinc-500 text-xs"> &middot; {data.video.codec_long_name}</span>
                  )}
                </span>
                <span className="text-zinc-500">Resolution</span>
                <span className="text-zinc-200">
                  {data.video.width}x{data.video.height}
                  {data.video.aspect_ratio && (
                    <span className="text-zinc-500 text-xs"> &middot; {data.video.aspect_ratio}</span>
                  )}
                </span>
                <span className="text-zinc-500">FPS</span>
                <span className="text-zinc-200">
                  {data.video.fps ? data.video.fps.toFixed(3) : "--"}
                </span>
                <span className="text-zinc-500">Pixel format</span>
                <span className="text-zinc-200">{data.video.pix_fmt || "--"}</span>
                <span className="text-zinc-500">Bitrate</span>
                <span className="text-zinc-200">{fmtBitrate(data.video.bit_rate)}</span>
                {data.video.has_bframes != null && (
                  <>
                    <span className="text-zinc-500">B-frames</span>
                    <span className="text-zinc-200">{data.video.has_bframes}</span>
                  </>
                )}
              </div>
            </Card>
          )}

          {data?.audio && (
            <Card>
              <CardTitle className="text-sm mb-3 flex items-center gap-2">
                <Volume2 className="w-4 h-4" /> Audio stream
              </CardTitle>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                <span className="text-zinc-500">Codec</span>
                <span className="text-zinc-200">{data.audio.codec_name || "--"}</span>
                <span className="text-zinc-500">Sample rate</span>
                <span className="text-zinc-200">
                  {data.audio.sample_rate ? `${data.audio.sample_rate} Hz` : "--"}
                </span>
                <span className="text-zinc-500">Channels</span>
                <span className="text-zinc-200">
                  {data.audio.channels ?? "--"}
                  {data.audio.channel_layout && (
                    <span className="text-zinc-500 text-xs"> &middot; {data.audio.channel_layout}</span>
                  )}
                </span>
                <span className="text-zinc-500">Bitrate</span>
                <span className="text-zinc-200">{fmtBitrate(data.audio.bit_rate)}</span>
              </div>
            </Card>
          )}

          {data && data.streams.length > 0 && (
            <Card>
              <CardTitle className="text-sm mb-3">All streams</CardTitle>
              <div className="space-y-1">
                {data.streams.map((s) => (
                  <div key={s.index} className="text-xs bg-zinc-900/60 border border-zinc-800 rounded-md px-3 py-2">
                    <span className="text-indigo-400 font-mono">#{s.index}</span>{" "}
                    <span className="uppercase text-zinc-500">{s.codec_type}</span>{" "}
                    <span className="text-zinc-200">{s.codec_name}</span>
                    {s.width && s.height && (
                      <span className="text-zinc-500"> {s.width}x{s.height}</span>
                    )}
                    {s.sample_rate && (
                      <span className="text-zinc-500"> {s.sample_rate} Hz</span>
                    )}
                    {s.channels != null && (
                      <span className="text-zinc-500"> {s.channels}ch</span>
                    )}
                    {s.bit_rate && (
                      <span className="text-zinc-500"> &middot; {fmtBitrate(s.bit_rate)}</span>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
