"use client";
import { useEffect, useState } from "react";
import { listFormats, type VideoFormatPreset } from "@/lib/api";

interface SafeZoneOverlayProps {
  /** Format preset id: reels, shorts, tiktok, square, feed_portrait, landscape, or "" to hide. */
  formatId: string;
  /** Show the preset label + dim summary in the corner. */
  showBadge?: boolean;
  className?: string;
}

/**
 * Translucent mask overlay for social-platform safe zones.
 *
 * The real platform UI (close button, action rail, caption bar) covers
 * specific regions of the frame. This overlay greys out those regions so a
 * designer can see at a glance whether a subject sits inside the viewable
 * safe area. The mask is purely cosmetic - it doesn't modify the video and
 * is rendered strictly for preview.
 */
export default function SafeZoneOverlay({ formatId, showBadge = true, className }: SafeZoneOverlayProps) {
  const [presets, setPresets] = useState<VideoFormatPreset[]>([]);

  useEffect(() => {
    listFormats().then((r) => setPresets(r.formats || [])).catch(() => setPresets([]));
  }, []);

  if (!formatId) return null;
  const preset = presets.find((p) => p.id === formatId);
  if (!preset) return null;

  // Convert pixel safe-zone offsets to percentages so the overlay stays
  // correct no matter what size the preview is rendered at.
  const topPct = (preset.safe_zone_px.top / preset.height) * 100;
  const bottomPct = (preset.safe_zone_px.bottom / preset.height) * 100;
  const rightPct = (preset.safe_zone_px.right / preset.width) * 100;
  const leftPct = (preset.safe_zone_px.left / preset.width) * 100;

  return (
    <div className={`pointer-events-none absolute inset-0 ${className ?? ""}`}>
      {/* Top gutter */}
      {topPct > 0 && (
        <div className="absolute left-0 right-0 top-0 bg-black/55 border-b border-yellow-400/60"
             style={{ height: `${topPct}%` }} />
      )}
      {/* Bottom gutter */}
      {bottomPct > 0 && (
        <div className="absolute left-0 right-0 bottom-0 bg-black/55 border-t border-yellow-400/60"
             style={{ height: `${bottomPct}%` }} />
      )}
      {/* Right rail (vertical formats) */}
      {rightPct > 0 && (
        <div className="absolute right-0 bg-black/40 border-l border-yellow-400/60"
             style={{ top: `${topPct}%`, bottom: `${bottomPct}%`, width: `${rightPct}%` }} />
      )}
      {/* Left rail (unused by current platforms but kept symmetric) */}
      {leftPct > 0 && (
        <div className="absolute left-0 bg-black/40 border-r border-yellow-400/60"
             style={{ top: `${topPct}%`, bottom: `${bottomPct}%`, width: `${leftPct}%` }} />
      )}
      {showBadge && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-black/70 text-[10px] text-yellow-300 rounded">
          {preset.label} · {preset.width}x{preset.height}
        </div>
      )}
    </div>
  );
}
