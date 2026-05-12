"use client";

import {
  Video, Wand2, BookOpen, Scissors, Music, ArrowUpCircle,
  Users, KeyRound, Briefcase, Settings, ChevronLeft, ChevronRight,
  Sparkles, Image as ImageIcon, Radio, ScanText,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const NAV_ITEMS = [
  { id: "generate", label: "Video Generator", icon: Video, group: "Create" },
  { id: "story", label: "Story Mode", icon: BookOpen, group: "Create" },
  { id: "images", label: "Images Studio", icon: ImageIcon, group: "Create" },
  { id: "edit", label: "Video Editor", icon: Scissors, group: "Tools" },
  { id: "audio", label: "Audio Studio", icon: Music, group: "Tools" },
  { id: "live-transcribe", label: "Live Transcribe", icon: Radio, group: "Tools" },
  { id: "screen-transcribe", label: "Screen Transcribe", icon: ScanText, group: "Tools" },
  { id: "upscale", label: "Upscale Video", icon: ArrowUpCircle, group: "Tools" },
  { id: "characters", label: "Characters", icon: Users, group: "Assets" },
  { id: "keyframes", label: "Keyframes", icon: KeyRound, group: "Assets" },
  { id: "jobs", label: "Jobs", icon: Briefcase, group: "System" },
  { id: "settings", label: "Settings", icon: Settings, group: "System" },
];

export default function Sidebar({ activeTab, onTabChange, isOpen, onToggle }: SidebarProps) {
  const groups = [...new Set(NAV_ITEMS.map((i) => i.group))];

  return (
    <aside
      className={cn(
        "fixed top-0 left-0 h-screen z-40 flex flex-col border-r border-zinc-800 bg-zinc-950 transition-all duration-300",
        isOpen ? "w-64" : "w-16"
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-zinc-800">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        {isOpen && (
          <div className="overflow-hidden">
            <h1 className="font-bold text-lg gradient-text">StudioLite</h1>
            <p className="text-[10px] text-zinc-500 -mt-0.5">AI Video Studio</p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {groups.map((group) => (
          <div key={group} className="mb-4">
            {isOpen && (
              <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider px-3 mb-1.5">
                {group}
              </p>
            )}
            {NAV_ITEMS.filter((i) => i.group === group).map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onTabChange(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all",
                    active
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/20"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 border border-transparent"
                  )}
                  title={item.label}
                >
                  <Icon className={cn("w-4 h-4 flex-shrink-0", active && "text-indigo-400")} />
                  {isOpen && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center py-3 border-t border-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
    </aside>
  );
}
