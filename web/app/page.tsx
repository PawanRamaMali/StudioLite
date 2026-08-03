"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import HomePanel from "@/components/panels/HomePanel";
import GeneratePanel from "@/components/panels/GeneratePanel";
import StoryPanel from "@/components/panels/StoryPanel";
import ImagesPanel from "@/components/panels/ImagesPanel";
import EditPanel from "@/components/panels/EditPanel";
import AudioPanel from "@/components/panels/AudioPanel";
import LiveTranscribePanel from "@/components/panels/LiveTranscribePanel";
import ScreenTranscribePanel from "@/components/panels/ScreenTranscribePanel";
import VideoTranscribePanel from "@/components/panels/VideoTranscribePanel";
import UpscalePanel from "@/components/panels/UpscalePanel";
import CharactersPanel from "@/components/panels/CharactersPanel";
import KeyframesPanel from "@/components/panels/KeyframesPanel";
import JobsPanel from "@/components/panels/JobsPanel";
import SettingsPanel from "@/components/panels/SettingsPanel";

const FIRST_VISIT_KEY = "studiolite:seen-home";

export default function Home() {
  // Default to "home" on first visit ever; otherwise last-picked tab or generate.
  const [activeTab, setActiveTab] = useState<string>("generate");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [ready, setReady] = useState(false);

  // Post-mount one-shot init: read localStorage (client-only), decide the
  // initial tab, then unblock render. This is exactly the case the strict
  // react-hooks/set-state-in-effect rule doesn't handle: reading a browser-only
  // API to seed initial UI state. Both setStates are safe here — they run once
  // and drive no dependent effect.
  useEffect(() => {
    let initialTab: string | null = null;
    try {
      const seen = window.localStorage.getItem(FIRST_VISIT_KEY);
      if (!seen) {
        initialTab = "home";
        window.localStorage.setItem(FIRST_VISIT_KEY, "1");
      }
    } catch {
      // localStorage disabled — stay on generate
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (initialTab) setActiveTab(initialTab);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setReady(true);
  }, []);

  const activePanel = (() => {
    switch (activeTab) {
      case "home":              return <HomePanel onNavigate={setActiveTab} />;
      case "generate":          return <GeneratePanel />;
      case "story":             return <StoryPanel />;
      case "images":            return <ImagesPanel />;
      case "edit":              return <EditPanel />;
      case "audio":             return <AudioPanel />;
      case "live-transcribe":   return <LiveTranscribePanel />;
      case "screen-transcribe": return <ScreenTranscribePanel />;
      case "video-transcribe":  return <VideoTranscribePanel />;
      case "upscale":           return <UpscalePanel />;
      case "characters":        return <CharactersPanel />;
      case "keyframes":         return <KeyframesPanel />;
      case "jobs":              return <JobsPanel />;
      case "settings":          return <SettingsPanel />;
      default:                  return <GeneratePanel />;
    }
  })();

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <main className={`flex-1 overflow-y-auto transition-all duration-300 ${sidebarOpen ? "ml-64" : "ml-16"}`}>
        <div className="p-6 max-w-7xl mx-auto">
          {ready && activePanel}
        </div>
      </main>
    </div>
  );
}
