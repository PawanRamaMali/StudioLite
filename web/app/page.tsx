"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import GeneratePanel from "@/components/panels/GeneratePanel";
import StoryPanel from "@/components/panels/StoryPanel";
import ImagesPanel from "@/components/panels/ImagesPanel";
import EditPanel from "@/components/panels/EditPanel";
import AudioPanel from "@/components/panels/AudioPanel";
import LiveTranscribePanel from "@/components/panels/LiveTranscribePanel";
import UpscalePanel from "@/components/panels/UpscalePanel";
import CharactersPanel from "@/components/panels/CharactersPanel";
import KeyframesPanel from "@/components/panels/KeyframesPanel";
import JobsPanel from "@/components/panels/JobsPanel";
import SettingsPanel from "@/components/panels/SettingsPanel";

const PANELS: Record<string, { component: React.FC; label: string }> = {
  generate: { component: GeneratePanel, label: "Video Generator" },
  story: { component: StoryPanel, label: "Story Mode" },
  images: { component: ImagesPanel, label: "Images Studio" },
  edit: { component: EditPanel, label: "Video Editor" },
  audio: { component: AudioPanel, label: "Audio Studio" },
  "live-transcribe": { component: LiveTranscribePanel, label: "Live Transcribe" },
  upscale: { component: UpscalePanel, label: "Upscale Video" },
  characters: { component: CharactersPanel, label: "Characters" },
  keyframes: { component: KeyframesPanel, label: "Keyframes" },
  jobs: { component: JobsPanel, label: "Jobs" },
  settings: { component: SettingsPanel, label: "Settings" },
};

export default function Home() {
  const [activeTab, setActiveTab] = useState("generate");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const ActivePanel = PANELS[activeTab]?.component || GeneratePanel;

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
          <ActivePanel />
        </div>
      </main>
    </div>
  );
}
