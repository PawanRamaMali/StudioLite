import { create } from "zustand";

interface AppState {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  jobs: Record<string, unknown>[];
  addJob: (job: Record<string, unknown>) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: "generate",
  setActiveTab: (tab) => set({ activeTab: tab }),
  jobs: [],
  addJob: (job) => set((s) => ({ jobs: [job, ...s.jobs].slice(0, 50) })),
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}));
