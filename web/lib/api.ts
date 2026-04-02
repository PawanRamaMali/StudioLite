const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Job {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  result?: Record<string, unknown>;
  error?: string;
  created_at: number;
  elapsed: number;
}

export interface SystemStatus {
  gpu_available: boolean;
  gpu_name: string;
  vram_total_gb: number;
  vram_free_gb: number;
  active_jobs: number;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// System
export const getSystemStatus = () => apiFetch<SystemStatus>("/api/v1/system/status");
export const getEngines = () => apiFetch<Record<string, unknown>>("/api/v1/system/engines");

// Jobs
export const getJob = (id: string) => apiFetch<Job>(`/api/v1/jobs/${id}`);
export const listJobs = (limit = 20) => apiFetch<Job[]>(`/api/v1/jobs?limit=${limit}`);

// Generation
export const generateText2Video = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/generate/text2video", { method: "POST", body: JSON.stringify(params) });

export const generateImage2Video = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/generate/image2video", { method: "POST", body: JSON.stringify(params) });

export const generateStory = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/generate/story", { method: "POST", body: JSON.stringify(params) });

// Audio
export const generateTTS = (params: { text: string; voice?: string; engine?: string }) =>
  apiFetch<Job>("/api/v1/audio/tts", { method: "POST", body: JSON.stringify(params) });

// Editing
export const trimVideo = (params: { video_path: string; start_time: number; end_time: number }) =>
  apiFetch<Job>("/api/v1/edit/trim", { method: "POST", body: JSON.stringify(params) });

export const mergeVideos = (params: { video_paths: string[]; transition?: string }) =>
  apiFetch<Job>("/api/v1/edit/merge", { method: "POST", body: JSON.stringify(params) });

export const upscaleVideo = (params: { video_path: string; scale?: number }) =>
  apiFetch<Job>("/api/v1/edit/upscale", { method: "POST", body: JSON.stringify(params) });

// Download
export const getDownloadUrl = (jobId: string) => `${API_BASE}/api/v1/jobs/${jobId}/download`;
