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
  status: string;
  gpu: {
    available: boolean;
    gpu_name: string;
    total_vram_gb: number;
    free_vram_gb: number;
    cuda_version: string;
  };
  active_jobs: number;
  total_jobs: number;
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
export const generateText2Video = (params: {
  prompt: string;
  negative_prompt?: string;
  engine?: string;
  model?: string;
  num_frames?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  fps?: number;
  seed?: number | null;
}) => apiFetch<Job>("/api/v1/generate/text2video", { method: "POST", body: JSON.stringify(params) });

export const generateStory = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/generate/story", { method: "POST", body: JSON.stringify(params) });

// Audio
export const generateTTS = (params: { text: string; voice?: string; engine?: string }) =>
  apiFetch<Job>("/api/v1/audio/tts", { method: "POST", body: JSON.stringify(params) });

// Characters
export const generateCharacterPortrait = (params: {
  name: string;
  description: string;
  visual_prompt?: string;
  views?: string[];
  style?: string;
  register_ip_adapter?: boolean;
}) => apiFetch<Job>("/api/v1/characters/generate-portrait", { method: "POST", body: JSON.stringify(params) });

export const getCharacterPortraits = (charName: string) =>
  apiFetch<{ character: string; portraits: { filename: string; url: string; view: string }[] }>(
    `/api/v1/characters/portraits/${encodeURIComponent(charName)}`
  );

// Download
export const getDownloadUrl = (jobId: string) => `${API_BASE}/api/v1/jobs/${jobId}/download`;

// Static assets
export const getPortraitUrl = (relativePath: string) => `${API_BASE}${relativePath}`;
