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

export const generateSFX = (params: { sfx_type: string; duration?: number }) =>
  apiFetch<Job>("/api/v1/audio/sfx", { method: "POST", body: JSON.stringify(params) });

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

// Trigger a real file download in the browser, preserving panel state.
// The native `download` attribute on <a> is ignored cross-origin unless the
// server sends Content-Disposition: attachment, so we fetch as blob and
// click a synthetic anchor with an object URL — works for any URL the
// browser can fetch (CORS permitting).
export async function downloadBlob(url: string, suggestedName?: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = suggestedName || url.split(/[\\/?#]/).pop() || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

// Static assets
export const getPortraitUrl = (relativePath: string) => `${API_BASE}${relativePath}`;
export const getImageUrl = (relativePath: string) => `${API_BASE}${relativePath}`;

// Images Studio
export const getImagesCatalog = () => apiFetch<{
  providers: { id: string; name: string; description: string }[];
  models: { id: string; name: string; path: string | null }[];
  sizes: { id: string; name: string; width: number; height: number }[];
  styles: string[];
  fooocus_styles: string[];
  edit_techniques: { id: string; name: string; description: string }[];
  qwen_edit_available: boolean;
  default_negatives: Record<string, string>;
  rembg_available: boolean;
  realesrgan_available: boolean;
}>("/api/v1/images/catalog");

export const enhancePrompt = (params: {
  prompt: string;
  style?: string | null;
  mode?: string;
  image_path?: string | null;
  negative_prompt?: string | null;
}) =>
  apiFetch<{ original: string; enhanced: string; negative: string; image_caption: string; mode: string }>(
    "/api/v1/images/enhance-prompt",
    { method: "POST", body: JSON.stringify(params) },
  );

export const generateImage = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/images/generate", { method: "POST", body: JSON.stringify(params) });

export const editImage = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/images/edit", { method: "POST", body: JSON.stringify(params) });

export const inpaintImage = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/images/inpaint", { method: "POST", body: JSON.stringify(params) });

export const variationImage = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/images/variation", { method: "POST", body: JSON.stringify(params) });

export const upscaleImage = (params: Record<string, unknown>) =>
  apiFetch<Job>("/api/v1/images/upscale", { method: "POST", body: JSON.stringify(params) });

export const removeBgImage = (params: { image_path: string }) =>
  apiFetch<Job>("/api/v1/images/remove-bg", { method: "POST", body: JSON.stringify(params) });

export const getImagesHistory = (limit = 50) =>
  apiFetch<{ images: { filename: string; path: string; url: string; size_bytes: number; mtime: number }[] }>(
    `/api/v1/images/history?limit=${limit}`,
  );

export async function uploadImage(file: File): Promise<{ image_path: string; url: string; size_bytes: number }> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/images/upload`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}
