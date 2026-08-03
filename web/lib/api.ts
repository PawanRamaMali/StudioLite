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
    gpu_name?: string;
    total_vram_gb?: number;
    free_vram_gb?: number;
    cuda_version?: string;
    message?: string;
  };
  capabilities: {
    cuda: boolean;
    platform: "windows" | "linux" | "darwin" | string;
    features: {
      video_generation: boolean;
      qwen_edit: boolean;
      lipsync: boolean;
      local_sdxl: boolean;
      local_character_portrait: boolean;
    };
  };
  active_jobs: number;
  total_jobs: number;
  output_dir?: string;
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

// Models
export interface ModelInventoryItem {
  key: string;
  name: string;
  installed: boolean;
  vram_min: number;
  engine: string;
  modes: string[];
  quality: string;
  speed: string;
  built_in: boolean;
  active_job: string | null;
}
export const getModelInventory = () =>
  apiFetch<{ models: ModelInventoryItem[]; installed_count: number; total_count: number }>(
    "/api/v1/models/inventory"
  );

export const downloadModel = (modelKey: string) =>
  apiFetch<Job>(`/api/v1/models/${encodeURIComponent(modelKey)}/download`, { method: "POST" });

export const deleteModel = (modelKey: string) =>
  apiFetch<{ key: string; hf_id: string; freed_bytes: number; found: boolean }>(
    `/api/v1/models/${encodeURIComponent(modelKey)}`,
    { method: "DELETE" }
  );

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
export interface TTSParams {
  text: string;
  voice?: string;
  engine?: string;
  persona?: string;
  speed?: number;
  pitch?: number;
  volume?: number;
  output_format?: "wav" | "mp3";
}
export const generateTTS = (params: TTSParams) =>
  apiFetch<Job>("/api/v1/audio/tts", { method: "POST", body: JSON.stringify(params) });

export const generateSFX = (params: { sfx_type: string; prompt?: string; duration?: number }) =>
  apiFetch<Job>("/api/v1/audio/sfx", { method: "POST", body: JSON.stringify(params) });

export interface PersonaInfo {
  id: string;
  label: string;
  voice: string;
  speed: number;
  pitch: number;
  volume: number;
  description: string;
}
export interface VoiceInfo {
  id: string;
  model: string;
  quality: string;
  gender: string;
  accent: string;
}
export const getAudioVoices = () =>
  apiFetch<{ personas: PersonaInfo[]; voices: VoiceInfo[] }>("/api/v1/audio/voices");

// Multipart-upload helper for file-based audio endpoints
async function uploadAudio<T>(path: string, file: File, extraFormData?: Record<string, string>): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  if (extraFormData) {
    for (const [k, v] of Object.entries(extraFormData)) fd.append(k, v);
  }
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload error: ${res.status}`);
  }
  return res.json();
}
export const isolateVoice = (file: File) => uploadAudio<Job>("/api/v1/audio/isolate", file);
export const normalizeAudio = (file: File, target_db: number) =>
  uploadAudio<Job>(`/api/v1/audio/normalize?target_db=${target_db}`, file);

// Video edit utilities
export interface EditUploadResponse {
  video_path: string;
  filename: string;
  size_bytes: number;
}

export interface EditAudioUploadResponse {
  audio_path: string;
  filename: string;
  size_bytes: number;
}

export async function uploadEditVideo(file: File): Promise<EditUploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/edit/upload`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function uploadEditAudio(file: File): Promise<EditAudioUploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/edit/upload-audio`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export const extractVideoAudio = (params: { video_path: string; format: "wav" | "mp3" }) =>
  apiFetch<Job>("/api/v1/edit/extract-audio", { method: "POST", body: JSON.stringify(params) });

export const removeVideoAudio = (params: { video_path: string }) =>
  apiFetch<Job>("/api/v1/edit/remove-audio", { method: "POST", body: JSON.stringify(params) });

export const compressVideo = (params: { video_path: string; preset: "high" | "medium" | "low" }) =>
  apiFetch<Job>("/api/v1/edit/compress", { method: "POST", body: JSON.stringify(params) });

export const rotateFlipVideo = (params: {
  video_path: string;
  rotate: 0 | 90 | 180 | 270;
  flip: "none" | "horizontal" | "vertical";
}) => apiFetch<Job>("/api/v1/edit/rotate-flip", { method: "POST", body: JSON.stringify(params) });

export const reverseVideo = (params: { video_path: string; include_audio: boolean }) =>
  apiFetch<Job>("/api/v1/edit/reverse", { method: "POST", body: JSON.stringify(params) });

export const loopVideo = (params: { video_path: string; count: number }) =>
  apiFetch<Job>("/api/v1/edit/loop", { method: "POST", body: JSON.stringify(params) });

export const stabilizeVideo = (params: { video_path: string; shakiness: number; accuracy: number }) =>
  apiFetch<Job>("/api/v1/edit/stabilize", { method: "POST", body: JSON.stringify(params) });

export const colorCorrectVideo = (params: {
  video_path: string;
  brightness: number;
  contrast: number;
  saturation: number;
  hue: number;
}) => apiFetch<Job>("/api/v1/edit/color-correction", { method: "POST", body: JSON.stringify(params) });

export const regionEffectVideo = (params: {
  video_path: string;
  x: number;
  y: number;
  width: number;
  height: number;
  mode: "blur" | "pixelate";
  strength: number;
}) => apiFetch<Job>("/api/v1/edit/region-effect", { method: "POST", body: JSON.stringify(params) });

export const pictureInPictureVideo = (params: {
  video_path: string;
  overlay_path: string;
  position: "top_left" | "top_right" | "bottom_left" | "bottom_right" | "center";
  size_percent: number;
  margin: number;
}) => apiFetch<Job>("/api/v1/edit/picture-in-picture", { method: "POST", body: JSON.stringify(params) });

export const backgroundMusicVideo = (params: {
  video_path: string;
  audio_path: string;
  music_volume: number;
  video_volume: number;
}) => apiFetch<Job>("/api/v1/edit/background-music", { method: "POST", body: JSON.stringify(params) });

export const speedVideo = (params: { video_path: string; speed: number; keep_audio: boolean }) =>
  apiFetch<Job>("/api/v1/edit/speed", { method: "POST", body: JSON.stringify(params) });

export const gifVideo = (params: {
  video_path: string;
  start_time: number;
  duration: number;
  fps: number;
  width: number;
}) => apiFetch<Job>("/api/v1/edit/gif", { method: "POST", body: JSON.stringify(params) });

export const thumbnailVideo = (params: { video_path: string; timestamp: number; width: number }) =>
  apiFetch<Job>("/api/v1/edit/thumbnail", { method: "POST", body: JSON.stringify(params) });

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
