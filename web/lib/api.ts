const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// The API token is either baked in at build time (NEXT_PUBLIC_API_TOKEN)
// or read from browser storage. When both are absent, requests still go
// out headerless — the server will 401 if auth is enabled, and the UI
// can then surface the auth-status probe to ask the user for it.
const _buildToken =
  typeof process !== "undefined" && process.env
    ? process.env.NEXT_PUBLIC_API_TOKEN || ""
    : "";
const TOKEN_STORAGE_KEY = "studiolite.api_token";

export function getApiToken(): string {
  if (_buildToken) return _buildToken;
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setApiToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    if (token) window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // Storage failed (private mode, quota) — the caller can retry.
  }
}

export interface AuthStatus {
  auth_enabled: boolean;
  auth_file_hint?: string | null;
  header: string;
  ws_query: string;
}

export const getAuthStatus = () => apiFetch<AuthStatus>("/api/v1/system/auth-status");

export interface Job {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
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
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options?.headers as Record<string, string> | undefined),
  };
  const token = getApiToken();
  if (token && !headers["X-StudioLite-Token"]) {
    headers["X-StudioLite-Token"] = token;
  }
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
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
export interface JobListEntry {
  job_id: string;
  kind: string;
  status: Job["status"];
  progress: number;
  message: string;
  created_at: number;
  elapsed: number;
}
export interface JobList {
  jobs: JobListEntry[];
  total: number;
}
export const getJob = (id: string) => apiFetch<Job>(`/api/v1/jobs/${id}`);
export const listJobs = (limit = 50) => apiFetch<JobList>(`/api/v1/jobs?limit=${limit}`);
export const cancelJob = (id: string) =>
  apiFetch<Job>(`/api/v1/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
export const jobDownloadUrl = (id: string) => `${API_BASE}/api/v1/jobs/${encodeURIComponent(id)}/download`;

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

// Uploads go through a bespoke helper because `apiFetch` sets a JSON
// content-type header that clobbers the multipart boundary the browser
// picks. Still inject the auth token if we have one.
async function _uploadFile<T>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append("file", file);
  const headers: Record<string, string> = {};
  const token = getApiToken();
  if (token) headers["X-StudioLite-Token"] = token;
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", headers, body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export const uploadEditVideo = (file: File) =>
  _uploadFile<EditUploadResponse>("/api/v1/edit/upload", file);

export const uploadEditAudio = (file: File) =>
  _uploadFile<EditAudioUploadResponse>("/api/v1/edit/upload-audio", file);

export const upscaleVideo = (params: { video_path: string; scale: 1 | 2 | 3 | 4; preset: string }) =>
  apiFetch<Job>("/api/v1/edit/upscale", { method: "POST", body: JSON.stringify(params) });

// Keyframe animation ------------------------------------------------------

export interface ImageUploadResponse {
  image_path: string;
  url: string;
  size_bytes: number;
}

export const uploadImage = (file: File) =>
  _uploadFile<ImageUploadResponse>("/api/v1/images/upload", file);

export const animateKeyframes = (params: {
  start_image_path: string;
  end_image_path: string;
  frames: number;
  fps: number;
  easing: string;
  method: string;
}) => apiFetch<Job>("/api/v1/edit/keyframe-animate", {
  method: "POST", body: JSON.stringify(params),
});

// Timeline / NLE ---------------------------------------------------------

export interface TimelineClipInput {
  video_path: string;
  in_point: number;
  out_point: number;
}

export interface TimelineRenderParams {
  clips: TimelineClipInput[];
  fps: number;
  width: number;
  height: number;
  codec: "h264" | "h265" | "prores";
  quality: "high" | "medium" | "low";
  apply_watermark: boolean;
}

export const renderTimeline = (params: TimelineRenderParams) =>
  apiFetch<Job>("/api/v1/edit/timeline-render", {
    method: "POST", body: JSON.stringify(params),
  });

// Licensing --------------------------------------------------------------

export interface LicenseStatus {
  valid: boolean;
  tier: "free" | "pro" | "studio" | string;
  features: string[];
  licensee: string;
  expires_at: number;
  reason: string;
  warning: string;
  device_fingerprint: string;
}

export const getLicenseStatus = () =>
  apiFetch<LicenseStatus>("/api/v1/system/license");

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

// uploadImage lives up in the edit section — same helper handles both.

// ---------------------------------------------------------------------------
// Film Studio
// ---------------------------------------------------------------------------

export type FilmStageKey =
  | "producer" | "screenwriter" | "story_editor" | "breakdown"
  | "storyboard" | "cinematographer" | "shots" | "editor";

export type FilmStageStatus =
  | "pending" | "running" | "paused" | "done"
  | "failed" | "stale" | "needs_review";

export interface FilmStageSpec {
  key: FilmStageKey;
  label: string;
  description: string;
  gated_by_default: boolean;
}

export interface FilmConfig {
  llm_backend: string;
  llm_model: string;
  llm_host: string | null;
  style: "stylized" | "photoreal";
  target_minutes: number;
  quality: "draft" | "standard" | "high" | "ultra";
  per_stage: Record<string, Record<string, unknown>>;
}

export interface FilmMeta {
  id: string;
  title: string;
  brief: string;
  created_at: number;
  updated_at: number;
  config: FilmConfig;
}

export interface FilmState {
  stage_status: Record<FilmStageKey, FilmStageStatus>;
  current_stage: FilmStageKey | null;
  is_paused: boolean;
  pause_requested: boolean;
  last_error: string | null;
  gates: Partial<Record<FilmStageKey, boolean>>;
  started_at: number | null;
  finished_at: number | null;
}

export interface FilmDetail {
  project: FilmMeta;
  state: FilmState;
  artifacts: Partial<Record<FilmStageKey, Record<string, unknown>>>;
  final_url: string | null;         // silent slideshow (editor stage output)
  final_mixed_url: string | null;   // mixed cut with dialogue + score (mixer stage output)
}

export interface FilmListItem {
  id: string;
  title: string;
  brief: string;
  created_at: number;
  updated_at: number;
}

export const filmListStages = () =>
  apiFetch<{ stages: FilmStageSpec[] }>("/api/v1/films/stages");

export const filmList = () =>
  apiFetch<{ projects: FilmListItem[] }>("/api/v1/films");

export const filmCreate = (params: { brief: string; title?: string; config?: Partial<FilmConfig> }) =>
  apiFetch<{ project: FilmMeta; state: FilmState }>("/api/v1/films", {
    method: "POST",
    body: JSON.stringify(params),
  });

export const filmGet = (id: string) =>
  apiFetch<FilmDetail>(`/api/v1/films/${encodeURIComponent(id)}`);

export const filmDelete = (id: string) =>
  apiFetch<{ deleted: string }>(`/api/v1/films/${encodeURIComponent(id)}`, { method: "DELETE" });

export const filmRun = (id: string) =>
  apiFetch<{ started: boolean; state: FilmState }>(
    `/api/v1/films/${encodeURIComponent(id)}/run`, { method: "POST" }
  );

export const filmPause = (id: string) =>
  apiFetch<{ pause_requested: boolean; state: FilmState }>(
    `/api/v1/films/${encodeURIComponent(id)}/pause`, { method: "POST" }
  );

export const filmRewind = (id: string, stage: FilmStageKey) =>
  apiFetch<{ rewound_to: FilmStageKey; state: FilmState }>(
    `/api/v1/films/${encodeURIComponent(id)}/rewind/${encodeURIComponent(stage)}`, { method: "POST" }
  );

export const filmEditArtifact = (id: string, stage: FilmStageKey, data: unknown) =>
  apiFetch<{ state: FilmState; artifact: unknown }>(
    `/api/v1/films/${encodeURIComponent(id)}/artifact/${encodeURIComponent(stage)}`,
    { method: "PUT", body: JSON.stringify({ data }) }
  );

export const filmSetGates = (id: string, gates: Partial<Record<FilmStageKey, boolean>>) =>
  apiFetch<{ gates: Record<string, boolean> }>(
    `/api/v1/films/${encodeURIComponent(id)}/gates`,
    { method: "POST", body: JSON.stringify({ gates }) }
  );

export function filmStreamUrl(id: string): string {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return `${wsBase}/api/v1/films/${encodeURIComponent(id)}/stream`;
}

export const FILM_API_BASE = API_BASE;

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

export interface LibraryStats {
  videos: number;
  roots: number;
  total_bytes: number;
  hashed: number;
  phashed: number;
}

export interface LibraryRoot {
  id: number;
  path: string;
  include_glob: string;
  exclude_glob: string;
  added_at: number;
  video_count: number;
}

export interface LibraryVideo {
  id: number;
  root_id: number | null;
  abs_path: string;
  rel_path: string;
  size_bytes: number;
  mtime: number;
  sha256: string | null;
  phash_hex: string | null;
  duration_sec: number | null;
  width: number | null;
  height: number | null;
  codec: string | null;
  fps: number | null;
  added_at: number;
  scanned_at: number | null;
  missing: boolean;
}

export interface LibraryCluster {
  kind: "exact" | "near";
  key: string;
  members: LibraryVideo[];
}

export interface LibraryDuplicates {
  exact_clusters: LibraryCluster[];
  near_clusters: LibraryCluster[];
  near_threshold: number;
  exact_saveable_bytes: number;
  near_saveable_bytes: number;
}

export interface LibraryScanResp {
  job_id: string;
  root_ids: number[];
}

export const libraryStats = () => apiFetch<LibraryStats>("/api/v1/library/stats");
export const libraryListRoots = () => apiFetch<{ roots: LibraryRoot[] }>("/api/v1/library/roots");
export const libraryAddRoot = (path: string, opts?: { include_glob?: string; exclude_glob?: string }) =>
  apiFetch<{ root_id: number; roots: LibraryRoot[] }>(
    "/api/v1/library/roots",
    { method: "POST", body: JSON.stringify({ path, ...opts }) },
  );
export const libraryAddRootsBatch = (paths: string[], opts?: { include_glob?: string; exclude_glob?: string }) =>
  apiFetch<{ added: { path: string; root_id: number }[]; skipped: { path: string; reason: string }[]; roots: LibraryRoot[] }>(
    "/api/v1/library/roots/batch",
    { method: "POST", body: JSON.stringify({ paths, ...opts }) },
  );
export const libraryDeleteRoot = (root_id: number) =>
  apiFetch<{ deleted: number; roots: LibraryRoot[] }>(
    `/api/v1/library/roots/${root_id}`, { method: "DELETE" },
  );

export const libraryListVideos = (params?: {
  limit?: number; offset?: number; root_id?: number; q?: string; include_missing?: boolean;
}) => {
  const qs = new URLSearchParams();
  if (params?.limit != null) qs.set("limit", String(params.limit));
  if (params?.offset != null) qs.set("offset", String(params.offset));
  if (params?.root_id != null) qs.set("root_id", String(params.root_id));
  if (params?.q) qs.set("q", params.q);
  if (params?.include_missing) qs.set("include_missing", "true");
  const suffix = qs.toString() ? `?${qs}` : "";
  return apiFetch<{ total: number; videos: LibraryVideo[] }>(`/api/v1/library/videos${suffix}`);
};

export const libraryDeleteVideo = (video_id: number, delete_file = false) =>
  apiFetch<{ deleted_index: boolean; file_deleted: boolean; file_error: string | null }>(
    `/api/v1/library/videos/${video_id}?delete_file=${delete_file}`, { method: "DELETE" },
  );

export const libraryDuplicates = (near_threshold = 8) =>
  apiFetch<LibraryDuplicates>(`/api/v1/library/duplicates?near_threshold=${near_threshold}`);

export type KeeperStrategy = "largest" | "smallest" | "oldest" | "newest" | "shortest_path";

export interface LibraryDeletionPlan {
  strategy: KeeperStrategy;
  near_threshold: number;
  clusters: Array<{
    kind: "exact" | "near";
    key: string;
    keeper: LibraryVideo;
    delete: LibraryVideo[];
    delete_bytes: number;
  }>;
  total_delete_files: number;
  total_delete_bytes: number;
  delete_ids: number[];
}

export const libraryDeletionPlan = (opts: {
  include_exact?: boolean;
  include_near?: boolean;
  near_threshold?: number;
  keeper_strategy?: KeeperStrategy;
  cluster_keeper_overrides?: Record<string, number>;
}) =>
  apiFetch<LibraryDeletionPlan>("/api/v1/library/duplicates/plan", {
    method: "POST",
    body: JSON.stringify({
      include_exact: opts.include_exact ?? true,
      include_near:  opts.include_near  ?? true,
      near_threshold: opts.near_threshold ?? 8,
      keeper_strategy: opts.keeper_strategy ?? "largest",
      cluster_keeper_overrides: opts.cluster_keeper_overrides ?? null,
    }),
  });

export interface LibraryBatchDeleteResult {
  results: Array<{
    id: number;
    abs_path?: string;
    status: "ok" | "failed" | "skipped";
    reason?: string;
    removed_index?: boolean;
    file_deleted?: boolean;
    file_error?: string | null;
  }>;
  files_deleted: number;
  bytes_freed: number;
}

export const libraryBatchDelete = (video_ids: number[], delete_file: boolean) =>
  apiFetch<LibraryBatchDeleteResult>("/api/v1/library/videos/batch-delete", {
    method: "POST",
    body: JSON.stringify({ video_ids, delete_file }),
  });

export const libraryStartScan = (root_ids?: number[]) =>
  apiFetch<LibraryScanResp>("/api/v1/library/scan", {
    method: "POST",
    body: JSON.stringify(root_ids && root_ids.length ? { root_ids } : {}),
  });

export function libraryThumbUrl(video_id: number): string {
  // Auth is enforced by middleware for /api routes. Token query param works
  // both for GETs and for the img src (browsers can't set custom headers there).
  const t = getApiToken();
  return `${API_BASE}/api/v1/library/videos/${video_id}/thumb${t ? `?token=${encodeURIComponent(t)}` : ""}`;
}

export function libraryStreamUrl(video_id: number): string {
  const t = getApiToken();
  return `${API_BASE}/api/v1/library/videos/${video_id}/stream${t ? `?token=${encodeURIComponent(t)}` : ""}`;
}

export const LIBRARY_API_BASE = API_BASE;
