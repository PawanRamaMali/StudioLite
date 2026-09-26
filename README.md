# StudioLite

[![CI](https://github.com/PawanRamaMali/StudioLite/actions/workflows/ci.yml/badge.svg)](https://github.com/PawanRamaMali/StudioLite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Runs locally](https://img.shields.io/badge/models-local%20%2F%20offline-brightgreen.svg)](#installation)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

## About

**StudioLite** is a local-first, open-source AI film studio. Give it a one-paragraph brief and it drives an 18-stage pipeline (producer, screenwriter, cinematographer, shot generator, motion, voice, composer, editor) to a finished short film on your own hardware. When you would rather work directly, there is also a timeline editor, a media library with semantic search and deduplication, an image studio, and a real-time transcription suite. The pipeline and the direct tools share the same backend, so a shot rendered in the pipeline can be trimmed on the timeline without leaving the app.

Everything runs offline against local models. Text: Ollama, Gemini, Groq, Hugging Face. Stills: SDXL family. Motion: Wan 2.2, HunyuanVideo, LTX-Video, AnimateDiff. Voice: IndexTTS-2, XTTS, Piper. Music: MusicGen, ACE-Step. Your brief, your characters, and your renders never leave the machine. There is no cloud account to sign up for and nothing calls home.

MIT licensed. Ships a Next.js UI on top of a FastAPI backend, with a Windows setup package for one-click install.

## Table of contents

- [Features](#features)
- [How it compares](#how-it-compares)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Windows setup package](#windows-setup-package)
- [Licensing model](#licensing-model)
- [Contributing](#contributing)
- [License](#license)

## Features

### Film Studio pipeline

18 stages, one artifact per stage, versioned on every write, editable in the panel:

1. **Producer**. Expands the brief into a pitch.
2. **Beats**. Three-act beat sheet.
3. **Outline**. Scene-by-scene structure.
4. **Cast**. Named characters with visual descriptions.
5. **Screenwriter**. Full script with dialogue and action.
6. **Cinematographer**. Per-scene visual language and shot list.
7. **Breakdown**. Shots with prompts, timing, camera notes.
8. **Character portraits**. SDXL front, three-quarter, side, and back sheets.
9. **Shot generator**. SDXL keyframes with adaptive OOM-halving batch sizing.
10. **Motion shots**. Wan 2.2, HunyuanVideo, LTX-Video, AnimateDiff, or SVD I2V clips.
11. **Continuity**. Cross-shot consistency verifier.
12. **Voice actor**. Per-character narration via IndexTTS-2, XTTS-v2, Qwen3-TTS, Chatterbox, or Piper.
13. **Composer**. MusicGen or ACE-Step score.
14. **SFX**. Procedural or generated sound effects.
15. **Editor**. Cuts the film to the chosen pacing.
16. **Mixer**. Ducks dialogue against music and SFX.
17. **Upscale**. Optional Real-ESRGAN 2x or 4x pass.
18. **Delivery**. Final mixed cut, ready to hand off.

Five starter templates ship in-box (short story, explainer, teaser trailer, product demo, draft scene) so you never open a blank config.

### Timeline editor

Multi-clip NLE-style timeline. Trim in and out per clip, reorder, export with codec presets: H.264, H.265, or ProRes at high, medium, or low quality. Free-tier renders carry a StudioLite watermark; a Pro or Studio license removes it, and the server verifies the license before honoring the client's request.

### Batch render

Queue N timeline exports as a single batch. Individual jobs stay cancellable and retryable, and the rollup endpoint reports done, in-flight, and per-status counts so a "render twelve shots overnight" workflow is one thing to watch, not twelve.

### Media Library

Point it at folders, and it walks them for videos and images. Then:

- Perceptual-hash duplicate detection with a review-and-delete plan.
- Semantic search using CLIP embeddings.
- Cluster view for visual grouping.
- Speech-to-text index with FTS5 full-text search.
- Batch re-encode of legacy codecs (mpeg2, wmv, rmvb, dv) to H.264.
- Per-video enhancement recommendations.

### Video Generator, Story Mode, Images Studio

Direct panels for one-shot diffusion runs when the full pipeline is more machinery than the job needs: T2V, I2V, multi-scene Story Mode, and an image studio for T2I, edit, inpaint, upscale, and background removal.

### Video Editor

Trim, merge, compress, rotate, stabilize, color-correct, region-effect, picture-in-picture, background music mix, speed change, GIF export, thumbnail extraction. All ffmpeg-backed and queued as background jobs with progress polling.

### Transcription

WhisperX and faster-whisper on files. Live mic and desktop-audio capture. Live screen OCR with RapidOCR. Optional LLM cleanup into Markdown, DOCX, or PDF.

### Delivery and portable projects

One click packages a project into a shippable zip: the final mixed cut, a CREDITS.md, and a JSON manifest recording tier, watermark state, and render settings. Alongside that, a portable `.studioproj` export bundles the whole project (artifacts, logs, state) so another machine can import it and continue where you left off.

### Jobs, licensing, telemetry

Every background runner is cancellable and retryable. Job state persists in SQLite, so a restart never loses in-flight work. Licensing is offline and Ed25519-signed, with per-key feature gating and a grace period for expired keys. Telemetry is off by default; when you turn it on, it writes a redacted diagnostic bundle you export by hand, and nothing ships without consent.

### Windows installer

Source-based IExpress setup with optional Authenticode signing (opt-in via env var). Every build writes a JSON release manifest with size, SHA-256, version, and signing state, so a future auto-updater has a canonical file to read.

## How it compares

StudioLite covers a range that today usually costs three or four separate SaaS subscriptions to cover. The tables below are honest: the commercial services do many things better, and the "where StudioLite trails" section calls out where.

### Cloud AI video generators

| Capability | StudioLite | RunwayML | Pika | Luma Dream Machine | OpenAI Sora |
|---|---|---|---|---|---|
| Text-to-video | Wan 2.2, LTX, AnimateDiff (local) | Gen-4 (cloud) | Pika 2.0 (cloud) | Ray 2 (cloud) | Sora (cloud, waitlisted) |
| Image-to-video | Wan I2V, SVD (local) | Gen-3 I2V | Yes | Yes | Yes |
| Motion fidelity | Good, backend-dependent | Strong | Strong | Strong | State of the art |
| Runs offline | Yes | No | No | No | No |
| Per-render cost | Your electricity | ~$0.05 / second | Credit tiers | Credit tiers | Credit tiers |
| Multi-scene film from a brief | Yes, 18-stage pipeline | No | No | No | No |
| Own the outputs | Yes, on disk | ToS-bound | ToS-bound | ToS-bound | ToS-bound |
| Watermark on free tier | Yes, license removes it | Yes | Yes | Yes | N/A |
| Open source | Yes (MIT) | No | No | No | No |
| Hardware | 12-24 GB VRAM (NVIDIA) | None | None | None | None |
| Batch export | Yes | Limited | Limited | Limited | Limited |

### Commercial NLEs

| Capability | StudioLite Timeline | Adobe Premiere Pro | DaVinci Resolve | Final Cut Pro |
|---|---|---|---|---|
| Multi-clip timeline | Yes (v1) | Full | Full | Full |
| Per-clip trim | Yes | Yes | Yes | Yes |
| Codec presets (H.264 / H.265 / ProRes) | Yes | Yes | Yes | Yes |
| Multi-track audio | Not yet | Yes | Yes | Yes |
| Color grading | Basic | Lumetri | Industry standard | Yes |
| Motion graphics | No | After Effects | Fusion | Motion |
| AI-generated shots as source clips | Native | Requires plugins | Requires plugins | Requires plugins |
| Runs offline | Yes | Yes | Yes | Yes |
| Cost | Free (MIT) | ~$23 / month | Free / $295 Studio | $300 one-time |
| Platforms | Windows, Linux | Windows, macOS | Windows, macOS, Linux | macOS only |
| Open source | Yes | No | No | No |

StudioLite ships a v1 timeline on purpose. If you need finishing-grade color or multi-track audio, run DaVinci Resolve on the exported cut. StudioLite is best at the stage before finishing: turning ideas into shots and stitching them together.

### Specialized AI tools

| Capability | StudioLite | Descript | HeyGen | ElevenLabs | Suno |
|---|---|---|---|---|---|
| Voice cloning | IndexTTS-2, XTTS-v2 (local) | Yes (cloud) | Yes (cloud) | Category leader (cloud) | N/A |
| Voice styling | Emotion via IndexTTS-2 | Overdub | Avatars | Extensive | N/A |
| Music generation | MusicGen, ACE-Step (local) | Stock library | Stock library | N/A | Category leader (cloud) |
| Transcription | WhisperX, faster-whisper | Yes | N/A | Yes | N/A |
| Talking-head avatars | Character portraits (lipsync not wired yet) | No | Category leader | No | No |
| Script, voice, edit in one product | Yes | Yes | Partial | Voice only | Music only |
| Per-project cost | Zero | $12-24 / month | $24-89 / month | $5-330 / month | $8-24 / month |
| Own the outputs | Yes | Yes with limits | ToS-bound | Yes with limits | ToS-bound |
| Open source | Yes | No | No | No | No |

### Where StudioLite wins and where it trails

**Wins**

- Local and private. Your brief, your characters, and your renders never leave the box.
- One product covers what today needs three or four subscriptions stitched together.
- Zero marginal cost. Once the hardware is paid for, iteration is free.
- Every model is swappable. You can pick faster, cheaper, or higher-quality per stage.
- MIT licensed. Telemetry is off by default and requires an explicit opt-in.

**Trails**

- Motion fidelity trails Sora and Runway Gen-4 by roughly a model generation.
- No real-time collaboration and no cloud project sharing beyond the portable export zip.
- Lipsync is not wired end-to-end yet.
- Timeline is v1: no multi-track audio, no compositor, no proxy workflow.
- Rendering is not real-time. A one-minute short takes minutes to hours depending on hardware and settings.
- The interesting bits assume a modern NVIDIA GPU. CPU-only mode is honest about what it can and cannot do.

## Architecture

```
                                 ┌──────────────┐
                                 │   Next.js    │  :3000
                                 │  (React UI)  │
                                 └──────┬───────┘
                                        │ REST + WebSocket
                                 ┌──────▼───────┐
                                 │   FastAPI    │  :8000
                                 │ api_server.py│
                                 └──────┬───────┘
             ┌──────────────┬──────────┼──────────────┬─────────────────┐
             │              │          │              │                 │
      ┌──────▼─────┐ ┌──────▼─────┐ ┌──▼──────────┐ ┌─▼─────────┐ ┌────▼──────┐
      │ filmmaker  │ │  library   │ │ api/routers │ │  models   │ │  jobs +   │
      │ (pipeline) │ │ (media db) │ │ (extracted) │ │ (SDXL /   │ │  license  │
      │  18 stages │ │ FTS5 + CLIP│ │             │ │ Wan / TTS)│ │  SQLite   │
      └────────────┘ └────────────┘ └─────────────┘ └───────────┘ └───────────┘
```

Runtime output lives under `.mp/` (git-ignored). Every long-running task is a job, every job reports progress, is cancellable, is retryable, and survives a process restart.

## Installation

### Prerequisites

- Python 3.11
- Node.js 22
- FFmpeg on PATH
- For the GPU-heavy features, an NVIDIA GPU with 12 GB or more of VRAM

Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg imagemagick python3-venv python3-dev build-essential
```

Windows:

```powershell
winget install Gyan.FFmpeg
winget install ImageMagick.ImageMagick
```

macOS:

```bash
brew install ffmpeg imagemagick
```

### Set up the app

```bash
git clone https://github.com/PawanRamaMali/StudioLite.git
cd StudioLite

python -m venv venv
# Windows: .\venv\Scripts\activate
source venv/bin/activate

# Pick one based on hardware:
pip install -r requirements-cuda.txt   # NVIDIA GPU
# or
pip install -r requirements-cpu.txt    # CPU-only

pip install -r requirements.txt

# Front-end
cd web && npm install && npm run build && cd ..
```

### Docker

```bash
docker compose --profile cuda up   # NVIDIA
docker compose --profile cpu  up   # CPU-only
```

Both profiles publish `:3000` (Next.js) and `:8000` (FastAPI). Model weights persist under bind-mounted `.models/` and `.mp/` volumes.

## Usage

Start everything with the one-click launcher:

```bash
# Linux / macOS
./launch.sh
# Windows
.\launch.ps1        # or double-click launch.bat
```

The launcher boots FastAPI on `:8000` and Next.js on `:3000`, waits for both to be reachable, and opens the browser. Ctrl+C cleanly stops both.

Manual start:

```bash
python api_server.py                           # backend on :8000
cd web && npm run dev                          # frontend on :3000
```

### Your first film

1. Open the **Film Studio** panel.
2. Pick a template. "Short Story" is a good first pick.
3. Edit the sample brief, or replace it with your own paragraph.
4. Click **Create**.
5. Click **Run**. Watch the pipeline advance stage-by-stage in the event stream.
6. Edit any stage's artifact between runs. Downstream stages automatically mark themselves stale so a re-run only touches the affected shots.
7. When it finishes, click **Package** to get a delivery zip.

## Configuration

Copy `config.example.json` to `config.json` (git-ignored) and edit.

Environment variables (also editable from the Settings panel):

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token for gated models |
| `HF_HOME` | Custom cache dir for HF weights |
| `NEXT_PUBLIC_API_URL` | Backend URL used by the Next.js UI |
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to expose |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory allocator tuning |
| `STUDIOLITE_AUTH` | `on` or `off` for API bearer auth |
| `STUDIOLITE_LICENSE_FILE` | Custom path for the offline license |
| `GEMINI_API_KEY`, `GROQ_API_KEY` | Cloud LLM backends (optional) |

## Windows setup package

`packaging/windows/build.py` builds a source-based IExpress installer:

```powershell
python packaging/windows/build.py
```

Outputs land under `dist/`: `StudioLite-Setup.exe`, a fallback zip, SHA-256 checksums, and `release-manifest.json` describing the build.

**Signing is opt-in.** Set `STUDIOLITE_SIGN_THUMBPRINT` (SHA-1 of the code-signing cert in your user store), optionally `STUDIOLITE_SIGN_TSA` (defaults to DigiCert), and optionally `STUDIOLITE_SIGNTOOL` (full path). If any of those are missing, the build finishes unsigned and records the reason in the manifest. See `packaging/windows/README.md` for the full release procedure.

## Licensing model

The code is MIT. The **product** ships a two-tier entitlement layer for feature gating:

| Tier | What you get | Watermark on export |
|---|---|---|
| Free | Full pipeline, all editors, all tools | Yes |
| Pro / Studio | Same, plus watermark removal and the `watermark_removal` feature flag | No |

Licenses are Ed25519-signed JSON payloads carrying tier, features, optional device fingerprint, expiry, and grace-days. Verification is offline. Nothing calls home.

For your own installs, the free tier is unlimited. For third-party distribution (packaging StudioLite as part of a product), the licensing scaffold gives you a clean way to gate features.

## Contributing

- Open an issue for anything that surprised you, especially wrong outputs.
- PRs welcome. Keep changes focused. New features should ship with tests under `tests/`.
- The test suite is 162 tests today and CI is green on both Ubuntu and Windows. Please keep both true.

## License

MIT for the source. See `LICENSE`.

Third-party dependencies, bundled fonts, and model weights each carry their own licenses. See `THIRD_PARTY_NOTICES.md` for the full list. Note that some model weights (Stable Diffusion XL, HunyuanVideo, LTX-Video, CogVideoX 5B) have restrictions on commercial use that supersede StudioLite's MIT grant.
