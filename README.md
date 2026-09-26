# StudioLite

[![CI](https://github.com/PawanRamaMali/StudioLite/actions/workflows/ci.yml/badge.svg)](https://github.com/PawanRamaMali/StudioLite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Runs locally](https://img.shields.io/badge/models-local%20%2F%20offline-brightgreen.svg)](#installation)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

## About

**StudioLite** is a local-first, open-source AI film studio. Write a one-paragraph brief, pick a template, and it runs an 18-stage pipeline (Producer, Screenwriter, Cinematographer, Shot Generator, Motion, Voice, Composer, Editor) into a finished short film on your own hardware. When you want to work directly instead of driving the pipeline, it also ships an NLE-style timeline editor, a media library with semantic search and deduplication, an image studio, and a real-time transcription suite.

Everything runs offline against local models: Ollama, Gemini, Groq, or Hugging Face for text; SDXL family for stills; Wan 2.2, HunyuanVideo, LTX-Video, AnimateDiff for motion; IndexTTS-2, XTTS, Piper for voice. Cloud accounts are never required. Your project data, drafts, characters, and renders never leave the machine.

MIT licensed. Two front-end surfaces sit on top of the same FastAPI engine.

## Table of contents

- [Features](#features)
- [Comparison with commercial tools](#comparison-with-commercial-tools)
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

1. **Producer**: expands the one-paragraph brief into a pitch.
2. **Beats**: three-act beat sheet.
3. **Outline**: scene-by-scene structure.
4. **Cast**: named characters with visual descriptions.
5. **Screenwriter**: full script with dialogue and action.
6. **Cinematographer**: per-scene visual language and shot list.
7. **Breakdown**: shots with prompts, timing, camera notes.
8. **Character portraits**: SDXL front / three-quarter / side / back reference sheets.
9. **Shot generator**: SDXL keyframes with adaptive OOM-halving batch sizing.
10. **Motion shots**: Wan 2.2, HunyuanVideo, LTX-Video, AnimateDiff, or SVD I2V clips.
11. **Continuity**: cross-shot consistency verifier.
12. **Voice actor**: per-character narration via IndexTTS-2, XTTS-v2, Qwen3-TTS, Chatterbox, or Piper.
13. **Composer**: MusicGen or ACE-Step score.
14. **SFX**: procedural or generated sound effects.
15. **Editor**: cut the film with the chosen pacing.
16. **Mixer**: ducked dialogue against music and SFX.
17. **Upscale**: optional Real-ESRGAN 2x or 4x neural pass.
18. **Delivery**: final mixed cut ready to ship.

Five starter templates (short story, explainer, teaser trailer, product demo, draft scene) so you never start from a blank config.

### Timeline editor

NLE-style multi-clip timeline. Trim in/out per clip, reorder, and export with codec presets: H.264, H.265, or ProRes at high / medium / low quality. Free-tier renders carry a StudioLite watermark. A Pro or Studio license removes it, and the server double-checks the license before honoring the client's request.

### Batch render

Queue N timeline exports as one batch. Individual jobs remain cancellable and retryable. One rollup endpoint reports done / in-flight / counts so you have one thing to watch instead of twelve.

### Media Library

Point it at folders, and it walks them for videos and images. Then:

- Perceptual-hash duplicate detection with review-and-delete plans.
- Semantic search with CLIP embeddings.
- Cluster view for visual grouping.
- Speech-to-text index with FTS5 full-text search.
- Batch re-encode of legacy codecs (mpeg2, wmv, rmvb, dv) to H.264.
- Per-video enhancement recommendations.

### Video Generator, Story Mode, Images Studio

Direct panels for one-shot diffusion runs when you do not need the full pipeline: T2V, I2V, multi-scene Story Mode, and an image studio for T2I, edit, inpaint, upscale, and background removal.

### Video Editor

Trim, merge, compress, rotate, stabilize, color-correct, region-effect, picture-in-picture, background music mix, speed change, GIF export, thumbnail extraction. All ffmpeg-backed; all queued as background jobs with progress polling.

### Transcription

WhisperX and faster-whisper on files. Live mic and desktop-audio transcription. Live screen OCR with RapidOCR. Optional LLM clean-up into Markdown, DOCX, or PDF.

### Delivery packaging and project export

One-click package produces a shippable zip: final mixed cut, CREDITS.md, and a JSON manifest with render metadata. The tier and watermark state are recorded so the recipient can see how it was made. A portable `.studioproj` export bundles the whole project (artifacts, logs, state) so another machine can import it and continue.

### Jobs, licensing, telemetry

- Every background runner is cancellable and retryable. Job state persists in SQLite so a restart never loses in-flight work.
- Offline signed licensing with Ed25519. Feature gating with per-key entitlements. Grace-period support for expired keys.
- Opt-in local telemetry writes a redacted diagnostic bundle you export by hand. Nothing ships without consent.

### Windows installer

Source-based IExpress setup with optional Authenticode signing (opt-in via env var). Every build writes a JSON release manifest with size, SHA-256, version, and signing state so a future auto-updater has a canonical file to read.

## Comparison with commercial tools

StudioLite covers ground that today usually requires stringing together three or four separate SaaS products. The tables below show how it lines up. Ratings are fair-witness: the commercial services do many things better, and StudioLite calls those out honestly.

### vs cloud AI video generators

| Capability | StudioLite | RunwayML | Pika | Luma Dream Machine | OpenAI Sora |
|---|---|---|---|---|---|
| Text-to-video | Wan 2.2, LTX, AnimateDiff (local) | Gen-4 (SaaS) | Pika 2.0 (SaaS) | Ray 2 (SaaS) | Sora (SaaS, waitlisted) |
| Image-to-video | Wan I2V, SVD (local) | Gen-3 I2V | Yes | Yes | Yes |
| Motion quality | Good, model-dependent | Excellent | Very good | Very good | Best-in-class |
| Runs offline | Yes | No | No | No | No |
| Per-render cost | Zero (your electricity) | ~$0.05/s | Credit tiers | Credit tiers | Credit tiers |
| Multi-scene film from a brief | Yes (18-stage pipeline) | No | No | No | No |
| Own the outputs | Yes, on disk | ToS-bound | ToS-bound | ToS-bound | ToS-bound |
| Watermark on free tier | Yes, removable with license | Yes | Yes | Yes | N/A |
| Open source | Yes (MIT) | No | No | No | No |
| Hardware needed | 12-24 GB VRAM (NVIDIA) | None | None | None | None |
| Batch export | Yes | Limited | Limited | Limited | Limited |

### vs commercial NLEs

| Capability | StudioLite Timeline | Adobe Premiere Pro | DaVinci Resolve | Final Cut Pro |
|---|---|---|---|---|
| Multi-clip timeline | Yes (v1) | Full | Full | Full |
| Trim per clip | Yes | Yes | Yes | Yes |
| Codec presets (H.264/H.265/ProRes) | Yes | Yes | Yes | Yes |
| Multi-track audio | Not yet | Yes | Yes | Yes |
| Color grading | Basic (color-correct utility) | Lumetri | Best-in-class | Yes |
| Motion graphics | No | Yes (After Effects) | Fusion | Motion |
| AI-generated shots as source clips | Yes, native pipeline | Requires plugins | Requires plugins | Requires plugins |
| Runs offline | Yes | Yes | Yes | Yes |
| Cost | Free (MIT) | ~$23/mo | Free / $295 (Studio) | $300 (one-time) |
| Platform | Windows, Linux | Windows, macOS | Windows, macOS, Linux | macOS only |
| Open source | Yes | No | No | No |

StudioLite is intentionally a v1 timeline. If you need finishing-grade color and multi-track audio, run DaVinci Resolve on the exported cut. StudioLite is best at the stage before finishing: turning ideas into shots and stitching them together.

### vs specialized AI tools

| Capability | StudioLite | Descript | HeyGen | ElevenLabs | Suno |
|---|---|---|---|---|---|
| Voice cloning | IndexTTS-2, XTTS-v2 (local) | Yes (cloud) | Yes (cloud) | Best-in-class (cloud) | N/A |
| Voice styling | Emotion via IndexTTS-2 | Overdub | Avatars | Extensive | N/A |
| Music generation | MusicGen, ACE-Step (local) | Stock library | Stock library | N/A | Best-in-class (cloud) |
| Transcription | WhisperX, faster-whisper | Yes | N/A | Yes | N/A |
| Talking-head avatars | Character portraits, no lipsync yet | No | Best-in-class | No | No |
| Script + voice + edit in one product | Yes | Yes | Partial | Voice only | Music only |
| Per-project cost | Zero | $12-24/mo | $24-89/mo | $5-330/mo | $8-24/mo |
| Own the outputs | Yes | Yes with limits | ToS-bound | Yes with limits | ToS-bound |
| Open source | Yes | No | No | No | No |

### Where StudioLite wins and where it does not

**Wins:**

- Local and private. Your brief, your characters, your renders never leave the box.
- One product covers what usually needs three or four subscriptions.
- Zero marginal cost. Once the hardware is paid for, iteration is free.
- Every model is swappable, so you can pick faster / cheaper / higher-quality per stage.
- MIT licensed with no telemetry unless you opt in.

**Where it does not:**

- Motion fidelity trails Sora and Runway Gen-4 by a full generation.
- Real-time collaboration is not a thing. There is no cloud project sharing beyond the portable export zip.
- Lipsync is not wired end-to-end yet.
- Timeline is v1: no multi-track audio, no compositor, no proxy workflow.
- Real-time rendering is not a thing either. A one-minute short takes minutes to hours depending on hardware and quality settings.
- Requires a modern NVIDIA GPU for the interesting bits. CPU-only mode is honest about what it can and cannot do.

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

Runtime output lives in `.mp/` (git-ignored). Every long-running task is a job. Every job carries progress, is cancellable, is retryable, and survives a process restart.

## Installation

### Prerequisites

- Python 3.11
- Node.js 22
- FFmpeg on your PATH
- Recommended for the GPU-heavy features: an NVIDIA GPU with 12+ GB VRAM

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

# Pick one, based on your hardware:
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

Run manually if you prefer:

```bash
python api_server.py                           # backend on :8000
cd web && npm run dev                          # frontend on :3000
```

### Your first film

1. Open the **Film Studio** panel.
2. Pick a template ("Short Story" is a good first pick).
3. Edit the sample brief or replace it with your own paragraph.
4. Click **Create**.
5. Click **Run**. Watch the pipeline advance stage-by-stage in the event stream.
6. Edit any stage's artifact between runs. Downstream stages automatically mark themselves stale so you can re-run only the affected shots.
7. When it finishes, click **Package** to get a delivery zip.

## Configuration

Copy `config.example.json` to `config.json` (git-ignored) and edit it.

Environment variables (also editable from the Settings panel):

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token for gated models |
| `HF_HOME` | Custom cache dir for HF weights |
| `NEXT_PUBLIC_API_URL` | Backend URL for the Next.js UI |
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to use |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory allocator tuning |
| `STUDIOLITE_AUTH` | `on` / `off` for API bearer auth |
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
| Pro / Studio | Same, plus watermark removal and `watermark_removal` feature flag | No |

Licenses are Ed25519-signed JSON payloads with tier, features, optional device fingerprint, expiry, and grace-days. Verification is offline. Nothing calls home to check.

For your own installs the free tier is unlimited. For third-party distribution (packaging StudioLite as part of a product) the licensing scaffold gives you a way to gate features cleanly.

## Contributing

- Open an issue for anything that surprised you, especially wrong outputs.
- PRs welcome. Keep changes focused. New features should ship with tests under `tests/`.
- The test suite is 162 tests today and CI is green on both Ubuntu and Windows. Please keep both true.

## License

MIT for the source. See `LICENSE`.

Third-party dependencies, bundled fonts, and model weights carry their own licenses. See `THIRD_PARTY_NOTICES.md` for the full list; note that some model weights (Stable Diffusion XL, HunyuanVideo, LTX-Video, CogVideoX 5B) have restrictions on commercial use that supersede StudioLite's MIT grant.
