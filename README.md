# StudioLite

A lightweight, web-based video editor and AI video generation studio built with Streamlit and FFmpeg. Edit videos directly in your browser with no installation required (beyond Python dependencies).

## Features

| Tool | Description |
|------|-------------|
| **Remove Watermark** | AI-powered inpainting to remove watermarks from videos, PDFs, and images |
| **Trim / Cut** | Cut portions of video with preview |
| **Add Image Overlay** | Add logos, watermarks, or images at custom positions |
| **Change Speed** | Speed up or slow down videos (0.25x - 4x) with audio sync |
| **Merge Videos** | Combine multiple videos into one |
| **Extract Frame** | Export single frames as PNG images |
| **Export Video** | Convert format (MP4, WebM, AVI, MOV, MKV), quality, and resolution |
| **Transcribe** | Extract text from audio/video or microphone recording using WhisperX (SRT, VTT, JSON, TSV) |
| **View & Publish** | Preview video and upload directly to YouTube via OAuth 2.0 |
| **ReelForge** | AI-powered short video generation (LLM script + image gen + TTS + subtitles + background music) |
| **Video Generator** | Real AI video generation using diffusion models (Wan 2.1/2.2, HunyuanVideo, LTX-Video, CogVideoX) |
| **Story Mode** | Multi-scene AI movie creator with storyboard editor, per-scene video generation, narration, and music |

---

## UI Options: Streamlit vs Next.js

StudioLite ships with **two front-ends** that share the underlying Python engines but cover different feature subsets.

| UI | Default URL | Stack | Status |
|----|-------------|-------|--------|
| **Streamlit** (legacy) | http://localhost:8501 | `app.py` | Feature-complete (~20 tools) |
| **Next.js** (modern) | http://localhost:3000 | `web/` + FastAPI `api_server.py` on :8000 | Curated subset focused on creative/GPU workflows |

### Running both

```bash
# Streamlit
streamlit run app.py

# Next.js (two processes)
python api_server.py            # FastAPI backend on :8000
cd web && npm install && npm run dev   # Next.js dev server on :3000
```

### Feature coverage

| Feature | Streamlit | Next.js |
|---------|:---------:|:-------:|
| Video Generator (T2V / I2V) | ✓ | ✓ |
| Story Mode (multi-scene movies) | ✓ | ✓ |
| Characters (portrait + IP-Adapter) | ✓ | ✓ |
| Audio Studio (TTS) | ✓ | ✓ (TTS only) |
| Audio Studio (SFX, voice isolation) | ✓ | stub — falls back to Streamlit |
| Images Studio (T2I / edit / inpaint / upscale / bg-remove) | — | ✓ |
| Jobs panel (live progress monitor) | — | ✓ |
| Video Editor (region edits, filters) | ✓ | stub — punts to Streamlit |
| Upscale Video | ✓ | stub (UI only, no handler) |
| Keyframes | ✓ | stub (UI only, no handler) |
| Trim / Cut | ✓ | API exists, no UI |
| Merge Videos | ✓ | API exists, no UI |
| Remove Watermark | ✓ | — |
| Add Image Overlay | ✓ | — |
| Change Speed | ✓ | — |
| Extract Frame | ✓ | — |
| Export Video (codec / resolution) | ✓ | — |
| Transcribe (WhisperX) | ✓ | — |
| View & Publish (YouTube OAuth) | ✓ | — |
| Motion Brush | ✓ | — |
| ReelForge (LLM-driven short videos) | ✓ | — |

### Which one should I use?

- **Streamlit** for: transcription, YouTube upload, watermark removal, frame/format export, image overlay, speed control, ReelForge, motion brush, video editor.
- **Next.js** for: image generation, multi-scene story mode, character portraits, video gen, live job monitoring.

Tracking work to close the gaps: see the GitHub issue **"Next.js UI feature parity with Streamlit"**.

---

## Video Generator - AI Video Diffusion

Video Generator creates **real AI-generated video** using state-of-the-art diffusion models. Unlike ReelForge (which stitches images), this generates actual motion video frame-by-frame.

### Supported Engines

| Engine | VRAM | Description |
|--------|------|-------------|
| **Wan 2.1/2.2** | 8-24GB | Best quality, supports 1.3B (fast) and 14B (quality) models |
| **HunyuanVideo** | 24GB+ | Tencent's 8.3B model, up to 1080p resolution |
| **LTX-Video** | 8-12GB | Fast generation with distilled models |
| **CogVideoX** | 8-16GB | Versatile with 2B and 5B variants |

### Generation Modes

- **Text to Video**: Generate video from text description
- **Image to Video**: Animate a static image with AI motion
- **Extend Video**: Continue/extend an existing video clip (Wan 2.2 only)

### Features

- **Scene-based Generation**: Create longer videos by generating multiple scenes
- **Audio Integration**: Add AI-generated music/audio to videos
- **Auto VRAM Detection**: Automatically configures optimal settings for your GPU
- **Multiple Resolutions**: 480p, 720p, 832p (Wan), up to 1080p (HunyuanVideo)
- **Quantization Support**: INT8 quantization for lower VRAM usage

### Quick Start

1. Select "Video Generator" from the sidebar
2. Choose your generation mode (Text/Image/Extend)
3. Enter a detailed prompt describing the video
4. Adjust settings (engine, resolution, frames)
5. Click "Generate Video"

### Model Downloads

Models are downloaded automatically on first use to `/mnt/hdd/huggingface/` (configurable via `HF_HOME`).

| Model | Size | Use Case |
|-------|------|----------|
| Wan2.1-T2V-1.3B | ~8GB | Fast text-to-video, good for testing |
| Wan2.1-T2V-14B | ~28GB | High-quality text-to-video |
| Wan2.2-T2V-A14B | ~28GB | Latest Wan model with improvements |
| Wan2.1-I2V-480P | ~28GB | Image-to-video animation |

---

## Story Mode - Multi-Scene AI Movie Creator

Story Mode lets you create complete AI-generated movies by planning multiple scenes, generating video for each one using diffusion models, and assembling them with voiceover narration and background music.

### Three-Phase Workflow

**Phase 1: Plan Your Story**
- Enter a movie concept/idea in natural language
- Choose genre (Cinematic, Sci-Fi, Fantasy, Horror, Documentary, etc.) and mood (Epic, Calm, Tense, Mysterious, etc.)
- Set number of scenes (2-8)
- Click **AI Generate Script** to have the LLM write a structured screenplay with scene titles, visual prompts, narration, and timing
- Or manually add blank scenes and write everything yourself

**Phase 2: Edit Storyboard**
- Visual timeline bar showing scene proportions with color coding
- Per-scene editing cards with:
  - **Title** - Short scene name
  - **Visual Prompt** - Detailed description for AI video generation (camera angles, lighting, motion)
  - **Narration** - Voiceover text spoken by TTS
  - **Duration** - Per-scene duration slider (2-12 seconds)
  - **Reference Image** - Optional upload for Image-to-Video generation
- Reorder scenes (Move Up/Down), Duplicate, or Remove
- Live preview of generated scene videos

**Phase 3: Generate Movie**
- **Video Engine**: Wan 2.1 (recommended), LTX-Video (fast), CogVideoX, HunyuanVideo (HD)
- **Engine Settings**: Model variant, resolution, frames per scene, inference steps, guidance scale
- **Scene Continuity** (configurable): Keep subjects, characters, and style consistent across scenes
- **Narration**: TTS voice selection (Piper or KittenTTS voices)
- **Background Music**: Track selection with volume control
- **Output**: Aspect ratio (Landscape/Portrait/Square/Instagram), FPS, negative prompt
- Step-by-step progress bar showing each scene being generated
- Final movie with download button and scene breakdown grid

### Scene Continuity Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| **None** | Each scene generated independently | Abstract/varied scenes |
| **Prompt Anchoring** | A "visual identity" description (subject appearance, color palette, art style) is injected into every scene's prompt | Consistent characters, settings, and style |
| **Scene Chaining** | Last frame of scene N is used as Image-to-Video input for scene N+1 | Smooth visual flow between scenes |
| **Both** | Combines prompt anchoring + scene chaining | Maximum consistency |
| **Shared Seed** | Same random seed used for all scenes (combinable with any method above) | Similar textures and patterns |

- **Visual Identity Anchor**: Auto-generated by the AI script writer, or manually written. Describes exact subject appearance, color palette, and camera style. Appended to every scene prompt.
- **Chaining Strength**: Controls how much the previous scene's last frame influences the next (0.3 = creative freedom, 0.9 = strong continuity).

### Story Mode Pipeline

```
Concept + Genre + Mood
    ↓
AI Script Generation (LLM → scene titles, visuals, narration, visual identity anchor)
    ↓
Storyboard Editing (manual refinement of each scene)
    ↓
Scene Continuity Engine:
  ├─ Prompt Anchoring: visual identity injected into every scene prompt
  ├─ Scene Chaining: last frame of scene N → I2V input for scene N+1
  └─ Shared Seed: same random seed across all scenes
    ↓
Per-Scene Video Generation (diffusion model: Wan/LTX/CogVideoX)
    ↓
TTS Narration (per-scene voiceover → concatenated audio)
    ↓
Video Assembly (concatenate scenes → add narration → mix music)
    ↓
Final Movie (.mp4 with synced video, narration, and music)
```

### VRAM Requirements for Story Mode

| Engine + Model | Min VRAM | Scenes | Notes |
|----------------|----------|--------|-------|
| Wan 2.1 1.3B (480p) | 8GB | 2-8 | Best quality/VRAM ratio, CPU offloading |
| LTX-Video base | 8-12GB | 2-8 | Fastest generation |
| CogVideoX 2B | 8-10GB | 2-8 | Good quality, versatile |
| Wan 2.1 14B (480p) | 16GB+ | 2-8 | Higher quality with CPU offloading |
| HunyuanVideo | 24GB+ | 2-8 | Up to 1080p resolution |

---

## ReelForge - AI Video Generation Engine

ReelForge is a complete AI-powered video generation pipeline that creates short-form videos from a simple topic prompt.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ReelForge Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │  Topic   │───▶│ LLM Backend  │───▶│   Script    │───▶│ Scene Data     │  │
│  │  Input   │    │ (llamacpp/   │    │ Generator   │    │ (narration +   │  │
│  └──────────┘    │  ollama)     │    │             │    │  visual desc)  │  │
│                  └──────────────┘    └─────────────┘    └───────┬────────┘  │
│                                                                  │           │
│  ┌───────────────────────────────────────────────────────────────┼─────────┐│
│  │                         For Each Scene                        ▼         ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  ││
│  │  │ Image       │◀───│ Image       │◀───│ Visual Description          │  ││
│  │  │ (SDXL/      │    │ Prompt      │    │ → Detailed Image Prompt     │  ││
│  │  │ Gemini)     │    │ Generator   │    │                             │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────┘  ││
│  │         │                                                                ││
│  │         ▼                                                                ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  ││
│  │  │ Audio       │◀───│ TTS Engine  │◀───│ Narration Text              │  ││
│  │  │ (.wav)      │    │ (Piper/     │    │ (what to speak)             │  ││
│  │  │             │    │ KittenTTS)  │    │                             │  ││
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                        Video Assembly                                    ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ ││
│  │  │ Images     │  │ Audio      │  │ Subtitles  │  │ Background Music   │ ││
│  │  │ + Motion   │ +│ Concat     │ +│ (PIL       │ +│ (AudioMixer with   │ ││
│  │  │ Effects    │  │ (numpy)    │  │ rendering) │  │ auto-ducking)      │ ││
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ ││
│  │                           │                                              ││
│  │                           ▼                                              ││
│  │                    ┌─────────────┐                                       ││
│  │                    │ Final Video │                                       ││
│  │                    │ (.mp4)      │                                       ││
│  │                    └─────────────┘                                       ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### ReelForge Features

- **Multiple LLM Backends**: llama.cpp (GGUF models, offline) or Ollama
- **Multiple TTS Engines**: Piper TTS (high-quality neural) or KittenTTS (lightweight)
- **Multiple Image Providers**: SDXL Turbo (local GPU), Gemini Image API, Fooocus API
- **Multiple Aspect Ratios**: 9:16 (Portrait), 16:9 (Landscape), 1:1 (Square), 4:5 (Instagram)
- **Background Music**: Auto-ducking mixer that lowers music during speech
- **Motion Effects**: Zoom in, zoom out, pan effects
- **Color Filters**: Warm, cool, vintage, vivid
- **Animated Subtitles**: PIL-rendered text overlays synced to speech

---

## Configuration

All settings are stored in `config.json`. Here's the complete configuration reference:

### config.json

```json
{
  "verbose": true,
  "headless": false,

  "llm_backend": "llamacpp",
  "gguf_model": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
  "ollama_base_url": "http://127.0.0.1:11434",
  "ollama_model": "",

  "tts_engine": "piper",
  "tts_voice": "Amy",

  "stt_provider": "local_whisper",
  "whisper_model": "base",
  "whisper_device": "auto",
  "whisper_compute_type": "int8",
  "assembly_ai_api_key": "",

  "image_provider": "sdxl_turbo",
  "nanobanana2_api_base_url": "https://generativelanguage.googleapis.com/v1beta",
  "nanobanana2_api_key": "",
  "nanobanana2_model": "gemini-3.1-flash-image-preview",
  "nanobanana2_aspect_ratio": "9:16",
  "fooocus_api_url": "http://127.0.0.1:8888",
  "fooocus_style": "Fooocus V2",

  "default_aspect_ratio": "9:16",
  "background_music_enabled": false,
  "background_music_volume": 0.15,

  "threads": 2,
  "font": "bold_font.ttf",
  "imagemagick_path": "/usr/bin/convert",
  "script_sentence_length": 4
}
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| **LLM Settings** ||||
| `llm_backend` | string | `"llamacpp"` | LLM backend: `"llamacpp"` or `"ollama"` |
| `gguf_model` | string | `""` | GGUF model filename for llama.cpp |
| `ollama_base_url` | string | `"http://127.0.0.1:11434"` | Ollama server URL |
| `ollama_model` | string | `""` | Ollama model name (e.g., `"llama3.2:3b"`) |
| **TTS Settings** ||||
| `tts_engine` | string | `"piper"` | TTS engine: `"piper"` (neural) or `"kitten"` (lightweight) |
| `tts_voice` | string | `"Amy"` | Voice name (engine-specific) |
| **STT Settings** ||||
| `stt_provider` | string | `"local_whisper"` | Speech-to-text provider |
| `whisper_model` | string | `"base"` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `whisper_device` | string | `"auto"` | Device: `"auto"`, `"cpu"`, `"cuda"` |
| `whisper_compute_type` | string | `"int8"` | Compute type: `"int8"`, `"float16"`, `"float32"` |
| **Image Generation** ||||
| `image_provider` | string | `"sdxl_turbo"` | Provider: `"sdxl_turbo"`, `"nanobanana2"`, `"fooocus"` |
| `nanobanana2_api_key` | string | `""` | Gemini API key for image generation |
| `fooocus_api_url` | string | `"http://127.0.0.1:8888"` | Fooocus API endpoint |
| **Audio Settings** ||||
| `background_music_enabled` | bool | `false` | Enable background music by default |
| `background_music_volume` | float | `0.15` | Music volume (0.0-1.0), ducked during speech |
| **Video Settings** ||||
| `default_aspect_ratio` | string | `"9:16"` | Default video format |
| `threads` | int | `2` | MoviePy encoding threads |
| `font` | string | `"bold_font.ttf"` | Font file for subtitles |

---

## Project Structure

```
StudioLite/
├── app.py                      # Streamlit web interface
├── reelforge.py                # ReelForge AI video generation engine
├── videogen.py                 # Video Generator (diffusion-based video gen)
├── remover.py                  # Video/image/PDF watermark removal
├── transcriber.py              # WhisperX speech-to-text
├── youtube_uploader.py         # YouTube OAuth 2.0 upload
├── config.json                 # Application configuration
├── requirements.txt            # Python dependencies
├── check_models.py             # Model download status checker
│
├── mpv2/                       # Core modules
│   ├── config.py               # Configuration getters
│   ├── utils.py                # Utility functions
│   ├── audio_mixer.py          # Background music mixer with auto-ducking
│   ├── llm_provider.py         # LLM abstraction (llama.cpp/Ollama)
│   │
│   └── classes/
│       ├── Tts.py              # KittenTTS wrapper
│       ├── PiperTts.py         # Piper TTS wrapper (neural voices)
│       └── TtsFactory.py       # TTS engine factory
│
├── models/                     # GGUF & SDXL models directory
├── fonts/                      # Font files for subtitles
└── music/                      # Background music files (.mp3, .wav)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- NVIDIA GPU with CUDA (recommended for SDXL image generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/PawanRamaMali/StudioLite.git
cd StudioLite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CUDA GPU support (recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For llama.cpp with GPU acceleration:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ReelForge Setup

### 1. LLM Backend

**Option A: llama.cpp (Recommended - Offline)**
1. Download a GGUF model (e.g., Mistral 7B Instruct)
2. Place it in the `models/` directory
3. Set `llm_backend: "llamacpp"` in config

**Option B: Ollama**
1. Install [Ollama](https://ollama.com/download)
2. Run `ollama pull llama3.2:3b`
3. Set `llm_backend: "ollama"` in config

### 2. TTS Engine

**Piper TTS (Default - High Quality)**
- Voice models download automatically on first use
- Available voices: Amy, Ryan, Lessac, Kristin, Bryce, Danny, Joe, Kathleen

**KittenTTS (Lightweight)**
- Faster but lower quality
- Available voices: Jasper, Luna, Marcus, Elena, Thomas, Sofia, Alex, Emma

### 3. Image Generation

**SDXL Turbo (Local GPU)**
- Download SDXL models to `models/` directory
- Recommended: [RealVisXL V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0), [Juggernaut XL](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)

**Gemini API (Cloud)**
- Get API key from [Google AI Studio](https://aistudio.google.com/)
- Set `nanobanana2_api_key` in config

### 4. Background Music

Place `.mp3` or `.wav` files in the `music/` directory. Music will automatically:
- Loop to match video duration
- Duck (reduce volume) when narration is playing
- Mix at the configured volume level

---

## API Reference

### ReelForge Generation Function

```python
from reelforge import rf_generate_full

result = rf_generate_full(
    topic="Benefits of meditation",      # Video topic
    language="English",                   # Script language
    sentence_count=4,                     # Sentences per scene
    image_provider="sdxl_turbo",          # Image provider
    sdxl_model="RealVisXL_V4.0.safetensors",
    progress_callback=on_progress,        # Progress updates
    image_style="photorealistic",         # Visual style preset
    image_steps=8,                        # SDXL inference steps
    image_guidance=2.0,                   # SDXL guidance scale
    subtitle_style="bold_yellow",         # Text style
    ken_burns_effect="zoom_in",           # Motion effect
    color_filter="none",                  # Color grading
    num_images=3,                         # Number of scenes
    aspect_ratio="9:16",                  # Video format
    music_enabled=True,                   # Add background music
    music_path=None,                      # Specific track or random
    music_volume=0.15,                    # Music volume (0.0-1.0)
)

# Result contains:
# - scenes: list of scene data with images, audio, timing
# - script: full narration text
# - video_path: path to generated video
# - title, description: AI-generated metadata
# - total_duration: video length in seconds
```

### Image Style Presets

| Style | Description |
|-------|-------------|
| `photorealistic` | Ultra-realistic photography |
| `cinematic` | Movie-like dramatic lighting |
| `digital_art` | Polished digital illustration |
| `anime` | Japanese anime style |
| `watercolor` | Soft watercolor painting |
| `oil_painting` | Classical oil painting |
| `3d_render` | 3D rendered graphics |
| `minimalist` | Clean, simple design |

### Subtitle Styles

| Style | Description |
|-------|-------------|
| `bold_yellow` | Yellow text with black outline |
| `white_shadow` | White text with drop shadow |
| `neon_glow` | Glowing neon effect |
| `minimal_white` | Clean white text |
| `bold_red` | Red text with outline |

### Motion Effects

- `zoom_in` - Slow zoom towards center
- `zoom_out` - Slow zoom outward
- `none` - Static image

### Color Filters

- `none` - No filter
- `warm` - Warm orange tones
- `cool` - Cool blue tones
- `vintage` - Faded retro look
- `vivid` - Enhanced saturation

---

## Tech Stack

- **Streamlit** - Web interface
- **OpenCV** - Video/image processing
- **FFmpeg** - Video encoding, trimming, merging
- **PyMuPDF** - PDF processing
- **WhisperX / faster-whisper** - Speech-to-text transcription
- **Google API** - YouTube upload integration
- **llama.cpp / Ollama** - LLM text generation
- **Stable Diffusion XL** - AI image generation via diffusers
- **Piper TTS** - High-quality neural text-to-speech
- **KittenTTS** - Lightweight text-to-speech
- **MoviePy** - Video compositing
- **NumPy / SoundFile** - Audio processing
- **SciPy** - Audio resampling and signal processing
- **Diffusers** - Video generation pipelines (Wan, HunyuanVideo, LTX, CogVideoX)
- **HuggingFace Hub** - Model downloading and caching

---

## Troubleshooting

### Audio Issues
- **Missing audio**: The pipeline uses numpy-based audio concatenation to ensure reliable playback
- **Silent scenes**: TTS failures are caught and fallback text is generated

### GPU Memory
- For SDXL on limited VRAM, reduce `image_steps` or use SDXL Turbo
- Piper TTS runs on CPU and doesn't require GPU

### Model Downloads
- GGUF models: Place in `models/` directory
- SDXL models: Place `.safetensors` files in `models/` directory
- Piper voices: Download automatically to `~/.local/share/piper/`

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## Credits

- [Piper TTS](https://github.com/rhasspy/piper) - Neural text-to-speech
- [Stable Diffusion XL](https://stability.ai/stable-diffusion) - Image generation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [WhisperX](https://github.com/m-bain/whisperX) - Speech recognition
