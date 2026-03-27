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
├── remover.py                  # Video/image/PDF watermark removal
├── transcriber.py              # WhisperX speech-to-text
├── youtube_uploader.py         # YouTube OAuth 2.0 upload
├── config.json                 # Application configuration
├── requirements.txt            # Python dependencies
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
