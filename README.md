# StudioLite

A lightweight, web-based video editor built with Streamlit and FFmpeg. Edit videos directly in your browser with no installation required (beyond Python dependencies).

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
| **ReelForge** | AI-powered short video generation (LLM script + image gen + TTS + subtitles) |

## Getting Started

### Prerequisites
- Python 3.8+
- FFmpeg installed on your system
- For ReelForge: Ollama running locally, ImageMagick installed, NVIDIA GPU (for SDXL image generation)

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

# For CUDA GPU support (recommended for ReelForge):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### ReelForge Setup

ReelForge integrates with [MoneyPrinterV2](https://github.com/FujiwaraChoki/MoneyPrinterV2) for AI video generation. To use it:

1. Clone MoneyPrinterV2 to `C:\Github\MoneyPrinterV2` (or update the path in `reelforge.py`)
2. Set up MoneyPrinterV2's `config.json` (copy from `config.example.json`)
3. Install and run [Ollama](https://ollama.com) with a model (e.g. `ollama pull llama3.2:3b`)
4. Install [ImageMagick](https://imagemagick.org) and set the path in config
5. Place SDXL `.safetensors` models in `MoneyPrinterV2/models/` for image generation (e.g. [RealVisXL V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0))

### Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Tech Stack

- **Streamlit** - Web interface
- **OpenCV** - Video/image processing
- **FFmpeg** - Video encoding, trimming, merging
- **PyMuPDF** - PDF processing
- **WhisperX** - Speech-to-text transcription
- **Google API** - YouTube upload integration
- **Ollama** - LLM text generation (ReelForge)
- **Stable Diffusion XL** - AI image generation via diffusers (ReelForge)
- **KittenTTS** - Text-to-speech (ReelForge)
- **MoviePy** - Video compositing (ReelForge)

## Project Structure

```
StudioLite/
├── app.py                # Streamlit web interface (all tools)
├── remover.py            # Core video/image/PDF processing logic
├── transcriber.py        # WhisperX speech-to-text transcription
├── youtube_uploader.py   # YouTube OAuth 2.0 upload integration
├── reelforge.py          # ReelForge AI video generation engine
├── requirements.txt      # Python dependencies
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
