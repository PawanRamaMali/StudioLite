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

## Getting Started

### Prerequisites
- Python 3.8+
- FFmpeg installed on your system

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
```

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

## Project Structure

```
StudioLite/
├── app.py           # Streamlit web interface
├── remover.py       # Core video processing logic
├── requirements.txt # Python dependencies
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
