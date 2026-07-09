# Third-Party Notices

StudioLite is licensed under the [MIT License](LICENSE). It builds on a number
of third-party projects, models, and assets, each distributed under its own
license. This document lists the significant components and their licenses.

Dependencies are installed by the user via `pip` / `npm` and are **not**
redistributed inside this repository, except where noted under "Bundled assets".
This list is provided in good faith; consult each project for authoritative and
up-to-date license terms.

## ⚠️ Copyleft components to be aware of

| Component | License | Notes |
|-----------|---------|-------|
| **PyMuPDF** (`pymupdf`) | **AGPL-3.0** (or commercial) | Used only for PDF watermark removal. Installed as a pip dependency. If you deploy StudioLite as a network service, the AGPL requires you to offer users the corresponding source. If you do not need PDF processing, you can omit this dependency. |
| **FFmpeg** (system binary) | **LGPL-2.1+ / GPL** (build-dependent) | Required to be installed separately on the host; it is not bundled or redistributed here. `ffmpeg-python` (the Python wrapper) is Apache-2.0. |
| **ImageMagick** (optional system binary) | ImageMagick License (Apache-2.0-style) | Optional external tool, installed separately. |

## Bundled assets (redistributed in this repo)

| Asset | Source | License |
|-------|--------|---------|
| `fonts/Anton-Regular.ttf` | [Anton](https://github.com/googlefonts/AntonFont) by Vernon Adams / The Anton Project Authors | SIL Open Font License 1.1 — see `fonts/Anton-OFL.txt` |
| `web/` frontend scaffold | Bootstrapped with [create-next-app](https://github.com/vercel/next.js) | MIT |

## Python dependencies

| Package | License |
|---------|---------|
| streamlit | Apache-2.0 |
| fastapi, uvicorn, python-multipart | MIT / BSD |
| numpy | BSD-3-Clause |
| Pillow | HPND (permissive) |
| opencv-python-headless | Apache-2.0 |
| ffmpeg-python | Apache-2.0 |
| moviepy | MIT |
| torch | BSD-3-Clause |
| torchaudio | BSD-2-Clause |
| faster-whisper | MIT |
| whisperx | BSD |
| diffusers, transformers, accelerate, safetensors | Apache-2.0 |
| huggingface-hub | Apache-2.0 |
| rapidocr-onnxruntime | Apache-2.0 |
| kittentts | Apache-2.0 |
| soundfile | BSD-3-Clause |
| ollama (python client) | MIT |
| requests | Apache-2.0 |
| tqdm | MPL-2.0 / MIT |
| srt-equalizer | MIT |
| termcolor | MIT |
| mss | MIT |
| imagehash | BSD-2-Clause |
| python-docx | MIT |
| reportlab | BSD-3-Clause |
| audio-recorder-streamlit | MIT |
| google-api-python-client, google-auth-oauthlib, google-auth-httplib2 | Apache-2.0 |

## Frontend (Next.js) dependencies

| Package | License |
|---------|---------|
| next, react, react-dom | MIT |
| @radix-ui/* | MIT |
| framer-motion | MIT |
| lucide-react | ISC |
| zustand, swr, clsx, tailwind-merge | MIT |
| tailwindcss | MIT |
| typescript | Apache-2.0 |
| eslint, eslint-config-next | MIT |

## Runtime engines and models (downloaded by the user, not redistributed)

| Component | License |
|-----------|---------|
| Piper TTS | MIT |
| KittenTTS | Apache-2.0 |
| llama.cpp / llama-cpp-python | MIT |
| Ollama (server) | MIT |
| OpenAI Whisper model weights | MIT |
| Stable Diffusion XL | CreativeML Open RAIL++-M |
| Wan 2.1 / 2.2 | Apache-2.0 |
| HunyuanVideo | Tencent Hunyuan Community License |
| LTX-Video | Custom (LTXV) / OpenRAIL-style |
| CogVideoX | Apache-2.0 (2B) / custom CogVideoX License (5B) |

Diffusion and LLM model weights carry their **own** licenses, some of which
restrict commercial or specific uses. You are responsible for complying with the
license of any model you download and use.
