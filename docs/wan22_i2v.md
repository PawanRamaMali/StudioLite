# Wan 2.2 TI2V-5B: enabling real Image-to-Video (I2V)

The Wan 2.2 motion backend has two modes:

- **T2V (default)** — text-only generation. Fast enough for a full film on 12GB
  VRAM (~10 min per shot). Character identity comes from the prompt alone;
  Wan reinvents the subject each shot, so shot-to-shot character consistency
  is weak.
- **I2V (opt-in)** — the SDXL keyframe conditions the generation, so the
  character in the still is the character in the clip. Consistency is
  preserved. Requires main-branch `diffusers` + `transformers` 5.x, which
  conflict with the `transformers` 4.x pinned by IndexTTS-2. To keep both
  working, I2V runs as a subprocess against an **isolated venv**.

I2V is uneconomical on a 12GB laptop card (~2-3 hours per shot). It's
practical on a **24GB desktop card** (~15-30 min per shot), where a
35-shot film lands in ~10-16 hours overnight instead of 3-4 days.

## Setup on a fresh box

1. Download the weights (34GB, one-time, no HF token required):

   ```
   hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
     --local-dir=~/models/wan22-ti2v-5b
   ```

2. Create the isolated venv:

   ```
   python -m venv ~/venvs/wan22
   ~/venvs/wan22/Scripts/python.exe -m pip install --upgrade pip

   # Match the CUDA build to your torch install
   ~/venvs/wan22/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 \
     --index-url https://download.pytorch.org/whl/cu128

   ~/venvs/wan22/Scripts/python.exe -m pip install \
     "git+https://github.com/huggingface/diffusers.git" \
     "transformers>=5.0" "accelerate>=0.30" "huggingface_hub>=1.0" \
     safetensors imageio imageio-ffmpeg sentencepiece tqdm protobuf ftfy
   ```

3. Verify the venv is discoverable from a StudioLite Python session:

   ```
   python -c "from filmmaker.agents import _wan22_i2v_available; print(_wan22_i2v_available())"
   ```

   Expected output: `True`.

## Running

Set the motion backend and start a film as normal:

```
"config": {"motion_backend": "wan22"}
```

The pipeline tries I2V first when the venv is present, then falls back to
in-process T2V per shot on any failure so the run never crashes.

## Environment variables

- `WAN22_MODEL_DIR` (default `~/models/wan22-ti2v-5b`) — weights location.
- `WAN22_VENV_PYTHON` (default `~/venvs/wan22/Scripts/python.exe`) — path to
  the venv's Python. Empty string forces T2V mode.
- `WAN22_MODE` — `auto` (default: try I2V, fall back to T2V), `t2v` (skip
  I2V entirely), `i2v` (fail the shot rather than fall back to T2V —
  useful when you're debugging the venv setup).
- `WAN22_I2V_TIMEOUT_SEC` (default `3600`) — per-shot ceiling in seconds.
  Bump on slower cards; drop on faster ones to fail fast.

## Wall-clock expectations

| Hardware | I2V mode? | Per-shot | 35-shot film |
| --- | --- | --- | --- |
| RTX 3060/4060 laptop, 12 GB | not recommended | ~120-180 min | 3-4 days |
| RTX 4070 laptop, 12 GB | not recommended | ~90-150 min | 2-3 days |
| RTX 4070 Ti desktop, 12 GB | tight | ~60-90 min | 1.5-2 days |
| RTX 3090/4090, 24 GB | comfortable | ~15-30 min | 10-16 hours overnight |
| RTX 5090, 32 GB | native res possible | ~5-10 min | 3-6 hours |

Numbers are extrapolations from the actual T2V performance on the reference
box (RTX 5070 Ti Laptop 12GB) and community reports for 24GB+ cards.
