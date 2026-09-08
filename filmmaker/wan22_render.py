"""Standalone Wan 2.2 TI2V-5B I2V renderer, called from motion_shots via
subprocess so main-branch diffusers can live in an isolated venv without
poisoning the StudioLite runtime env (which is pinned to older transformers
for IndexTTS-2 / XTTS compatibility).

CLI:
    python wan22_render.py --model-dir DIR --image IN.png --prompt "..."
                           --out OUT.mp4 --duration-sec N --camera-move MOVE

Exits non-zero on any failure so the caller can fall back to T2V or
Ken Burns cleanly.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to Wan 2.2 diffusers dir")
    p.add_argument("--image", required=True, help="Keyframe PNG to condition on")
    p.add_argument("--prompt", required=True, help="Motion prompt")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--duration-sec", type=int, default=3)
    p.add_argument("--camera-move", default="static")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    for path in (args.model_dir, args.image):
        if not os.path.exists(path):
            print(f"error: missing {path}", file=sys.stderr)
            return 2

    try:
        import torch
        from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
        from diffusers.utils import export_to_video, load_image
    except Exception as e:
        print(f"error: import failure: {e}", file=sys.stderr)
        return 3

    # Motion hint appended to the prompt so the model biases toward the
    # camera move the cinematographer picked. Same table as the T2V renderer.
    hints = {
        "pan":     "smooth horizontal camera pan",
        "dolly":   "slow dolly-in",
        "crane":   "camera cranes upward",
        "handheld":"subtle handheld shake",
        "whip":    "fast whip pan",
        "static":  "gentle atmospheric motion, minimal camera movement",
    }
    hint = hints.get(args.camera_move, "gentle atmospheric motion")
    full_prompt = f"{args.prompt}. {hint}."

    # Round frame count to 4k+1 (temporal-compression requirement) and cap
    # at 81 to keep VRAM under 12GB. 49 frames = ~2s at 24fps, 81 = ~3.4s.
    fps = 24
    num_frames = min(81, max(49, args.duration_sec * fps + 1))
    num_frames = ((num_frames - 1) // 4) * 4 + 1

    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_dir,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # model_cpu_offload is what worked for T2V at these sizes; sequential
    # is slower but if we hit OOM we retry once via the fallback below.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    image = load_image(args.image)
    with torch.no_grad():
        try:
            out = pipe(
                image=image,
                prompt=full_prompt,
                height=args.height,
                width=args.width,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=args.steps,
                negative_prompt="low quality, distorted, deformed, watermark",
            )
        except torch.cuda.OutOfMemoryError:
            # Free everything and retry with the sequential offload — much
            # slower per step but survives at the memory boundary.
            del pipe
            torch.cuda.empty_cache()
            pipe = WanImageToVideoPipeline.from_pretrained(
                args.model_dir, vae=vae, torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.enable_sequential_cpu_offload()
            pipe.set_progress_bar_config(disable=True)
            out = pipe(
                image=image,
                prompt=full_prompt,
                height=args.height,
                width=args.width,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=args.steps,
                negative_prompt="low quality, distorted, deformed, watermark",
            )

    frames = out.frames[0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_to_video(frames, args.out, fps=fps)
    print(f"OK {args.out} frames={num_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
