"""
LatentSync 1.6 lip-sync post-processing.

Runs AFTER video generation: takes a generated clip + its narration audio
and produces a lip-synced version.

Integration model: subprocess into the bundled LatentSync repo.

Weights live at .models/latentsync-16/ (symlinked into third_party/LatentSync/checkpoints/
via a Windows directory junction). Source cloned to third_party/LatentSync/ via:
    git clone https://github.com/bytedance/LatentSync.git third_party/LatentSync

If either the source tree or the weights are missing, ``lip_sync_video``
raises with a clear message so callers can surface it upstream.
"""

import os
import sys
import subprocess
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LATENTSYNC_REPO = os.path.join(ROOT_DIR, "third_party", "LatentSync")
LATENTSYNC_WEIGHTS = os.path.join(ROOT_DIR, ".models", "latentsync-16")
UNET_CKPT = os.path.join(LATENTSYNC_WEIGHTS, "latentsync_unet.pt")
WHISPER_TINY = os.path.join(LATENTSYNC_WEIGHTS, "whisper", "tiny.pt")
DEFAULT_CONFIG = "configs/unet/stage2_512.yaml"  # relative to LATENTSYNC_REPO


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _try_link_checkpoints() -> None:
    """Best-effort auto-create the ``third_party/LatentSync/checkpoints`` link.

    On Linux/macOS uses a real symlink (no privileges required). On Windows
    tries a directory junction via ``mklink /J`` (no dev-mode needed for
    junctions to shared paths, unlike symbolic links). Silent on failure —
    ``available()`` will still return False and ``install_hint()`` surfaces
    the manual step.
    """
    if not os.path.isdir(LATENTSYNC_REPO) or not os.path.isdir(LATENTSYNC_WEIGHTS):
        return
    link = os.path.join(LATENTSYNC_REPO, "checkpoints")
    if os.path.exists(link):
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", link, LATENTSYNC_WEIGHTS],
                check=False, capture_output=True,
            )
        else:
            os.symlink(LATENTSYNC_WEIGHTS, link, target_is_directory=True)
    except Exception:
        pass


def available() -> bool:
    """True iff LatentSync source + weights are all present, the checkpoint
    link resolves to our weights, and a CUDA GPU is present.

    LatentSync's bundled inference script hardcodes ``device="cuda"`` so it
    cannot run without a GPU regardless of what we pass in.
    """
    if not _has_cuda():
        return False
    if not os.path.isdir(LATENTSYNC_REPO):
        return False
    if not os.path.isfile(UNET_CKPT) or not os.path.isfile(WHISPER_TINY):
        return False
    if not os.path.isfile(os.path.join(LATENTSYNC_REPO, "scripts", "inference.py")):
        return False
    ckpt_link = os.path.join(LATENTSYNC_REPO, "checkpoints", "latentsync_unet.pt")
    if not os.path.isfile(ckpt_link):
        _try_link_checkpoints()
    return os.path.isfile(ckpt_link)


def install_hint() -> str:
    steps = []
    if not _has_cuda():
        steps.append("LatentSync requires an NVIDIA GPU with CUDA drivers.")
    if not os.path.isdir(LATENTSYNC_REPO):
        steps.append(
            "Clone the repo: git clone https://github.com/bytedance/LatentSync.git "
            f"{LATENTSYNC_REPO}"
        )
    if not (os.path.isfile(UNET_CKPT) and os.path.isfile(WHISPER_TINY)):
        steps.append(
            "Download weights from https://huggingface.co/ByteDance/LatentSync-1.6 "
            f"into {LATENTSYNC_WEIGHTS}"
        )
    ckpt_link = os.path.join(LATENTSYNC_REPO, "checkpoints")
    if os.path.isdir(LATENTSYNC_REPO) and not os.path.exists(ckpt_link):
        if os.name == "nt":
            steps.append(f'Create junction: cmd /c mklink /J "{ckpt_link}" "{LATENTSYNC_WEIGHTS}"')
        else:
            steps.append(f'Create symlink: ln -s "{LATENTSYNC_WEIGHTS}" "{ckpt_link}"')
    if not steps:
        return "LatentSync is set up correctly."
    return "LatentSync setup incomplete:\n  - " + "\n  - ".join(steps)


def lip_sync_video(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: Optional[int] = None,
    enable_deepcache: bool = True,
    unet_config_path: str = DEFAULT_CONFIG,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Run LatentSync on one video + audio pair.

    Returns the path of the lip-synced mp4. Defaults match ``inference.sh``
    shipped in the LatentSync repo (stage2_512, 20 steps, guidance 1.5).
    """
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + ".lipsync.mp4"

    if not available():
        raise RuntimeError(install_hint())

    video_abs = os.path.abspath(video_path)
    audio_abs = os.path.abspath(audio_path)
    output_abs = os.path.abspath(output_path)

    if not os.path.isfile(video_abs):
        raise FileNotFoundError(f"video_path not found: {video_abs}")
    if not os.path.isfile(audio_abs):
        raise FileNotFoundError(f"audio_path not found: {audio_abs}")

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", unet_config_path,
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", video_abs,
        "--audio_path", audio_abs,
        "--video_out_path", output_abs,
        "--seed", str(seed) if seed is not None else "-1",
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")

    if progress_callback:
        progress_callback(10, 100, "Starting LatentSync inference...")

    # LatentSync's inference.py uses relative paths ('configs', 'checkpoints/...'),
    # so we must run it with cwd set to the repo root.
    env = os.environ.copy()
    env["PYTHONPATH"] = LATENTSYNC_REPO + os.pathsep + env.get("PYTHONPATH", "")

    logger.info("LatentSync cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=LATENTSYNC_REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to launch LatentSync inference: {e}") from e

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").splitlines()[-20:]
        raise RuntimeError(
            f"LatentSync exited {proc.returncode}. Last output:\n  "
            + "\n  ".join(tail)
        )

    if not os.path.isfile(output_abs):
        raise RuntimeError(f"LatentSync produced no output at {output_abs}")

    if progress_callback:
        progress_callback(100, 100, "Lip sync complete.")
    return output_abs


def lip_sync_scene_clips(
    scene_clips: list,
    scene_audios: list,
    progress_callback: Optional[Callable] = None,
) -> list:
    """Batch-apply lip sync across a list of (video, audio) pairs.

    A scene whose audio is missing or silent is passed through unchanged.
    Raises on first hard failure so job error reporting surfaces the issue.
    """
    if len(scene_clips) != len(scene_audios):
        raise ValueError(
            f"scene_clips ({len(scene_clips)}) != scene_audios ({len(scene_audios)})"
        )

    out = []
    n = len(scene_clips)
    for i, (vid, aud) in enumerate(zip(scene_clips, scene_audios)):
        if not vid:
            out.append(vid)
            continue
        if not aud or not os.path.isfile(aud):
            out.append(vid)
            continue
        if progress_callback:
            progress_callback(i, n, f"Lip sync scene {i+1}/{n}...")
        synced = lip_sync_video(vid, aud, progress_callback=None)
        out.append(synced)
    return out
