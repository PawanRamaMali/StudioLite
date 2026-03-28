#!/usr/bin/env python3
"""Check Wan model download status."""
import os
import sys

# Set HF cache location
os.environ["HF_HOME"] = "/mnt/hdd/huggingface"

MODELS = {
    "Wan2.1-T2V-1.3B": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 8),
    "Wan2.1-T2V-14B": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", 28),
    "Wan2.2-T2V-A14B": ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", 28),
    "Wan2.1-I2V-480P": ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 28),
    "Wan2.2-I2V-A14B": ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", 28),
}

def get_dir_size_gb(path):
    """Get directory size in GB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)

def check_model_ready(repo_id):
    """Check if model is fully downloaded."""
    try:
        from huggingface_hub import snapshot_download
        # This returns immediately if model is cached
        path = snapshot_download(repo_id, local_files_only=True)
        return True, path
    except Exception:
        return False, None

def main():
    print("=" * 60)
    print("Wan Model Download Status")
    print("=" * 60)

    hub_path = "/mnt/hdd/huggingface/hub"
    total_ready = 0

    for name, (repo_id, expected_gb) in MODELS.items():
        model_dir = os.path.join(hub_path, f"models--{repo_id.replace('/', '--')}")

        if os.path.exists(model_dir):
            current_gb = get_dir_size_gb(model_dir)
            ready, _ = check_model_ready(repo_id)

            if ready:
                status = "READY"
                total_ready += 1
            else:
                pct = min(100, int(current_gb / expected_gb * 100))
                status = f"Downloading... {current_gb:.1f}GB / ~{expected_gb}GB ({pct}%)"
        else:
            status = "Not started"

        print(f"{name:20} {status}")

    print("=" * 60)
    print(f"Ready: {total_ready}/{len(MODELS)} models")

    if total_ready == len(MODELS):
        print("\nAll models downloaded! You can use Video Generator now.")
    else:
        print("\nTip: Run this script again to check progress.")

    return total_ready == len(MODELS)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
