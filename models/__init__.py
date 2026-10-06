"""StudioLite model registry.

Central inventory of every weight StudioLite can use at runtime. Each
entry declares where its file is expected on disk, how big it should be,
and the official source URL - so the Settings panel can tell the user
exactly what's installed, what's missing, and where to grab the rest.

The registry is pure data + a disk-probe helper. It does NOT trigger any
downloads: that's the job of model_hub (video engines) and of the lazy
loaders that live next to each backend module (whisper, SDXL, etc).
"""

from .registry import (  # noqa: F401
    REGISTRY,
    ModelKind,
    ModelSpec,
    expected_path_for,
    hf_cache_dir,
    hf_hub_model_path,
    probe_model,
    probe_registry,
    registry_summary,
)
