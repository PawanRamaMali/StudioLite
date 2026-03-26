"""
LLM Provider - Supports multiple backends:
- llama.cpp (GGUF models) - Default, works offline
- Ollama - If installed and running
"""
import os
from pathlib import Path

from .config import get_ollama_base_url, ROOT_DIR

# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------
_selected_model: str | None = None
_selected_backend: str = "llamacpp"  # "llamacpp" or "ollama"
_llama_model = None  # Loaded llama-cpp-python model


def get_models_dir() -> Path:
    """Get the models directory path."""
    return Path(ROOT_DIR) / "models"


def get_gguf_models() -> list[str]:
    """List available GGUF models in the models directory."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted([f.name for f in models_dir.glob("*.gguf")])


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------
def check_llamacpp_available() -> tuple[bool, str]:
    """Check if llama-cpp-python is installed."""
    try:
        from llama_cpp import Llama
        return True, "llama-cpp-python available"
    except ImportError:
        return False, "llama-cpp-python not installed. Run: pip install llama-cpp-python"


def load_llamacpp_model(model_path: str = None, n_gpu_layers: int = -1, n_ctx: int = 4096):
    """
    Load a GGUF model with llama-cpp-python.

    Args:
        model_path: Path to .gguf file, or model name in models/ dir
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        n_ctx: Context window size
    """
    global _llama_model

    from llama_cpp import Llama

    # Find the model file
    if model_path is None:
        # Use selected model or first available
        models = get_gguf_models()
        if not models:
            raise RuntimeError("No GGUF models found. Download one first.")
        model_path = str(get_models_dir() / models[0])
    elif not os.path.isabs(model_path):
        # Relative path - look in models dir
        model_path = str(get_models_dir() / model_path)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    # Unload previous model
    if _llama_model is not None:
        del _llama_model
        _llama_model = None

    _llama_model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    return _llama_model


def generate_llamacpp(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Generate text using llama-cpp-python."""
    global _llama_model

    if _llama_model is None:
        load_llamacpp_model()

    response = _llama_model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
def _ollama_client():
    import ollama
    return ollama.Client(host=get_ollama_base_url())


def check_ollama_connection() -> tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        client = _ollama_client()
        client.list()
        return True, "Ollama is running"
    except Exception:
        return False, "Ollama not running. Install from https://ollama.com/download or use llama.cpp backend."


def list_ollama_models() -> list[str]:
    """Lists all models available on the local Ollama server."""
    try:
        response = _ollama_client().list()
        return sorted(m.model for m in response.models)
    except Exception:
        return []


def generate_ollama(prompt: str, model: str = None) -> str:
    """Generate text using Ollama."""
    connected, msg = check_ollama_connection()
    if not connected:
        raise RuntimeError(msg)

    if not model:
        raise RuntimeError("No Ollama model selected.")

    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Model downloading
# ---------------------------------------------------------------------------
def download_model(model_id: str, filename: str = None, progress_callback=None) -> str:
    """
    Download a GGUF model from Hugging Face.

    Args:
        model_id: HuggingFace model ID (e.g., "TheBloke/Mistral-7B-v0.1-GGUF")
        filename: Specific file to download (e.g., "mistral-7b-v0.1.Q4_K_M.gguf")
        progress_callback: Optional callback(downloaded, total) for progress

    Returns:
        Path to the downloaded model file
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # If no filename specified, find a good GGUF file
    if filename is None:
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]

        # Prefer Q4_K_M quantization (good balance of quality/size)
        preferred = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f]
        if preferred:
            filename = preferred[0]
        elif gguf_files:
            filename = gguf_files[0]
        else:
            raise RuntimeError(f"No GGUF files found in {model_id}")

    # Download
    local_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )

    return local_path


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------
def select_backend(backend: str) -> None:
    """Select the LLM backend: 'llamacpp' or 'ollama'."""
    global _selected_backend
    if backend not in ("llamacpp", "ollama"):
        raise ValueError(f"Unknown backend: {backend}. Use 'llamacpp' or 'ollama'.")
    _selected_backend = backend


def get_backend() -> str:
    """Get the current backend."""
    return _selected_backend


def select_model(model: str) -> None:
    """Sets the model to use for all subsequent generate_text calls."""
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    """Returns the currently selected model."""
    return _selected_model


def list_models() -> list[str]:
    """List available models for the current backend."""
    if _selected_backend == "llamacpp":
        return get_gguf_models()
    else:
        return list_ollama_models()


def check_backend_status() -> tuple[bool, str]:
    """Check if the current backend is ready."""
    if _selected_backend == "llamacpp":
        ok, msg = check_llamacpp_available()
        if ok:
            models = get_gguf_models()
            if models:
                return True, f"llama.cpp ready with {len(models)} model(s)"
            return False, "llama.cpp ready but no GGUF models found"
        return ok, msg
    else:
        return check_ollama_connection()


def generate_text(prompt: str, model_name: str = None) -> str:
    """
    Generate text using the selected backend.

    Args:
        prompt: The prompt to send to the model
        model_name: Optional model override

    Returns:
        Generated text response
    """
    model = model_name or _selected_model

    if _selected_backend == "llamacpp":
        # For llama.cpp, load model if needed
        global _llama_model
        if _llama_model is None and model:
            load_llamacpp_model(model)
        elif _llama_model is None:
            load_llamacpp_model()  # Will use first available
        return generate_llamacpp(prompt)
    else:
        return generate_ollama(prompt, model)


# Recommended models for download
RECOMMENDED_MODELS = [
    {
        "name": "Mistral 7B Q4",
        "id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "4.4 GB",
        "description": "Fast, good quality, great for scripts",
    },
    {
        "name": "Phi-3 Mini Q4",
        "id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4.gguf",
        "size": "2.2 GB",
        "description": "Small, fast, good instruction following",
    },
    {
        "name": "Llama 3.2 3B Q4",
        "id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "2.0 GB",
        "description": "Latest Llama, compact and capable",
    },
    {
        "name": "Qwen2.5 7B Q4",
        "id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size": "4.7 GB",
        "description": "Excellent for structured output (JSON)",
    },
]
