import os
from .local_causal import LocalCausalLMBackend

# Global cache for the active backend
_active_llm_backend = None

def get_available_backends() -> dict:
    """Returns a dictionary of all available LLM backends."""
    backends = {
        "Qwen2.5-7B-Instruct": LocalCausalLMBackend(
            name="Qwen2.5-7B-Instruct",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            capabilities={
                "languages": "all", # Supports everything
                "vram_gb": 4.5
            }
        ),
        "Qwen3.5-9B-Instruct": LocalCausalLMBackend(
            name="Qwen3.5-9B-Instruct",
            model_id="Qwen/Qwen3.5-9B-Instruct",
            capabilities={
                "languages": "all",
                "vram_gb": 6.0
            }
        ),
        "EuroLLM-9B-Instruct": LocalCausalLMBackend(
            name="EuroLLM-9B-Instruct",
            model_id="utter-project/EuroLLM-9B-Instruct",
            capabilities={
                "languages": ["en", "fr", "es", "de", "it", "pt", "nl", "pl", "sv", "cs", "ro", "hu"],
                "vram_gb": 6.0
            }
        )
    }
    
    # Filter only those whose requirements are installed
    available = {}
    for name, backend in backends.items():
        if backend.is_available():
            available[name] = backend
            
    return available

def get_backend(name: str):
    """
    Returns the requested backend.
    If a different backend is currently loaded, it unloads it to free VRAM.
    """
    global _active_llm_backend
    
    available = get_available_backends()
    if name not in available:
        raise ValueError(f"LLM Backend '{name}' is not available. Ensure dependencies are installed.")
        
    requested_backend = available[name]
    
    if _active_llm_backend is not None and _active_llm_backend.name != requested_backend.name:
        print(f"Switching LLM Backend: Unloading {_active_llm_backend.name} to free VRAM...")
        _active_llm_backend.unload()
        
    _active_llm_backend = requested_backend
    
    # Redirect HuggingFace Cache to user's local directory to avoid C: drive bloat
    models_dir = os.path.expanduser("~/.zasttranslate/models")
    os.makedirs(models_dir, exist_ok=True)
    os.environ["HF_HOME"] = models_dir
    
    return _active_llm_backend
