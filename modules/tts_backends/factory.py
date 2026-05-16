import os
from .voxcpm2 import VoxCPM2Backend

# Redirect HuggingFace cache to shared directory
HF_HOME = os.path.expanduser("~/.zasttranslate/models")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

class TTSFactory:
    def __init__(self):
        self._backends = {
            "VoxCPM 2": VoxCPM2Backend(),
        }
        self._current_backend_name = None

    def get_available_backends(self):
        """Return a dict of name -> backend for those available on the system."""
        return {name: backend for name, backend in self._backends.items() if backend.is_available()}

    def get_backend(self, name: str):
        if name not in self._backends:
            raise ValueError(f"Unknown TTS Backend: {name}")

        backend = self._backends[name]
        
        # If switching backends, unload the previous one to free VRAM
        if self._current_backend_name and self._current_backend_name != name:
            print(f"Switching TTS backend: Unloading {self._current_backend_name} to free VRAM...")
            self._backends[self._current_backend_name].unload()
            
        self._current_backend_name = name
        return backend

# Global factory instance
_factory = TTSFactory()

def get_backend(name: str):
    return _factory.get_backend(name)

def get_available_backends():
    return _factory.get_available_backends()
