import os
from abc import ABC, abstractmethod

class TTSBackend(ABC):
    """Abstract base class for all TTS backends in ZastTranslate."""
    
    def __init__(self):
        self.model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the user-facing name of the backend."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> dict:
        """
        Return a dict describing capabilities:
        {
            "languages": ["en", "fr", "es", ...], # or "all"
            "speed_control": bool, # True if natively supports speed 
            "duration_control": bool, # True if natively supports target duration
            "multi_speaker": bool, # True if allows mapping speakers natively
            "voice_design": bool, # True if zero-shot voice cloning
            "vram_gb": float, # Estimated VRAM requirement
            "sample_rate": int # Output sample rate
        }
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if dependencies are installed and the backend can be used."""
        pass

    @abstractmethod
    def load(self, ref_audio_path=None):
        """Load the model into VRAM/memory. 
        ref_audio_path is optional but some models might need it at load time (e.g. Qwen3)"""
        pass

    @abstractmethod
    def unload(self):
        """Free VRAM/memory."""
        pass

    @abstractmethod
    def generate(self, text: str, language: str, output_path: str, ref_audio_path: str = None, speed: float = 1.0, duration: float = None) -> dict:
        """
        Generate audio and save it to output_path.
        Must return a dict {"duration": float, "path": str, "sample_rate": int}
        """
        pass
