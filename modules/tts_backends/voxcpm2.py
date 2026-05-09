import torch
import os
import soundfile as sf
import numpy as np
from .base import TTSBackend
from config import DEVICE, TEMP_DIR

class VoxCPM2Backend(TTSBackend):
    def __init__(self):
        super().__init__()
        self.device = DEVICE

    @property
    def name(self) -> str:
        return "VoxCPM 2"

    @property
    def capabilities(self) -> dict:
        return {
            "languages": ["en", "zh", "de", "fr", "es", "it", "pt", "pl", "nl", "ru"],
            "speed_control": False, # We inject speed instructions
            "duration_control": False,
            "multi_speaker": False,
            "voice_design": True,
            "vram_gb": 8.0,
            "sample_rate": 24000
        }

    def is_available(self) -> bool:
        try:
            import voxcpm
            return True
        except ImportError:
            return False

    def load(self, ref_audio_path=None):
        if self.model is not None:
            return
        
        print("Loading VoxCPM 2 Backend...")
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

    def unload(self):
        if self.model is not None:
            # voxcpm keeps things in model.tts_model
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _inject_speed_prompt(self, text: str, speed: float) -> str:
        """Inject speed instructions at the start of the text for VoxCPM 2."""
        if speed > 1.15:
            return f"(faster) {text}"
        elif speed > 1.05:
            return f"(slightly faster) {text}"
        elif speed < 0.85:
            return f"(slower) {text}"
        elif speed < 0.95:
            return f"(slightly slower) {text}"
        return text

    def generate(self, text: str, language: str, output_path: str, ref_audio_path: str = None, speed: float = 1.0, duration: float = None) -> dict:
        if self.model is None:
            self.load()
            
        modified_text = self._inject_speed_prompt(text, speed)
        print(f"VoxCPM 2 generate: modified_text='{modified_text}', lang={language}")
        
        # Real generation
        # We pass reference_wav_path for voice cloning
        wav = self.model.generate(
            text=modified_text,
            reference_wav_path=ref_audio_path if ref_audio_path and os.path.exists(ref_audio_path) else None,
            normalize=True
        )
        
        sr = self.capabilities["sample_rate"]
        sf.write(output_path, wav, sr)
        
        # Calculate generated duration
        generated_duration = len(wav) / sr
        
        return {
            "duration": generated_duration,
            "path": output_path,
            "sample_rate": sr
        }
