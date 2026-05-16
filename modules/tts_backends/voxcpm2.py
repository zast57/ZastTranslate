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
        self._sample_rate = None

    @property
    def name(self) -> str:
        return "VoxCPM 2"

    @property
    def capabilities(self) -> dict:
        return {
            "languages": "all",  # VoxCPM2 supports 30 languages natively, no language tag needed
            "speed_control": False,
            "duration_control": False,
            "multi_speaker": False,
            "voice_design": True,
            "vram_gb": 8.0,
            "sample_rate": self._sample_rate or 24000,
            "fitted_speed_factor": 1.0  # No speed control available; CPS handled by fitted_cps_config.py
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
        
        self.default_ref_audio = ref_audio_path
        
        print("Loading VoxCPM 2 Backend...")
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
        # Get actual sample rate from the underlying model
        self._sample_rate = getattr(self.model.tts_model, 'sample_rate', 24000)
        print(f"VoxCPM 2 sample rate: {self._sample_rate}")

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
            
        final_ref = ref_audio_path if ref_audio_path and os.path.exists(ref_audio_path) else getattr(self, 'default_ref_audio', None)
        
        # If no reference is provided at all (e.g., Default Voice mode), VoxCPM 2 will use a random voice for EVERY segment.
        # To prevent this, we fallback to the video's own vocals, since gradio samples cause voice hallucinations.
        if not final_ref or not os.path.exists(final_ref):
            import glob
            vocals_files = glob.glob(os.path.join(TEMP_DIR, "htdemucs", "*", "vocals.wav"))
            if vocals_files:
                final_ref = vocals_files[0]
            else:
                final_ref = None
                
        modified_text = self._inject_speed_prompt(text, speed)
        print(f"VoxCPM 2 generate: modified_text='{modified_text}', lang={language}, ref={final_ref}")
        
        # Generate speech — normalize=False to avoid Chinese text normalizer mangling English
        wav = self.model.generate(
            text=modified_text,
            reference_wav_path=final_ref,
            normalize=False,
            retry_badcase=True,
            retry_badcase_max_times=3,
        )
        
        sr = self._sample_rate or 24000
        sf.write(output_path, wav, sr)
        
        # Calculate generated duration
        generated_duration = len(wav) / sr
        
        return {
            "duration": generated_duration,
            "path": output_path,
            "sample_rate": sr
        }

