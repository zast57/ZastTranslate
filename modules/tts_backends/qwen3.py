import torch
import os
import soundfile as sf
import numpy as np
from .base import TTSBackend
from config import DEVICE, TEMP_DIR

class Qwen3Backend(TTSBackend):
    def __init__(self):
        super().__init__()
        self.device = DEVICE
        self.voice_clone_prompt = None

    @property
    def name(self) -> str:
        return "Qwen3-TTS"

    @property
    def capabilities(self) -> dict:
        return {
            "languages": ["en", "fr", "es", "de", "it", "pt", "ja", "ko", "zh", "ru"],
            "speed_control": False, # Uses native instruct for slight adjustments, but librosa for strict control
            "duration_control": False,
            "multi_speaker": False,
            "voice_design": True,
            "vram_gb": 4.5,
            "sample_rate": 24000
        }

    def is_available(self) -> bool:
        try:
            import qwen_tts
            return True
        except ImportError:
            return False

    def load(self, ref_audio_path=None):
        if self.model is not None:
            return

        from qwen_tts import Qwen3TTSModel
        
        # Choose model variant: Base for cloning, CustomVoice for preset speakers
        if ref_audio_path and os.path.exists(ref_audio_path):
            model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            print(f"Loading Qwen3-TTS Base (voice cloning mode)...")
            self._mode = "clone"
        else:
            model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            print(f"Loading Qwen3-TTS CustomVoice (preset voices)...")
            self._mode = "custom"
        
        load_kwargs = {
            "device_map": "cuda:0" if self.device == "cuda" else "cpu",
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }
        
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_id, **load_kwargs,
                attn_implementation="flash_attention_2",
            )
            print(f"Qwen3-TTS loaded with FlashAttention 2.")
        except Exception:
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id, **load_kwargs,
                    attn_implementation="sdpa",
                )
                print(f"Qwen3-TTS loaded with SDPA (Fast Attention).")
            except Exception:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id, **load_kwargs,
                )
                print(f"Qwen3-TTS loaded (Standard Attention).")

        # Cache the voice prompt if cloning
        if self._mode == "clone" and ref_audio_path and os.path.exists(ref_audio_path):
            print("Caching voice prompt for Qwen3-TTS...")
            self.voice_clone_prompt = self.model.get_voice_clone_prompt(ref_audio_path)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.voice_clone_prompt = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _map_language(self, lang_code):
        short_map = {
            "fr": "French", "en": "English", "es": "Spanish",
            "de": "German", "it": "Italian", "pt": "Portuguese",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
            "ru": "Russian",
        }
        if lang_code in short_map:
            return short_map[lang_code]
        return "French" # fallback

    def _build_speed_instruct(self, speed_factor):
        if speed_factor > 1.15:
            return "语速快速"
        elif speed_factor > 1.02:
            return "语速偏快"
        return None

    def generate(self, text: str, language: str, output_path: str, ref_audio_path: str = None, speed: float = 1.0, duration: float = None) -> dict:
        if self.model is None:
            self.load(ref_audio_path)

        lang_name = self._map_language(language)
        instruct_text = self._build_speed_instruct(speed)
        
        if self._mode == "clone":
            kwargs = {
                "text": text,
                "language": lang_name,
                "x_vector_only_mode": True,
            }
            
            if self.voice_clone_prompt:
                kwargs["voice_clone_prompt"] = self.voice_clone_prompt
            elif ref_audio_path and os.path.exists(ref_audio_path):
                kwargs["ref_audio"] = ref_audio_path
            
            if instruct_text:
                instruct_ids = self.model._tokenize_texts(
                    [self.model._build_instruct_text(instruct_text)]
                )
                kwargs["instruct_ids"] = instruct_ids
            
            wavs, sr = self.model.generate_voice_clone(**kwargs)
        else:
            speaker_map = {
                "French": "serena", "English": "ryan", "Chinese": "vivian",
                "Japanese": "ono_anna", "Korean": "sohee", "German": "eric",
                "Spanish": "serena", "Italian": "serena", "Portuguese": "dylan",
                "Russian": "aiden",
            }
            speaker = speaker_map.get(lang_name, "ryan")
            
            kwargs = {
                "text": text,
                "language": lang_name,
                "speaker": speaker,
            }
            if instruct_text:
                kwargs["instruct"] = instruct_text
            
            wavs, sr = self.model.generate_custom_voice(**kwargs)

        audio_data = wavs[0].cpu().numpy()
        sf.write(output_path, audio_data, sr)
        
        return {
            "duration": len(audio_data) / sr,
            "path": output_path,
            "sample_rate": sr
        }
