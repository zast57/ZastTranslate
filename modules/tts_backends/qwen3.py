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
            "speed_control": False,  # instruct_ids has no effect in clone mode (Base model)
            "duration_control": False,
            "multi_speaker": False,
            "voice_design": True,
            "vram_gb": 4.5,
            "sample_rate": 24000,
            "fitted_cps": 7.0,         # Calibrated to Qwen3-TTS actual speaking rate ~7.17 chars/sec
            "fitted_speed_factor": 1.0  # No speed control: use factor=1.0
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

        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                def _compute_default_rope_parameters(config, device=None, **kwargs):
                    base = getattr(config, "rope_theta", 10000.0)
                    dim = getattr(config, "head_dim", getattr(config, "hidden_size", 1024) // getattr(config, "num_attention_heads", 16))
                    import torch
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
                    return inv_freq, 1.0
                ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
                print("Patched ROPE_INIT_FUNCTIONS to include custom 'default'")
        except ImportError:
            pass

        from qwen_tts import Qwen3TTSModel
        
        # Fix compatibility: newer transformers requires pad_token_id on all configs
        try:
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerConfig
            if not hasattr(Qwen3TTSTalkerConfig, 'pad_token_id'):
                Qwen3TTSTalkerConfig.pad_token_id = None
                print("Patched Qwen3TTSTalkerConfig.pad_token_id for transformers compatibility")
        except ImportError:
            pass
        
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
                print(f"Qwen3-TTS loaded with SDPA (PyTorch native).")
            except Exception:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id, **load_kwargs,
                    attn_implementation="eager",
                )
                print(f"Qwen3-TTS loaded (Standard Attention).")

        # Cache the voice prompt if cloning
        if self._mode == "clone" and ref_audio_path and os.path.exists(ref_audio_path):
            # Always use x_vector_only_mode=True for cross-lingual dubbing.
            # ICL mode (x_vector_only_mode=False) encodes the phoneme patterns of the
            # reference language into the voice prompt. When the reference is in a
            # DIFFERENT language than the target (e.g. French ref → English output),
            # the encoded phonemes corrupt generation, producing German/Polish gibberish
            # instead of the target language. x_vector_only captures only the speaker
            # timbre (pitch, resonance) without language-specific phonetics — the
            # `language` parameter in generate() then correctly controls the output language.
            print("Caching voice prompt for Qwen3-TTS (x-vector only — cross-lingual mode)...")
            self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio_path,
                x_vector_only_mode=True,
            )
            self._icl_mode = False

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
                # x_vector_only_mode is NOT passed here — it is already encoded
                # inside each VoiceClonePromptItem built during load().
            }

            # Set max_new_tokens from TEXT LENGTH, not slot duration.
            # Qwen3-TTS generates ~95% of max_new_tokens regardless of content — the model
            # fills silence tokens to reach the cap. So max_new_tokens directly controls
            # the audio duration: 12 tokens/sec ÷ 7.17 chars/sec ÷ 0.95 ≈ 1.76 tokens/char
            max_tok = max(16, int(len(text) * 1.76))
            kwargs["max_new_tokens"] = max_tok

            if self.voice_clone_prompt:
                kwargs["voice_clone_prompt"] = self.voice_clone_prompt
            elif ref_audio_path and os.path.exists(ref_audio_path):
                kwargs["ref_audio"] = ref_audio_path

            # Note: instruct_ids has no effect on speaking rate in clone mode (Base model)
            # Speed calibration is handled upstream via fitted_cps=7.0 in capabilities

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
            # Apply max_new_tokens to prevent runaway generation.
            # CustomVoice model may generate indefinitely without this cap.
            # Same formula as clone mode: ~1.76 tokens/char at 12 Hz, 7.17 chars/sec.
            max_tok = max(16, int(len(text) * 1.76))
            kwargs["max_new_tokens"] = max_tok
            if instruct_text:
                kwargs["instruct"] = instruct_text
            
            wavs, sr = self.model.generate_custom_voice(**kwargs)

        wav = wavs[0]
        audio_data = wav.cpu().numpy() if hasattr(wav, 'cpu') else wav
        sf.write(output_path, audio_data, sr)
        
        return {
            "duration": len(audio_data) / sr,
            "path": output_path,
            "sample_rate": sr
        }
