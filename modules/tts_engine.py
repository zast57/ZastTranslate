import torch
import os
import soundfile as sf
import numpy as np
from modules.utils import cleanup_model, get_exact_duration
from config import DEVICE, TEMP_DIR, VOICES_DIR

class TTSEngine:
    def __init__(self):
        self.model = None
        self.mms_pipeline = None
        self.engine = None
        self.device = DEVICE
        self.ref_audio_path = None
        self.voice_clone_prompt = None  # Cached prompt for Qwen3-TTS

    def load_model(self, voice_path=None):
        """Load the best available TTS engine."""
        if self.engine is not None:
            return

        if voice_path:
            self.ref_audio_path = voice_path

        # Try Qwen3-TTS first (best quality + cloning)
        try:
            from qwen_tts import Qwen3TTSModel
            
            # Choose model variant: Base for cloning, CustomVoice for preset speakers
            if voice_path and os.path.exists(voice_path):
                model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                print(f"Loading Qwen3-TTS Base (voice cloning mode)...")
            else:
                model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                print(f"Loading Qwen3-TTS CustomVoice (preset voices)...")
            
            load_kwargs = {
                "device_map": "cuda:0" if self.device == "cuda" else "cpu",
                "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }
            
            # Try with flash_attention_2 first, fallback to default
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id, **load_kwargs,
                    attn_implementation="flash_attention_2",
                )
                print(f"Qwen3-TTS loaded with FlashAttention 2.")
            except Exception:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id, **load_kwargs,
                )
                print(f"Qwen3-TTS loaded (without FlashAttention).")
            
            self.engine = "qwen3-tts-clone" if "Base" in model_id else "qwen3-tts-custom"
            
            # Pre-compute voice clone prompt if ref audio available
            if self.ref_audio_path and os.path.exists(self.ref_audio_path) and "Base" in model_id:
                self._prepare_clone_prompt()
            
            return
        except ImportError:
            print("Qwen3-TTS not installed. Install with: pip install qwen-tts")
        except Exception as e:
            print(f"Qwen3-TTS loading failed: {e}")

        # Try Chatterbox (good quality + cloning)
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            print("Loading Chatterbox Multilingual TTS...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self.engine = "chatterbox"
            print("Chatterbox loaded.")
            return
        except ImportError:
            print("Chatterbox not installed. Install with: pip install chatterbox-tts")
        except Exception as e:
            print(f"Chatterbox loading failed: {e}")

        # Fallback to MMS-TTS (no cloning)
        print("Falling back to MMS-TTS (no voice cloning)...")
        try:
            from transformers import pipeline
            self.mms_pipeline = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-fra",
                device=0 if self.device == "cuda" else -1
            )
            self.engine = "mms-tts"
            print("MMS-TTS loaded.")
        except Exception as e:
            print(f"MMS-TTS also failed: {e}")
            self.engine = "silence"

    def _prepare_clone_prompt(self):
        """Pre-compute voice clone prompt from reference audio (Qwen3-TTS only)."""
        if self.engine != "qwen3-tts-clone" or not self.ref_audio_path:
            return
        try:
            print(f"Preparing voice clone prompt from: {self.ref_audio_path}")
            self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=self.ref_audio_path,
                ref_text="",  # Empty = auto transcribe
                x_vector_only_mode=True,  # Don't need ref_text
            )
            print("Voice clone prompt ready.")
        except Exception as e:
            print(f"Failed to prepare clone prompt: {e}")
            self.voice_clone_prompt = None

    def set_reference_audio(self, voice_path):
        """Set or change the reference audio for voice cloning."""
        self.ref_audio_path = voice_path
        self.voice_clone_prompt = None  # Reset cached prompt
        if self.engine == "qwen3-tts-clone" and self.model:
            self._prepare_clone_prompt()

    def get_engine_type(self):
        """Return the current engine type for speed control decisions."""
        return self.engine or "unknown"

    def synthesize_segment(self, text, language, output_path, voice_path=None, speaker_id=None, speed_factor=1.0):
        """Generate TTS audio for a text segment. speed_factor > 1.0 = faster speech."""
        if self.engine is None:
            self.load_model(voice_path=voice_path)

        ref_audio = voice_path or self.ref_audio_path

        if self.engine in ("qwen3-tts-clone", "qwen3-tts-custom"):
            return self._synthesize_qwen3(text, language, output_path, ref_audio, speed_factor=speed_factor)
        elif self.engine == "chatterbox":
            return self._synthesize_chatterbox(text, language, output_path, ref_audio)
        elif self.engine == "mms-tts":
            return self._synthesize_mms(text, output_path)
        else:
            return self._synthesize_silence(text, output_path)

    def _map_language(self, lang_code):
        """Map NLLB/internal language codes to Qwen3-TTS language names.
        Qwen3-TTS supports: Chinese, English, Japanese, Korean, German, French,
        Russian, Portuguese, Spanish, Italian.
        """
        # NLLB codes (fra_Latn, eng_Latn, etc.) — match by prefix
        nllb_map = {
            "fra": "French", "eng": "English", "spa": "Spanish",
            "deu": "German", "ita": "Italian", "por": "Portuguese",
            "jpn": "Japanese", "kor": "Korean", "zho": "Chinese",
            "rus": "Russian",
        }
        # Short codes (fr, en, etc.)
        short_map = {
            "fr": "French", "en": "English", "es": "Spanish",
            "de": "German", "it": "Italian", "pt": "Portuguese",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
            "ru": "Russian",
        }
        # Display names (Francais, Espanol, etc.) — case-insensitive
        display_map = {
            "francais": "French", "english": "English", "espanol": "Spanish",
            "deutsch": "German", "italiano": "Italian", "portugues": "Portuguese",
            "japanese": "Japanese", "korean": "Korean", "russian": "Russian",
            "chinese": "Chinese",
        }
        
        for prefix, lang_name in nllb_map.items():
            if lang_code.startswith(prefix):
                return lang_name
        if lang_code in short_map:
            return short_map[lang_code]
        lower = lang_code.lower().split(" ")[0]  # "Chinese (Simplifie)" -> "chinese"
        if lower in display_map:
            return display_map[lower]
        
        print(f"WARNING: Unknown language '{lang_code}', defaulting to French")
        return "French"

    def _build_speed_instruct(self, speed_factor):
        """Build a speed instruction string based on the speed factor.
        Uses Chinese descriptors matching the official Qwen TTS API docs:
        快速 / 偏快 / 中速 / 偏慢 / 缓慢
        """
        if speed_factor > 1.15:
            return "语速快速"
        elif speed_factor > 1.02:
            return "语速偏快"
        return None

    def _synthesize_qwen3(self, text, language, output_path, ref_audio=None, speed_factor=1.0):
        """Synthesize using Qwen3-TTS (clone or custom voice) with optional speed control.
        
        Speed control uses the native 'instruct' mechanism for both modes:
        - CustomVoice: passes instruct= parameter directly
        - Voice Clone: injects instruct_ids into **kwargs → core model.generate()
        """
        try:
            lang_name = self._map_language(language)
            instruct_text = self._build_speed_instruct(speed_factor)
            
            if self.engine == "qwen3-tts-clone":
                # Voice cloning mode (Base model)
                kwargs = {
                    "text": text,
                    "language": lang_name,
                    "x_vector_only_mode": True,
                }
                
                if self.voice_clone_prompt:
                    kwargs["voice_clone_prompt"] = self.voice_clone_prompt
                elif ref_audio and os.path.exists(ref_audio):
                    kwargs["ref_audio"] = ref_audio
                
                # Inject instruct_ids for speed control via **kwargs
                # The core model.generate() accepts instruct_ids for all model types
                if instruct_text:
                    instruct_ids = self.model._tokenize_texts(
                        [self.model._build_instruct_text(instruct_text)]
                    )
                    kwargs["instruct_ids"] = instruct_ids
                    print(f"  → Clone mode: native instruct='{instruct_text}' ({speed_factor:.2f}x)")
                
                wavs, sr = self.model.generate_voice_clone(**kwargs)
            else:
                # CustomVoice mode (preset speakers) with native speed control
                speaker_map = {
                    "French": "serena",
                    "English": "ryan",
                    "Chinese": "vivian",
                    "Japanese": "ono_anna",
                    "Korean": "sohee",
                    "German": "eric",
                    "Spanish": "serena",
                    "Italian": "serena",
                    "Portuguese": "dylan",
                    "Russian": "aiden",
                }
                speaker = speaker_map.get(lang_name, "ryan")
                
                if instruct_text:
                    print(f"  → CustomVoice: instruct='{instruct_text}' ({speed_factor:.2f}x)")
                
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=lang_name,
                    speaker=speaker,
                    instruct=instruct_text,
                )
            
            # Save output
            audio = wavs[0] if isinstance(wavs, list) else wavs
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            sf.write(output_path, audio, sr)
            return {"path": output_path, "duration": get_exact_duration(output_path)}
            
        except Exception as e:
            print(f"Qwen3-TTS error: {e}")
            return self._synthesize_silence(text, output_path)

    def _synthesize_chatterbox(self, text, language, output_path, ref_audio=None):
        """Synthesize using Chatterbox Multilingual TTS."""
        try:
            import torchaudio as ta
            
            lang_id_map = {
                "fra": "fr", "fr": "fr",
                "eng": "en", "en": "en",
                "spa": "es", "es": "es",
                "deu": "de", "de": "de",
                "ita": "it", "it": "it",
            }
            
            lang_id = "fr"  # default
            for prefix, lid in lang_id_map.items():
                if language.startswith(prefix):
                    lang_id = lid
                    break
            
            kwargs = {"language_id": lang_id}
            if ref_audio and os.path.exists(ref_audio):
                kwargs["audio_prompt_path"] = ref_audio
            
            wav = self.model.generate(text, **kwargs)
            ta.save(output_path, wav, self.model.sr)
            return {"path": output_path, "duration": get_exact_duration(output_path)}
            
        except Exception as e:
            print(f"Chatterbox error: {e}")
            return self._synthesize_silence(text, output_path)

    def _synthesize_mms(self, text, output_path):
        """Synthesize using MMS-TTS (no cloning)."""
        try:
            output = self.mms_pipeline(text)
            audio_data = output['audio']
            sample_rate = output['sampling_rate']
            
            if audio_data.ndim > 1 and audio_data.shape[0] < 10:
                audio_data = audio_data.T
            
            sf.write(output_path, audio_data, sample_rate)
            return {"path": output_path, "duration": get_exact_duration(output_path)}
        except Exception as e:
            print(f"MMS-TTS error: {e}")
            return self._synthesize_silence(text, output_path)

    def _synthesize_silence(self, text, output_path):
        """Last resort: generate silence."""
        estimated_duration = max(0.5, len(text) / 15.0)
        sr = 24000
        audio = np.zeros(int(estimated_duration * sr), dtype=np.float32)
        sf.write(output_path, audio, sr)
        return {"path": output_path, "duration": estimated_duration}

    def cleanup(self):
        """Release all models from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.mms_pipeline is not None:
            cleanup_model(self.mms_pipeline)
            self.mms_pipeline = None
        self.engine = None
        self.ref_audio_path = None
        self.voice_clone_prompt = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    tts = TTSEngine()
    tts.load_model()
    print(f"Engine: {tts.engine}")
