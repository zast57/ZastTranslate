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
        
        print("Loading VoxCPM 2 Backend (with denoiser/ZipEnhancer)...")
        from voxcpm import VoxCPM
        # load_denoiser=True: enables ZipEnhancer post-processing for noise reduction.
        # Costs ~2 GB VRAM but improves audio quality noticeably.
        self.model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=True)
        # Get actual sample rate from the underlying model
        self._sample_rate = getattr(self.model.tts_model, 'sample_rate', 24000)
        print(f"VoxCPM 2 sample rate: {self._sample_rate}")





        self._prompt_cache_key = None
        self._cached_prompt_cache = None

    def unload(self):
        if self.model is not None:
            # voxcpm keeps things in model.tts_model
            del self.model
            self.model = None
            self._prompt_cache_key = None
            self._cached_prompt_cache = None
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

    def _get_trimmed_ref(self, ref_path: str, max_seconds: float = 30.0) -> str:
        """
        Return a trimmed reference audio path (max 30s).
        VoxCPM2 docs: "5 to 30 seconds is a practical range".
        Passing a full 10-minute vocals.wav creates thousands of KV-cache tokens → 23+ GB VRAM.
        Trimmed to 30s: ~4-6 GB VRAM. Trimmed file is cached per source path.
        """
        cache_key = f"{ref_path}__trim{int(max_seconds)}s"
        if getattr(self, '_trimmed_ref_cache', {}).get(cache_key):
            return self._trimmed_ref_cache[cache_key]

        if not hasattr(self, '_trimmed_ref_cache'):
            self._trimmed_ref_cache = {}

        data, sr = sf.read(ref_path, always_2d=False)
        duration = len(data) / sr
        if duration <= max_seconds:
            self._trimmed_ref_cache[cache_key] = ref_path
            return ref_path

        # Take the first max_seconds
        max_samples = int(max_seconds * sr)
        trimmed = data[:max_samples]
        trim_path = ref_path.replace(".wav", f"__ref30s.wav")
        sf.write(trim_path, trimmed, sr)
        print(f"VoxCPM 2: reference audio trimmed {duration:.0f}s -> {max_seconds:.0f}s -> {trim_path}")
        self._trimmed_ref_cache[cache_key] = trim_path
        return trim_path

    def _get_or_build_prompt_cache(self, ref_path: str, denoise: bool = True):
        """
        Build and cache prompt_cache (voice embedding and audio features) for reference audio.
        VoxCPM normally re-runs ZipEnhancer denoiser (~1.5s) and build_prompt_cache (~1.0s)
        on EVERY single segment. Caching it here cuts segment latency by ~2.5s without quality loss.
        """
        if not ref_path or not os.path.exists(ref_path):
            return None

        cache_key = f"{os.path.abspath(ref_path)}__denoise_{denoise}"
        if getattr(self, '_prompt_cache_key', None) == cache_key and getattr(self, '_cached_prompt_cache', None) is not None:
            return self._cached_prompt_cache

        print(f"VoxCPM 2: Building voice embedding & prompt cache for: {ref_path} (denoise={denoise})...")
        actual_ref_path = ref_path

        if denoise and hasattr(self.model, 'denoiser') and self.model.denoiser is not None:
            denoised_cache_path = ref_path.replace(".wav", "__denoised.wav")
            if os.path.exists(denoised_cache_path):
                actual_ref_path = denoised_cache_path
            else:
                try:
                    self.model.denoiser.enhance(ref_path, output_path=denoised_cache_path)
                    actual_ref_path = denoised_cache_path
                except Exception as e:
                    print(f"VoxCPM 2: Denoiser warning: {e}, using raw reference audio.")
                    actual_ref_path = ref_path

        try:
            prompt_cache = self.model.tts_model.build_prompt_cache(
                reference_wav_path=actual_ref_path
            )
            self._prompt_cache_key = cache_key
            self._cached_prompt_cache = prompt_cache
            print(f"VoxCPM 2: Voice prompt cache successfully built and cached.")
            return self._cached_prompt_cache
        except Exception as e:
            print(f"VoxCPM 2: Could not build prompt cache: {e}, falling back to standard generate.")
            return None

    def generate(self, text: str, language: str, output_path: str, ref_audio_path: str = None, speed: float = 1.0, duration: float = None, gender: str = "Woman") -> dict:
        if self.model is None:
            self.load()
            
        final_ref = ref_audio_path if ref_audio_path and os.path.exists(ref_audio_path) else getattr(self, 'default_ref_audio', None)
        if final_ref and not os.path.exists(final_ref):
            final_ref = None

        # CRITICAL: trim reference audio to 30s max.
        # Full-length vocals.wav (10+ min) fills the LM KV-cache with thousands of tokens → 23+ GB VRAM.
        # 30s reference: same quality per VoxCPM2 docs, ~4-6 GB VRAM instead of 24 GB.
        if final_ref and os.path.exists(final_ref):
            final_ref = self._get_trimmed_ref(final_ref, max_seconds=30.0)

        modified_text = self._inject_speed_prompt(text, speed)
        # If no reference is provided (no cloning mode), we MUST use a cached default wav.
        # Otherwise, Voice Design will generate a slightly different voice for every sentence!
        if not final_ref:
            voices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "voices")
            os.makedirs(voices_dir, exist_ok=True)
            
            # Use gender-specific default voice
            gender_str = str(gender).lower()
            default_wav = os.path.join(voices_dir, f"default_{gender_str}.wav")
            
            if not os.path.exists(default_wav):
                print(f"VoxCPM 2: Generating a persistent default voice (default_{gender_str}.wav) to ensure consistency...")
                prompt = f"(A clear, professional and expressive {gender_str} voice) Hello, this is the default voice for your translations."
                wav = self.model.generate(
                    text=prompt,
                    normalize=False,
                    inference_timesteps=10,
                    denoise=True
                )
                sr = getattr(self.model.tts_model, 'sample_rate', 24000)
                sf.write(default_wav, wav, sr)
                print(f"VoxCPM 2: Saved default voice to {default_wav}")
            final_ref = default_wav
            
        safe_preview = modified_text[:60].encode('ascii', 'replace').decode('ascii')
        print(f"VoxCPM 2 generate: text='{safe_preview}', lang={language}, ref={final_ref}")
        
        # Check if we can use pre-built prompt cache (saves ~2.5s of redundant denoising & audio encoding)
        prompt_cache = self._get_or_build_prompt_cache(final_ref, denoise=True)
        if prompt_cache is not None:
            gen_iter = self.model.tts_model._generate_with_prompt_cache(
                target_text=modified_text,
                prompt_cache=prompt_cache,
                min_len=2,
                max_len=4096,
                inference_timesteps=10,
                cfg_value=2.0,
                retry_badcase=True,
                retry_badcase_max_times=2,
                retry_badcase_ratio_threshold=6.0,
                streaming=False,
            )
            wav_chunks = []
            for wav_chunk, _, _ in gen_iter:
                wav_chunks.append(wav_chunk.squeeze(0).cpu().numpy())
            wav = wav_chunks[0] if wav_chunks else np.zeros(int((self._sample_rate or 24000) * 0.5), dtype=np.float32)
        else:
            # Fallback to standard generate
            wav = self.model.generate(
                text=modified_text,
                reference_wav_path=final_ref,
                normalize=False,
                inference_timesteps=10,
                denoise=True,
                retry_badcase=True,
                retry_badcase_max_times=2,
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

