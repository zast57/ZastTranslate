import os
import torch

# Nom du projet
APP_NAME = "ZastTranslate"
APP_VERSION = "0.9-beta"

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOICES_DIR = os.path.join(BASE_DIR, "voices")

# Creer les dossiers s'ils n'existent pas
for d in [TEMP_DIR, OUTPUT_DIR, MODELS_DIR, VOICES_DIR]:
    os.makedirs(d, exist_ok=True)

# GPU/CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    GPU_VRAM = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
except Exception:
    GPU_NAME = "Unknown"
    GPU_VRAM = "N/A"

# Synchronization constants
TOLERANCE_TOO_LONG = 0.15    # 150ms: beyond this, adjust speed
MAX_SPEED_FACTOR = 1.25      # Max native speedup (CustomVoice instruct)
MAX_STRETCH_FACTOR = 1.2     # Max librosa time_stretch (voice clone fallback)
MIN_SEGMENT_DURATION = 0.3   # Ignore segments < 300ms

# Never Cut Vocal mode
NEVER_CUT_WARNING = (
    "⚠️ 'Never Cut Vocal' mode enabled: all text will be spoken in full, "
    "but dubbing may drift out of sync with on-screen actions."
)

# Estimated characters per second by language family (for LLM pre-check)
CHARS_PER_SECOND = {
    "latin": 13,      # FR, EN, ES, DE, IT, PT (balance: reformulation quality vs TTS fit)
    "cjk": 7,         # ZH, JA, KO
    "arabic": 12,     # AR
    "default": 13
}

# Audio
WHISPERX_SAMPLE_RATE = 16000
DEMUCS_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 44100
OUTPUT_CHANNELS = 2  # stereo
OUTPUT_BIT_DEPTH = 16
MP3_BITRATE = "320k"

# Supported languages (language codes)
# Limited to languages supported by Qwen3-TTS for dubbing
LANGUAGES = {
    "Francais": "fra_Latn",
    "English": "eng_Latn",
    "Espanol": "spa_Latn",
    "Deutsch": "deu_Latn",
    "Italiano": "ita_Latn",
    "Portugues": "por_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Chinese (Simplifie)": "zho_Hans",
    "Russian": "rus_Cyrl",
}
