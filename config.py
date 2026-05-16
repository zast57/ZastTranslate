import os
import torch

# Nom du projet
APP_NAME = "ZastTranslate"
APP_VERSION = "1.00"

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

# NVIDIA Ampere+ (RTX 30xx/40xx) Acceleration
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("NVIDIA TF32 Acceleration Enabled (Massive speedup for RTX 3000/4000+ GPUs)")

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
# Full list covering VoxCPM2 (30 langs) + Qwen3-TTS (10 langs)
# The UI dynamically filters based on selected TTS + LLM backend capabilities
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
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva",
    "Dutch": "nld_Latn",
    "Polish": "pol_Latn",
    "Turkish": "tur_Latn",
    "Swedish": "swe_Latn",
    "Czech": "ces_Latn",
    "Romanian": "ron_Latn",
    "Hungarian": "hun_Latn",
    # VoxCPM2 additional languages
    "Burmese": "mya_Mymr",
    "Danish": "dan_Latn",
    "Finnish": "fin_Latn",
    "Greek": "ell_Grek",
    "Hebrew": "heb_Hebr",
    "Indonesian": "ind_Latn",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    "Malay": "zsm_Latn",
    "Norwegian": "nob_Latn",
    "Swahili": "swh_Latn",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn",
}
