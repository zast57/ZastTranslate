"""
ZastTranslate - Environment Verification
Checks all critical dependencies after installation.
"""
import sys

errors = []

print("=" * 50)
print("ZastTranslate - Environment Check")
print("=" * 50)

# 1. PyTorch + CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_ok else "None"
    print(f"[OK] torch {torch.__version__} | CUDA: {cuda_ok} | GPU: {gpu}")
    if not cuda_ok:
        print("  [WARN] No CUDA - will run on CPU (much slower)")
except ImportError:
    errors.append("torch")
    print("[FAIL] torch not installed")

# 2. torchvision
try:
    import torchvision
    print(f"[OK] torchvision {torchvision.__version__}")
except ImportError:
    errors.append("torchvision")
    print("[FAIL] torchvision not installed")

# 3. torchaudio
try:
    import torchaudio
    print(f"[OK] torchaudio {torchaudio.__version__}")
except ImportError:
    errors.append("torchaudio")
    print("[FAIL] torchaudio not installed")

# 4. Gradio
try:
    import gradio
    print(f"[OK] gradio {gradio.__version__}")
except ImportError:
    errors.append("gradio")
    print("[FAIL] gradio not installed")

# 5. Transformers
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline
    import transformers
    print(f"[OK] transformers {transformers.__version__}")
except ImportError as e:
    errors.append("transformers")
    print(f"[FAIL] transformers: {e}")

# 6. WhisperX
try:
    import whisperx
    ver = getattr(whisperx, "__version__", "installed")
    print(f"[OK] whisperx {ver}")
except ImportError:
    errors.append("whisperx")
    print("[FAIL] whisperx not installed")

# 7. Demucs
try:
    import demucs
    ver = getattr(demucs, "__version__", "installed")
    print(f"[OK] demucs {ver}")
except ImportError:
    errors.append("demucs")
    print("[FAIL] demucs not installed")

# 8. Audio processing
try:
    import librosa
    print(f"[OK] librosa {librosa.__version__}")
except ImportError:
    errors.append("librosa")
    print("[FAIL] librosa not installed")

try:
    import soundfile
    print(f"[OK] soundfile {soundfile.__version__}")
except ImportError:
    errors.append("soundfile")
    print("[FAIL] soundfile not installed")

try:
    import pydub
    ver = getattr(pydub, "__version__", "installed")
    print(f"[OK] pydub {ver}")
except ImportError:
    errors.append("pydub")
    print("[FAIL] pydub not installed")

# 9. numpy version check
try:
    import numpy
    ver = tuple(int(x) for x in numpy.__version__.split(".")[:2])
    if ver[0] >= 2:
        errors.append("numpy")
        print(f"[FAIL] numpy {numpy.__version__} - must be < 2.0")
    else:
        print(f"[OK] numpy {numpy.__version__}")
except ImportError:
    errors.append("numpy")
    print("[FAIL] numpy not installed")

# 10. yt-dlp
try:
    import yt_dlp
    print(f"[OK] yt-dlp {yt_dlp.version.__version__}")
except ImportError:
    errors.append("yt-dlp")
    print("[FAIL] yt-dlp not installed")

# 11. ffmpeg-python
try:
    import ffmpeg
    print(f"[OK] ffmpeg-python")
except ImportError:
    errors.append("ffmpeg-python")
    print("[FAIL] ffmpeg-python not installed")

# 12. accelerate
try:
    import accelerate
    print(f"[OK] accelerate {accelerate.__version__}")
except ImportError:
    errors.append("accelerate")
    print("[FAIL] accelerate not installed")

# 13. Qwen3-TTS (optional)
print("-" * 50)
try:
    from qwen_tts import Qwen3TTSModel
    print(f"[OK] qwen-tts (voice cloning available)")
except ImportError:
    print("[INFO] qwen-tts not installed (voice cloning disabled)")
    print("  Run 'Install Voice Cloning (Qwen3-TTS)' from menu")

# Summary
print("=" * 50)
if errors:
    print(f"[FAIL] {len(errors)} missing: {', '.join(errors)}")
    sys.exit(1)
else:
    print("[OK] All dependencies verified!")
    sys.exit(0)
