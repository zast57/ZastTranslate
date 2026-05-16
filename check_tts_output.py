import whisperx
import torch
import soundfile as sf
import numpy as np
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading WhisperX on {device}...")
model = whisperx.load_model("small", device=device, compute_type="float16" if device == "cuda" else "float32")

def load_audio_sf(path):
    """Load audio with soundfile+librosa and resample to 16kHz mono float32 (WhisperX format)."""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
    return data.astype(np.float32)

for fname in ["temp/seg_14.62_temp.wav", "temp/seg_21.73_temp.wav", "temp/seg_29.04_temp.wav"]:
    print(f"\n--- {fname} ---")
    audio = load_audio_sf(fname)
    result = model.transcribe(audio, batch_size=4)
    print("Language detected:", result.get("language"))
    for seg in result["segments"]:
        print(f"  {seg['start']:.1f}-{seg['end']:.1f}: {seg['text']}")
