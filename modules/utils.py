import subprocess
import os
import json
import gc
import torch
import librosa
import soundfile as sf
import numpy as np

def get_exact_duration(filepath):
    """Return the EXACT duration in seconds via ffprobe (high-precision float)"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"ffprobe error on {filepath}: {e}")
        return 0.0

def convert_sample_rate(input_path, output_path, target_sr, channels=1):
    """Convert sample rate and channel count of an audio file via ffmpeg"""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', str(target_sr), '-ac', str(channels),
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)

def cleanup_model(model):
    """Release a model from VRAM/RAM cleanly"""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_progress(progress_data, filepath="temp/progress.json"):
    """Save pipeline state for error recovery"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def load_progress(filepath="temp/progress.json"):
    """Load a saved pipeline state"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def format_timestamp(seconds):
    """Convert seconds to SRT format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_from_srt_timestamp(ts):
    """Convert an SRT timestamp (HH:MM:SS,mmm) to float seconds"""
    parts = ts.replace(',', '.').split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

if __name__ == "__main__":
    # Quick unit test
    print("Test utils.py...")
    print(f"Format timestamp 125.5s -> {format_timestamp(125.5)}")
    print(f"Seconds from timestamp 00:02:05,500 -> {seconds_from_srt_timestamp('00:02:05,500')}")
