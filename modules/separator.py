import torch
import os
import subprocess
from modules.utils import cleanup_model
from config import DEVICE, TEMP_DIR

class VocalSeparator:
    def __init__(self, model_name="htdemucs"):
        self.model_name = model_name

    def separate(self, audio_path):
        """
        Separate vocals from background audio using Demucs.
        Uses the demucs command line for simplicity and robustness.
        Returns {"vocals": str, "background": str}
        """
        print(f"Separating audio with {self.model_name}...")
        
        # Demucs command with --two-stems to get vocals.wav + no_vocals.wav
        cmd = [
            "demucs",
            "-n", self.model_name,
            "--out", TEMP_DIR,
            "--device", DEVICE,
            "--two-stems", "vocals",
            audio_path
        ]
        subprocess.run(cmd, check=True)

        # Default Demucs output path: out_dir/model_name/track_name/
        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(TEMP_DIR, self.model_name, track_name)
        
        vocals_path = os.path.join(output_dir, "vocals.wav")
        background_path = os.path.join(output_dir, "no_vocals.wav")
        
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocals not found at {vocals_path}")
            
        return {"vocals": vocals_path, "background": background_path}

    def cleanup(self):
        # Nothing to clean up since we use subprocess
        pass

if __name__ == "__main__":
    import sys
    sep = VocalSeparator()
    if len(sys.argv) > 1:
        res = sep.separate(sys.argv[1])
        print(f"Vocals: {res['vocals']}")
        print(f"Background: {res['background']}")
    else:
        print("Usage: python separator.py [AUDIO_PATH]")
