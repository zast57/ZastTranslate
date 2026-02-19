import whisperx
import torch
import os
from modules.utils import cleanup_model
from config import DEVICE, GPU_VRAM

class Transcriber:
    def __init__(self, model_size="large-v3", compute_type="float16"):
        self.model_size = model_size
        self.compute_type = compute_type if DEVICE == "cuda" else "int8"
        self.device = DEVICE

    def transcribe(self, audio_path, language=None, enable_diarization=True):
        """
        Transcribe audio with WhisperX.
        Returns {"language": str, "segments": list}
        """
        print(f"Loading WhisperX {self.model_size} on {self.device}...")
        try:
            model = whisperx.load_model(
                self.model_size, 
                self.device, 
                compute_type=self.compute_type,
                language=language
            )
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

        print("Transcription in progress...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        detected_lang = result.get("language", "unknown")
        
        # Alignment
        print("Word-level alignment...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=self.device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        # Cleanup alignment models
        cleanup_model(model_a)
        cleanup_model(model)
        return {
            "language": detected_lang,
            "segments": result["segments"]
        }

    def cleanup(self):
        cleanup_model(None)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        t = Transcriber(model_size="base") # petit modele pour test
        res = t.transcribe(sys.argv[1], enable_diarization=False)
        print(f"Langue: {res['language']}")
        for s in res['segments']:
            print(f"{s['start']}-{s['end']}: {s['text']}")
