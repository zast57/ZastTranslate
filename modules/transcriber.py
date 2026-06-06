import whisperx
import torch
import os
from modules.utils import cleanup_model
from config import DEVICE, GPU_VRAM

def split_segment(seg, max_duration=8.0, max_chars=75):
    """
    Split a single segment into smaller segments if it exceeds max_duration or max_chars.
    Uses word-level timestamps if available, otherwise falls back to character-proportional splitting.
    """
    duration = seg.get('end', 0.0) - seg.get('start', 0.0)
    text = seg.get('text', '')
    
    if duration <= max_duration and len(text) <= max_chars:
        return [seg]
        
    words = seg.get('words', [])
    # If we have word-level timestamps, use them
    if words:
        sub_segs = []
        curr_words = []
        
        # Helper to build a segment from a list of word dicts
        def build_sub_seg(word_list):
            if not word_list:
                return None
            # Find first and last word with valid start/end
            valid_starts = [w['start'] for w in word_list if 'start' in w]
            valid_ends = [w['end'] for w in word_list if 'end' in w]
            
            sub_start = valid_starts[0] if valid_starts else seg.get('start', 0.0)
            sub_end = valid_ends[-1] if valid_ends else seg.get('end', 0.0)
            
            # Reconstruct text
            sub_text = " ".join([w['word'] for w in word_list])
            
            return {
                'start': sub_start,
                'end': sub_end,
                'text': sub_text,
                'words': word_list
            }
            
        for i, w in enumerate(words):
            curr_words.append(w)
            
            # Determine if we should split after this word
            word_text = w.get('word', '')
            has_strong_punct = any(word_text.endswith(p) for p in ('.', '?', '!', ';'))
            has_weak_punct = any(word_text.endswith(p) for p in (',', ':', '-'))
            
            curr_sub_seg_temp = build_sub_seg(curr_words)
            curr_duration = curr_sub_seg_temp['end'] - curr_sub_seg_temp['start']
            curr_len = len(curr_sub_seg_temp['text'])
            
            # Check if there's a next word to decide if we split
            is_last = (i == len(words) - 1)
            
            should_split = False
            if not is_last:
                if has_strong_punct and curr_duration >= 3.0:
                    should_split = True
                elif has_weak_punct and curr_duration >= 5.0:
                    should_split = True
                elif curr_duration >= max_duration:
                    should_split = True
                elif curr_len >= max_chars:
                    should_split = True
                    
            if should_split:
                sub_seg = build_sub_seg(curr_words)
                if sub_seg:
                    sub_segs.append(sub_seg)
                curr_words = []
                
        # Append remaining
        if curr_words:
            sub_seg = build_sub_seg(curr_words)
            if sub_seg:
                sub_segs.append(sub_seg)
                
        # Adjust contiguous segment boundaries to avoid gaps/overlaps
        for idx in range(len(sub_segs) - 1):
            if sub_segs[idx]['end'] > sub_segs[idx+1]['start']:
                mid = (sub_segs[idx]['end'] + sub_segs[idx+1]['start']) / 2
                sub_segs[idx]['end'] = mid
                sub_segs[idx+1]['start'] = mid
                
        return sub_segs if sub_segs else [seg]
        
    else:
        # Fallback: split by text length proportionally
        tokens = text.split()
        if not tokens:
            return [seg]
            
        # Group tokens into chunks that fit max_chars
        chunks = []
        curr_chunk = []
        curr_len = 0
        for t in tokens:
            if curr_len + len(t) + 1 > max_chars and curr_chunk:
                chunks.append(" ".join(curr_chunk))
                curr_chunk = [t]
                curr_len = len(t)
            else:
                curr_chunk.append(t)
                curr_len += len(t) + 1
        if curr_chunk:
            chunks.append(" ".join(curr_chunk))
            
        # Distribute time proportionally to character lengths of chunks
        total_chars = sum(len(c) for c in chunks)
        if total_chars == 0:
            return [seg]
            
        sub_segs = []
        start_time = seg.get('start', 0.0)
        seg_duration = seg.get('end', 0.0) - start_time
        
        for c in chunks:
            chunk_dur = (len(c) / total_chars) * seg_duration
            end_time = start_time + chunk_dur
            sub_segs.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'text': c
            })
            start_time = end_time
            
        return sub_segs

def split_long_segments(segments, max_duration=8.0, max_chars=75):
    """Split all segments in the list that exceed threshold duration or character count."""
    new_segments = []
    for seg in segments:
        new_segments.extend(split_segment(seg, max_duration, max_chars))
    return new_segments

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
        
        # Re-enable TF32 — pyannote disables it for reproducibility,
        # but we need it for fast LLM/TTS inference on Ampere+ GPUs
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 re-enabled after transcription")
        
        split_segs = split_long_segments(result["segments"], max_duration=8.0, max_chars=75)
        print(f"Split long segments: {len(result['segments'])} -> {len(split_segs)}")
        return {
            "language": detected_lang,
            "segments": split_segs
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
