import whisperx
import torch
import os
from modules.utils import cleanup_model
from config import DEVICE, GPU_VRAM

MIN_CUE_WORDS = 3
MIN_CUE_DURATION_MS = 400

def split_segment(seg, max_duration=8.0, max_chars=75, min_words=MIN_CUE_WORDS, min_duration_ms=MIN_CUE_DURATION_MS):
    """
    Split a single segment into smaller cues if it exceeds max_duration or max_chars.
    Uses word-level timestamps from WhisperX to strictly avoid orphans:
    - No cue with fewer than min_words (3) words unless followed by strong punctuation (. ? ! … ; :).
    - No cue with duration under min_duration_ms (400ms) unless followed by strong punctuation.
    - Remaining words under threshold are kept on the current cue or merged into the previous cue.
    """
    duration = seg.get('end', 0.0) - seg.get('start', 0.0)
    text = seg.get('text', '')
    min_duration = min_duration_ms / 1000.0
    
    if duration <= max_duration and len(text) <= max_chars:
        return [seg]
        
    words = seg.get('words', [])
    if words:
        # 1. Fill missing word-level timestamps via interpolation
        seg_start = float(seg.get('start', 0.0))
        seg_end = float(seg.get('end', seg_start + 1.0))
        
        # Ensure words have start and end
        for idx, w in enumerate(words):
            if 'start' not in w or w['start'] is None:
                w['start'] = seg_start if idx == 0 else words[idx-1].get('end', seg_start)
            if 'end' not in w or w['end'] is None:
                w['end'] = w['start'] + 0.1
            if w['end'] < w['start']:
                w['end'] = w['start'] + 0.1

        sub_segs = []
        curr_words = []
        
        def build_sub_seg(word_list):
            if not word_list:
                return None
            sub_start = word_list[0]['start']
            sub_end = word_list[-1]['end']
            sub_text = " ".join([w.get('word', '').strip() for w in word_list if w.get('word', '').strip()])
            return {
                'start': round(sub_start, 3),
                'end': round(sub_end, 3),
                'text': sub_text,
                'words': list(word_list)
            }
            
        strong_puncts = ('.', '?', '!', ';', '…')
        weak_puncts = (',', ':', '-')
        
        for i, w in enumerate(words):
            curr_words.append(w)
            
            word_text = w.get('word', '').strip()
            has_strong_punct = any(word_text.endswith(p) for p in strong_puncts)
            has_weak_punct = any(word_text.endswith(p) for p in weak_puncts)
            
            curr_dur = curr_words[-1]['end'] - curr_words[0]['start']
            curr_len = sum(len(cw.get('word', '')) for cw in curr_words) + (len(curr_words) - 1)
            
            is_last = (i == len(words) - 1)
            if is_last:
                break
                
            rem_words = words[i+1:]
            rem_count = len(rem_words)
            rem_dur = rem_words[-1]['end'] - rem_words[0]['start']
            rem_last_text = rem_words[-1].get('word', '').strip()
            rem_ends_strong = any(rem_last_text.endswith(p) for p in strong_puncts)
            
            # Check if splitting here would leave an invalid orphan remainder
            rem_is_too_short = (rem_count < min_words or rem_dur < min_duration) and not rem_ends_strong
            
            should_split = False
            if not rem_is_too_short:
                if has_strong_punct and curr_dur >= 2.5:
                    should_split = True
                elif has_weak_punct and curr_dur >= 4.5:
                    should_split = True
                elif curr_dur >= max_duration:
                    should_split = True
                elif curr_len >= max_chars:
                    should_split = True
                    
            if should_split:
                sub_seg = build_sub_seg(curr_words)
                if sub_seg:
                    sub_segs.append(sub_seg)
                curr_words = []
                
        # Handle remaining words
        if curr_words:
            curr_dur = curr_words[-1]['end'] - curr_words[0]['start']
            curr_last_text = curr_words[-1].get('word', '').strip()
            curr_ends_strong = any(curr_last_text.endswith(p) for p in strong_puncts)
            
            is_orphan = (len(curr_words) < min_words or curr_dur < min_duration) and not curr_ends_strong
            
            if is_orphan and sub_segs:
                # Merge into the previous sub-segment instead of expelling an orphan
                prev = sub_segs[-1]
                prev['words'].extend(curr_words)
                prev['end'] = round(curr_words[-1]['end'], 3)
                prev['text'] = " ".join([w.get('word', '').strip() for w in prev['words'] if w.get('word', '').strip()])
            else:
                sub_seg = build_sub_seg(curr_words)
                if sub_seg:
                    sub_segs.append(sub_seg)
                    
        # Adjust contiguous segment boundaries to prevent overlaps
        for idx in range(len(sub_segs) - 1):
            if sub_segs[idx]['end'] > sub_segs[idx+1]['start']:
                # Set previous end strictly at or before next start
                sub_segs[idx]['end'] = round(max(sub_segs[idx]['start'] + 0.1, sub_segs[idx+1]['start']), 3)
                
        return sub_segs if sub_segs else [seg]
        
    else:
        # Fallback: split by text length proportionally without word timestamps
        tokens = text.split()
        if not tokens:
            return [seg]
            
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
            # Check if last chunk is too small (< min_words), merge with previous chunk if so
            if len(curr_chunk) < min_words and chunks:
                chunks[-1] += " " + " ".join(curr_chunk)
            else:
                chunks.append(" ".join(curr_chunk))
            
        total_chars = sum(len(c) for c in chunks)
        if total_chars == 0:
            return [seg]
            
        sub_segs = []
        start_time = seg.get('start', 0.0)
        seg_duration = max(0.1, seg.get('end', 0.0) - start_time)
        
        for c in chunks:
            chunk_dur = (len(c) / total_chars) * seg_duration
            end_time = start_time + chunk_dur
            sub_segs.append({
                'start': round(start_time, 3),
                'end': round(end_time, 3),
                'text': c
            })
            start_time = end_time
            
        return sub_segs

def split_long_segments(segments, max_duration=8.0, max_chars=75, min_words=MIN_CUE_WORDS, min_duration_ms=MIN_CUE_DURATION_MS):
    """Split all segments in the list that exceed threshold duration or character count."""
    new_segments = []
    for seg in segments:
        new_segments.extend(split_segment(seg, max_duration, max_chars, min_words, min_duration_ms))
    return new_segments

DEFAULT_INITIAL_PROMPT = (
    "ZastTranslate, Pinokio, WhisperX, large-v3, Qwen, Qwen3.5-9B, VoxCPM 2, "
    "FLUX.1-schnell, Fitted, Bulk Mode, Translation et Dubbing, Blog Studio, "
    "Clean Fillers, 16:9, 9:16, longues traînes, tics IA, Demucs, pyannote, Ollama, vLLM, "
    "Claude, Claude.ai, ChatGPT, OpenAI, GPT-4, DeepSeek, PyTorch, LoRA, ElevenLabs, "
    "Kokoro, F5-TTS, CosyVoice, Midjourney, Stable Diffusion, ComfyUI, Hugging Face, "
    "FFmpeg, VRAM, CUDA, GGUF, Python, GitHub, YouTube, Gradio, LLM, IA, API, SRT, VTT, TTS, "
    "WordPress, TikTok."
)

def merge_orphan_punctuation_segments(segments):
    """
    Merge segments that consist only of punctuation (e.g. '?', '.', '!', '...', ',', ';', ':')
    into the previous segment, eliminating isolated single-punctuation subtitle lines.
    """
    if not segments:
        return segments

    import re
    cleaned = []
    for seg in segments:
        text = seg.get("text", "").strip()
        is_only_punct = bool(re.match(r"^[\s\.\?\!\,\;\:\-\–\—\…\¿\¡\'\"\(\)\[\]\«\»]+$", text))
        
        if is_only_punct:
            if cleaned:
                prev = cleaned[-1]
                prev_text = prev.get("text", "").rstrip()
                # In French/English typography, handle spacing before punctuation
                if text in ("?", "!", ";", ":"):
                    if not prev_text.endswith(text):
                        prev["text"] = f"{prev_text} {text}"
                else:
                    if not prev_text.endswith(text):
                        prev["text"] = f"{prev_text}{text}"
                # Extend previous segment end timestamp
                if seg.get("end") and seg.get("end") > prev.get("end", 0.0):
                    prev["end"] = seg["end"]
                # Also merge words if present
                if "words" in prev and "words" in seg:
                    prev["words"] = prev.get("words", []) + seg.get("words", [])
            else:
                continue
        else:
            cleaned.append(seg)
            
    return cleaned

class Transcriber:
    def __init__(self, model_size="large-v3", compute_type="float16"):
        self.model_size = model_size
        self.compute_type = compute_type if DEVICE == "cuda" else "int8"
        self.device = DEVICE

    def transcribe(self, audio_path, language=None, enable_diarization=True, initial_prompt=None):
        """
        Transcribe audio with WhisperX.
        Returns {"language": str, "segments": list}
        """
        prompt = initial_prompt or DEFAULT_INITIAL_PROMPT
        print(f"Loading WhisperX {self.model_size} on {self.device} with initial_prompt context...")
        asr_options = {
            "initial_prompt": prompt,
            "condition_on_previous_text": False,
        }
        try:
            model = whisperx.load_model(
                self.model_size, 
                self.device, 
                compute_type=self.compute_type,
                language=language,
                asr_options=asr_options
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
        
        # Step 3: Merge solitary orphan punctuation lines and remove empty cues
        split_segs = merge_orphan_punctuation_segments(split_segs)
        from modules.srt_cleaner import (
            remove_empty_cues_and_redistribute,
            fix_inter_cue_casing,
            apply_asr_corrections_cross_cues,
            normalize_timecodes,
        )
        split_segs = remove_empty_cues_and_redistribute(split_segs, text_key="text")
        
        # Step 4: Inter-cue casing (protects acronyms, proper nouns, single accented letters)
        split_segs = fix_inter_cue_casing(split_segs, text_key="text", lang_code=detected_lang)
        
        # Step 5: Post-ASR dictionary corrections on reconstituted text
        split_segs = apply_asr_corrections_cross_cues(split_segs, text_key="text")
        
        # Step 6: Strict timecode normalization (end[i] <= start[i+1] - 40ms, min 400ms duration)
        split_segs = normalize_timecodes(split_segs, min_gap_ms=40, min_cue_duration_ms=MIN_CUE_DURATION_MS, text_key="text")
            
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
