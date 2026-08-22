import re
import os
from typing import List, Dict, Any, Optional
from modules.utils import format_timestamp, seconds_from_srt_timestamp

# Multilingual conversational filler patterns (case-insensitive)
FILLER_PATTERNS_BY_LANG = {
    "fr": [
        r"^(?:donc\s+voil[aà]|donc\s+alors|voil[aà]\s+donc|en\s+fait\s+donc|du\s+coup\s+donc|alors\s+voil[aà])\b[,;\s]*",
        r"^(?:donc|voil[aà]|alors|en\s+fait|du\s+coup|euh|ben|bah|bon\s+ben|bon\s+bah)\b[,;\s]*",
    ],
    "en": [
        r"^(?:so\s+basically|so\s+like|you\s+know\s+like|actually\s+so|well\s+so)\b[,;\s]*",
        r"^(?:so|well|actually|basically|you\s+know|like|um|uh|er|ah)\b[,;\s]*",
    ],
    "es": [
        r"^(?:bueno\s+pues|o\s+sea\s+que|entonces\s+bueno)\b[,;\s]*",
        r"^(?:bueno|pues|entonces|o\s+sea|eh|em|este)\b[,;\s]*",
    ],
    "de": [
        r"^(?:also\s+quasi|also\s+halt|sozusagen\s+also)\b[,;\s]*",
        r"^(?:also|halt|quasi|sozusagen|ähm|äh|na\s+ja)\b[,;\s]*",
    ],
    "it": [
        r"^(?:quindi\s+ecco|allora\s+quindi|cioè\s+quindi)\b[,;\s]*",
        r"^(?:quindi|allora|ecco|cioè|ehm|eh|insomma)\b[,;\s]*",
    ],
    "pt": [
        r"^(?:então\s+tipo|bom\s+então|tipo\s+assim)\b[,;\s]*",
        r"^(?:então|tipo|bom|né|tipo\s+assim|aí)\b[,;\s]*",
    ],
}

FILLER_WORDS_BY_LANG = {
    "fr": {"donc", "voila", "voilà", "alors", "euh", "ben", "bah", "quoi"},
    "en": {"so", "um", "uh", "er", "ah", "like", "basically", "actually"},
    "es": {"bueno", "pues", "entonces", "eh", "em", "este"},
    "de": {"also", "halt", "quasi", "ähm", "äh"},
    "it": {"quindi", "allora", "ecco", "cioè", "ehm"},
    "pt": {"então", "tipo", "bom", "né", "aí"},
}

MID_HESITATION_PATTERNS = {
    "fr": (r'[,;\s]*\b(?:euh|ben|bah)\b[,;\s]*', r'[,;\s]+\b(?:voil[aà]|quoi)\s*$'),
    "en": (r'[,;\s]*\b(?:um|uh|er|ah)\b[,;\s]*', r'[,;\s]+\b(?:you\s+know|right)\s*$'),
    "es": (r'[,;\s]*\b(?:eh|em|este)\b[,;\s]*', r'[,;\s]+\b(?:viste|sabes)\s*$'),
    "de": (r'[,;\s]*\b(?:ähm|äh)\b[,;\s]*', r'[,;\s]+\b(?:oder\s+so|nicht\s+wahr)\s*$'),
    "it": (r'[,;\s]*\b(?:ehm|eh)\b[,;\s]*', r'[,;\s]+\b(?:capito|sai)\s*$'),
    "pt": (r'[,;\s]*\b(?:é|ahn)\b[,;\s]*', r'[,;\s]+\b(?:né|sabe)\s*$'),
}

class SRTCleaner:
    """
    Professional Subtitle Cleaner & Ergonomic Formatter.
    - Removes oral filler words while preserving exact word-level audio sync.
    - Supports multiple languages (FR, EN, ES, DE, IT, PT, etc.).
    - Offers contextual AI correction (via LLM) for technical terms & typo polish.
    - Splits text into standard TV/YouTube ergonomic cues (max 40 chars/line, 2 lines max).
    - Exports to clean .srt, .vtt, and single-line .sbv formats.
    """

    def __init__(self, max_chars_per_line: int = 40, max_lines_per_cue: int = 2, 
                 min_cue_duration: float = 1.0, max_cue_duration: float = 6.0):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue
        self.min_cue_duration = min_cue_duration
        self.max_cue_duration = max_cue_duration

    def _normalize_lang(self, lang_code: str) -> str:
        if not lang_code:
            return "en"
        return lang_code.lower()[:2]

    def clean_text_heuristics(self, text: str, lang_code: str = "en") -> str:
        """Clean oral fillers and normalize spacing/punctuation using fast regex."""
        if not text:
            return ""
        
        cleaned = text.strip()
        lang = self._normalize_lang(lang_code)
        patterns = FILLER_PATTERNS_BY_LANG.get(lang, [])
        
        # Iteratively strip leading filler phrases for the specific language
        for _ in range(3):
            for pat in patterns:
                cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        
        # Remove mid-sentence & trailing hesitations if language pattern exists
        if lang in MID_HESITATION_PATTERNS:
            mid_pat, end_pat = MID_HESITATION_PATTERNS[lang]
            cleaned = re.sub(mid_pat, ' ', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(end_pat, '', cleaned, flags=re.IGNORECASE)

        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([,;.!?])', r'\1', cleaned)
        cleaned = re.sub(r'([,;.!?])(?=[^\s\d])', r'\1 ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            
        return cleaned

    def clean_words_list(self, words: List[Dict[str, Any]], lang_code: str = "en") -> List[Dict[str, Any]]:
        """
        Filter filler words from a word-level timestamp list (from WhisperX alignment).
        Returns filtered words list with adjusted timestamps.
        """
        if not words:
            return []

        lang = self._normalize_lang(lang_code)
        fillers = FILLER_WORDS_BY_LANG.get(lang, set())
        universal_hesitations = {"um", "uh", "er", "ah", "euh", "ähm", "äh", "ehm"}

        cleaned_words = []
        skip_leading = True
        
        for w in words:
            raw_word = re.sub(r'[^\w]', '', w.get("word", "").lower())
            if skip_leading and raw_word in fillers:
                # Drop leading filler and let timestamp shift to next word
                continue
            skip_leading = False
            
            # Skip pure hesitation tokens
            if raw_word in universal_hesitations:
                continue
                
            cleaned_words.append(w)

        return cleaned_words if cleaned_words else words

    def clean_segments_heuristic(self, segments: List[Dict[str, Any]], lang_code: str = "en") -> List[Dict[str, Any]]:
        """
        Clean an entire segment list while maintaining time synchronization.
        """
        cleaned_segments = []
        for seg in segments:
            text = seg.get("text", "")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            words = seg.get("words", [])

            if words:
                cleaned_words = self.clean_words_list(words, lang_code)
                if cleaned_words:
                    start = cleaned_words[0].get("start", start)
                    end = cleaned_words[-1].get("end", end)
                    text = " ".join(w.get("word", "").strip() for w in cleaned_words)

            cleaned_text = self.clean_text_heuristics(text, lang_code)
            if cleaned_text:
                new_seg = dict(seg)
                new_seg["start"] = start
                new_seg["end"] = end
                new_seg["text"] = cleaned_text
                if words:
                    new_seg["words"] = cleaned_words if 'cleaned_words' in locals() else words
                cleaned_segments.append(new_seg)

        return cleaned_segments

    def clean_with_llm(self, text: str, lang_code: str = "en", llm_backend=None) -> str:
        """
        Contextual AI correction pass with LLM:
        - Resolves phonetic confusions (e.g., 'Pinocchio' -> 'Pinokio', 'Cloud' -> 'Claude').
        - Polishes typography without altering the natural flow or duration.
        """
        if not text or len(text.strip()) < 5:
            return text
            
        if llm_backend is None:
            # Fallback to heuristic cleaning if LLM is not provided
            return self.clean_text_heuristics(text, lang_code)

        prompt = f"""You are an expert video subtitle editor. Clean up the following speech-to-text transcript.
Rules:
1. Fix speech recognition mistakes, phonetic errors, and technical terms based on the context (e.g. AI tools, tech brands).
2. Remove oral hesitations and filler tics ('donc', 'voilà', 'alors', 'euh', 'you know', 'um').
3. Keep the EXACT same language ({lang_code}) and meaning.
4. Keep the output close in length to the original.
5. Output ONLY the cleaned text, no explanations, no quotes.

Transcript:
"{text}"

Cleaned text:"""

        messages = [
            {"role": "system", "content": "You are a professional subtitle proofreader. Return ONLY the cleaned transcript with zero meta-commentary."},
            {"role": "user", "content": prompt}
        ]

        try:
            res = llm_backend.generate(messages, max_new_tokens=max(60, int(len(text) * 1.3)), temperature=0.2)
            cleaned = res.strip().strip('"\'«»')
            if cleaned and len(cleaned) >= 3:
                return cleaned
        except Exception as e:
            print(f"[SRTCleaner] LLM cleaning fallback due to: {e}")

        return self.clean_text_heuristics(text, lang_code)

    def split_into_ergonomic_cues(self, segments: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Split or re-group subtitle segments into standard TV/YouTube ergonomic cues:
        - Max ~40 chars per line
        - Max 2 lines per cue
        - Ensures minimum readability duration (1.0s) and natural pauses
        """
        if not segments:
            return []

        cues = []
        for seg in segments:
            raw_text = seg.get(text_key, "").strip()
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = max(0.1, end - start)

            if not raw_text:
                continue

            words = raw_text.split()
            if not words:
                continue

            # If segment is short enough, format directly
            if len(raw_text) <= self.max_chars_per_line * self.max_lines_per_cue:
                formatted_lines = self._wrap_lines(raw_text)
                cues.append({
                    "start": start,
                    "end": max(end, start + self.min_cue_duration),
                    "text": "\n".join(formatted_lines),
                    "lines": formatted_lines
                })
                continue

            # Segment is long: split into multiple balanced sub-cues
            sub_chunks = self._chunk_words(words, max_chars=self.max_chars_per_line * self.max_lines_per_cue)
            total_chars = sum(len(c) for c in sub_chunks)
            curr_start = start

            for idx, chunk in enumerate(sub_chunks):
                chunk_len = len(chunk)
                ratio = chunk_len / max(1, total_chars)
                chunk_duration = duration * ratio
                curr_end = curr_start + chunk_duration

                if idx == len(sub_chunks) - 1:
                    curr_end = end  # snap to segment end

                formatted_lines = self._wrap_lines(chunk)
                cues.append({
                    "start": curr_start,
                    "end": max(curr_end, curr_start + self.min_cue_duration),
                    "text": "\n".join(formatted_lines),
                    "lines": formatted_lines
                })
                curr_start = curr_end

        return cues

    def _wrap_lines(self, text: str) -> List[str]:
        """Wrap text into 1 or 2 balanced lines under max_chars_per_line."""
        text = text.strip()
        if len(text) <= self.max_chars_per_line:
            return [text]

        words = text.split()
        if len(words) <= 1:
            return [text]

        mid = len(text) // 2
        best_split = len(words) // 2
        min_diff = float("inf")
        curr_len = 0

        for i, w in enumerate(words[:-1]):
            curr_len += len(w) + 1
            diff = abs(curr_len - mid)
            if diff < min_diff:
                min_diff = diff
                best_split = i + 1

        line1 = " ".join(words[:best_split])
        line2 = " ".join(words[best_split:])
        return [line1, line2]

    def _chunk_words(self, words: List[str], max_chars: int) -> List[str]:
        """Group words into chunks not exceeding max_chars, breaking on punctuation when possible."""
        chunks = []
        curr = []
        curr_len = 0

        for w in words:
            w_len = len(w) + (1 if curr else 0)
            # Break if adding word exceeds max_chars or if last word had strong punctuation
            if (curr_len + w_len > max_chars and curr) or (curr and curr[-1].endswith(('.', '!', '?')) and curr_len >= max_chars * 0.6):
                chunks.append(" ".join(curr))
                curr = [w]
                curr_len = len(w)
            else:
                curr.append(w)
                curr_len += w_len

        if curr:
            chunks.append(" ".join(curr))

        return chunks

    def export_srt(self, cues: List[Dict[str, Any]], output_path: str):
        """Export cues to standard UTF-8 SRT file."""
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            for i, cue in enumerate(cues, 1):
                s = format_timestamp(cue["start"])
                e = format_timestamp(cue["end"])
                f.write(f"{i}\n{s} --> {e}\n{cue['text']}\n\n")

    def export_vtt(self, cues: List[Dict[str, Any]], output_path: str):
        """Export cues to WebVTT format."""
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write("WEBVTT\n\n")
            for i, cue in enumerate(cues, 1):
                s = format_timestamp(cue["start"]).replace(',', '.')
                e = format_timestamp(cue["end"]).replace(',', '.')
                f.write(f"{i}\n{s} --> {e}\n{cue['text']}\n\n")

    def export_sbv(self, cues: List[Dict[str, Any]], output_path: str):
        """Export cues to single-line YouTube SBV format (avoids YouTube Studio multi-line parser warnings)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for cue in cues:
                start_s = cue["start"]
                end_s = cue["end"]
                
                # Format: H:MM:SS.mmm
                def _fmt_sbv(secs):
                    hrs = int(secs // 3600)
                    mins = int((secs % 3600) // 60)
                    scs = int(secs % 60)
                    ms = int(round((secs - int(secs)) * 1000))
                    return f"{hrs}:{mins:02d}:{scs:02d}.{ms:03d}"

                single_line_text = " ".join(cue["text"].split())
                f.write(f"{_fmt_sbv(start_s)},{_fmt_sbv(end_s)}\n{single_line_text}\n\n")
