import re
import os
import json
from typing import List, Dict, Any, Optional
from modules.utils import format_timestamp, seconds_from_srt_timestamp

# Load external ASR corrections
def load_asr_corrections(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load external ASR corrections from config/asr_corrections.json."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "asr_corrections.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[SRTCleaner] Error loading {config_path}: {e}")
    return []

def remove_empty_cues_and_redistribute(cues: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """
    Remove cues whose text stripped of punctuation and whitespace is empty (e.g. solitary '.').
    Redistribute their duration to the previous cue.
    """
    if not cues:
        return []
    cleaned = []
    for cue in cues:
        raw_text = cue.get(text_key, "")
        # Strip all punctuation and whitespace
        stripped = re.sub(r'[\s\.\?\!\,\;\:\-\–\—\…\¿\¡\'\"\(\)\[\]\«\»]+', '', raw_text)
        if not stripped:
            if cleaned:
                prev = cleaned[-1]
                if cue.get("end", 0.0) > prev.get("end", 0.0):
                    prev["end"] = cue["end"]
                # If the deleted cue had punctuation (e.g. '.'), ensure previous cue ends cleanly
                p = raw_text.strip()
                if p in ('.', '?', '!', '…', ':', ';') and not re.search(r'[\.\?\!\…\:\;]$', prev.get(text_key, '')):
                    prev[text_key] = f"{prev.get(text_key, '').rstrip()}{p}"
            continue
        cleaned.append(cue)
    return cleaned

GERMAN_LOWERCASE_WORDS = {
    "und", "oder", "aber", "denn", "weil", "dass", "da", "wenn", "ob", "wie", "als",
    "wir", "sie", "er", "es", "ich", "du", "ihr", "mein", "dein", "sein", "unser", "euer",
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "eines",
    "in", "an", "auf", "aus", "bei", "mit", "nach", "von", "zu", "für", "über", "unter",
    "vor", "zwischen", "durch", "ohne", "um", "gegen", "ist", "sind", "war", "waren",
    "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte", "hatten",
    "kann", "können", "muss", "müssen", "soll", "sollen", "will", "wollen",
    "nicht", "auch", "noch", "schon", "sehr", "nur", "so", "dann", "hier", "da", "jetzt"
}

def fix_inter_cue_casing(cues: List[Dict[str, Any]], text_key: str = "text", lang_code: str = "fr", glossary_proper_nouns: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Fix capitalization across cues for all supported languages:
    - Capitalize the first letter if previous cue ends with strong punctuation (. ? ! … : ») or if it's the first cue.
    - Otherwise (mid-sentence continuation), lowercase the first character.
    - Multilingual intelligence:
      * English: preserves pronoun 'I' and contractions ('I'm', 'I've', 'I'll', 'I'd').
      * German: preserves capitalization of German nouns (only lowercases known function words/verbs/pronouns).
      * Accented letters: preserves single accented letters ('À', 'Á', 'É', etc.).
      * Preserves proper nouns, mixed-case words, and acronyms (SEO, IA, URL, AI, etc.).
    """
    if not cues:
        return cues
        
    norm_lang = (lang_code or "fr").lower()[:2]
        
    glossary = {
        "ZastTranslate", "YouTube", "VoxCPM", "Qwen", "FLUX", "FLUX.1", "Pinokio",
        "Whisper", "WhisperX", "Claude", "OpenAI", "ChatGPT", "WordPress", "TikTok",
        "Midjourney", "ComfyUI", "Gradio", "SEO", "IA", "AI", "URL", "API", "TTS", "LLM",
        "SRT", "VTT", "SBV", "CPS", "HD", "4K", "TF32", "NVIDIA", "CUDA", "JSON",
        "YAML", "HTML", "CSS", "JS", "Auto", "Hermès", "Blog", "Bulk", "Mode",
        "Studio", "Clean", "Fillers", "Translation", "Dubbing", "Fitted"
    }
    if glossary_proper_nouns:
        glossary.update(glossary_proper_nouns)
        
    # Auto-add words from external ASR corrections glossary
    try:
        for corr in load_asr_corrections():
            rep = corr.get("replacement", "")
            for word in rep.split():
                cleaned_w = re.sub(r'[^\w]', '', word)
                if cleaned_w:
                    glossary.add(cleaned_w)
    except Exception:
        pass

    for i, cue in enumerate(cues):
        text = cue.get(text_key, "").strip()
        if not text:
            continue
            
        if i == 0:
            if text[0].isalpha() and text[0].islower():
                cue[text_key] = text[0].upper() + text[1:]
            continue
            
        prev_text = cues[i - 1].get(text_key, "").strip()
        prev_ends_strong = bool(re.search(r'[\.\?\!\…\:\»][\'"\)\]\»]*$', prev_text))
        
        if prev_ends_strong:
            if text[0].isalpha() and text[0].islower():
                cue[text_key] = text[0].upper() + text[1:]
        else:
            # Mid-sentence continuation: check if first token should be lowercased
            m = re.match(r'^([^\w]*)([\w\-\.\'\’]+)(.*)$', text, re.DOTALL)
            if m:
                prefix, first_tok, suffix = m.group(1), m.group(2), m.group(3)
                
                # Reasons to keep uppercase:
                # 1. Single accented letter like 'À', 'Á', 'É', etc.
                is_single_accented = (len(first_tok) == 1 and first_tok in "ÀÁÉÈÊËÍÎÏÓÔÙÚÛÜÇÑ")
                # 2. English pronoun 'I' and contractions ('I'm', 'I've', 'I'll', 'I'd')
                is_english_i = (norm_lang == "en" and (first_tok in ("I", "I'm", "I've", "I'll", "I'd", "I'd've") or first_tok.startswith("I'")))
                # 3. German: keep uppercase for nouns, only lowercase known lowercase words
                if norm_lang == "de":
                    if first_tok.lower() not in GERMAN_LOWERCASE_WORDS:
                        continue
                # 4. Acronym: len >= 2 and all uppercase (e.g. SEO, IA, AI, URL)
                is_acronym = (len(first_tok) >= 2 and first_tok.isupper())
                # 5. Mixed case: has uppercase inside (e.g. YouTube, VoxCPM, ZastTranslate)
                is_mixed_case = any(c.isupper() for c in first_tok[1:])
                # 6. In glossary set
                clean_tok = re.sub(r'[^\w]', '', first_tok)
                is_glossary = (first_tok in glossary or clean_tok in glossary)
                
                if not (is_single_accented or is_english_i or is_acronym or is_mixed_case or is_glossary):
                    lowered = first_tok[0].lower() + first_tok[1:]
                    cue[text_key] = prefix + lowered + suffix

    return cues

def apply_asr_corrections_cross_cues(cues: List[Dict[str, Any]], text_key: str = "text", corrections: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Apply external ASR corrections across reconstituted text to repair terms even when
    split across consecutive cues (e.g. 'QN 3.5.' in cue i and '9B' in cue i+1).
    Applied AFTER inter-cue casing so case-sensitive patterns match accurately.
    """
    if not cues:
        return cues
    if corrections is None:
        corrections = load_asr_corrections()
    if not corrections:
        return cues

    for corr in corrections:
        pattern_str = corr.get("pattern", "")
        rep = corr.get("replacement", "")
        case_sensitive = corr.get("case_sensitive", False)
        if not pattern_str or rep is None:
            continue
            
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(pattern_str, flags)
        except Exception as e:
            print(f"[SRTCleaner] Invalid regex pattern '{pattern_str}': {e}")
            continue

        max_replaces = 50
        while max_replaces > 0:
            max_replaces -= 1
            # Build full text and map character indices back to (cue_idx, offset_in_cue)
            char_map = []
            pieces = []
            for c_idx, cue in enumerate(cues):
                t = cue.get(text_key, "")
                if c_idx > 0:
                    pieces.append(" ")
                    char_map.append((c_idx - 1, len(cues[c_idx - 1].get(text_key, ""))))
                for off, ch in enumerate(t):
                    pieces.append(ch)
                    char_map.append((c_idx, off))
            full_text = "".join(pieces)
            
            match = rx.search(full_text)
            if not match:
                break
                
            m_start = match.start()
            m_end = match.end() - 1
            if m_start >= len(char_map) or m_end >= len(char_map):
                break
                
            start_c_idx, start_off = char_map[m_start]
            end_c_idx, end_off = char_map[m_end]
            end_off += 1
            
            if start_c_idx == end_c_idx:
                t = cues[start_c_idx].get(text_key, "")
                cues[start_c_idx][text_key] = (t[:start_off] + rep + t[end_off:]).strip()
            else:
                t_start = cues[start_c_idx].get(text_key, "")
                t_end = cues[end_c_idx].get(text_key, "")
                
                prefix = t_start[:start_off].rstrip()
                cues[start_c_idx][text_key] = f"{prefix} {rep}".strip() if prefix else rep
                
                for mid_idx in range(start_c_idx + 1, end_c_idx):
                    cues[mid_idx][text_key] = ""
                    
                remainder = t_end[end_off:].lstrip()
                cues[end_c_idx][text_key] = remainder
                
    cues = remove_empty_cues_and_redistribute(cues, text_key=text_key)
    return cues

def normalize_timecodes(cues: List[Dict[str, Any]], min_gap_ms: int = 40, min_cue_duration_ms: int = 400, text_key: str = "text") -> List[Dict[str, Any]]:
    """
    Ensure strict timecode normalization:
    - end[i] <= start[i+1] - MIN_GAP_MS (40ms)
    - end[i] > start[i]
    - strict chronological ordering
    - if clamping creates a cue shorter than min_cue_duration_ms (400ms), merge with neighbor.
    """
    if not cues:
        return []
        
    min_gap_s = min_gap_ms / 1000.0
    min_dur_s = min_cue_duration_ms / 1000.0

    normalized = []
    for c in cues:
        start = float(c.get("start", 0.0))
        end = float(c.get("end", start + min_dur_s))
        if end <= start:
            end = start + min_dur_s
        new_c = dict(c)
        new_c["start"] = round(start, 3)
        new_c["end"] = round(end, 3)
        normalized.append(new_c)

    normalized.sort(key=lambda x: x["start"])

    changed = True
    iteration = 0
    max_iterations = 300
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        merged_list = []
        skip_next = False
        
        for i in range(len(normalized)):
            if skip_next:
                skip_next = False
                continue
                
            curr = normalized[i]
            if i == len(normalized) - 1:
                merged_list.append(curr)
                break
                
            next_cue = normalized[i + 1]
            max_allowed_end = round(next_cue["start"] - min_gap_s, 3)
            
            if curr["end"] > max_allowed_end:
                if max_allowed_end - curr["start"] >= min_dur_s:
                    curr["end"] = max_allowed_end
                    merged_list.append(curr)
                    changed = True
                else:
                    # Clamping would make cue too short -> merge with next_cue
                    t1 = curr.get(text_key, "").strip()
                    t2 = next_cue.get(text_key, "").strip()
                    next_cue[text_key] = f"{t1} {t2}".strip()
                    next_cue["start"] = curr["start"]
                    next_cue["end"] = max(curr["end"], next_cue["end"])
                    skip_next = True
                    changed = True
            else:
                merged_list.append(curr)
                
        normalized = merged_list

    for i in range(len(normalized) - 1):
        if normalized[i]["end"] > normalized[i + 1]["start"] - min_gap_s:
            normalized[i]["end"] = round(max(normalized[i]["start"] + 0.1, normalized[i + 1]["start"] - min_gap_s), 3)

    return normalized

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

TECH_BRAND_REPLACEMENTS = [
    (r'\b(?:CloudEye|Cloud\s*Eye|Cloud\s*A\s*I|Cloude\s*AI|Cloud\s*ai)\b', 'Claude.ai'),
    (r'\bClaude\s*dot\s*ai\b', 'Claude.ai'),
    (r'\bChat\s+GPT\b', 'ChatGPT'),
    (r'\bOpen\s+AI\b', 'OpenAI'),
    (r'\bMid\s+journey\b', 'Midjourney'),
    (r'\bStable\s+Diffusion\b', 'Stable Diffusion'),
    (r'\bComfy\s*UI\b', 'ComfyUI'),
    (r'\bPinokio\b', 'Pinokio'),
    (r'\bWhisper\s*X\b', 'WhisperX'),
    (r'\bHugging\s*Face\b', 'Hugging Face'),
]

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
                 min_cue_duration: float = 0.4, max_cue_duration: float = 6.0):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue
        self.min_cue_duration = min_cue_duration
        self.max_cue_duration = max_cue_duration

    def _normalize_lang(self, lang_code: str) -> str:
        if not lang_code:
            return "en"
        return lang_code.lower()[:2]

    def clean_text_heuristics(self, text: str, lang_code: str = "en") -> str:
        """Clean oral fillers, fix tech brand phonetics, and normalize spacing/punctuation."""
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

        # Fix spacing around punctuation (avoid breaking domain names like .ai, .com)
        cleaned = re.sub(r'\s+([,;.!?])', r'\1', cleaned)
        cleaned = re.sub(r'([,;!?])(?=[^\s\d])', r'\1 ', cleaned)
        cleaned = re.sub(r'\.(?=[A-ZÀ-ÖØ-ß])', '. ', cleaned)

        # Apply tech / AI brand phonetic normalizations from static list & external JSON
        for pat, rep in TECH_BRAND_REPLACEMENTS:
            cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
            
        try:
            for corr in load_asr_corrections():
                p = corr.get("pattern", "")
                r = corr.get("replacement", "")
                cs = corr.get("case_sensitive", False)
                if p and r is not None:
                    fl = 0 if cs else re.IGNORECASE
                    cleaned = re.sub(p, r, cleaned, flags=fl)
        except Exception:
            pass

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # NOTE: Blind capitalization was intentionally removed here!
        # Inter-cue capitalization must be handled by fix_inter_cue_casing
        # to prevent capitalizing mid-sentence cues (Bug 3).
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
        Never leaves empty cues behind (Bug 4).
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

        # Remove cues that were reduced to empty / punctuation-only after filler removal,
        # redistributing duration to previous cue (Bug 4)
        cleaned_segments = remove_empty_cues_and_redistribute(cleaned_segments, text_key="text")
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
        - Preserves exact WhisperX word audio timestamps without inventing duration (Bug 1 & 2)
        """
        if not segments:
            return []

        cues = []
        for seg in segments:
            raw_text = seg.get(text_key, "").strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + self.min_cue_duration))
            duration = max(0.1, end - start)

            if not raw_text:
                continue

            words = raw_text.split()
            if not words:
                continue

            # If segment is short enough, wrap lines directly preserving real audio timing
            if len(raw_text) <= self.max_chars_per_line * self.max_lines_per_cue:
                formatted_lines = self._wrap_lines(raw_text)
                cues.append({
                    "start": start,
                    "end": end,
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
                    "end": curr_end,
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
        """Export cues to standard UTF-8 SRT file after final timecode normalization pass."""
        normalized_cues = normalize_timecodes(cues, min_gap_ms=40, min_cue_duration_ms=400, text_key="text")
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            for i, cue in enumerate(normalized_cues, 1):
                s = format_timestamp(cue["start"])
                e = format_timestamp(cue["end"])
                f.write(f"{i}\n{s} --> {e}\n{cue['text']}\n\n")

    def export_vtt(self, cues: List[Dict[str, Any]], output_path: str):
        """Export cues to WebVTT format after final timecode normalization pass."""
        normalized_cues = normalize_timecodes(cues, min_gap_ms=40, min_cue_duration_ms=400, text_key="text")
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write("WEBVTT\n\n")
            for i, cue in enumerate(normalized_cues, 1):
                s = format_timestamp(cue["start"]).replace(',', '.')
                e = format_timestamp(cue["end"]).replace(',', '.')
                f.write(f"{i}\n{s} --> {e}\n{cue['text']}\n\n")

    def export_sbv(self, cues: List[Dict[str, Any]], output_path: str):
        """Export cues to single-line YouTube SBV format after final timecode normalization pass."""
        normalized_cues = normalize_timecodes(cues, min_gap_ms=40, min_cue_duration_ms=400, text_key="text")
        with open(output_path, 'w', encoding='utf-8') as f:
            for cue in normalized_cues:
                start_s = cue["start"]
                end_s = cue["end"]
                
                def _fmt_sbv(secs):
                    hrs = int(secs // 3600)
                    mins = int((secs % 3600) // 60)
                    scs = int(secs % 60)
                    ms = int(round((secs - int(secs)) * 1000))
                    return f"{hrs}:{mins:02d}:{scs:02d}.{ms:03d}"

                single_line_text = " ".join(cue["text"].split())
                f.write(f"{_fmt_sbv(start_s)},{_fmt_sbv(end_s)}\n{single_line_text}\n\n")
