import torch
import re
from config import DEVICE, CHARS_PER_SECOND
from modules.llm_backends import get_backend
from config import DEVICE, CHARS_PER_SECOND

class Reformulator:
    """
    Unified LLM-based translator + reformulator.
    Uses Qwen3-8B to translate and fit text to time constraints in a single pass.
    Replaces the old NLLB + separate reformulation pipeline.
    """
    def __init__(self, backend_name="Qwen2.5-7B-Instruct"):
        self.backend_name = backend_name
        self.llm = None

    def load_model(self):
        if self.llm is None or self.llm.name != self.backend_name:
            self.llm = get_backend(self.backend_name)
            self.llm.load()

    def _language_name(self, lang_code):
        """Convert language code to human-readable name."""
        name_map = {
            "fra": "French", "eng": "English", "spa": "Spanish",
            "deu": "German", "ita": "Italian", "por": "Portuguese",
            "jpn": "Japanese", "kor": "Korean", "zho": "Chinese",
            "rus": "Russian", "arb": "Arabic", "hin": "Hindi",
            "nld": "Dutch", "pol": "Polish", "tur": "Turkish",
            "swe": "Swedish", "ces": "Czech", "ron": "Romanian",
            "hun": "Hungarian", "mya": "Burmese", "dan": "Danish",
            "fin": "Finnish", "ell": "Greek", "heb": "Hebrew",
            "ind": "Indonesian", "khm": "Khmer", "lao": "Lao",
            "zsm": "Malay", "nob": "Norwegian", "swh": "Swahili",
            "tgl": "Tagalog", "tha": "Thai", "vie": "Vietnamese",
            "fr": "French", "en": "English", "es": "Spanish", 
            "de": "German", "it": "Italian", "pt": "Portuguese", 
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese", 
            "ru": "Russian", "ar": "Arabic", "hi": "Hindi", 
            "nl": "Dutch", "pl": "Polish", "tr": "Turkish", 
            "sv": "Swedish", "cs": "Czech", "ro": "Romanian", 
            "hu": "Hungarian", "my": "Burmese", "da": "Danish",
            "fi": "Finnish", "el": "Greek", "he": "Hebrew",
            "id": "Indonesian", "km": "Khmer", "lo": "Lao",
            "ms": "Malay", "no": "Norwegian", "sw": "Swahili",
            "tl": "Tagalog", "th": "Thai", "vi": "Vietnamese"
        }
        for prefix, name in name_map.items():
            if lang_code.startswith(prefix):
                return name
        return "Unknown"

    def _source_language_name(self, lang_code):
        """Get source language name from Whisper-style codes."""
        short_map = {
            "fr": "French", "en": "English", "es": "Spanish",
            "de": "German", "it": "Italian", "pt": "Portuguese",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
            "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
            "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
            "sv": "Swedish", "cs": "Czech", "ro": "Romanian",
            "hu": "Hungarian", "my": "Burmese", "da": "Danish",
            "fi": "Finnish", "el": "Greek", "he": "Hebrew",
            "id": "Indonesian", "km": "Khmer", "lo": "Lao",
            "ms": "Malay", "no": "Norwegian", "sw": "Swahili",
            "tl": "Tagalog", "th": "Thai", "vi": "Vietnamese"
        }
        return short_map.get(lang_code, "Unknown")

    def _generate(self, messages, max_new_tokens=120, multiline=False):
        """Run LLM generation with standard settings."""
        self.load_model()
        response = self.llm.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.05,
            multiline=multiline
        )
        return self._clean_response(response, multiline)

    def _clean_response(self, response: str, multiline: bool = False) -> str:
        """Apply all post-processing to a raw LLM response."""
        # Strip <think> blocks (Qwen3 reasoning artifacts)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL|re.IGNORECASE).strip()
        response = re.sub(r'<think>.*', '', response, flags=re.DOTALL|re.IGNORECASE).strip()
        response = response.replace('<think>', '').replace('</think>', '').strip()
        
        # Strip DeepSeek/Qwen3 textual thinking blocks
        response = re.sub(r'Thinking Process:.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL|re.IGNORECASE).strip()
        response = re.sub(r'Thinking Process:.*', '', response, flags=re.DOTALL|re.IGNORECASE).strip()
        
        # Strip LLM meta-comment lines ("Here's the translation:", "Translated:", etc.)
        _META_PATTERNS = [
            r"^here'?s?\s+(the|my|a)?\s*(translated|translation).*?:\s*",
            r"^translation\s*:\s*",
            r"^translated\s*(sentence|text)?\s*:\s*",
            r"^voici\s+(la|ma)?\s*traduction.*?:\s*",
            r"^traduction\s*:\s*",
            r"^перевод\s*:\s*",
            r"^вот\s+(мой\s+)?перевод\s*:\s*",
            r"^übersetzung\s*:\s*",
            r"^hier\s+ist\s+die\s+übersetzung\s*:\s*",
        ]
        lines = response.strip().split('\n')
        # If first line matches a meta-pattern, skip it
        if len(lines) > 1:
            for pat in _META_PATTERNS:
                if re.match(pat, lines[0], re.IGNORECASE):
                    lines = lines[1:]
                    break
        
        if multiline:
            result = '\n'.join(lines).strip()
        else:
            # Take first non-empty line, strip quotes
            result = ''
            for line in lines:
                line = line.strip()
                if line:
                    result = line
                    break
        
        for q in ['"', "'", '\u201c', '\u201d', '\u00ab', '\u00bb']:
            if result.startswith(q) and result.endswith(q):
                result = result[1:-1].strip()
        
        # Strip orphan leading/trailing quotes (unpaired)
        _all_quotes = '"\'\u201c\u201d\u00ab\u00bb'
        while result and result[0] in _all_quotes:
            result = result[1:].strip()
        while result and result[-1] in _all_quotes:
            result = result[:-1].strip()
        
        if not multiline:
            # Strip markdown artifacts that LLM sometimes leaks
            result = re.sub(r'^[\*]+\s*', '', result)   # leading asterisks
            result = re.sub(r'\s*[\*]+$', '', result)   # trailing asterisks
            result = result.replace('**', '').strip()    # bold markers
        
        # Final meta-comment check: if result IS a meta-comment, return empty
        for pat in _META_PATTERNS:
            if re.match(pat, result, re.IGNORECASE) and len(result) < 40:
                return ''
        
        return result

    def _build_translate_messages(self, text, source_lang, target_lang_code, duration, max_chars):
        """Build messages + max_new_tokens for a translate_and_fit call (used for batching)."""
        src_name = self._source_language_name(source_lang)
        tgt_name = self._language_name(target_lang_code)
        prompt = f"""You are an expert video dubbing translator. Translate from {src_name} to {tgt_name}.

Source text: "{text}"
Strict target duration: {duration:.1f}s → translation MAX {max_chars} characters (spaces included).
ABSOLUTE RULES:
- Keep the main meaning and tone.
- Be BRUTALLY concise: remove ALL fillers, unnecessary words, secondary details, repetitions.
- Paraphrase short, use contractions, fast spoken language, abbreviations if natural.
- Priority: fit within the duration, even if it requires simplification.
- Output ONLY the {tgt_name} translation, nothing else.

{tgt_name}:"""
        messages = [
            {"role": "system", "content": f"You are a video dubbing translator. Translate from {src_name} to {tgt_name} as BRUTALLY CONCISE as possible to fit in {duration:.1f}s. MAX {max_chars} characters. Output: ONLY the {tgt_name} translation."},
            {"role": "user", "content": prompt}
        ]
        return messages, max(15, int(max_chars * 1.2))

    def translate_and_fit(self, text, source_lang, target_lang_code, duration, max_chars):
        """
        Translate text and fit it to time constraints in a single LLM pass.
        Retries once with higher temperature if first attempt fails.
        """
        self.load_model()
        
        src_name = self._source_language_name(source_lang)
        tgt_name = self._language_name(target_lang_code)
        
        # Same language? Just shorten if needed
        if src_name == tgt_name:
            if len(text) <= max_chars * 1.1:
                return text
            return self.shorten(text, max_chars, target_lang_code)
        
        for attempt in range(2):
            prompt = f"""You are an expert video dubbing translator. Translate from {src_name} to {tgt_name}.

Source text: "{text}"
Strict target duration: {duration:.1f}s → translation MAX {max_chars} characters (spaces included).
ABSOLUTE RULES:
- Keep the main meaning and tone.
- Be BRUTALLY concise: remove ALL fillers, unnecessary words, secondary details, repetitions.
- Paraphrase short, use contractions, fast spoken language, abbreviations if natural.
- Priority: fit within the duration, even if it requires simplification.
- Output ONLY the {tgt_name} translation, nothing else.

{tgt_name}:"""

            messages = [
                {"role": "system", "content": f"You are a video dubbing translator. Translate from {src_name} to {tgt_name} as BRUTALLY CONCISE as possible to fit in {duration:.1f}s. MAX {max_chars} characters. Output: ONLY the {tgt_name} translation."},
                {"role": "user", "content": prompt}
            ]
            
            result = self._generate(messages, max_new_tokens=max(15, int(max_chars * 1.2)))
            
            if not result or len(result) < 3:
                if attempt == 0:
                    print(f"  [RETRY] Empty result, retrying...")
                    continue
                return None
            if result.strip() == text.strip():
                if attempt == 0:
                    print(f"  [RETRY] LLM returned source text unchanged, retrying...")
                    continue
                return None
            
            # Detect LLM leak: if result is clearly not target language or is a meta-comment
            _leak_indicators = ["here's", "translated sentence", "translation:", "voici la"]
            result_lower = result.lower()
            leaked = False
            for indicator in _leak_indicators:
                if result_lower.startswith(indicator):
                    print(f"  [WARN] LLM meta-leak detected: '{result[:50]}'")
                    leaked = True
                    break
            if leaked:
                if attempt == 0:
                    continue
                return None
            
            return result
        
        return None

    def translate_segments(self, segments, source_lang, target_lang_name, 
                           target_lang_code, cps, speed_factor=1.15):
        """
        Translate all segments with brutal concision for timing.
        Uses GPU-batched inference (BATCH_SIZE=8) for ~5-8x speedup on 4090.
        Failed/overflowed segments are retried individually.
        """
        self.load_model()
        
        tgt_name = self._language_name(target_lang_code)
        src_name = self._source_language_name(source_lang)
        same_lang = (src_name == tgt_name)
        
        BATCH_SIZE = 8
        print(f"LLM Translation → {tgt_name} ({len(segments)} segments, batch={BATCH_SIZE}, CPS={cps})...")

        aggressive_cps = cps

        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]

            # --- Separate trivial (empty / same-lang) from segments needing LLM ---
            pending_idx = []   # indices within batch that need LLM
            batch_messages = []
            batch_max_tokens = []
            batch_meta = []    # (text, duration, max_chars) per pending segment

            for local_i, seg in enumerate(batch):
                text = seg.get("text", "").strip()
                if not text:
                    seg["translated_text"] = ""
                    continue

                duration = seg["end"] - seg["start"]
                max_chars = int(duration * aggressive_cps * speed_factor)

                if same_lang:
                    seg["translated_text"] = text if len(text) <= max_chars * 1.1 else text[:max_chars]
                    continue

                msgs, mnt = self._build_translate_messages(text, source_lang, target_lang_code, duration, max_chars)
                pending_idx.append(local_i)
                batch_messages.append(msgs)
                batch_max_tokens.append(mnt)
                batch_meta.append((text, duration, max_chars))

            if not batch_messages:
                continue

            # --- Single GPU call for the whole sub-batch ---
            raw_responses = self.llm.generate_batch(
                batch_messages, batch_max_tokens,
                do_sample=True, temperature=0.3, repetition_penalty=1.05
            )

            # --- Post-process and validate each response ---
            for local_i, raw, (text, duration, max_chars) in zip(pending_idx, raw_responses, batch_meta):
                seg = batch[local_i]
                result = self._clean_response(raw)

                # Validate
                is_empty   = not result or len(result) < 3
                is_source  = result.strip() == text.strip()
                is_leak    = any(result.lower().startswith(ind) for ind in ["here's", "translated sentence", "translation:", "voici la"])
                is_overflow = result and len(result) > max_chars * 1.4

                if is_empty or is_source or is_leak:
                    # Individual retry via translate_and_fit (has its own 2-attempt loop)
                    print(f"  [RETRY] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: batch result invalid, retrying individually")
                    result = self.translate_and_fit(text, source_lang, target_lang_code, duration, max_chars)

                elif is_overflow:
                    shorter_max = int(max_chars * 0.7)
                    print(f"  [RETRY] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: {len(result)} chars > {max_chars} limit, retrying with {shorter_max}")
                    retry = self.translate_and_fit(text, source_lang, target_lang_code, duration, shorter_max)
                    if retry and len(retry) < len(result):
                        result = retry

                if result:
                    seg["translated_text"] = result
                else:
                    seg["translated_text"] = f"[TRANSLATION FAILED: {text}]"
                    print(f"  [WARN] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: LLM translation FAILED after retries")

            done = min(batch_start + BATCH_SIZE, len(segments))
            print(f"  {done}/{len(segments)} segments translated")

        print(f"Translation complete: {len(segments)} segments")
        return segments

    def _build_normal_messages(self, text, src_name, tgt_name):
        """Build messages + max_new_tokens for a translate_normal call (used for batching)."""
        prompt = f"""Translate the following text from {src_name} to {tgt_name}.
Translate naturally and faithfully, preserving the full meaning, tone, and nuance.
Do NOT shorten or simplify. Output ONLY the {tgt_name} translation.

Source: "{text}"

{tgt_name}:"""
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate from {src_name} to {tgt_name} naturally and faithfully. Output ONLY the translation."},
            {"role": "user", "content": prompt}
        ]
        return messages, max(30, int(len(text) * 2))

    def translate_normal(self, segments, source_lang, target_lang_code):
        """
        Translate all segments naturally without any length constraint.
        Produces a faithful, full translation (no shortening, no concision).
        Stores result in 'normal_text' key of each segment.
        Uses GPU-batched inference (BATCH_SIZE=8) matching translate_segments.
        """
        self.load_model()
        
        src_name = self._source_language_name(source_lang)
        tgt_name = self._language_name(target_lang_code)
        same_lang = (src_name == tgt_name)
        BATCH_SIZE = 8
        print(f"Normal Translation → {tgt_name} ({len(segments)} segments, batch={BATCH_SIZE}, natural/full)...")
        
        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]

            pending_idx = []
            batch_messages = []
            batch_max_tokens = []

            for local_i, seg in enumerate(batch):
                text = seg.get("text", "").strip()
                if not text:
                    seg["normal_text"] = ""
                    continue
                if same_lang:
                    seg["normal_text"] = text
                    continue
                msgs, mnt = self._build_normal_messages(text, src_name, tgt_name)
                pending_idx.append(local_i)
                batch_messages.append(msgs)
                batch_max_tokens.append(mnt)

            if not batch_messages:
                continue

            raw_responses = self.llm.generate_batch(
                batch_messages, batch_max_tokens,
                do_sample=False, temperature=0.0, repetition_penalty=1.0
            )

            for local_i, raw in zip(pending_idx, raw_responses):
                seg = batch[local_i]
                text = seg.get("text", "").strip()
                result = self._clean_response(raw, multiline=False)

                if result and len(result) >= 3 and result.strip() != text.strip():
                    seg["normal_text"] = result
                else:
                    # Individual retry on failure
                    msgs, mnt = self._build_normal_messages(text, src_name, tgt_name)
                    retry = self._generate(msgs, max_new_tokens=mnt)
                    if retry and len(retry) >= 3 and retry.strip() != text.strip():
                        seg["normal_text"] = retry
                    else:
                        fitted = seg.get("translated_text", "")
                        if fitted and not fitted.startswith("[TRANSLATION FAILED"):
                            seg["normal_text"] = fitted
                            print(f"  [WARN] Normal translation failed [{seg['start']:.1f}-{seg['end']:.1f}], using fitted version")
                        else:
                            seg["normal_text"] = f"[TRANSLATION FAILED: {text}]"
                            print(f"  [WARN] Normal translation FAILED [{seg['start']:.1f}-{seg['end']:.1f}], no fallback")

            done = min(batch_start + BATCH_SIZE, len(segments))
            print(f"  {done}/{len(segments)} segments translated (normal)")

        print(f"Normal translation complete: {len(segments)} segments")
        return segments

    def shorten(self, text, target_chars, language):
        """
        Shorten text using LLM to fit within target_chars.
        Used as fallback or for same-language reformulation.
        """
        self.load_model()
        
        lang_name = self._language_name(language)
        
        if lang_name == "French":
            examples = (
                'Exemple: "Maintenant, essayons de le tester." → "Essayons de le tester."\n'
                'Exemple: "Nous allons mettre en place notre application." → "On met en place l\'appli."\n'
                'Exemple: "Merci d\'avoir regardé jusqu\'à la fin." → "Merci d\'avoir regardé."\n'
            )
        elif lang_name == "English":
            examples = (
                'Example: "Now let\'s test it out and see." → "Let\'s test it out."\n'
                'Example: "We\'ll set up our app and player." → "We set up the app and player."\n'
            )
        else:
            examples = ""

        prompt = f"""Shorten this {lang_name} sentence to {target_chars} characters or fewer.

Rules:
- Output ONLY the shortened sentence in {lang_name}
- Must be grammatically correct and natural
- Keep the same meaning
- Remove fillers, use shorter forms ("nous allons" → "on")
- Do NOT translate to another language

{examples}Sentence ({len(text)} chars): {text}
Shortened ({target_chars} chars max):"""

        messages = [
            {"role": "system", "content": f"You shorten {lang_name} sentences. Remove filler words. Keep meaning. Output ONLY the result."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._generate(messages, max_new_tokens=min(80, len(text)))
        
        # Validation
        if not result or len(result) < 3 or len(result) >= len(text):
            return None
        if len(result) > target_chars * 1.3:
            return None
        
        return result

    def check_timing_batch(self, segments, language_family="default"):
        """Estimate and flag segments that are too long."""
        cps = CHARS_PER_SECOND.get(language_family, 13)
        for seg in segments:
            text = seg.get("translated_text", seg.get("text", ""))
            duration = seg["end"] - seg["start"]
            estimated_duration = len(text) / cps
            seg["estimated_too_long"] = estimated_duration > (duration * 1.2)
        return segments

    def translate_text(self, text, source_lang_code, target_lang_code):
        """Translate a generic text (like a video title or description)."""
        if not text or not text.strip():
            return ""
        self.load_model()
        target_lang = self._language_name(target_lang_code)
        source_lang = self._source_language_name(source_lang_code)
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert native translator and localization specialist for YouTube content. "
                    f"Translate the user's text from {source_lang} to {target_lang}.\n"
                    f"CRITICAL RULES:\n"
                    f"- Output ONLY the translated text, nothing else.\n"
                    f"- DO NOT output your internal reasoning, thinking process, notes, quotes, or preambles.\n"
                    f"- Keep the tone engaging, natural, and idiomatic for YouTube titles and descriptions.\n"
                    f"- PRESERVE EXACTLY all URLs, links, social media handles, emojis, and paragraph formatting.\n"
                    f"- Translate relevant hashtags naturally into {target_lang} (e.g. #PromptInjection, #Cybersecurity) while keeping brand names and standard acronyms intact (#ChatGPT, #LLM, #SEO).\n"
                    f"- Do not alter or translate website URLs, affiliate links, or channel handles."
                )
            },
            {"role": "user", "content": text}
        ]
        
        result = self._generate(messages, max_new_tokens=4096, multiline=True)
        return result if result else ""

    def cleanup(self):
        if self.llm is not None:
            self.llm.unload()
            self.llm = None

if __name__ == "__main__":
    r = Reformulator()
    # r.load_model() # Heavy to load
    text = "In this full Python game development course, you will learn how to code a playable Minecraft clone."
    # print(r.translate_and_fit(text, "en", "fra_Latn", 3.5, 50))
