import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modules.utils import cleanup_model
from config import DEVICE, CHARS_PER_SECOND

class Reformulator:
    """
    Unified LLM-based translator + reformulator.
    Uses Qwen3-8B to translate and fit text to time constraints in a single pass.
    Replaces the old NLLB + separate reformulation pipeline.
    """
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = DEVICE

    def load_model(self):
        if self.model is None:
            print(f"Loading Translation LLM ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )

    def _language_name(self, lang_code):
        """Convert language code to human-readable name."""
        name_map = {
            "fra": "French", "eng": "English", "spa": "Spanish",
            "deu": "German", "ita": "Italian", "por": "Portuguese",
            "jpn": "Japanese", "kor": "Korean", "zho": "Chinese",
            "rus": "Russian", "fr": "French", "en": "English",
            "es": "Spanish", "de": "German", "it": "Italian",
            "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
            "zh": "Chinese", "ru": "Russian",
        }
        for prefix, name in name_map.items():
            if lang_code.startswith(prefix):
                return name
        return "French"

    def _source_language_name(self, lang_code):
        """Get source language name from Whisper-style codes."""
        short_map = {
            "fr": "French", "en": "English", "es": "Spanish",
            "de": "German", "it": "Italian", "pt": "Portuguese",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
            "ru": "Russian",
        }
        return short_map.get(lang_code, "English")

    def _generate(self, messages, max_new_tokens=120):
        """Run LLM generation with standard settings."""
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.2,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Strip <think> blocks (Qwen3 reasoning artifacts)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()
        response = response.replace('<think>', '').replace('</think>', '').strip()
        
        # Strip LLM meta-comment lines ("Here's the translation:", "Translated:", etc.)
        _META_PATTERNS = [
            r"^here'?s?\s+(the|my|a)?\s*(translated|translation).*?:\s*",
            r"^translation\s*:\s*",
            r"^translated\s*(sentence|text)?\s*:\s*",
            r"^voici\s+(la|ma)?\s*traduction.*?:\s*",
            r"^traduction\s*:\s*",
        ]
        lines = response.strip().split('\n')
        # If first line matches a meta-pattern, skip it
        if len(lines) > 1:
            for pat in _META_PATTERNS:
                if re.match(pat, lines[0], re.IGNORECASE):
                    lines = lines[1:]
                    break
        
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
        
        # Strip markdown artifacts that LLM sometimes leaks
        result = re.sub(r'^[\*]+\s*', '', result)   # leading asterisks
        result = re.sub(r'\s*[\*]+$', '', result)   # trailing asterisks
        result = result.replace('**', '').strip()    # bold markers
        
        # Final meta-comment check: if result IS a meta-comment, return empty
        for pat in _META_PATTERNS:
            if re.match(pat, result, re.IGNORECASE) and len(result) < 40:
                return ''
        
        return result

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
        CPS forced to 9 for ultra-short output.
        If overflow estimate >40%, retry with 30% shorter constraint.
        """
        self.load_model()
        
        tgt_name = self._language_name(target_lang_code)
        print(f"LLM Translation → {tgt_name} ({len(segments)} segments, BRUTAL concision CPS=9)...")
        
        # Force CPS to 9 for aggressive brevity
        aggressive_cps = 9
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                seg["translated_text"] = ""
                continue
            
            duration = seg["end"] - seg["start"]
            max_chars = int(duration * aggressive_cps * speed_factor)
            
            # First pass
            result = self.translate_and_fit(
                text, source_lang, target_lang_code, duration, max_chars
            )
            
            # Overflow retry: if result is too long (>40% over max_chars), retry shorter
            if result and len(result) > max_chars * 1.4:
                shorter_max = int(max_chars * 0.7)
                print(f"  [RETRY] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: {len(result)} chars > {max_chars} limit, retrying with {shorter_max} chars")
                retry = self.translate_and_fit(
                    text, source_lang, target_lang_code, duration, shorter_max
                )
                if retry and len(retry) < len(result):
                    result = retry
            
            if result:
                seg["translated_text"] = result
            else:
                # Mark as failed — do NOT use original source text as fallback
                seg["translated_text"] = f"[TRANSLATION FAILED: {text}]"
                print(f"  [WARN] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: LLM translation FAILED after retries")
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(segments)} segments translated")
        
        print(f"Translation complete: {len(segments)} segments")
        return segments

    def translate_normal(self, segments, source_lang, target_lang_code):
        """
        Translate all segments naturally without any length constraint.
        Produces a faithful, full translation (no shortening, no concision).
        Stores result in 'normal_text' key of each segment.
        """
        self.load_model()
        
        src_name = self._source_language_name(source_lang)
        tgt_name = self._language_name(target_lang_code)
        print(f"Normal Translation → {tgt_name} ({len(segments)} segments, natural/full)...")
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                seg["normal_text"] = ""
                continue
            
            # Same language? Keep as-is
            if src_name == tgt_name:
                seg["normal_text"] = text
                continue
            
            prompt = f"""Translate the following text from {src_name} to {tgt_name}.
Translate naturally and faithfully, preserving the full meaning, tone, and nuance.
Do NOT shorten or simplify. Output ONLY the {tgt_name} translation.

Source: "{text}"

{tgt_name}:"""

            messages = [
                {"role": "system", "content": f"You are a professional translator. Translate from {src_name} to {tgt_name} naturally and faithfully. Output ONLY the translation."},
                {"role": "user", "content": prompt}
            ]
            
            # Try up to 2 times
            final_result = None
            for attempt in range(2):
                result = self._generate(messages, max_new_tokens=max(30, int(len(text) * 2)))
                if result and len(result) >= 3 and result.strip() != text.strip():
                    final_result = result
                    break
                if attempt == 0:
                    print(f"  [RETRY] Normal translation attempt failed [{seg['start']:.1f}-{seg['end']:.1f}], retrying...")
            
            if final_result:
                seg["normal_text"] = final_result
            else:
                # Fallback: use fitted version (which is already translated, never original)
                fitted = seg.get("translated_text", "")
                if fitted and not fitted.startswith("[TRANSLATION FAILED"):
                    seg["normal_text"] = fitted
                    print(f"  [WARN] Normal translation failed [{seg['start']:.1f}-{seg['end']:.1f}], using fitted version")
                else:
                    seg["normal_text"] = f"[TRANSLATION FAILED: {text}]"
                    print(f"  [WARN] Normal translation FAILED [{seg['start']:.1f}-{seg['end']:.1f}], no fallback available")
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(segments)} segments translated (normal)")
        
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

    def cleanup(self):
        cleanup_model(self.model)
        cleanup_model(self.tokenizer)
        self.model = None
        self.tokenizer = None

if __name__ == "__main__":
    r = Reformulator()
    # r.load_model() # Heavy to load
    text = "In this full Python game development course, you will learn how to code a playable Minecraft clone."
    # print(r.translate_and_fit(text, "en", "fra_Latn", 3.5, 50))
