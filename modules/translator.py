import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from modules.utils import cleanup_model
from config import DEVICE, LANGUAGES

class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = DEVICE

    def load_model(self):
        if self.model is None:
            print(f"Loading NLLB-200 ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def translate_segments(self, segments, source_lang_name, target_lang_name):
        """
        Traduit une liste de segments.
        Ajoute 'translated_text' a chaque segment.
        """
        if not segments:
            return segments

        self.load_model()
        
        src_code = LANGUAGES.get(source_lang_name, "eng_Latn")
        tgt_code = LANGUAGES.get(target_lang_name, "eng_Latn")
        
        print(f"Traduction de {source_lang_name} ({src_code}) vers {target_lang_name} ({tgt_code})...")

        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        if tgt_token_id == self.tokenizer.unk_token_id:
            print(f"WARNING: Language code '{tgt_code}' not found in tokenizer. Trying alternatives...")
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(f"__{tgt_code}__")
        
        self.tokenizer.src_lang = src_code
        
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                seg["translated_text"] = ""
                continue
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            translated_tokens = self.model.generate(
                **inputs, 
                forced_bos_token_id=tgt_token_id, 
                max_length=512
            )
            
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            seg["translated_text"] = translated_text
            
        return segments

    def retranslate_constrained(self, segments, source_lang_name, target_lang_name, 
                                 chars_per_second=15, speed_factor=1.25):
        """
        Re-translate segments that overflow their time slots with a tighter max_length.
        Only re-translates segments where translated_text is too long to fit.
        Returns (segments, retranslated_count).
        """
        if not segments:
            return segments, 0

        self.load_model()
        
        src_code = LANGUAGES.get(source_lang_name, "eng_Latn")
        tgt_code = LANGUAGES.get(target_lang_name, "eng_Latn")
        
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        if tgt_token_id == self.tokenizer.unk_token_id:
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(f"__{tgt_code}__")
        
        self.tokenizer.src_lang = src_code
        
        retranslated = 0
        for seg in segments:
            text = seg.get("text", "").strip()
            translated = seg.get("translated_text", "")
            if not text or not translated:
                continue
            
            duration = seg["end"] - seg["start"]
            # Max chars that fit = duration × cps × speed_factor  
            max_chars = duration * chars_per_second * speed_factor
            
            if len(translated) <= max_chars * 1.1:  # 10% tolerance
                continue
            
            # Re-translate with tighter max_length
            # NLLB subword ratio: ~1.3-1.8 tokens per char, use 1.5
            max_tokens = max(10, int(max_chars * 1.5))
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            constrained_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=max_tokens
            )
            
            constrained_text = self.tokenizer.batch_decode(constrained_tokens, skip_special_tokens=True)[0]
            
            # Only use constrained version if it's actually shorter and non-empty
            if constrained_text.strip() and len(constrained_text) < len(translated):
                old_len = len(translated)
                seg["translated_text"] = constrained_text
                seg["retranslated"] = True
                retranslated += 1
                print(f"  Constrained [{seg['start']:.1f}-{seg['end']:.1f}]: {old_len}→{len(constrained_text)} chars")
        
        return segments, retranslated

    def cleanup(self):
        cleanup_model(self.model)
        cleanup_model(self.tokenizer)
        self.model = None
        self.tokenizer = None

if __name__ == "__main__":
    t = Translator()
    segs = [{"text": "Bonjour tout le monde", "start": 0, "end": 2}]
    res = t.translate_segments(segs, "Francais", "English")
    print(res[0]["translated_text"])
