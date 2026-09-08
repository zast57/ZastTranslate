import os
import re
from modules.utils import format_timestamp, seconds_from_srt_timestamp

DANGLING_CONNECTORS = {
    "si", "que", "de", "le", "la", "les", "des", "pour", "avec", "dans", "sans", "sur",
    "sous", "en", "par", "à", "au", "aux", "et", "ou", "mais", "car", "donc", "un", "une",
    "du", "d'", "l'", "c'", "qu'", "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur",
    "après", "avant", "pendant", "vers", "chez",
    "if", "and", "the", "to", "of", "in", "for", "on", "with", "that", "this", "as", "at", "by", "or", "but"
}

SENTENCE_STARTERS = {
    "parce", "car", "mais", "or", "donc", "ensuite", "puis", "enfin", "d'ailleurs", "selon", "d'après", "en"
}

def merge_sentence_fragments(segments, max_gap=2.2, max_combined_duration=16.0):
    """
    Merge sentence fragments, dangling clauses, and stranded orphan words into grammatically complete sentences.
    Guarantees no broken mid-clause cuts (e.g. '...exonération si' + 'votre vente...', '...ticket de' + 'caisse.', etc.).
    """
    if not segments or len(segments) <= 1:
        return segments

    terminals = ('.', '?', '!', '...', '…', ':', '»', '"')

    # Pre-pass: If a segment ends in a comma and next begins with a clear sentence starter ("D'après", "Selon"),
    # convert comma to a period so independent sentences don't merge across unrelated topics.
    for i in range(len(segments) - 1):
        curr_text = segments[i].get('text', '').strip()
        next_text = segments[i + 1].get('text', '').strip()
        next_first = next_text.split()[0].lower() if next_text.split() else ""
        if next_first in ["d'après", "selon", "voilà", "maintenant"] and curr_text.endswith(','):
            segments[i]['text'] = curr_text.rstrip(',') + '.'

    # Pass 1: Forward-merge orphan sentence starters (e.g. "Parce que le BOFIP," with next segment)
    pass1 = []
    skip_next = False
    for i in range(len(segments)):
        if skip_next:
            skip_next = False
            continue
        curr = dict(segments[i])
        curr_text = curr.get('text', '').strip()
        curr_words = curr_text.split()
        first_word = curr_words[0].lower().rstrip(",;:") if curr_words else ""
        
        if i < len(segments) - 1 and (len(curr_words) <= 4 or (curr.get('end', 0) - curr.get('start', 0)) < 1.8):
            next_seg = segments[i + 1]
            comb_dur = next_seg.get('end', 0.0) - curr.get('start', 0.0)
            gap = next_seg.get('start', 0.0) - curr.get('end', 0.0)
            if (first_word in SENTENCE_STARTERS or not any(curr_text.endswith(p) for p in terminals)) and comb_dur <= 16.0 and gap <= 2.5:
                curr['end'] = next_seg['end']
                curr['text'] = f"{curr_text} {next_seg.get('text', '').strip()}"
                if 'words' in curr and 'words' in next_seg:
                    curr['words'] = curr.get('words', []) + next_seg.get('words', [])
                pass1.append(curr)
                skip_next = True
                continue
        pass1.append(curr)

    # Pass 2: Backward-merge dangling ends and incomplete sentences
    merged = []
    for s in pass1:
        if not merged:
            merged.append(dict(s))
            continue
        
        prev = merged[-1]
        gap = s.get('start', 0.0) - prev.get('end', 0.0)
        combined_dur = s.get('end', 0.0) - prev.get('start', 0.0)
        prev_text = prev.get('text', '').strip()
        curr_text = s.get('text', '').strip()
        
        prev_words = prev_text.split()
        prev_last_word = prev_words[-1].lower().rstrip(".,;:\"'«»…") if prev_words else ""
        is_dangling_tail = prev_last_word in DANGLING_CONNECTORS
        prev_ends_terminal = any(prev_text.endswith(p) for p in ('.', '?', '!')) and not is_dangling_tail
        
        curr_words = curr_text.split()
        is_trailing_orphan = len(curr_words) <= 3 or (s.get('end', 0) - s.get('start', 0)) < 2.0
        
        should_merge = False
        
        if prev_ends_terminal:
            if len(curr_words) == 1 and combined_dur <= 14.0:
                should_merge = True
        else:
            if is_dangling_tail and combined_dur <= 18.5:
                should_merge = True
            elif is_trailing_orphan and combined_dur <= 18.5:
                should_merge = True
            elif len(curr_words) <= 6 and curr_text.lower().startswith(('ou ', 'et ', 'mais ', 'car ', 'donc ', 'par ', 'de ')) and combined_dur <= 18.5:
                should_merge = True
            elif gap <= max_gap and combined_dur <= max_combined_duration:
                should_merge = True
            elif len(curr_words) <= 6 and combined_dur <= 17.0:
                should_merge = True
            
        if should_merge:
            prev['end'] = s['end']
            prev['text'] = f"{prev_text} {curr_text}"
            if 'words' in prev and 'words' in s:
                prev['words'] = prev.get('words', []) + s.get('words', [])
        else:
            merged.append(dict(s))
            
    for idx, m in enumerate(merged, 1):
        m["index"] = idx
        
    return merged

class SRTParser:
    def merge_sentence_fragments(self, segments, max_gap=2.2, max_combined_duration=16.0):
        return merge_sentence_fragments(segments, max_gap=max_gap, max_combined_duration=max_combined_duration)

    def parse_srt(self, srt_path):
        """
        Parse an SRT file and return a list of segments.
        Returns a list of dict: {"index": int, "start": float, "end": float, "text": str}
        """
        if hasattr(srt_path, "name"):
            srt_path = srt_path.name
        elif not isinstance(srt_path, str):
            srt_path = str(srt_path)
        segments = []
        if not os.path.exists(srt_path):
            return segments

        with open(srt_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        # Regex to parse SRT blocks
        pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\d+\n\d{2}:\d{2}:\d{2},\d{3}).)*)', re.DOTALL)
        matches = pattern.findall(content)

        for match in matches:
            idx, start_ts, end_ts, text = match
            segments.append({
                "index": int(idx),
                "start": seconds_from_srt_timestamp(start_ts),
                "end": seconds_from_srt_timestamp(end_ts),
                "text": text.strip()
            })
        
        from modules.transcriber import merge_orphan_punctuation_segments
        segments = merge_orphan_punctuation_segments(segments)
        segments = merge_sentence_fragments(segments)
        for i, s in enumerate(segments, 1):
            s["index"] = i

        return segments

    def segments_to_srt(self, segments, output_path, text_key="text"):
        """
        Write an SRT file from a list of segments with timecode normalization.
        """
        from modules.srt_cleaner import normalize_timecodes
        normalized = normalize_timecodes(segments, min_gap_ms=40, min_cue_duration_ms=400, text_key=text_key)
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            for i, seg in enumerate(normalized, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                text = seg.get(text_key, "").strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def segments_to_clean_srt(self, segments, output_path, text_key="text", lang_code="fr", clean_fillers=True):
        """
        Write an ergonomically wrapped and optionally filler-cleaned SRT file following
        the strict 7-step pipeline:
        1. WhisperX (already transcribed)
        2. Cues splitting on word_timestamps
        3. Remove empty / punctuation-only cues and redistribute duration
        4. Inter-cue casing correction
        5. External ASR dictionary corrections on reconstituted text across cues
        6. Final timecode normalization (end[i] <= start[i+1] - 40ms, min 400ms duration)
        7. Export clean SRT
        """
        from modules.srt_cleaner import (
            SRTCleaner,
            remove_empty_cues_and_redistribute,
            fix_inter_cue_casing,
            apply_asr_corrections_cross_cues,
            normalize_timecodes,
        )
        cleaner = SRTCleaner()
        
        target_segs = segments
        if clean_fillers:
            target_segs = cleaner.clean_segments_heuristic(segments, lang_code=lang_code)
            
        # Step 3: Remove empty cues & redistribute duration
        target_segs = remove_empty_cues_and_redistribute(target_segs, text_key=text_key)
        
        # Step 4: Inter-cue casing
        target_segs = fix_inter_cue_casing(target_segs, text_key=text_key, lang_code=lang_code)
        
        # Step 5: ASR dictionary corrections across reconstituted text
        target_segs = apply_asr_corrections_cross_cues(target_segs, text_key=text_key)
        
        # Wrap into ergonomic cues without inventing durations
        cues = cleaner.split_into_ergonomic_cues(target_segs, text_key=text_key)
        
        # Step 6: Strict timecode normalization
        cues = normalize_timecodes(cues, min_gap_ms=40, min_cue_duration_ms=400, text_key=text_key)
        
        # Step 7: Export SRT
        cleaner.export_srt(cues, output_path)
        return cues

    def segments_to_bilingual_srt(self, segments, output_path, original_key="text", translated_key="translated_text"):
        """
        Generate a bilingual SRT (original + translation).
        """
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                original = seg.get(original_key, "").strip()
                translated = seg.get(translated_key, "").strip()
                f.write(f"{i}\n{start} --> {end}\n{original}\n{translated}\n\n")

    def validate_srt(self, segments):
        """
        Validate segment consistency.
        Returns a list of errors.
        """
        errors = []
        for i, seg in enumerate(segments):
            if seg["end"] <= seg["start"]:
                errors.append(f"Segment {i+1}: invalid duration ({seg['start']} -> {seg['end']})")
            if i > 0 and seg["start"] < segments[i-1]["end"]:
                errors.append(f"Segment {i+1}: overlaps with segment {i}")
        return errors

    def convert_user_srt_to_segments(self, srt_path):
        """
        Load and validate a user SRT file.
        """
        segments = self.parse_srt(srt_path)
        errors = self.validate_srt(segments)
        return segments, errors

if __name__ == "__main__":
    # Test simple
    print("Test srt_parser.py...")
    parser = SRTParser()
    test_segments = [
        {"index": 1, "start": 0.5, "end": 2.5, "text": "Hello"},
        {"index": 2, "start": 3.0, "end": 4.0, "text": "World"}
    ]
    parser.segments_to_srt(test_segments, "temp/test.srt")
    parsed = parser.parse_srt("temp/test.srt")
    print(f"Segments parsed: {len(parsed)}")
    print(f"First text: {parsed[0]['text']}")
