import os
import re
from modules.utils import format_timestamp, seconds_from_srt_timestamp

class SRTParser:
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
