import os
import re
from modules.utils import format_timestamp, seconds_from_srt_timestamp

class SRTParser:
    def parse_srt(self, srt_path):
        """
        Parse an SRT file and return a list of segments.
        Returns a list of dict: {"index": int, "start": float, "end": float, "text": str}
        """
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
        
        return segments

    def segments_to_srt(self, segments, output_path, text_key="text"):
        """
        Write an SRT file from a list of segments.
        """
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                text = seg.get(text_key, "").strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

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
