import numpy as np
import soundfile as sf
import librosa
import os
from modules.utils import get_exact_duration
from config import (
    TOLERANCE_TOO_LONG, MIN_SEGMENT_DURATION, TEMP_DIR,
    OUTPUT_SAMPLE_RATE, CHARS_PER_SECOND, MAX_SPEED_FACTOR,
    NEVER_CUT_WARNING
)

# Two-pass: natural voice first, native speed instruction if overflow

class TimeSync:
    def __init__(self, tts_engine, reformulator=None):
        self.tts = tts_engine
        self.reformulator = reformulator

    def _resample_to_target(self, audio, orig_sr, target_sr=OUTPUT_SAMPLE_RATE):
        """Resample audio to target sample rate if needed."""
        if orig_sr == target_sr:
            return audio, target_sr
        audio_resampled = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        return audio_resampled, target_sr



    @staticmethod
    def estimate_duration(text, language_family="latin"):
        """
        Estimate TTS duration from text length using chars/second ratios.
        Returns estimated duration in seconds.
        """
        cps = CHARS_PER_SECOND.get(language_family, CHARS_PER_SECOND["default"])
        return len(text) / cps

    @staticmethod
    def get_language_family(lang_code):
        """Map language codes to language families for duration estimation."""
        cjk = ["jpn", "zho", "kor", "ja", "zh", "ko"]
        arabic = ["arb", "ar"]
        for prefix in cjk:
            if lang_code.startswith(prefix):
                return "cjk"
        for prefix in arabic:
            if lang_code.startswith(prefix):
                return "arabic"
        return "latin"

    def _compute_effective_durations(self, segments, total_duration=None):
        """
        Compute effective duration for each segment using up to 2 gaps.
        
        1-gap: extend into the gap between this segment and the next.
        2-gap: if the NEXT segment also has slack (its gap > its strict duration),
               borrow that slack so the current segment has more room.
        
        This avoids truncating dense segments when followed by sparse ones.
        """
        GAP_MARGIN = 0.03  # 30ms safety margin per boundary
        
        for i, seg in enumerate(segments):
            strict_duration = seg["end"] - seg["start"]
            
            # 1-gap: space to next segment
            if i + 1 < len(segments):
                gap_1 = segments[i + 1]["start"] - seg["start"] - GAP_MARGIN
            elif total_duration:
                gap_1 = total_duration - seg["start"]
            else:
                gap_1 = strict_duration
            
            available = gap_1
            
            # 2-gap: borrow slack from next segment's gap
            # If next segment has more space than it needs, we can use the extra
            if i + 1 < len(segments) and i + 2 < len(segments):
                next_strict = segments[i + 1]["end"] - segments[i + 1]["start"]
                next_gap = segments[i + 2]["start"] - segments[i + 1]["start"]
                next_slack = max(0, next_gap - next_strict - GAP_MARGIN)
                if next_slack > 0:
                    available = gap_1 + next_slack
            elif i + 1 < len(segments) and total_duration:
                # Last pair: next segment can extend to total_duration
                next_strict = segments[i + 1]["end"] - segments[i + 1]["start"]
                next_gap = total_duration - segments[i + 1]["start"]
                next_slack = max(0, next_gap - next_strict - GAP_MARGIN)
                if next_slack > 0:
                    available = gap_1 + next_slack
            
            seg["effective_duration"] = max(strict_duration, available)
            seg["strict_duration"] = strict_duration

    def pre_check_and_shorten(self, segments, language):
        """
        Pre-check timing with GLOBAL gap awareness.
        Computes effective durations and logs overflow warnings.
        Text fitting is handled upstream by translator + reformulator.
        """
        lang_family = self.get_language_family(language)
        
        # Global pass: compute effective durations
        self._compute_effective_durations(segments)
        
        warning_count = 0
        for seg in segments:
            text = seg.get("translated_text", seg.get("text", ""))
            effective = seg["effective_duration"]
            estimated_duration = self.estimate_duration(text, lang_family)
            
            seg["estimated_duration"] = round(estimated_duration, 2)
            seg["pre_shortened"] = False
            
            # Log warnings for segments that may still overflow
            if estimated_duration > effective and effective > MIN_SEGMENT_DURATION:
                overflow_pct = (estimated_duration - effective) / effective * 100
                print(f"[WARN] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: ~{overflow_pct:.0f}% over, may overflow")
                warning_count += 1
        
        if warning_count > 0:
            print(f"{warning_count} segments may still overflow (speed instruct + truncation will handle)")
        
        return segments, 0

    def sync_segment(self, segment, language, voice_path=None):
        """
        Synchronize a segment with two-pass speed control:
        1. Generate TTS at natural speed
        2. If overflow detected, re-generate with native speed instruction
        3. If still overflows → truncate with fade-out to prevent overlap
        """
        strict_duration = segment.get("strict_duration", segment["end"] - segment["start"])
        
        if strict_duration < MIN_SEGMENT_DURATION:
            return {
                "synced_path": None,
                "start": segment["start"],
                "end": segment["end"],
                "final_text": "",
                "sped_up": False,
                "truncated": False,
            }

        text = segment.get("translated_text", segment.get("text", ""))
        effective_duration = segment.get("effective_duration", strict_duration)
        was_sped_up = False
        was_truncated = False
        
        # PASS 1: Generate TTS at natural speed
        temp_tts_path = os.path.join(TEMP_DIR, f"seg_{segment['start']:.2f}_temp.wav")
        res = self.tts.synthesize_segment(text, language, temp_tts_path, voice_path=voice_path)
        current_duration = res["duration"]
        
        overflow = current_duration - effective_duration
        
        # PASS 2: If overflows, re-generate with speed instruction
        if overflow > TOLERANCE_TOO_LONG and effective_duration > MIN_SEGMENT_DURATION:
            speed_factor = min(current_duration / effective_duration, MAX_SPEED_FACTOR)
            print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}]: {overflow:.2f}s over → re-generating at {speed_factor:.2f}x")
            
            temp_fast_path = os.path.join(TEMP_DIR, f"seg_{segment['start']:.2f}_fast.wav")
            res_fast = self.tts.synthesize_segment(
                text, language, temp_fast_path, voice_path=voice_path, speed_factor=speed_factor
            )
            
            # Use the faster version if it's shorter (instruct may not always work perfectly)
            if res_fast["duration"] < current_duration:
                res = res_fast
                temp_tts_path = temp_fast_path
                current_duration = res["duration"]
                was_sped_up = True
                new_overflow = current_duration - effective_duration
                if new_overflow > TOLERANCE_TOO_LONG:
                    print(f"  Still {new_overflow:.2f}s over after speedup — will truncate with fade-out")
                else:
                    print(f"  Speed pass OK: {current_duration:.2f}s fits in {effective_duration:.2f}s slot")
            else:
                print(f"  Speed instruct didn't help — keeping natural version")
        elif current_duration > strict_duration + TOLERANCE_TOO_LONG:
            gap_used = current_duration - strict_duration
            print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}]: using {gap_used:.2f}s from gap")

        # READ, RESAMPLE, SAVE
        final_synced_path = os.path.join(TEMP_DIR, f"seg_{segment['start']:.2f}_synced.wav")
        
        audio, sr = sf.read(temp_tts_path)
        audio, sr = self._resample_to_target(audio, sr)
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # PASS 3: Truncate with fade-out if STILL overflowing
        # This prevents the next segment from overwriting the tail of this one
        final_overflow = len(audio) / sr - effective_duration
        if final_overflow > TOLERANCE_TOO_LONG and effective_duration > MIN_SEGMENT_DURATION:
            max_samples = int(effective_duration * sr)
            if max_samples < len(audio):
                FADE_OUT_MS = 80  # 80ms smooth fade-out
                fade_samples = min(int(FADE_OUT_MS / 1000 * sr), max_samples // 2)
                audio = audio[:max_samples]
                # Apply fade-out at the end
                fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                audio[-fade_samples:] *= fade
                was_truncated = True
                print(f"  Truncated [{segment['start']:.1f}-{segment['end']:.1f}]: {len(audio)/sr:.2f}s (fade-out {FADE_OUT_MS}ms)")

        sf.write(final_synced_path, audio, sr)
        
        actual_duration = len(audio) / sr
        overflow_secs = max(0, actual_duration - effective_duration)
        
        return {
            "synced_path": final_synced_path,
            "start": segment["start"],
            "end": segment["end"],
            "final_text": text,
            "sped_up": was_sped_up,
            "truncated": was_truncated,
            "final_duration": round(actual_duration, 3),
            "slot_duration": round(effective_duration, 3),
            "overflow": round(overflow_secs, 3),
        }

    def sync_all(self, segments, language, voice_mapping=None, total_duration=None):
        """Sync all segments with global gap awareness.
        
        Returns:
            synced_segments: list of synced segment dicts
            stats: dict with sped_up, truncated, overflow counts and details
        """
        self._compute_effective_durations(segments, total_duration=total_duration)
        
        synced_segments = []
        sped_up_count = 0
        truncated_count = 0
        overflow_count = 0  # Audio overflows slot (will lose tail in crossfade)
        sped_up_list = []
        truncated_list = []
        overflow_list = []
        
        for seg in segments:
            voice = None
            if voice_mapping:
                spk = seg.get("speaker", "SPEAKER_00")
                voice = voice_mapping.get(spk)
                
            res = self.sync_segment(seg, language, voice_path=voice)
            synced_segments.append(res)
            
            if res.get("sped_up"):
                sped_up_count += 1
                sped_up_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]")
            if res.get("truncated"):
                truncated_count += 1
                truncated_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]")
            elif res.get("overflow", 0) > 0.05:  # >50ms overflow (not truncated but will crossfade)
                overflow_count += 1
                overflow_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] +{res['overflow']:.2f}s")
        
        cut_total = truncated_count + overflow_count
        stats = {
            "total": len(segments),
            "sped_up": sped_up_count,
            "truncated": truncated_count,
            "overflow": overflow_count,
            "cut_total": cut_total,
            "perfect": len(segments) - sped_up_count - cut_total,
            "sped_up_segments": sped_up_list,
            "truncated_segments": truncated_list,
            "overflow_segments": overflow_list,
        }
        
        print(f"\n[Sync Stats] {stats['perfect']}/{stats['total']} perfect, "
              f"{sped_up_count} sped up, {truncated_count} truncated, {overflow_count} overflow")
        if truncated_list:
            print(f"  Truncated (cut with fade-out): {', '.join(truncated_list)}")
        if overflow_list:
            print(f"  Overflow (tail lost in crossfade): {', '.join(overflow_list)}")
        
        return synced_segments, stats

    # ---- NEVER CUT VOCAL MODE ----

    def _generate_all_natural(self, segments, language, voice_mapping=None):
        """
        Phase 1: Generate ALL TTS at natural speed (no speedup, no truncation).
        Returns segments enriched with tts_path and tts_duration (real measured).
        """
        results = []
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg.get("text", ""))
            strict_duration = seg["end"] - seg["start"]

            if strict_duration < MIN_SEGMENT_DURATION or not text.strip():
                results.append({
                    **seg,
                    "tts_path": None,
                    "tts_duration": 0.0
                })
                continue

            voice = None
            if voice_mapping:
                spk = seg.get("speaker", "SPEAKER_00")
                voice = voice_mapping.get(spk)

            tts_path = os.path.join(TEMP_DIR, f"nc_seg_{seg['start']:.2f}.wav")
            res = self.tts.synthesize_segment(
                text, language, tts_path, voice_path=voice, speed_factor=1.0
            )

            results.append({
                **seg,
                "tts_path": tts_path,
                "tts_duration": res["duration"]
            })
            print(f"[NeverCut] Seg [{seg['start']:.1f}-{seg['end']:.1f}]: "
                  f"TTS={res['duration']:.2f}s (slot={strict_duration:.2f}s)")

        return results

    def _plan_cascade_placement(self, segments, total_duration):
        """
        Phase 2: Global gap analysis + cascade placement with pre-absorption.
        
        Pass 1: Naive cascade — place each segment at max(cursor, original_start).
        Pass 2 (if overflow): Pull segments earlier by consuming available gaps,
                so the audio fits within total_duration.
        
        Returns:
            segments: enriched with real_start / real_end
            drift_info: dict with max_drift, final_overflow, absorbed_by_gaps
        """
        # ---- PASS 1: Naive cascade to measure overflow ----
        cursor = 0.0
        gap_info = []  # [(segment_index, available_gap)]

        for i, seg in enumerate(segments):
            original_start = seg["start"]
            tts_dur = seg.get("tts_duration", 0.0)

            if tts_dur <= 0:
                seg["real_start"] = original_start
                seg["real_end"] = seg["end"]
                seg["drift"] = 0.0
                continue

            if cursor < original_start:
                available_gap = original_start - cursor
                gap_info.append((i, available_gap))
                real_start = original_start
            else:
                real_start = cursor

            seg["real_start"] = real_start
            seg["real_end"] = real_start + tts_dur
            seg["drift"] = real_start - original_start
            cursor = seg["real_end"]

        naive_overflow = max(0.0, cursor - total_duration)

        # ---- PASS 2: Pre-absorb overflow by consuming gaps ----
        if naive_overflow > 0.05 and gap_info:
            total_available = sum(g for _, g in gap_info)
            # How much gap to consume (up to the overflow amount)
            to_absorb = min(naive_overflow, total_available)
            
            if to_absorb > 0.01:
                print(f"[NeverCut] Overflow={naive_overflow:.2f}s — "
                      f"pre-absorbing {to_absorb:.2f}s from {len(gap_info)} gaps")
                
                # Distribute absorption proportionally across gaps
                absorbed_per_gap = {}
                for idx, gap_size in gap_info:
                    proportion = gap_size / total_available
                    absorbed_per_gap[idx] = proportion * to_absorb
                
                # Re-run placement with reduced gaps
                cursor = 0.0
                for i, seg in enumerate(segments):
                    original_start = seg["start"]
                    tts_dur = seg.get("tts_duration", 0.0)

                    if tts_dur <= 0:
                        seg["real_start"] = original_start
                        seg["real_end"] = seg["end"]
                        seg["drift"] = 0.0
                        continue

                    if cursor < original_start:
                        # Reduce gap by absorbed amount (start earlier)
                        gap_reduction = absorbed_per_gap.get(i, 0)
                        earliest_start = max(cursor, original_start - gap_reduction)
                        real_start = earliest_start
                    else:
                        real_start = cursor

                    seg["real_start"] = real_start
                    seg["real_end"] = real_start + tts_dur
                    seg["drift"] = real_start - original_start
                    cursor = seg["real_end"]

        # ---- Compute final stats ----
        max_drift = 0.0
        total_gap_absorbed = 0.0
        
        for seg in segments:
            if seg.get("tts_duration", 0) <= 0:
                continue
            drift = seg.get("drift", 0)
            max_drift = max(max_drift, drift)
            # Negative drift means we started early (pre-absorbed)
            early_start = max(0, seg["start"] - seg.get("real_start", seg["start"]))
            total_gap_absorbed += early_start
            
            if abs(drift) > 0.1:
                direction = "early" if drift < 0 else "late"
                print(f"[NeverCut] Seg [{seg['start']:.1f}-{seg['end']:.1f}]: "
                      f"placed at {seg['real_start']:.2f}s ({direction}: {abs(drift):.2f}s)")

        final_overflow = max(0.0, cursor - total_duration)

        drift_info = {
            "max_drift": round(max_drift, 2),
            "final_overflow": round(final_overflow, 2),
            "absorbed_by_gaps": round(total_gap_absorbed, 2),
            "pre_absorbed": round(naive_overflow - final_overflow, 2),
        }

        print(f"[NeverCut] Summary: max_drift={max_drift:.2f}s, "
              f"final_overflow={final_overflow:.2f}s, "
              f"pre-absorbed={drift_info['pre_absorbed']:.2f}s")

        return segments, drift_info

    def sync_all_never_cut(self, segments, language, total_duration, voice_mapping=None):
        """
        Never Cut Vocal mode: generate all TTS at natural speed, then
        plan cascade placement using global gap analysis.
        
        Returns:
            synced_segments: list of dicts with synced_path, real_start, real_end
            drift_info: dict with max_drift, final_overflow, absorbed_by_gaps
        """
        print(f"\n{'='*60}")
        print(f"NEVER CUT VOCAL MODE — {len(segments)} segments")
        print(f"{'='*60}")

        # Phase 1: Generate all TTS at natural speed
        enriched = self._generate_all_natural(segments, language, voice_mapping)

        # Phase 2: Plan cascade placement
        placed, drift_info = self._plan_cascade_placement(enriched, total_duration)

        # Phase 3: Build synced_segments list (same format as sync_all output)
        synced_segments = []
        for seg in placed:
            tts_path = seg.get("tts_path")

            if tts_path and os.path.exists(tts_path):
                # Resample to target SR
                audio, sr = sf.read(tts_path)
                audio, sr = self._resample_to_target(audio, sr)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Save resampled version
                synced_path = os.path.join(TEMP_DIR, f"nc_seg_{seg['start']:.2f}_synced.wav")
                sf.write(synced_path, audio, sr)
            else:
                synced_path = None

            synced_segments.append({
                "synced_path": synced_path,
                "start": seg["start"],
                "end": seg["end"],
                "real_start": seg.get("real_start", seg["start"]),
                "real_end": seg.get("real_end", seg["end"]),
                "final_text": seg.get("translated_text", seg.get("text", "")),
                "drift": seg.get("drift", 0.0)
            })

        print(f"{'='*60}\n")
        return synced_segments, drift_info

    def build_full_audio(self, synced_segments, total_duration, sr=OUTPUT_SAMPLE_RATE, use_real_positions=False):
        """Assemble the full voice track with crossfade at overlaps.
        
        If use_real_positions=True (Never Cut mode), uses seg['real_start'] for placement
        instead of seg['start'], allowing cascade-shifted positioning.
        """
        CROSSFADE_MS = 50  # 50ms crossfade at overlap boundaries
        crossfade_samples = int(CROSSFADE_MS / 1000 * sr)
        
        output_path = os.path.join(TEMP_DIR, "full_voice_track.wav")
        total_samples = int(total_duration * sr)
        full_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Track where previous audio actually ends (for overlap detection)
        prev_audio_end = 0

        for seg in synced_segments:
            path = seg["synced_path"]
            if path is None or not os.path.exists(path):
                continue
                
            audio, seg_sr = sf.read(path)
            
            if seg_sr != sr:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=seg_sr, target_sr=sr)
            
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Use real_start for Never Cut mode, original start otherwise
            placement_time = seg.get("real_start", seg["start"]) if use_real_positions else seg["start"]
            start_sample = int(placement_time * sr)
            end_sample = min(start_sample + len(audio), total_samples)
            audio_len = end_sample - start_sample
            
            if audio_len <= 0:
                continue
            
            # Check if previous segment's audio overlaps into this one
            overlap = prev_audio_end - start_sample
            
            if overlap > crossfade_samples:
                # Crossfade zone: blend old (fade out) with new (fade in)
                xfade_len = min(crossfade_samples, audio_len, overlap)
                fade_out = np.linspace(1.0, 0.0, xfade_len, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, xfade_len, dtype=np.float32)
                
                # Apply crossfade
                full_audio[start_sample:start_sample + xfade_len] = (
                    full_audio[start_sample:start_sample + xfade_len] * fade_out
                    + audio[:xfade_len] * fade_in
                )
                # Write remainder normally
                if audio_len > xfade_len:
                    full_audio[start_sample + xfade_len:end_sample] = audio[xfade_len:audio_len]
            else:
                # No significant overlap — write normally
                full_audio[start_sample:end_sample] = audio[:audio_len]
            
            # Update the actual end of written audio
            prev_audio_end = max(prev_audio_end, end_sample)

        sf.write(output_path, full_audio, sr)
        return output_path

if __name__ == "__main__":
    pass
