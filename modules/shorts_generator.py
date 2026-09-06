import os
import sys
import json
import re
import subprocess
import shutil
from typing import List, Dict, Any, Optional

try:
    from scenedetect import open_video, SceneManager, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False


def get_ffmpeg_binary() -> str:
    """Return the path to an FFmpeg binary with full filter and libass support."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_ffmpeg = os.path.join(root_dir, "env", "Scripts", "ffmpeg.exe")
    if os.path.exists(env_ffmpeg):
        return env_ffmpeg

    winget_ffmpeg = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe")
    if os.path.exists(winget_ffmpeg):
        return winget_ffmpeg

    found = shutil.which("ffmpeg")
    if found:
        return found
    return "ffmpeg"


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS subtitle timestamp: H:MM:SS.cs"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _format_time_display(seconds: float) -> str:
    """Format seconds into MM:SS display string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_short_title(title: str) -> str:
    """Clean and polish Short hook title: sentence case, brand capitalization, no AI clichés."""
    if not title:
        return "High-impact video highlight"
    t = title.strip().strip('"\'«»#*')
    
    # Remove sensationalist AI clichés (both French and English)
    cliches = [
        r"\bshocking\s+(?:truth|revelation)\s*[:\-–—]?\s*",
        r"\bultimate\s+guide\s*[:\-–—]?\s*",
        r"\byou\s+won'?t\s+believe\s*[:\-–—]?\s*",
        r"\bsecret\s+revealed\s*[:\-–—]?\s*",
        r"\brévélation\s+choc\s*[:\-–—]?\s*",
        r"\ble\s+twist\s+final\s*[:\-–—]?\s*",
        r"\bla\s+preuve\s+par\s+la\s+démo\s*[:\-–—]?\s*",
        r"\bsecret(?:s)?\s+dévoilé(?:s)?\s*[:\-–—]?\s*",
        r"\bincroyable\s+découverte\s*[:\-–—]?\s*",
        r"\bce\s+secret\s+que\s+personne\s+ne\s+vous\s+dit\s*[:\-–—]?\s*",
        r"\ble\s+guide\s+ultime\s*[:\-–—]?\s*"
    ]
    for c in cliches:
        t = re.sub(c, "", t, flags=re.IGNORECASE)
    
    t = re.sub(r'[\s\-:_]+$', '', t).strip()
    t = re.sub(r'^[\s\-:_]+', '', t).strip()
    
    if not t:
        return "High-impact video highlight"
        
    # Sentence case formatting for French
    parts = t.split(":")
    formatted_parts = []
    for p in parts:
        p_strip = p.strip()
        if not p_strip:
            continue
        words = p_strip.split()
        fixed_words = []
        for idx, w in enumerate(words):
            if idx == 0:
                fixed_words.append(w.capitalize())
            else:
                w_lower = w.lower()
                if w_lower in ["hermes", "hermès"]:
                    fixed_words.append("Hermès")
                elif w_lower in ["hermesagent", "hermèsagent"]:
                    fixed_words.append("Hermès Agent")
                elif w_lower in ["windows", "linux", "macos"]:
                    fixed_words.append(w.capitalize())
                elif w_lower in ["ollama", "qwen", "telegram", "chatgpt", "claude", "whisper", "whisperx", "demucs", "python", "docker", "pinokio"]:
                    fixed_words.append(w.capitalize())
                elif w_lower in ["ia", "ai", "api", "gpu", "vram", "cpu", "llm", "tts", "srt", "seo", "ass", "hd", "4k"]:
                    fixed_words.append(w.upper())
                else:
                    fixed_words.append(w.lower())
        formatted_parts.append(" ".join(fixed_words))
    t = " : ".join(formatted_parts)

    # Brand maps
    brand_map = {
        r"\bherm[eè]s\s+agent\b": "Hermès Agent",
        r"\bherm[eè]s\b": "Hermès",
        r"\bwindows\b": "Windows",
        r"\bia\b": "IA",
        r"\bai\b": "AI",
        r"\bapi\b": "API",
        r"\bchatgpt\b": "ChatGPT",
        r"\bclaude\b": "Claude",
        r"\bpython\b": "Python",
        r"\byoutube\b": "YouTube",
        r"\bgpu\b": "GPU",
        r"\bvram\b": "VRAM",
        r"\bllm\b": "LLM",
        r"\bollama\b": "Ollama",
        r"\bqwen\b": "Qwen",
        r"\btelegram\b": "Telegram",
        r"\bwhisperx\b": "WhisperX",
        r"\bpinokio\b": "Pinokio"
    }
    for pat, rep in brand_map.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    return t


class ViralShortsStudio:
    """Automated engine for detecting, snapping, recropping and rendering viral 9:16 shorts."""

    def __init__(self):
        self.scenedetect_available = SCENEDETECT_AVAILABLE

    def detect_viral_moments(
        self,
        segments: List[Dict[str, Any]],
        llm_backend=None,
        num_shorts: int = 3,
        min_duration: float = 25.0,
        max_duration: float = 55.0,
        source_lang: str = "fr",
        text_key: str = "text",
        target_lang: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze the full transcript with the LLM to identify the Top N most viral, standalone moments.
        Returns a list of dicts with: title, reason, start, end, duration, score.
        """
        if not segments:
            return []

        # Prepare condensed transcript with uniform sampling across the full timeline
        total_duration = segments[-1]["end"]
        max_samples = 35
        if len(segments) <= max_samples:
            sampled_segments = segments
        else:
            step = len(segments) / max_samples
            sampled_segments = [segments[int(i * step)] for i in range(max_samples)]

        transcript_lines = []
        for s in sampled_segments:
            sec_val = int(s["start"])
            m = sec_val // 60
            sec_rem = sec_val % 60
            if text_key != "text":
                text_clean = (s.get(text_key) or s.get("translated_text") or s.get("fitted_text") or s.get("normal_text") or s.get("text", "")).strip()
            else:
                text_clean = s.get('text', '').strip()
            if text_clean:
                transcript_lines.append(f"[{sec_val}s ({m:02d}:{sec_rem:02d})] {text_clean}")
        condensed_transcript = "\n".join(transcript_lines)

        desired_lang = target_lang if (target_lang and text_key != "text") else source_lang
        lang_rule = (
            f"- LANGUAGE REQUIREMENT: You MUST write the 'title' and 'reason' strictly in the target language ({desired_lang.upper()}). "
            f"Do NOT output titles or explanations in any other language.\n"
        )

        prompt = (
            f"You are a world-class viral video editor (YouTube Shorts, TikTok, Instagram Reels).\n"
            f"Analyze the transcript below and identify exactly the top {num_shorts} standalone, high-impact moments for vertical video clips.\n\n"
            f"TOTAL VIDEO DURATION: {int(total_duration)} seconds.\n"
            f"STRICT TIMESTAMP RULES:\n"
            f"- Transcript timestamps are provided as [Xs (MM:SS)], where X is SECONDS.\n"
            f"- You MUST provide 'start_sec' and 'end_sec' in REAL SECONDS (e.g., 325, 629, 1227).\n"
            f"- NEVER exceed total video duration ({int(total_duration)}s)!\n"
            f"- Each clip duration must be between {int(min_duration)} and {int(max_duration)} seconds.\n"
            f"- Each clip must be 100% standalone and understandable without the rest of the video.\n"
            f"- Prioritize key demonstrations, concrete steps, or practical tips.\n"
            f"- Hook title: descriptive, natural, high engagement. RULE: NEVER use sensational clickbait clichés ('Shocking truth', 'Ultimate guide', 'Insane hack'). Use natural sentence case.\n"
            f"{lang_rule}"
            f"- Respond STRICTLY with the JSON array below, with NO wrapping markdown or commentary:\n\n"
            f"[\n"
            f'  {{"title": "How to set up Hermes Agent on Windows in 2 minutes", "reason": "Clear, concise walkthrough of the setup flow", "start_sec": 325.0, "end_sec": 375.0, "score": 95}},\n'
            f'  {{"title": "Controlling your PC remotely via Telegram", "reason": "Impressive real-world practical use case", "start_sec": 1227.0, "end_sec": 1265.0, "score": 93}}\n'
            f"]\n\n"
            f"TRANSCRIPT:\n{condensed_transcript}"
        )

        def _parse_time_val(val, default_val=0.0):
            v = None
            if isinstance(val, (int, float)):
                v = float(val)
            elif isinstance(val, str):
                val_clean = val.strip().replace(",", ".")
                if ":" in val_clean:
                    parts = val_clean.split(":")
                    if len(parts) == 2:
                        try:
                            v = float(parts[0]) * 60.0 + float(parts[1])
                        except Exception:
                            pass
                    elif len(parts) == 3:
                        try:
                            v = float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                        except Exception:
                            pass
                if v is None:
                    try:
                        v = float(val_clean)
                    except Exception:
                        pass
            if v is None:
                return default_val

            # Handle case where LLM wrote MMSS flat integer (e.g. 2027 for 20:27, 1029 for 10:29, 525 for 05:25)
            if total_duration > 0:
                # 1. If v exceeds video duration, it is likely flat MMSS
                if v > total_duration and v < 6000:
                    m = int(v // 100)
                    s = int(v % 100)
                    if s < 60:
                        conv = m * 60.0 + s
                        if conv <= total_duration:
                            print(f"[SHORTS] Auto-converted MMSS flat integer {v} -> {conv:.1f}s ({m}m{s:02d}s)")
                            return conv
                # 2. Check if decimal MM.SS (e.g. 19.14 when total_duration > 300)
                if total_duration > 300 and 0.0 < v < 60.0:
                    dec_part = round((v - int(v)) * 100)
                    if dec_part < 60:
                        conv = int(v) * 60.0 + dec_part
                        if conv <= total_duration:
                            return conv
                # 3. Check if v without exceeding total_duration is actually flat MMSS (e.g. 525 -> 325, 1029 -> 629)
                if v >= 100 and v < 6000:
                    m = int(v // 100)
                    s = int(v % 100)
                    if s < 60:
                        conv = m * 60.0 + s
                        has_nearby_segment = any(abs(seg["start"] - conv) <= 6.0 for seg in segments)
                        has_nearby_v = any(abs(seg["start"] - v) <= 6.0 for seg in segments)
                        if has_nearby_segment and not has_nearby_v:
                            print(f"[SHORTS] Auto-converted probable MMSS {v} -> {conv:.1f}s ({m}m{s:02d}s)")
                            return conv

            return v

        detected_clips = []
        if llm_backend:
            try:
                messages = [
                    {"role": "system", "content": "Tu es un expert mondial en création de contenu viral (YouTube Shorts, TikTok, Instagram Reels). Tu réponds STRICTEMENT au format JSON demandé, sans aucun texte introductif ni conclusion."},
                    {"role": "user", "content": prompt}
                ]
                
                raw_out = llm_backend.generate(messages, max_new_tokens=700, do_sample=True, temperature=0.3)
                print(f"[SHORTS] LLM raw output:\n{raw_out}")
                
                # Extract JSON array
                match = re.search(r"\[\s*\{.*?\}\s*\]", raw_out, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    for item in parsed:
                        st = _parse_time_val(item.get("start") or item.get("start_sec") or item.get("start_time"), 0.0)
                        en = _parse_time_val(item.get("end") or item.get("end_sec") or item.get("end_time"), 0.0)
                        
                        if en <= st:
                            en = st + 35.0
                        dur = en - st
                        if dur < min_duration:
                            en = st + min_duration
                        elif dur > max_duration:
                            en = st + max_duration
                        if total_duration > 0:
                            st = max(0.0, min(st, total_duration - 15.0))
                            en = max(st + 15.0, min(en, total_duration))
                        
                        raw_title = str(item.get("title", f"Highlight ({_format_time_display(st)})")).strip()
                        detected_clips.append({
                            "title": format_short_title(raw_title),
                            "reason": str(item.get("reason", "High-impact highlight detected")).strip(),
                            "start": max(0.0, st),
                            "end": en,
                            "duration": round(en - st, 1),
                            "score": int(item.get("score", 90))
                        })
            except Exception as e:
                print(f"[SHORTS] LLM extraction error: {e}")

        # Fallback heuristic if LLM failed or not provided
        if not detected_clips:
            print(f"[SHORTS] Using heuristic milestone detection fallback for {num_shorts} shorts")
            heuristic_titles = [
                "Catchy Hook & Core Concept",
                "Quick Setup & Getting Started",
                "Pro Tips & Key Configuration",
                "Hands-on Demo & Practical Walkthrough",
                "Key Takeaways & Final Results"
            ]
            if num_shorts == 1:
                ratios = [0.35]
            elif num_shorts == 2:
                ratios = [0.25, 0.70]
            elif num_shorts == 3:
                ratios = [0.18, 0.50, 0.82]
            elif num_shorts == 4:
                ratios = [0.15, 0.38, 0.62, 0.85]
            else: # 5
                ratios = [0.12, 0.30, 0.50, 0.70, 0.88]

            for i in range(min(num_shorts, len(ratios))):
                title = heuristic_titles[i % len(heuristic_titles)]
                t_center = total_duration * ratios[i]
                st = max(0.0, t_center - 18.0)
                en = min(total_duration, st + 38.0)
                detected_clips.append({
                    "title": format_short_title(title),
                    "reason": "Auto-selected viral segment from video timeline",
                    "start": round(st, 1),
                    "end": round(en, 1),
                    "duration": round(en - st, 1),
                    "score": 88
                })

        return detected_clips[:num_shorts]

    def refine_boundaries(
        self,
        video_path: Optional[str],
        candidate_clips: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Refine candidate start and end times using PySceneDetect visual cuts
        and WhisperX speech/silence boundaries to avoid chopped speech and mid-cut scenes.
        """
        visual_cuts = []
        if video_path and os.path.exists(video_path) and self.scenedetect_available:
            try:
                print(f"[SHORTS] Running PySceneDetect on {video_path}...")
                video = open_video(video_path)
                sm = SceneManager()
                sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
                sm.detect_scenes(video)
                scene_list = sm.get_scene_list()
                for scene in scene_list:
                    visual_cuts.append(scene[0].get_seconds())
                print(f"[SHORTS] Detected {len(visual_cuts)} visual camera cuts")
            except Exception as e:
                print(f"[SHORTS] PySceneDetect detection failed: {e}")

        refined_clips = []
        for clip in candidate_clips:
            raw_start = clip["start"]
            raw_end = clip["end"]

            # 1. Snap to nearest visual cut if within +/- 1.5 seconds
            best_start = raw_start
            best_end = raw_end
            if visual_cuts:
                for cut in visual_cuts:
                    if abs(cut - raw_start) <= 1.5:
                        best_start = cut
                        break
                for cut in visual_cuts:
                    if abs(cut - raw_end) <= 1.5:
                        best_end = cut
                        break

            # 2. Snap to nearest WhisperX speech segment boundaries
            if segments:
                # Find segment starting near best_start
                nearest_seg_start = min(segments, key=lambda s: abs(s["start"] - best_start))
                if abs(nearest_seg_start["start"] - best_start) <= 2.0:
                    best_start = nearest_seg_start["start"]

                # Find segment ending near best_end
                nearest_seg_end = min(segments, key=lambda s: abs(s["end"] - best_end))
                if abs(nearest_seg_end["end"] - best_end) <= 2.0:
                    best_end = nearest_seg_end["end"]

            # Guarantee valid duration between 20s and 59s and strictly within video duration
            total_duration = segments[-1]["end"] if segments else 0.0
            if total_duration > 0:
                best_start = max(0.0, min(best_start, total_duration - 20.0))
                best_end = min(best_end, total_duration)

            if best_end <= best_start:
                best_end = best_start + 35.0
            dur = best_end - best_start
            if dur < 20.0:
                best_end = best_start + 25.0
            elif dur > 59.0:
                best_end = best_start + 55.0

            if total_duration > 0:
                best_end = min(best_end, total_duration)

            clip_text = self.extract_clip_text(segments, best_start, best_end, text_key=text_key)
            refined_clips.append({
                "title": clip["title"],
                "reason": clip.get("reason", ""),
                "start": round(best_start, 2),
                "end": round(best_end, 2),
                "duration": round(best_end - best_start, 1),
                "score": clip.get("score", 90),
                "subtitles": clip_text
            })

        return refined_clips

    def extract_clip_text(
        self,
        segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
        text_key: str = "text"
    ) -> str:
        """Extract continuous subtitle text for a clip interval."""
        matched = []
        for s in segments:
            s_st = s.get("start", 0.0)
            s_en = s.get("end", 0.0)
            if s_en <= start_time or s_st >= end_time:
                continue
            if text_key != "text":
                txt = (s.get(text_key) or s.get("translated_text") or s.get("fitted_text") or s.get("normal_text") or s.get("text", "")).strip()
            else:
                txt = s.get("text", "").strip()
            if txt:
                matched.append(txt)
        return " ".join(matched)

    def generate_ass_subtitles(
        self,
        segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
        ass_output_path: str,
        style: str = "tiktok_yellow",
        text_key: str = "text",
        custom_text: Optional[str] = None
    ) -> str:
        """
        Generate a stylized .ass subtitle file for 9:16 vertical video.
        Supports TikTok / Shorts Viraux animated word-by-word karaoke with {\\kf<cs>} tags.
        Styles:
          - 'tiktok_yellow' / 'yellow_glow': Vibrant Neon Yellow active highlight with white base (Viral Shorts style).
          - 'tiktok_mint' / 'mint_green': Mint Green active highlight with white base.
          - 'tiktok_cyan': Electric Cyan active highlight with white base.
          - 'tiktok_magenta': Hot Fuchsia Pink active highlight with white base.
          - 'clean_white': Minimalist crisp white text without karaoke wiping.
        """
        is_karaoke = True
        if style in ("tiktok_mint", "mint_green", "mint"):
            highlight_color = "&H00D4FF70" # Neon mint green (&H00BBGGRR)
            base_color = "&H00FFFFFF"      # Crisp white
        elif style in ("tiktok_cyan", "cyan"):
            highlight_color = "&H00FFFF00" # Electric cyan
            base_color = "&H00FFFFFF"
        elif style in ("tiktok_magenta", "magenta", "pink"):
            highlight_color = "&H00D420FF" # Hot fuchsia pink
            base_color = "&H00FFFFFF"
        elif style in ("clean_white", "white"):
            highlight_color = "&H00FFFFFF"
            base_color = "&H00D0D0D0"
            is_karaoke = False
        else: # tiktok_yellow, yellow_glow, default
            highlight_color = "&H003BF5FF" # Vibrant TikTok/Shorts yellow (&H00BBGGRR)
            base_color = "&H00FFFFFF"      # Crisp white

        outline_color = "&H00000000"
        back_color = "&HA0000000"

        ass_header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial Black,72,{highlight_color},{base_color},{outline_color},{back_color},"
            "-1,0,0,0,100,100,1,0,1,7,4,2,40,40,360,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        all_words = []
        if custom_text and custom_text.strip():
            # User edited subtitles take priority: distribute words proportionally across clip duration
            raw_tokens = custom_text.strip().split()
            clip_dur = max(1.0, end_time - start_time)
            total_chars = sum(len(t) for t in raw_tokens)
            curr_t = start_time
            for tok in raw_tokens:
                weight = len(tok) / max(1, total_chars)
                tok_dur = max(0.14, weight * clip_dur)
                tok_end = min(end_time, curr_t + tok_dur)
                all_words.append({
                    "word": tok,
                    "start": curr_t,
                    "end": tok_end
                })
                curr_t = tok_end
        else:
            # Collect all individual words overlapping the short clip window [start_time, end_time]
            for s in segments:
                s_start = s.get("start", 0.0)
                s_end = s.get("end", 0.0)
                if s_end <= start_time or s_start >= end_time:
                    continue

                # If segment has native WhisperX word-level alignments (only for original audio matching source text)
                words_in_seg = s.get("words")
                if words_in_seg and isinstance(words_in_seg, list) and len(words_in_seg) > 0 and text_key == "text":
                    for w in words_in_seg:
                        w_start = w.get("start", s_start)
                        w_end = w.get("end", s_end)
                        w_text = w.get("word", "").strip()
                        if w_text and w_end > start_time and w_start < end_time:
                            all_words.append({
                                "word": w_text,
                                "start": w_start,
                                "end": w_end
                            })
                else:
                    # Proportional fallback for segments without word alignment or translated/dubbed text
                    if text_key != "text":
                        text = (s.get(text_key) or s.get("translated_text") or s.get("fitted_text") or s.get("normal_text") or s.get("text", "")).strip()
                    else:
                        text = s.get("text", "").strip()
                    tokens = text.split()
                    if tokens:
                        total_chars = sum(len(t) for t in tokens)
                        seg_dur = max(0.2, s_end - s_start)
                        curr_t = s_start
                        for tok in tokens:
                            tok_dur = max(0.12, (len(tok) / max(1, total_chars)) * seg_dur)
                            tok_end = curr_t + tok_dur
                            if tok_end > start_time and curr_t < end_time:
                                all_words.append({
                                    "word": tok,
                                    "start": curr_t,
                                    "end": tok_end
                                })
                            curr_t = tok_end

        # Clean words: merge isolated punctuation tokens into preceding word
        cleaned_words = []
        for w in all_words:
            w_text = w["word"].strip()
            if not w_text:
                continue
            is_only_punct = bool(re.match(r"^[\s\.\?\!\,\;\:\-\–\—\…\¿\¡\'\"\(\)\[\]\«\»]+$", w_text))
            if is_only_punct and cleaned_words:
                cleaned_words[-1]["word"] += f" {w_text}"
                cleaned_words[-1]["end"] = max(cleaned_words[-1]["end"], w["end"])
            else:
                cleaned_words.append(w)
        all_words = cleaned_words

        # Group words into high-impact TikTok / Shorts chunks (3-5 words max, or ~20 chars)
        chunks = []
        curr_chunk = []
        curr_chars = 0
        for w in all_words:
            curr_chunk.append(w)
            curr_chars += len(w["word"]) + 1
            word_str = w["word"]
            has_break_punct = any(word_str.endswith(p) for p in ('.', '?', '!', ';', ':', '...'))
            if len(curr_chunk) >= 4 or curr_chars >= 20 or has_break_punct:
                chunks.append(curr_chunk)
                curr_chunk = []
                curr_chars = 0
        if curr_chunk:
            chunks.append(curr_chunk)

        dialogue_lines = []
        clip_duration = end_time - start_time

        for chunk in chunks:
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]

            # Hold the completed chunk for 0.25s so viewers can absorb the punchline
            chunk_end_hold = chunk_end + 0.25

            # Re-baseline relative to clip start
            rel_start = max(0.0, chunk_start - start_time)
            rel_end = min(clip_duration, chunk_end_hold - start_time)
            if rel_end <= rel_start:
                continue

            if is_karaoke:
                # Build karaoke timing tags: {\kf<cs>}WORD
                karaoke_parts = []
                for idx, w in enumerate(chunk):
                    w_st = max(chunk_start, w["start"])
                    w_en = max(w_st + 0.05, w["end"])
                    # If there's a gap to the next word in the chunk, absorb it into this word's highlight
                    if idx < len(chunk) - 1:
                        next_st = chunk[idx + 1]["start"]
                        if next_st > w_en:
                            w_en = next_st

                    dur_cs = max(6, int(round((w_en - w_st) * 100)))
                    word_clean = w["word"].upper()
                    karaoke_parts.append(f"{{\\kf{dur_cs}}}{word_clean}")

                text_content = " ".join(karaoke_parts)
            else:
                # Static format
                text_content = " ".join(w["word"].upper() for w in chunk)

            start_ass = _format_ass_time(rel_start)
            end_ass = _format_ass_time(rel_end)
            dialogue_lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_content}")

        # Fallback if no words matched
        if not dialogue_lines:
            for s in segments:
                s_start = s.get("start", 0.0)
                s_end = s.get("end", 0.0)
                if s_end <= start_time or s_start >= end_time:
                    continue
                rel_start = max(0.0, s_start - start_time)
                rel_end = min(clip_duration, s_end - start_time)
                text = s.get(text_key, s.get("text", "")).strip().upper()
                if text:
                    start_ass = _format_ass_time(rel_start)
                    end_ass = _format_ass_time(rel_end)
                    dialogue_lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")

        with open(ass_output_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(dialogue_lines) + "\n")

        return ass_output_path

    def render_vertical_short(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        audio_path: Optional[str] = None,
        crop_mode: str = "blur_stack",
        ass_subtitles_path: Optional[str] = None
    ) -> bool:
        """
        Cut and render a 9:16 vertical short (1080x1920) using FFmpeg.
        Crop modes:
          - 'blur_stack': 16:9 video centered with stacked blurred background (recommended).
          - 'crop_center': 1080x1920 center cropped view.
        """
        duration = end_time - start_time
        if duration <= 0:
            return False

        def _build_and_run(use_subtitles: bool = True) -> bool:
            sub_filter = ""
            if use_subtitles and ass_subtitles_path and os.path.exists(ass_subtitles_path):
                escaped_ass = ass_subtitles_path.replace("\\", "/").replace(":", "\\:")
                sub_filter = f",subtitles='{escaped_ass}'"

            if crop_mode == "crop_center":
                vf = f"[0:v]scale=-1:1920,crop=1080:1920:(in_w-1080)/2:0{sub_filter}[v_out]"
            else: # blur_stack
                vf = (
                    f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
                    f"[0:v]scale=1080:-1[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2{sub_filter}[v_out]"
                )

            # Base FFmpeg command
            ffmpeg_bin = get_ffmpeg_binary()
            cmd = [
                ffmpeg_bin, "-y",
                "-ss", f"{start_time:.2f}",
                "-t", f"{duration:.2f}",
                "-i", video_path,
            ]

            if audio_path and os.path.exists(audio_path):
                cmd.extend([
                    "-ss", f"{start_time:.2f}",
                    "-t", f"{duration:.2f}",
                    "-i", audio_path,
                    "-filter_complex", vf,
                    "-map", "[v_out]",
                    "-map", "1:a:0?",
                ])
            else:
                cmd.extend([
                    "-filter_complex", vf,
                    "-map", "[v_out]",
                    "-map", "0:a:0?",
                ])

            # Codec candidate list in order of performance and compatibility
            codec_candidates = [
                # 1. NVIDIA Hardware NVENC
                ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-c:a", "aac", "-b:a", "192k", output_path],
                # 2. Standard software x264
                ["-c:v", "libx264", "-preset", "fast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", output_path],
                # 3. OpenH264 (conda-forge default)
                ["-c:v", "libopenh264", "-b:v", "4000k", "-c:a", "aac", "-b:a", "192k", output_path],
                # 4. Windows Media Foundation Hardware
                ["-c:v", "h264_mf", "-b:v", "4000k", "-c:a", "aac", output_path],
                # 5. Generic H.264
                ["-c:v", "h264", "-c:a", "aac", output_path]
            ]

            print(f"[SHORTS] Rendering short: {output_path} ({start_time:.2f}s -> {end_time:.2f}s, mode={crop_mode}, subs={'on' if sub_filter else 'off'})")
            
            for idx, codec_opts in enumerate(codec_candidates):
                full_cmd = list(cmd) + codec_opts
                try:
                    res = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                        encoder_name = codec_opts[1]
                        print(f"[SHORTS] Successfully rendered with {encoder_name}: {output_path}")
                        return True
                    else:
                        err_snippet = res.stderr[-300:].strip() if res.stderr else "unknown error"
                        print(f"[SHORTS] Encoder {codec_opts[1]} failed: {err_snippet}")
                except Exception as e:
                    print(f"[SHORTS] Encoder {codec_opts[1]} exception: {e}")

            return False

        # Try with subtitles first
        success = _build_and_run(use_subtitles=True)
        if not success and ass_subtitles_path:
            print("[SHORTS] Retrying render without embedded subtitles as fallback...")
            success = _build_and_run(use_subtitles=False)

        return success


shorts_studio = ViralShortsStudio()
