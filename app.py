import sys
# Force UTF-8 encoding on Windows console to prevent charmap UnicodeEncodeErrors with Asian/Cyrillic/Arabic text
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import warnings
import logging
# Suppress noisy 3rd-party startup warnings
warnings.filterwarnings("ignore", message=".*rope_config_validation.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*In 2.9, this function.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*save_with_torchcodec.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*\ntorchcodec is not installed.*")  # pyannote wraps torchcodec RuntimeError
warnings.filterwarnings("ignore", message=".*fast path is not available.*")
warnings.filterwarnings("ignore", message=".*You are using a Python version.*", category=FutureWarning)  # google-api-core Python EOL warning
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.WARNING)

import gradio as gr
import os
import glob
import shutil
import time
import json
import re
import sys
from config import *
from config import NEVER_CUT_WARNING
from modules.downloader import VideoDownloader
from modules.separator import VocalSeparator
from modules.transcriber import Transcriber
from modules.translator import Translator
from modules.reformulator import Reformulator
from modules.tts_backends.factory import get_backend as get_tts_backend, get_available_backends as get_available_tts_backends
from modules.llm_backends.factory import get_backend as get_llm_backend, get_available_backends as get_available_llm_backends
from modules.time_sync import TimeSync
from modules.audio_mixer import AudioMixer
from modules.video_assembler import VideoAssembler
from modules.srt_parser import SRTParser
from modules.youtube_publisher import YouTubePublisher
from modules.seo_assistant import seo_assistant
from modules.shorts_generator import shorts_studio
from modules.blog_generator import blog_generator, sync_humanizer_rules_from_github
from modules.flux_generator import flux_studio
from fitted_cps_config import get_fitted_cps, get_effective_cps, load_user_cps, save_user_cps, FITTED_CPS_BY_LANG

import pandas as pd

# --- HELPERS ---

# NLLB code → ISO 639-1 uppercase
_NLLB_TO_ISO = {
    "fra_Latn": "FR", "eng_Latn": "EN", "spa_Latn": "ES",
    "deu_Latn": "DE", "ita_Latn": "IT", "por_Latn": "PT",
    "jpn_Jpan": "JA", "kor_Hang": "KO", "zho_Hans": "ZH",
    "rus_Cyrl": "RU", "arb_Arab": "AR", "hin_Deva": "HI",
    "nld_Latn": "NL", "pol_Latn": "PL", "tur_Latn": "TR",
    "swe_Latn": "SV", "ces_Latn": "CS", "ron_Latn": "RO",
    "hun_Latn": "HU",
    # VoxCPM2 additional languages
    "mya_Mymr": "MY", "dan_Latn": "DA", "fin_Latn": "FI",
    "ell_Grek": "EL", "heb_Hebr": "HE", "ind_Latn": "ID",
    "khm_Khmr": "KM", "lao_Laoo": "LO", "zsm_Latn": "MS",
    "nob_Latn": "NO", "swh_Latn": "SW", "tgl_Latn": "TL",
    "tha_Thai": "TH", "vie_Latn": "VI",
}

def _get_iso_code(lang_code):
    """Normalize any language code (Whisper 'en', NLLB 'fra_Latn') to ISO 639-1 uppercase."""
    if not lang_code:
        return "XX"
    # Check NLLB mapping first
    if lang_code in _NLLB_TO_ISO:
        return _NLLB_TO_ISO[lang_code]
    # Whisper-style short code (en, fr, es...)
    return lang_code[:2].upper()

def _get_segments_json():
    import json
    out = []
    # If translated_segments exists, use it, else use segments
    segs = state.translated_segments if state.translated_segments else state.segments
    for s in segs:
        out.append({
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "text": s.get("text", ""),
            "translated_text": s.get("translated_text", ""),
            "normal_text": s.get("normal_text", "")
        })
    return json.dumps(out)

def _get_segments_json_html():
    import base64
    json_str = _get_segments_json()
    b64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return f'<div id="segments_json_data" style="display:none;">{b64_str}</div>'

def _get_empty_segments_html():
    import base64
    b64_str = base64.b64encode(b"[]").decode('utf-8')
    return f'<div id="segments_json_data" style="display:none;">{b64_str}</div>'

def find_segment_audio_path(start, lang=None, never_cut=None):
    """Find the cached synced audio path for a given segment start time, supporting multiple naming conventions."""
    try:
        f_start = float(start)
    except (ValueError, TypeError):
        f_start = 0.0

    # Build candidate start time strings (primary + slight float rounding tolerances)
    start_strs = [f"{f_start:.2f}"]
    for delta in (-0.01, 0.01):
        alt = f"{f_start + delta:.2f}"
        if alt not in start_strs:
            start_strs.append(alt)
    
    tags = []
    tgt = lang or (state.video_info.get('target_language') if state.video_info else None)
    if tgt:
        target_code = LANGUAGES.get(tgt, tgt)
        iso2 = _get_iso_code(target_code).lower()
        tag3 = str(target_code).lower().strip()[:3]
        lang_str = str(tgt).lower().strip()[:3]
        for t in [tag3, iso2, lang_str]:
            if t and t not in tags:
                tags.append(t)

    prefixes = ["nc_seg_", "seg_"] if never_cut else ["seg_", "nc_seg_"]

    for s_str in start_strs:
        # 1. Check with candidate language tags
        for pfx in prefixes:
            for t in tags:
                p = os.path.join(TEMP_DIR, f"{pfx}{t}_{s_str}_synced.wav")
                if os.path.exists(p):
                    return p

        # 2. Check without language tag
        for pfx in prefixes:
            p = os.path.join(TEMP_DIR, f"{pfx}{s_str}_synced.wav")
            if os.path.exists(p):
                return p

        # 3. Fallback: check any file matching this start time in TEMP_DIR
        for pfx in prefixes:
            matches = glob.glob(os.path.join(TEMP_DIR, f"{pfx}*_{s_str}_synced.wav"))
            if matches:
                return matches[0]

    return None

def _build_dubbing_df_data(text_source):
    if not state.translated_segments:
        return []
    target_lang = state.video_info.get('target_language', 'fr') if state.video_info else 'fr'
    
    rows = []
    for idx, seg in enumerate(state.translated_segments):
        text = seg.get("translated_text", "") # fitted
        if text_source == "Normal Translation":
            text = seg.get("normal_text", text)
            
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        
        audio_path = find_segment_audio_path(start, lang=target_lang)
        status = "Ready" if (audio_path and os.path.exists(audio_path)) else "Not Generated"
            
        rows.append([
            idx + 1,
            round(start, 2),
            round(end, 2),
            text,
            status
        ])
    return rows

def toggle_dubbing_text_source_warning(choice):
    if choice == "Normal Translation":
        return gr.update(value="⚠️ **Warning**: Normal translation is not adapted to segment duration. The synthesized speech may overflow the video slot, causing truncation or desync. Fitted translation is recommended.", visible=True)
    return gr.update(value="", visible=False)

def load_segment_to_editor(*args, **kwargs):
    evt = kwargs.get("evt", None)
    remaining_args = []
    for a in args:
        if hasattr(a, "index") and not isinstance(a, str):
            evt = a
        else:
            remaining_args.append(a)
    
    text_source = remaining_args[0] if len(remaining_args) > 0 else "Fitted (Dubbing-optimized)"
    voice_mode = remaining_args[1] if len(remaining_args) > 1 else "Default voice"
    voice_file = remaining_args[2] if len(remaining_args) > 2 else None
    never_cut = remaining_args[3] if len(remaining_args) > 3 else False
    default_voice_gender = remaining_args[4] if len(remaining_args) > 4 else "Male"

    if evt is None or not hasattr(evt, "index"):
        return gr.update(visible=False), 0, "", 0, 0.0, 0, 0.0, None, "Error: No selection", _get_empty_segments_html()

    if isinstance(evt.index, (list, tuple)):
        row_idx = evt.index[0]
    else:
        row_idx = int(evt.index)

    if not state.translated_segments or row_idx >= len(state.translated_segments):
        return gr.update(visible=False), 0, "", 0, 0.0, 0, 0.0, None, "Error: Index out of range", _get_empty_segments_html()
    
    seg = state.translated_segments[row_idx]
    text = seg.get("translated_text", "")
    if text_source == "Normal Translation":
        text = seg.get("normal_text", text)
    
    start_min = int(seg["start"] // 60)
    start_sec = round(seg["start"] % 60, 2)
    end_min = int(seg["end"] // 60)
    end_sec = round(seg["end"] % 60, 2)
    
    target_lang = state.video_info.get('target_language', 'fr') if state.video_info else 'fr'
    
    # Check if the audio file already exists in cache
    audio_path = find_segment_audio_path(seg['start'], lang=target_lang, never_cut=never_cut)
        
    if not audio_path or not os.path.exists(audio_path):
        audio_path = None
        status_msg = f"Selected segment #{row_idx + 1}. Audio not generated yet. Click 'Regenerate Segment Audio' to generate it."
    else:
        status_msg = f"Selected segment #{row_idx + 1}."
        
    return (
        gr.update(visible=True),
        row_idx + 1,
        text,
        start_min,
        start_sec,
        end_min,
        end_sec,
        audio_path,
        status_msg,
        _get_segments_json_html()
    )

# --- GLOBAL STATE ---
class AppState:
    def __init__(self):
        self.video_info = None
        self.segments = []  # Transcriptions
        self.translated_segments = []
        self.synced_segments = []
        self.temp_dir = TEMP_DIR
        self.keep_models = False
        self.bulk_results = {'localizations': {}, 'srts': {}}
        self.bulk_translated_segments = {}
        self.seo_package = None
        self.blog_package = None
        self.detected_shorts = []
        
state = AppState()

# --- MODULE INSTANCES (lazy loading) ---
downloader = VideoDownloader()
separator = VocalSeparator()
transcriber = Transcriber()
translator = Translator()
reformulator = Reformulator()

# TTS Backend Initialization
def load_config():
    config_path = os.path.expanduser("~/.zasttranslate/config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def save_config(config_data):
    config_path = os.path.expanduser("~/.zasttranslate/config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f)

user_config = load_config()
current_tts_backend = user_config.get("tts_backend", "VoxCPM 2")
current_llm_backend = user_config.get("llm_backend", "Qwen3.5-9B")

available_tts_backends = get_available_tts_backends()
available_llm_backends = get_available_llm_backends()

if current_tts_backend not in available_tts_backends:
    current_tts_backend = "VoxCPM 2"
if current_llm_backend not in available_llm_backends:
    current_llm_backend = "Qwen3.5-9B"

tts_engine = get_tts_backend(current_tts_backend)
# Reformulator will load the LLM internally using its backend_name
reformulator = Reformulator(backend_name=current_llm_backend)

time_sync = TimeSync(tts_engine, reformulator)
audio_mixer = AudioMixer()
video_assembler = VideoAssembler()
srt_parser = SRTParser()
youtube_publisher = YouTubePublisher(BASE_DIR)

# User CPS overrides (loaded once at startup, persisted to user_cps_config.json)
_user_cps_overrides: dict = load_user_cps()


def _build_cps_dataframe():
    """Build the CPS config DataFrame shown in the Config tab."""
    rows = []
    for lang_name, nllb_code in sorted(LANGUAGES.items()):
        iso = _get_iso_code(nllb_code).lower()
        default_cps = FITTED_CPS_BY_LANG.get(iso, FITTED_CPS_BY_LANG["_default"])
        override = _user_cps_overrides.get(iso, "")
        rows.append({
            "Language": lang_name,
            "ISO": iso,
            "Default CPS": default_cps,
            "Your CPS": "" if override == "" else override,
        })
    return pd.DataFrame(rows)


def zast_tooltip(text: str) -> str:
    """Helper to render an interactive tooltip badge."""
    clean = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    escaped = clean.replace('"', '&quot;').replace("'", "&#39;")
    attr_val = escaped.replace("\n", "&#10;")
    return f"<span class='zast-tooltip' title='{attr_val}' data-tooltip='{attr_val}'>ℹ️</span>"


# --- UI FUNCTIONS ---

def reset_project():
    """Reset all state for a new project."""
    state.video_info = None
    state.segments = []
    state.translated_segments = []
    state.synced_segments = []
    state.seo_package = None
    # Clean temp directory
    from config import TEMP_DIR, BASE_DIR
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
    # Clean Gradio temporary uploads cache to prevent disk bloat
    gradio_temp = os.path.join(BASE_DIR, "cache", "GRADIO_TEMP_DIR")
    if os.path.exists(gradio_temp):
        shutil.rmtree(gradio_temp, ignore_errors=True)
        os.makedirs(gradio_temp, exist_ok=True)
    return (
        "",           # url_input
        None,         # file_input
        "",           # local_title_input
        "Ready for a new project.",  # status_dl
        gr.update(visible=True, value=None),  # video_preview
        gr.update(visible=False, value=None), # audio_preview
        gr.update(interactive=False),  # btn_transcribe
        gr.update(interactive=False),  # btn_translate
        gr.update(interactive=False),  # btn_synth
        gr.update(interactive=False),  # btn_bulk_run
        gr.update(visible=False),      # btn_import_metadata
        gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only"]), # bulk_output_type
        gr.update(visible=True, value=None), # final_video_out
        _get_empty_segments_html(),          # segments_json_holder
        [],                                  # dubbing_segments_df
        "",                                  # original_title_input
        "",                                  # original_desc_input
        "",                                  # translated_title_input
        "",                                  # translated_desc_input
        gr.update(visible=False),            # btn_import_metadata_single
        "",                                  # seo_title_out
        "",                                  # seo_tags_out
        "",                                  # seo_chapters_out
        "",                                  # seo_desc_out
        ""                                   # seo_status
    )


def step0_check_url(url):
    """Check YouTube URL and return available resolutions."""
    if not url:
        return "Please enter a YouTube URL.", gr.update(choices=["1080p"], value="1080p"), gr.update(interactive=False)
    try:
        info = downloader.check_url(url)
        resolutions = info["resolutions"] or ["Best"]
        title = info["title"]
        duration = info["duration"]
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        status = f"✅ {title} ({minutes}:{seconds:02d})"
        # Default to 1080p if available, else first option
        default = "1080p" if "1080p" in resolutions else resolutions[0]
        return status, gr.update(choices=resolutions, value=default, interactive=True), gr.update(interactive=True)
    except Exception as e:
        return f"Error: {str(e)}", gr.update(choices=["1080p"], value="1080p"), gr.update(interactive=False)


def step1_download(url, local_file, resolution, custom_title="", progress=gr.Progress()):
    try:
        is_audio = False
        if url:
            progress(0.02, "Connecting to YouTube...")
            info = downloader.download(url, resolution=resolution, progress_callback=progress)
            show_btn = gr.update(visible=True)
        elif local_file:
            progress(0.05, "Reading uploaded file...")
            filepath = local_file.name if hasattr(local_file, 'name') else local_file
            ext = os.path.splitext(filepath)[1].lower()
            if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
                is_audio = True
            info = downloader.import_local(filepath, progress_callback=progress)
            if custom_title and str(custom_title).strip():
                info['title'] = str(custom_title).strip()
            show_btn = gr.update(visible=False)
        else:
            raise ValueError("Please provide a YouTube URL or a local file.")
        
        info['is_audio_only'] = is_audio
        state.video_info = info

        if is_audio:
            return (
                f"Audio loaded: {info['title']}",
                gr.update(visible=False, value=None),
                gr.update(visible=True, value=info['video_path']),
                gr.update(interactive=True),
                show_btn,
                gr.update(value="Audio Only", choices=["Audio Only", "Subtitles & Metadata Only"]),
                gr.update(visible=False, value=None),
                show_btn
            )
        else:
            return (
                f"Video loaded: {info['title']}",
                gr.update(visible=True, value=info['video_path']),
                gr.update(visible=False, value=None),
                gr.update(interactive=True),
                show_btn,
                gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only", "Subtitles & Metadata Only"]),
                gr.update(visible=True, value=None),
                show_btn
            )
    except Exception as e:
        return (
            f"Error: {str(e)}",
            None,
            None,
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only", "Subtitles & Metadata Only"]),
            gr.update(visible=True),
            gr.update(visible=False)
        )

def step2_transcribe(lang_source, model_size, progress=gr.Progress()):
    if not state.video_info:
        return "Error: No video loaded.", None, _get_empty_segments_html(), None
    
    progress(0.1, "Separating vocals...")
    stems = separator.separate(state.video_info['audio_44k'])
    state.video_info['vocals'] = stems['vocals']
    state.video_info['background'] = stems['background']
    if not state.keep_models:
        separator.cleanup()
    
    progress(0.4, "Transcribing with WhisperX...")
    # Map display names to WhisperX language codes
    source_lang_map = {
        "Auto": None, "French": "fr", "English": "en", "Spanish": "es",
        "German": "de", "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
        "Korean": "ko", "Chinese": "zh", "Russian": "ru", "Arabic": "ar",
        "Hindi": "hi", "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
        "Swedish": "sv", "Czech": "cs", "Romanian": "ro", "Hungarian": "hu",
    }
    lang_code = source_lang_map.get(lang_source)
    
    video_title = state.video_info.get('title', '')
    from modules.transcriber import DEFAULT_INITIAL_PROMPT
    custom_prompt = f"{video_title}. {DEFAULT_INITIAL_PROMPT}" if video_title else DEFAULT_INITIAL_PROMPT

    res = transcriber.transcribe(
        state.video_info['audio_16k'], 
        language=lang_code,
        enable_diarization=False,
        initial_prompt=custom_prompt
    )
    state.segments = res['segments']
    state.video_info['detected_language'] = res['language']
    if not state.keep_models:
        transcriber.cleanup()

    # Save source transcription for ICL voice cloning
    ref_text_parts = [seg.get("text", "").strip() for seg in state.segments if seg.get("text", "").strip()]
    ref_audio_text = " ".join(ref_text_parts)
    ref_audio_text_path = os.path.join(TEMP_DIR, "ref_audio_text.txt")
    with open(ref_audio_text_path, "w", encoding="utf-8") as _f:
        _f.write(ref_audio_text)
    print(f"Saved source transcription for voice cloning ({len(ref_audio_text)} chars)")

    # Extract a clean 5-15s segment for high-quality voice cloning
    try:
        import soundfile as sf
        best_seg = None
        max_duration = 0
        # First pass: try to find a segment between 5 and 15 seconds
        for seg in state.segments:
            duration = seg['end'] - seg['start']
            if 5 <= duration <= 15 and duration > max_duration:
                max_duration = duration
                best_seg = seg
        # Fallback: take the longest segment available
        if not best_seg and state.segments:
            best_seg = max(state.segments, key=lambda s: s['end'] - s['start'])
            
        if best_seg and state.video_info and 'vocals' in state.video_info and os.path.exists(state.video_info['vocals']):
            data, sr = sf.read(state.video_info['vocals'])
            start_sample = int(best_seg['start'] * sr)
            end_sample = int(best_seg['end'] * sr)
            extracted = data[start_sample:end_sample]
            out_path = os.path.join(TEMP_DIR, "ref_audio_extracted.wav")
            sf.write(out_path, extracted, sr)
            state.video_info['ref_audio_extracted'] = out_path
            
            out_text_path = os.path.join(TEMP_DIR, "ref_audio_extracted.txt")
            with open(out_text_path, "w", encoding="utf-8") as f:
                f.write(best_seg['text'].strip())
            state.video_info['ref_audio_text'] = best_seg['text'].strip()
            print(f"Extracted best reference audio: {best_seg['start']:.1f}s to {best_seg['end']:.1f}s")
    except Exception as e:
        print(f"Failed to extract clean reference audio: {e}")

    # Prepare dataframe for editor
    data = []
    for seg in state.segments:
        data.append([
            round(seg['start'], 2), 
            round(seg['end'], 2), 
            seg['text']
        ])
    
    # Auto-generate SRT file for instant download
    src_lang = state.video_info.get('detected_language', 'fr') if state.video_info else 'fr'
    iso = _get_iso_code(src_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"transcription_{iso}.srt")
    try:
        srt_parser.segments_to_clean_srt(state.segments, srt_path, text_key="text", lang_code=src_lang, clean_fillers=False)
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"transcription_{iso}.srt"))
    except Exception as e:
        print(f"Auto-export transcription SRT error: {e}")
        srt_path = None
    
    return f"Transcription complete ({len(data)} segments). Subtitles ready to download below. Click 'Validate Transcription' to proceed.", data, _get_segments_json_html(), srt_path

def step2b_import_srt(srt_file, lang_source):
    """Import an SRT file as transcription."""
    if srt_file is None:
        return "Error: No SRT file selected.", None, _get_empty_segments_html(), None
    
    file_path = srt_file.name if hasattr(srt_file, "name") else str(srt_file)
    try:
        segments, errors = srt_parser.convert_user_srt_to_segments(file_path)
        
        if not segments:
            return "Error: No segments found in SRT file.", None, _get_empty_segments_html(), None
        
        if state.video_info is None:
            state.video_info = {}
            
        source_lang_map = {
            "Auto": "Auto", "French": "fr", "English": "en", "Spanish": "es",
            "German": "de", "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
            "Korean": "ko", "Chinese": "zh", "Russian": "ru", "Arabic": "ar",
            "Hindi": "hi", "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
            "Swedish": "sv", "Czech": "cs", "Romanian": "ro", "Hungarian": "hu",
        }
        state.video_info['detected_language'] = source_lang_map.get(lang_source, "Auto")
        
        # Convert to internal format
        state.segments = []
        data = []
        for seg in segments:
            state.segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })
            data.append([
                round(seg['start'], 2), 
                round(seg['end'], 2), 
                seg['text']
            ])
        
        warning = ""
        if errors:
            warning = f" Warnings: {'; '.join(errors)}"
            
        # Extract a clean 5-15s segment for high-quality voice cloning
        try:
            import soundfile as sf
            best_seg = None
            max_duration = 0
            for seg in state.segments:
                duration = seg['end'] - seg['start']
                if 5 <= duration <= 15 and duration > max_duration:
                    max_duration = duration
                    best_seg = seg
            if not best_seg and state.segments:
                best_seg = max(state.segments, key=lambda s: s['end'] - s['start'])
                
            if best_seg and state.video_info and 'vocals' in state.video_info and os.path.exists(state.video_info['vocals']):
                data_sf, sr = sf.read(state.video_info['vocals'])
                start_sample = int(best_seg['start'] * sr)
                end_sample = int(best_seg['end'] * sr)
                extracted = data_sf[start_sample:end_sample]
                out_path = os.path.join(TEMP_DIR, "ref_audio_extracted.wav")
                sf.write(out_path, extracted, sr)
                state.video_info['ref_audio_extracted'] = out_path
                
                out_text_path = os.path.join(TEMP_DIR, "ref_audio_extracted.txt")
                with open(out_text_path, "w", encoding="utf-8") as f:
                    f.write(best_seg['text'].strip())
                state.video_info['ref_audio_text'] = best_seg['text'].strip()
                print(f"Extracted best reference audio: {best_seg['start']:.1f}s to {best_seg['end']:.1f}s")
        except Exception as e:
            print(f"Failed to extract clean reference audio: {e}")
        
        src_lang = state.video_info.get('detected_language', 'fr')
        iso = _get_iso_code(src_lang)
        srt_path = os.path.join(OUTPUT_DIR, f"transcription_{iso}.srt")
        try:
            srt_parser.segments_to_clean_srt(state.segments, srt_path, text_key="text", lang_code=src_lang, clean_fillers=False)
            shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"transcription_{iso}.srt"))
        except Exception:
            srt_path = None

        return f"SRT imported ({len(data)} segments).{warning} Subtitles ready below. Click 'Validate Transcription'.", data, _get_segments_json_html(), srt_path
    except Exception as e:
        return f"Error importing SRT: {str(e)}", None, _get_empty_segments_html(), None

def step2_clean_transcription(df_data, lang_source, progress=gr.Progress()):
    """Clean oral fillers and polish transcription text in the editor."""
    if df_data is not None:
        rows = _dataframe_to_rows(df_data)
        new_segments = []
        for row in rows:
            try:
                start = float(row[0])
                end = float(row[1])
                text = str(row[2])
                if text.strip():
                    new_segments.append({"start": start, "end": end, "text": text})
            except Exception:
                pass
        if new_segments:
            state.segments = new_segments

    if not state.segments:
        return "No transcription to clean.", df_data, _get_segments_json_html(), None
    
    progress(0.2, "Cleaning filler words & oral tics...")
    from modules.srt_cleaner import SRTCleaner
    cleaner = SRTCleaner()
    
    src_lang = state.video_info.get('detected_language', 'fr') if state.video_info else 'fr'
    if lang_source != "Auto" and lang_source:
        source_lang_map = {
            "French": "fr", "English": "en", "Spanish": "es", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
            "Chinese": "zh", "Russian": "ru", "Arabic": "ar", "Hindi": "hi",
            "Dutch": "nl", "Polish": "pl", "Turkish": "tr", "Swedish": "sv",
            "Czech": "cs", "Romanian": "ro", "Hungarian": "hu",
        }
        src_lang = source_lang_map.get(lang_source, src_lang)
    
    from modules.srt_cleaner import (
        remove_empty_cues_and_redistribute,
        fix_inter_cue_casing,
        apply_asr_corrections_cross_cues,
        normalize_timecodes,
    )
    state.segments = cleaner.clean_segments_heuristic(state.segments, lang_code=src_lang)
    state.segments = remove_empty_cues_and_redistribute(state.segments, text_key="text")
    state.segments = fix_inter_cue_casing(state.segments, text_key="text")
    state.segments = apply_asr_corrections_cross_cues(state.segments, text_key="text")
    state.segments = normalize_timecodes(state.segments, min_gap_ms=40, min_cue_duration_ms=400, text_key="text")
    
    # Update df data
    data = []
    for seg in state.segments:
        data.append([
            round(seg['start'], 2), 
            round(seg['end'], 2), 
            seg['text']
        ])

    # Auto-export cleaned SRT
    iso = _get_iso_code(src_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"transcription_{iso}.srt")
    try:
        srt_parser.segments_to_clean_srt(state.segments, srt_path, text_key="text", lang_code=src_lang, clean_fillers=False)
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"transcription_{iso}.srt"))
    except Exception:
        srt_path = None

    return f"Transcription cleaned ({len(data)} segments). Fillers removed and timings synced. Clean SRT ready below.", data, _get_segments_json_html(), srt_path

def step2_generate_seo_metadata(df_data, selected_pack, lang_source="Auto", progress=gr.Progress()):
    """Generate complete YouTube SEO Kit (Title, Chapters, 4 Hashtags Packs, Description, Tags)."""
    # Sync segments from dataframe if available
    if df_data is not None:
        rows = _dataframe_to_rows(df_data)
        new_segments = []
        for row in rows:
            try:
                start = float(row[0])
                end = float(row[1])
                text = str(row[2])
                if text.strip():
                    new_segments.append({"start": start, "end": end, "text": text})
            except Exception:
                pass
        if new_segments:
            state.segments = new_segments

    if not state.segments:
        return "", "", "", "", "⚠️ No transcription available. Please transcribe a video or import an SRT first."

    progress(0.2, "Searching live YouTube trends & Analyzing topics...")
    current_title = state.video_info.get("title", "") if state.video_info else ""
    
    # Determine exact source language
    source_lang_map = {
        "French": "fr", "English": "en", "Spanish": "es", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
        "Chinese": "zh", "Russian": "ru", "Arabic": "ar", "Hindi": "hi",
        "Dutch": "nl", "Polish": "pl", "Turkish": "tr", "Swedish": "sv",
        "Czech": "cs", "Romanian": "ro", "Hungarian": "hu",
    }
    src_lang = None
    if lang_source and lang_source != "Auto":
        src_lang = source_lang_map.get(lang_source)
    if not src_lang and state.video_info:
        src_lang = state.video_info.get("detected_language") or state.video_info.get("language")
    if not src_lang and state.segments:
        sample_text = " ".join(s.get("text", "") for s in state.segments[:10]).lower()
        fr_markers = [" le ", " la ", " les ", " un ", " une ", " des ", " dans ", " pour ", " avec ", " vous ", " nous ", " est ", " sont ", " c'est "]
        if any(m in sample_text for m in fr_markers):
            src_lang = "fr"
        else:
            src_lang = "en"
    if not src_lang:
        src_lang = "fr"
    
    # Load LLM backend if available for high-quality title and chapter generation
    llm_backend = getattr(reformulator, "llm", None)
    if llm_backend is None:
        try:
            reformulator.load_model()
            llm_backend = getattr(reformulator, "llm", None)
        except Exception as e:
            print(f"[SEO Assistant] LLM load note: {e}")

    progress(0.6, "Generating SEO title, timestamped chapters, description and tags...")
    pkg = seo_assistant.generate_full_seo_package(
        state.segments,
        current_title=current_title,
        source_lang=src_lang,
        llm_backend=llm_backend
    )
    state.seo_package = pkg

    # Apply selected hashtag pack if different from default
    packs = pkg.get("hashtag_packs", {})
    p1 = packs.get("Pack 1: Subject & Specific", "#Tech #IA")
    p2 = packs.get("Pack 2: Review & Unboxing", "#Tutoriel #Guide")
    p3 = packs.get("Pack 3: Collector & Tech", "#IntelligenceArtificielle #Productivite")
    p4 = packs.get("Pack 4: Community & Trends", "#Innovation #Dev #Tendance")
    
    pack_choices = [
        f"Pack 1 (Sujet & Outil) : {p1}",
        f"Pack 2 (Format & Tuto) : {p2}",
        f"Pack 3 (Tech & Écosystème) : {p3}",
        f"Pack 4 (Tendances & Dev) : {p4}"
    ]

    trends_str = ", ".join(pkg.get("trends", [])[:6]) if pkg.get("trends") else "Auto-detected"
    status_msg = f"✅ **SEO Kit & Chapters generated successfully (Humanizer Anti-AI Certified)!**\n\n*Live YouTube Trends:* `{trends_str}`"

    return (
        pkg["title"],
        pkg["tags"],
        pkg["chapters"],
        pkg["description"],
        gr.update(choices=pack_choices, value=pack_choices[0]),
        status_msg
    )

def step2_change_hashtag_pack(selected_pack, current_desc):
    """Dynamically switch the hashtag pack in the generated description."""
    if not selected_pack or not current_desc:
        return current_desc
    
    # Extract hashtags directly if present in choice string
    if ":" in selected_pack and "#" in selected_pack:
        new_hashtags = selected_pack.split(":", 1)[1].strip()
    elif state.seo_package:
        packs = state.seo_package.get("hashtag_packs", {})
        new_hashtags = packs.get(selected_pack, "")
    else:
        new_hashtags = ""
        
    if not new_hashtags:
        return current_desc
    
    lines = current_desc.strip().splitlines()
    if lines and lines[-1].startswith("#"):
        lines[-1] = new_hashtags
    else:
        lines.append("")
        lines.append(new_hashtags)
    return "\n".join(lines)

def step2_apply_seo_metadata(seo_title, seo_desc):
    """Apply generated SEO title and description to state and translation inputs."""
    if not seo_title and not seo_desc:
        return "", "", "", "", "⚠️ Please generate or enter a title and description first."
        
    if state.video_info is None:
        state.video_info = {}
    if seo_title:
        state.video_info["title"] = seo_title
    if seo_desc:
        state.video_info["description"] = seo_desc
            
    status_msg = "✅ **Title and Description applied successfully!** Ready for Translation (Tab 3) and Bulk Mode (Tab 5)."
    return seo_title, seo_desc, seo_title, seo_desc, status_msg

def _dataframe_to_rows(data):
    """Convert Gradio Dataframe output to list of lists, handling all formats."""
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        return data.values.tolist()
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    elif isinstance(data, list):
        return data
    else:
        print(f"WARNING: Unknown dataframe format: {type(data)}")
        return []

def step3_save_transcription(data):
    rows = _dataframe_to_rows(data)
    new_segments = []
    for row in rows:
        try:
            start = float(row[0])
            end = float(row[1])
            text = str(row[2])
            if text.strip():  # Skip empty text
                new_segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
        except Exception as e:
            print(f"WARNING: Skipping row {row}: {e}")
    state.segments = new_segments
    if len(new_segments) == 0:
        return "⚠️ No segments found. Make sure the transcription table has data.", gr.update(interactive=False), gr.update(interactive=False), _get_empty_segments_html(), None

    src_lang = state.video_info.get('detected_language', 'fr') if state.video_info else 'fr'
    iso = _get_iso_code(src_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"transcription_{iso}.srt")
    try:
        srt_parser.segments_to_clean_srt(state.segments, srt_path, text_key="text", lang_code=src_lang, clean_fillers=False)
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"transcription_{iso}.srt"))
    except Exception:
        srt_path = None

    return f"✅ Transcription validated ({len(new_segments)} segments). Subtitles saved to output/{os.path.basename(srt_path) if srt_path else ''}. Go to 'Translation' tab.", gr.update(interactive=True), gr.update(interactive=True), _get_segments_json_html(), srt_path

def step4_translate(target_lang, original_title="", original_desc="", progress=gr.Progress()):
    if not state.segments:
        return "Error: No transcription available.", None, _get_empty_segments_html(), "", "", None

    # Clear old synced segment audio cache when running a new translation
    import glob
    for f in glob.glob(os.path.join(TEMP_DIR, "seg_*.wav")) + glob.glob(os.path.join(TEMP_DIR, "nc_seg_*.wav")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing cached file {f}: {e}")

    progress(0, f"Translating to {target_lang}...")
    
    # Detect source language from transcription
    source_lang = state.video_info.get('detected_language', 'en') if state.video_info else 'en'
    target_lang_code = LANGUAGES.get(target_lang, target_lang)
    lang_iso = _get_iso_code(target_lang_code).lower()
    cps = get_effective_cps(lang_iso, _user_cps_overrides)
    print(f"[INFO] Using fitted_cps={cps} for target language '{lang_iso}' (user override: {lang_iso in _user_cps_overrides})")
    speed_factor = tts_engine.capabilities.get("fitted_speed_factor", MAX_SPEED_FACTOR)

    if state.video_info:
        state.video_info['target_language'] = target_lang_code

    translated_title = ""
    translated_desc = ""
    if original_title or original_desc:
        progress(0.02, "Translating metadata...")
        if original_title:
            translated_title = reformulator.translate_text(original_title, source_lang, target_lang_code)
        if original_desc:
            translated_desc = reformulator.translate_text(original_desc, source_lang, target_lang_code)
        if state.video_info:
            state.video_info['translated_title'] = translated_title
            state.video_info['translated_description'] = translated_desc
    
    # PHASE 1: LLM fitted translation — concision calibrated to TTS speaking rate
    progress(0.1, "Phase 1/3: Fitted translation (time-constrained)...")
    translated = reformulator.translate_segments(
        state.segments, source_lang, target_lang, target_lang_code,
        cps=cps, speed_factor=speed_factor
    )
    state.translated_segments = translated
    
    # PHASE 2: LLM reformulation for segments STILL too long (safety net)
    progress(0.4, "Phase 2/3: Reformulating remaining long segments...")
    reformulated_count = 0
    for seg in state.translated_segments:
        text = seg.get("translated_text", "")
        duration = seg["end"] - seg["start"]
        max_chars = int(duration * cps * speed_factor)
        
        if len(text) > max_chars * 1.1 and text.strip():
            try:
                shortened = reformulator.shorten(text, max_chars, target_lang_code)
                if shortened and len(shortened) < len(text):
                    seg["translated_text"] = shortened
                    seg["reformulated"] = True
                    reformulated_count += 1
                    print(f"  Reformulated [{seg['start']:.1f}-{seg['end']:.1f}]: {len(text)}→{len(shortened)} chars")
            except Exception as e:
                print(f"  Reformulation failed [{seg['start']:.1f}-{seg['end']:.1f}]: {e}")
    
    if reformulated_count > 0:
        print(f"Reformulated {reformulated_count} segments to fit timing")
    
    if original_title.strip():
        try:
            translated_title = reformulator.translate_text(original_title, source_lang, target_lang_code)
        except Exception:
            try:
                translated_title = translator.translate_text(original_title, source_lang, target_lang)
            except Exception as e:
                print(f"Title translation error: {e}")
                translated_title = original_title
    if original_desc.strip():
        try:
            translated_desc = reformulator.translate_text(original_desc, source_lang, target_lang_code)
        except Exception:
            try:
                translated_desc = translator.translate_text(original_desc, source_lang, target_lang)
            except Exception as e:
                print(f"Description translation error: {e}")
                translated_desc = original_desc

    state.video_info['translated_title'] = translated_title
    state.video_info['translated_description'] = translated_desc

    # Fit subtitles using reformulator (LLM or heuristic)
    progress(0.5, "Fitting translations to speech timing...")
    state.translated_segments, reformulated_count = reformulator.fit_segments(
        state.translated_segments, cps, target_lang=target_lang_code, progress=progress
    )

    # Build Dataframe with 5 columns: Start, End, Original, Translation (normal), Fitted
    data = []
    for seg in state.translated_segments:
        normal_text = seg.get("normal_text", seg.get("translated_text", ""))
        fitted = seg["translated_text"]
        
        # Show fit status on fitted column
        duration = seg["end"] - seg["start"]
        max_chars = int(duration * cps * MAX_SPEED_FACTOR)
        status = "✅" if len(fitted) <= max_chars * 1.1 else "⚠️"
        
        data.append([
            round(seg['start'], 2), 
            round(seg['end'], 2), 
            seg['text'],
            normal_text,
            f"{status} {fitted}"
        ])
    
    status_msg = f"Translation complete ({len(data)} segments"
    if reformulated_count > 0:
        status_msg += f", {reformulated_count} adapted to fit timing"
    status_msg += "). Review below — ✅ fits, ⚠️ may overflow."

    tgt_lang = state.video_info.get('target_language', 'en') if state.video_info else 'en'
    iso = _get_iso_code(tgt_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"fitted_{iso}.srt")
    try:
        srt_parser.segments_to_clean_srt(state.translated_segments, srt_path, text_key="translated_text", lang_code=tgt_lang, clean_fillers=False)
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"fitted_{iso}.srt"))
    except Exception:
        srt_path = None
        
    return status_msg, data, _get_segments_json_html(), translated_title, translated_desc, srt_path

def step5_save_translation(data, dubbing_text_source="Fitted Translation", translated_title="", translated_desc=""):
    rows = _dataframe_to_rows(data)
    count = 0
    import re
    for i, row in enumerate(rows):
        if i < len(state.translated_segments):
            # Backup original fitted
            if 'fitted_text' not in state.translated_segments[i]:
                state.translated_segments[i]['fitted_text'] = state.translated_segments[i].get('translated_text', '')
            
            # Column 3 = Translation (normal/full)
            normal_text = str(row[3]).strip()
            if normal_text:
                state.translated_segments[i]['normal_text'] = normal_text
            
            # Column 4 = Fitted (with status emoji prefix — strip it)
            text = str(row[4])
            text = re.sub(r'^[✅⚠️\ufe0f]+\s*', '', text)
            state.translated_segments[i]['fitted_text'] = text
            state.translated_segments[i]['translated_text'] = text
            count += 1
            
    if count == 0:
        return "⚠️ No translation data found.", gr.update(interactive=False), _get_empty_segments_html(), []
        
    # Set active text based on source selection
    for seg in state.translated_segments:
        if dubbing_text_source == "Normal Translation":
            seg['translated_text'] = seg.get('normal_text', seg.get('fitted_text', ''))
        else:
            seg['translated_text'] = seg.get('fitted_text', '')
            
    if state.video_info:
        state.video_info['translated_title'] = translated_title
        state.video_info['translated_description'] = translated_desc

    df_data = _build_dubbing_df_data(dubbing_text_source)
    return (
        f"✅ Translation validated ({count} segments). Go to the 'Dubbing & Export' tab.",
        gr.update(interactive=True),
        _get_segments_json_html(),
        df_data
    )

def export_transcription_srt(df_data=None):
    """Export current transcription as clean & ergonomically wrapped SRT file with source language ISO code."""
    if df_data is not None:
        rows = _dataframe_to_rows(df_data)
        new_segments = []
        for row in rows:
            try:
                start = float(row[0])
                end = float(row[1])
                text = str(row[2])
                if text.strip():
                    new_segments.append({"start": start, "end": end, "text": text})
            except Exception:
                pass
        if new_segments:
            state.segments = new_segments

    if not state.segments:
        return "No transcription to export.", None
    src_lang = state.video_info.get('detected_language', 'fr') if state.video_info else 'fr'
    iso = _get_iso_code(src_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"transcription_{iso}.srt")
    srt_parser.segments_to_clean_srt(state.segments, srt_path, text_key="text", lang_code=src_lang, clean_fillers=False)
    try:
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"transcription_{iso}.srt"))
    except Exception:
        pass
    return f"Exported {len(state.segments)} segments to output/{os.path.basename(srt_path)}.", srt_path

def export_translation_srt(df_data=None):
    """Export normal/full translation as clean & ergonomically wrapped SRT file."""
    if df_data is not None:
        try:
            step5_save_translation(df_data, dubbing_text_source="Normal Translation")
        except Exception:
            pass
    if not state.translated_segments:
        return "No translation to export.", None
    tgt_lang = state.video_info.get('target_language', 'en') if state.video_info else 'en'
    iso = _get_iso_code(tgt_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"translation_{iso}.srt")
    srt_parser.segments_to_clean_srt(state.translated_segments, srt_path, text_key="normal_text", lang_code=tgt_lang, clean_fillers=False)
    try:
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"translation_{iso}.srt"))
    except Exception:
        pass
    return f"Exported {len(state.translated_segments)} segments to output/{os.path.basename(srt_path)}.", srt_path

def export_fitted_srt(df_data=None):
    """Export fitted/concise translation as clean & ergonomically wrapped SRT file (used for dubbing)."""
    if df_data is not None:
        try:
            step5_save_translation(df_data, dubbing_text_source="Fitted Translation")
        except Exception:
            pass
    if not state.translated_segments:
        return "No translation to export.", None
    tgt_lang = state.video_info.get('target_language', 'en') if state.video_info else 'en'
    iso = _get_iso_code(tgt_lang)
    srt_path = os.path.join(OUTPUT_DIR, f"fitted_{iso}.srt")
    srt_parser.segments_to_clean_srt(state.translated_segments, srt_path, text_key="translated_text", lang_code=tgt_lang, clean_fillers=False)
    try:
        shutil.copy2(srt_path, os.path.join(TEMP_DIR, f"fitted_{iso}.srt"))
    except Exception:
        pass
    return f"Exported {len(state.translated_segments)} segments to output/{os.path.basename(srt_path)}.", srt_path

def ensure_vocals_and_reference_audio():
    if not state.video_info:
        return
    
    # Check if background separation is done and both files exist on disk
    vocals_exist = 'vocals' in state.video_info and state.video_info['vocals'] and os.path.exists(state.video_info['vocals'])
    bg_exist = 'background' in state.video_info and state.video_info['background'] and os.path.exists(state.video_info['background'])
    
    if not vocals_exist or not bg_exist:
        print("[INFO] Separating background and vocals on the fly...")
        audio_44k = state.video_info.get('audio_44k')
        
        # If audio_44k is missing or not on disk, try to re-extract from video_path
        if not audio_44k or not os.path.exists(audio_44k):
            video_path = state.video_info.get('video_path')
            if video_path and os.path.exists(video_path):
                print(f"[INFO] Re-extracting audio from video: {video_path}...")
                audio_paths = downloader.extract_audio(video_path)
                state.video_info['audio_16k'] = audio_paths['audio_16k']
                state.video_info['audio_44k'] = audio_paths['audio_44k']
                audio_44k = audio_paths['audio_44k']
            else:
                raise FileNotFoundError(
                    "Source video/audio file not found. "
                    "Cannot separate vocals. Please re-import your video or audio file."
                )
        
        # Run separation
        stems = separator.separate(audio_44k)
        state.video_info['vocals'] = stems['vocals']
        state.video_info['background'] = stems['background']
        if not state.keep_models:
            separator.cleanup()
                
    # Check if clean reference audio is extracted
    if 'ref_audio_extracted' not in state.video_info or not state.video_info['ref_audio_extracted'] or not os.path.exists(state.video_info['ref_audio_extracted']):
        try:
            import soundfile as sf
            best_seg = None
            max_duration = 0
            # Try to find a segment between 5 and 15 seconds
            segs = state.translated_segments if state.translated_segments else state.segments
            for seg in segs:
                duration = seg['end'] - seg['start']
                if 5 <= duration <= 15 and duration > max_duration:
                    max_duration = duration
                    best_seg = seg
            if not best_seg and segs:
                best_seg = max(segs, key=lambda s: s['end'] - s['start'])
                
            if best_seg and 'vocals' in state.video_info and os.path.exists(state.video_info['vocals']):
                data, sr = sf.read(state.video_info['vocals'])
                start_sample = int(best_seg['start'] * sr)
                end_sample = int(best_seg['end'] * sr)
                extracted = data[start_sample:end_sample]
                out_path = os.path.join(TEMP_DIR, "ref_audio_extracted.wav")
                sf.write(out_path, extracted, sr)
                state.video_info['ref_audio_extracted'] = out_path
                
                out_text_path = os.path.join(TEMP_DIR, "ref_audio_extracted.txt")
                with open(out_text_path, "w", encoding="utf-8") as f:
                    f.write(best_seg.get('text', best_seg.get('translated_text', '')).strip())
                state.video_info['ref_audio_text'] = best_seg.get('text', best_seg.get('translated_text', '')).strip()
                print(f"[INFO] Extracted reference audio: {best_seg['start']:.1f}s to {best_seg['end']:.1f}s")
        except Exception as e:
            print(f"[WARN] Failed to extract reference audio: {e}")

def step6_synthesize(voice_mode, voice_file, never_cut, default_voice_gender="Woman", dubbing_text_source="Fitted Translation", progress=gr.Progress()):
    if not state.translated_segments:
        return "Error: No translation available.", None, None, None, None, [], _get_empty_segments_html()
        
    try:
        # Set active text based on source selection
        for seg in state.translated_segments:
            if 'fitted_text' not in seg:
                seg['fitted_text'] = seg.get('translated_text', '')
            if dubbing_text_source == "Normal Translation":
                seg['translated_text'] = seg.get('normal_text', seg.get('fitted_text', ''))
            else:
                seg['translated_text'] = seg.get('fitted_text', '')
                
        progress(0.05, "Initializing TTS & Sync...")
        
        progress(0.05, "Preparing original vocals and background audio...")
        ensure_vocals_and_reference_audio()

        # Determine voice path based on mode
        voice_path = None
        if voice_mode == "Clone from original" and state.video_info and 'vocals' in state.video_info:
            if 'ref_audio_extracted' in state.video_info and os.path.exists(state.video_info['ref_audio_extracted']):
                voice_path = state.video_info['ref_audio_extracted']
                print(f"Using cleanly extracted original vocals for cloning: {voice_path}")
            else:
                voice_path = state.video_info['vocals']
                print(f"Using original vocals for cloning: {voice_path}")
        elif voice_mode == "Clone from file" and voice_file:
            voice_path = voice_file
            print(f"Using uploaded voice file for cloning: {voice_path}")
        else:
            print("Using default TTS voice (no cloning).")
        
        # Detect target language from translated segments
        target_lang = state.video_info.get('target_language', 'fr') if state.video_info else 'fr'
        
        # Load models
        tts_engine.load(ref_audio_path=voice_path)
        
        if never_cut:
            # ---- NEVER CUT VOCAL MODE ----
            print(NEVER_CUT_WARNING)
            
            progress(0.2, "[Never Cut] Generating all audio at natural speed...")
            synced, drift_info = time_sync.sync_all_never_cut(
                state.translated_segments, target_lang,
                state.video_info['duration'], voice_mapping=None, gender=default_voice_gender
            )
            state.synced_segments = synced
            if not state.keep_models:
                tts_engine.cleanup()
            
            # Assembly with real (cascade) positions
            progress(0.8, "[Never Cut] Assembling with cascade placement...")
            full_audio = time_sync.build_full_audio(
                state.synced_segments,
                state.video_info['duration'],
                use_real_positions=True
            )
            
            # Build status with drift info
            drift_msg = f"max drift: {drift_info['max_drift']}s"
            if drift_info['final_overflow'] > 0:
                drift_msg += f", overflow: {drift_info['final_overflow']}s"
            status = f"Done! {len(synced)} segments — {drift_msg} (Never Cut mode)"
        else:
            # ---- NORMAL MODE (unchanged) ----
            # PHASE 1: Pre-check timing and shorten long segments BEFORE TTS
            progress(0.15, "Pre-checking segment timing...")
            state.translated_segments, shortened_count = time_sync.pre_check_and_shorten(
                state.translated_segments, target_lang
            )
            
            if shortened_count > 0:
                print(f"Pre-shortened {shortened_count} segments to fit timing.")
            
            # PHASE 2: TTS + sync with speed control for minor overflows
            progress(0.3, f"Generating audio ({len(state.translated_segments)} segments)...")
            state.synced_segments, sync_stats = time_sync.sync_all(
                state.translated_segments, target_lang, voice_mapping=None,
                total_duration=state.video_info.get('duration'), gender=default_voice_gender
            )
            
            if not state.keep_models:
                tts_engine.cleanup()
            
            # Assembly
            progress(0.8, "Mixing audio...")
            full_audio = time_sync.build_full_audio(
                state.synced_segments, 
                state.video_info['duration']
            )
            
            # Build detailed status
            parts = [f"Done! {sync_stats['total']} segments"]
            if sync_stats['perfect'] == sync_stats['total']:
                parts.append("— ✅ all fit perfectly")
            else:
                if sync_stats['perfect'] > 0:
                    parts.append(f"— ✅ {sync_stats['perfect']} perfect")
                if sync_stats['sped_up'] > 0:
                    parts.append(f"⚡ {sync_stats['sped_up']} sped up")
                if sync_stats['cut_total'] > 0:
                    parts.append(f"✂️ {sync_stats['cut_total']} cut")
            if sync_stats.get("reformulated", 0) > 0:
                parts.append(f"✍️ {sync_stats['reformulated']} segment(s) reformulé(s)")
            status = " ".join(parts)
        
        # Common: mix voice + background and assemble video
        mixed_audio = os.path.join(TEMP_DIR, "final_mix.wav")
        audio_mixer.mix(full_audio, state.video_info['background'], mixed_audio)
        
        tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
        iso = _get_iso_code(tgt_lang)
        
        # Copy final audio to output dir for easy export
        final_audio_export = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
        shutil.copy2(mixed_audio, final_audio_export)
        
        if state.video_info.get('is_audio_only', False):
            df_data = _build_dubbing_df_data(dubbing_text_source)
            return status, gr.update(visible=False, value=None), mixed_audio, gr.update(visible=False, value=None), gr.update(visible=True, value=mixed_audio), df_data, _get_segments_json_html(), None, final_audio_export
        else:
            progress(0.9, "Assembling final video...")
            final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
            video_assembler.assemble(
                state.video_info['video_path'], 
                mixed_audio, 
                final_video
            )
            df_data = _build_dubbing_df_data(dubbing_text_source)
            return status, gr.update(visible=True, value=final_video), mixed_audio, gr.update(visible=True, value=final_video), gr.update(visible=False, value=None), df_data, _get_segments_json_html(), final_video, final_audio_export
    except Exception as e:
        import traceback
        traceback.print_exc()
        if not state.keep_models:
            tts_engine.cleanup()
        return f"❌ Synthesis failed: {str(e)}", gr.update(), None, gr.update(), gr.update(), gr.update(), gr.update(), None, None


def export_video():
    """Export the assembled dubbed video as a downloadable file."""
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    video_path = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        return f"🎬 Dubbed video ready ({size_mb:.1f} MB).", video_path
    vids = glob.glob(os.path.join(OUTPUT_DIR, "final_video_*.mp4"))
    if vids:
        latest_vid = max(vids, key=os.path.getmtime)
        size_mb = os.path.getsize(latest_vid) / 1024 / 1024
        return f"🎬 Dubbed video ready ({size_mb:.1f} MB).", latest_vid
    return "❌ No dubbed video found. Run 'Assemble Final Video & Audio' first.", None

def export_audio():
    """Export the mixed audio as a downloadable file."""
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    audio_path = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
    if os.path.exists(audio_path):
        size_mb = os.path.getsize(audio_path) / 1024 / 1024
        return f"🎵 Dubbed audio ready ({size_mb:.1f} MB).", audio_path
    auds = glob.glob(os.path.join(OUTPUT_DIR, "final_audio_*.wav"))
    if auds:
        latest_aud = max(auds, key=os.path.getmtime)
        size_mb = os.path.getsize(latest_aud) / 1024 / 1024
        return f"🎵 Dubbed audio ready ({size_mb:.1f} MB).", latest_aud
    return "❌ No audio available. Run Dubbing first.", None

def step6_regenerate_segment(
    row_idx_1based, text, start_min, start_sec, end_min, end_sec,
    voice_mode, voice_file, never_cut, default_voice_gender, dubbing_text_source
):
    if not state.translated_segments:
        return None, "Error: No segments found.", [], _get_empty_segments_html(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    idx = int(row_idx_1based) - 1
    if idx < 0 or idx >= len(state.translated_segments):
        return None, f"Error: Segment index {idx+1} out of bounds.", [], _get_empty_segments_html(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    seg = state.translated_segments[idx]
    old_start = seg["start"]
    
    # Calculate new start/end times
    new_start = float(start_min * 60 + start_sec)
    new_end = float(end_min * 60 + end_sec)
    
    # Update state values
    seg["start"] = new_start
    seg["end"] = new_end
    
    # Update text to speak based on dubbing text source selection
    if dubbing_text_source == "Normal Translation":
        seg["normal_text"] = text
    else:
        seg["translated_text"] = text
        seg["fitted_text"] = text
        
    # Update transcription state start/end
    if idx < len(state.segments):
        state.segments[idx]["start"] = new_start
        state.segments[idx]["end"] = new_end
        
    target_lang = state.video_info.get('target_language', 'fr') if state.video_info else 'fr'
    target_lang_code = LANGUAGES.get(target_lang, target_lang)
    
    # Delete old cache files if timing changed across all possible language tags
    if abs(old_start - new_start) > 0.01:
        for f in glob.glob(os.path.join(TEMP_DIR, f"*seg_*_{old_start:.2f}*.wav")) + glob.glob(os.path.join(TEMP_DIR, f"*seg_{old_start:.2f}*.wav")):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing old cache file {f}: {e}")

    voice_path = None
    if voice_mode == "Clone from original" and state.video_info:
        # Check if ref_audio_extracted is missing, if so generate it on the fly
        if 'ref_audio_extracted' not in state.video_info or not os.path.exists(state.video_info['ref_audio_extracted']):
            ensure_vocals_and_reference_audio()
            
        if 'ref_audio_extracted' in state.video_info and os.path.exists(state.video_info['ref_audio_extracted']):
            voice_path = state.video_info['ref_audio_extracted']
        elif 'vocals' in state.video_info and os.path.exists(state.video_info['vocals']):
            voice_path = state.video_info['vocals']
    elif voice_mode == "Clone from file" and voice_file:
        voice_path = voice_file.name if hasattr(voice_file, 'name') else voice_file

    print(f"[REGEN] row_idx={row_idx_1based}, voice_mode='{voice_mode}', voice_path='{voice_path}'")
    tts_engine.load(ref_audio_path=voice_path)
    
    try:
        time_sync._compute_effective_durations(state.translated_segments, total_duration=state.video_info.get('duration'))
        
        # Override temporary file name check to bypass cache for this specific regeneration
        for f in glob.glob(os.path.join(TEMP_DIR, f"*seg_*_{new_start:.2f}*.wav")) + glob.glob(os.path.join(TEMP_DIR, f"*seg_{new_start:.2f}*.wav")):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing temp cache file to force regen: {e}")
                
        res = time_sync.sync_segment(seg, target_lang_code, voice_path=voice_path, gender=default_voice_gender)
        
        seg_audio_path = res.get("synced_path")
        if not seg_audio_path or not os.path.exists(seg_audio_path):
            raise FileNotFoundError("Synthesized audio file not found.")
            
        if never_cut:
            tag = str(target_lang_code).lower().strip()[:3]
            nc_synced_path = os.path.join(TEMP_DIR, f"nc_seg_{tag}_{new_start:.2f}_synced.wav")
            shutil.copy2(seg_audio_path, nc_synced_path)
            seg_audio_path = nc_synced_path
            
        if state.synced_segments and idx < len(state.synced_segments):
            res["synced_path"] = seg_audio_path
            state.synced_segments[idx] = res
            
        status_msg = f"✅ Segment #{idx+1} regenerated. Duration: {res['final_duration']:.2f}s (slot: {res['slot_duration']:.2f}s)."
        if res.get("reformulated"):
            status_msg += f" ⚠️ **Texte raccourci pour s'adapter au timing** : \"{res['final_text']}\""
        
        if not state.keep_models:
            tts_engine.cleanup()
            reformulator.cleanup()
            
        df_data = _build_dubbing_df_data(dubbing_text_source)
        
        # Default updates if assembly is not ready
        video_update = gr.update()
        audio_update = gr.update()
        video_out_update = gr.update()
        audio_out_update = gr.update()
        export_video_update = gr.update()
        export_audio_update = gr.update()
        
        # If assembly was already run, re-assemble in background
        if state.synced_segments and idx < len(state.synced_segments):
            try:
                print("[REGEN] Re-assembling mixed audio and video in background...")
                full_audio = time_sync.build_full_audio(
                    state.synced_segments,
                    state.video_info['duration'],
                    use_real_positions=never_cut
                )
                mixed_audio = os.path.join(TEMP_DIR, "final_mix.wav")
                audio_mixer.mix(full_audio, state.video_info['background'], mixed_audio)
                
                tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
                iso = _get_iso_code(tgt_lang)
                
                final_audio_export = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
                shutil.copy2(mixed_audio, final_audio_export)
                
                if not state.video_info.get('is_audio_only', False):
                    final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
                    video_assembler.assemble(
                        state.video_info['video_path'], 
                        mixed_audio, 
                        final_video
                    )
                    video_update = gr.update(value=final_video)
                    audio_update = gr.update(visible=False, value=None)
                    video_out_update = gr.update(value=final_video)
                    audio_out_update = gr.update(visible=True, value=mixed_audio)
                    export_video_update = gr.update(value=final_video)
                    export_audio_update = gr.update(value=final_audio_export)
                else:
                    video_update = gr.update(visible=False, value=None)
                    audio_update = gr.update(value=mixed_audio)
                    video_out_update = gr.update(visible=False, value=None)
                    audio_out_update = gr.update(visible=True, value=mixed_audio)
                    export_audio_update = gr.update(value=final_audio_export)
            except Exception as assemble_err:
                print(f"[ERROR] Background assembly failed: {assemble_err}")
                
        return seg_audio_path, status_msg, df_data, _get_segments_json_html(), video_update, audio_update, video_out_update, audio_out_update, export_video_update, export_audio_update
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if not state.keep_models:
            tts_engine.cleanup()
            reformulator.cleanup()
        return None, f"❌ Synthesis failed: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def step5_bulk_run(target_langs, voice_mode, voice_file, never_cut, output_type, bulk_title, bulk_desc, default_voice_gender="Woman", generate_shorts=False, progress=gr.Progress()):
    if not state.segments:
        yield "Error: No transcription available.", None
        return
        
    if not target_langs:
        yield "Error: Please select at least one target language.", None, ""
        return

    output_files = []
    metadata_display = ""
    total_langs = len(target_langs)
    state.bulk_results = {'localizations': {}, 'srts': {}}
    if not hasattr(state, "bulk_translated_segments") or not isinstance(state.bulk_translated_segments, dict):
        state.bulk_translated_segments = {}
    
    # Detect source language from transcription
    source_lang = state.video_info.get('detected_language', 'en') if state.video_info else 'en'
    
    # Clear old synced segment audio cache when running a bulk process
    import glob
    for f in glob.glob(os.path.join(TEMP_DIR, "seg_*.wav")) + glob.glob(os.path.join(TEMP_DIR, "nc_seg_*.wav")):
        try:
            os.remove(f)
        except Exception:
            pass

    # Ensure vocals and background audio are prepared and present on disk
    if state.video_info and output_type != "Subtitles & Metadata Only":
        yield "Ensuring background and vocals are prepared...", output_files, metadata_display
        try:
            ensure_vocals_and_reference_audio()
        except Exception as prep_err:
            yield f"Error preparing audio: {prep_err}", output_files, metadata_display
            return
    
    # Phase 1: Translate ALL languages first to prevent VRAM fragmentation
    all_translated_segments = {}
    for idx, target_lang in enumerate(target_langs):
        # Progress math setup
        if output_type == "Subtitles & Metadata Only":
            base_progress = (idx / total_langs)
            prog_step = 1.0 / total_langs
        else:
            base_progress = (idx / total_langs) * 0.5
            prog_step = 0.5 / total_langs
        
        target_lang_code = LANGUAGES.get(target_lang, target_lang)
        iso = _get_iso_code(target_lang_code)
        lang_iso = iso.lower()
        cps = get_effective_cps(lang_iso, _user_cps_overrides)
        print(f"[INFO] Using fitted_cps={cps} for target language '{lang_iso}' (user override: {lang_iso in _user_cps_overrides})")
        speed_factor = tts_engine.capabilities.get("fitted_speed_factor", MAX_SPEED_FACTOR)

        if state.video_info:
            state.video_info['target_language'] = target_lang_code
            
        yield f"[{idx+1}/{total_langs}] TRANSLATION PHASE: {target_lang}...", output_files, ""
        
        # Translate Title and Description if provided
        translated_title = ""
        translated_desc = ""
        if bulk_title or bulk_desc:
            progress(base_progress + prog_step * 0.1, f"[{target_lang}] Translating metadata...")
            if bulk_title:
                translated_title = reformulator.translate_text(bulk_title, source_lang, target_lang_code)
            if bulk_desc:
                translated_desc = reformulator.translate_text(bulk_desc, source_lang, target_lang_code)
            
            meta_text = f"### {target_lang} Metadata\n\n**Title:**\n```text\n{translated_title}\n```\n\n**Description:**\n```text\n{translated_desc}\n```\n\n---\n"
            metadata_display += meta_text
            
            # Store for YouTube Publishing
            state.bulk_results['localizations'][target_lang_code] = {
                'title': translated_title,
                'description': translated_desc
            }
            
            # Save metadata to disk immediately so they are never lost on reload
            meta_md_path = os.path.join(OUTPUT_DIR, "metadata_translations.md")
            meta_json_path = os.path.join(OUTPUT_DIR, "metadata_translations.json")
            try:
                with open(meta_md_path, "w", encoding="utf-8") as _fmd:
                    _fmd.write(metadata_display)
                with open(meta_json_path, "w", encoding="utf-8") as _fjson:
                    import json as _json
                    _json.dump(state.bulk_results['localizations'], _fjson, ensure_ascii=False, indent=2)
            except Exception as _meta_err:
                print(f"Error saving metadata file: {_meta_err}")
            
            yield f"[{idx+1}/{total_langs}] TRANSLATION PHASE: {target_lang}...", output_files, metadata_display
        
        progress(base_progress + prog_step * 0.2, f"[{target_lang}] Fitted translation...")
        translated = reformulator.translate_segments(
            state.segments, source_lang, target_lang, target_lang_code,
            cps=cps, speed_factor=speed_factor
        )
        
        progress(base_progress + prog_step * 0.5, f"[{target_lang}] Reformulating long segments...")
        for seg in translated:
            text = seg.get("translated_text", "")
            duration = seg["end"] - seg["start"]
            max_chars = int(duration * cps * speed_factor)
            
            if len(text) > max_chars * 1.1 and text.strip():
                try:
                    shortened = reformulator.shorten(text, max_chars, target_lang_code)
                    if shortened and len(shortened) < len(text):
                        seg["translated_text"] = shortened
                        seg["reformulated"] = True
                except Exception as e:
                    print(f"Reformulation failed for {target_lang}: {e}")
                    
        progress(base_progress + prog_step * 0.8, f"[{target_lang}] Natural full translation...")
        reformulator.translate_normal(translated, source_lang, target_lang_code)
        
        # Save SRTs
        trans_srt = os.path.join(TEMP_DIR, f"translation_{iso}.srt")
        srt_parser.segments_to_clean_srt(translated, trans_srt, text_key="normal_text", lang_code=target_lang_code, clean_fillers=False)
        output_files.append(trans_srt)
        
        fitted_srt = os.path.join(TEMP_DIR, f"fitted_{iso}.srt")
        srt_parser.segments_to_clean_srt(translated, fitted_srt, text_key="translated_text", lang_code=target_lang_code, clean_fillers=False)
        output_files.append(fitted_srt)
        
        # Store SRT path for YouTube Publishing (We prefer natural translation for subtitles)
        state.bulk_results['srts'][target_lang_code] = trans_srt

        # Store for the synthesis phase and for Tab 6 Shorts
        import copy
        all_translated_segments[target_lang] = copy.deepcopy(translated)
        state.bulk_translated_segments[target_lang] = copy.deepcopy(translated)
        state.bulk_translated_segments[target_lang_code] = copy.deepcopy(translated)
        state.translated_segments = copy.deepcopy(translated)

    # CRITICAL: Clean up LLM from VRAM completely before loading TTS
    if not state.keep_models:
        reformulator.cleanup()
        
    # If Subtitles & Metadata Only, skip TTS and finish immediately
    if output_type == "Subtitles & Metadata Only":
        meta_md_path = os.path.join(OUTPUT_DIR, "metadata_translations.md")
        if os.path.exists(meta_md_path) and meta_md_path not in output_files:
            output_files.append(meta_md_path)
        meta_json_path = os.path.join(OUTPUT_DIR, "metadata_translations.json")
        if os.path.exists(meta_json_path) and meta_json_path not in output_files:
            output_files.append(meta_json_path)

        import zipfile
        zip_path = os.path.join(OUTPUT_DIR, "bulk_export_all.zip")
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for f in output_files:
                    if os.path.exists(f):
                        zipf.write(f, os.path.basename(f))
            output_files.insert(0, zip_path)
        except Exception as e:
            print(f"Failed to create ZIP: {e}")

        yield f"Completed! Generated subtitles & metadata for {total_langs} languages.", output_files, metadata_display
        return
    
    # Phase 2: Synthesize ALL languages
    # Determine voice path ONCE before loading model
    initial_voice_path = None
    if voice_mode == "Clone from original" and state.video_info and 'vocals' in state.video_info:
        if 'ref_audio_extracted' in state.video_info and os.path.exists(state.video_info['ref_audio_extracted']):
            initial_voice_path = state.video_info['ref_audio_extracted']
        else:
            initial_voice_path = state.video_info['vocals']
    elif voice_mode == "Clone from file" and voice_file:
        initial_voice_path = voice_file

    tts_engine.load(ref_audio_path=initial_voice_path) # Load once with correct model type
    
    for idx, target_lang in enumerate(target_langs):
        base_progress = 0.5 + (idx / total_langs) * 0.5
        prog_step = 0.5 / total_langs

        target_lang_code = LANGUAGES.get(target_lang, target_lang)
        iso = _get_iso_code(target_lang_code)

        yield f"[{idx+1}/{total_langs}] SYNTHESIS PHASE: {target_lang}...", output_files, metadata_display
        progress(base_progress + prog_step * 0.2, f"[{target_lang}] Initializing TTS...")

        try:
            translated_for_lang = all_translated_segments[target_lang]

            if never_cut:
                progress(base_progress + prog_step * 0.4, f"[{target_lang}] Audio sync (Never Cut)...")
                synced, _ = time_sync.sync_all_never_cut(
                    translated_for_lang, iso, state.video_info['duration'], voice_mapping=None, gender=default_voice_gender
                )
                
                progress(base_progress + prog_step * 0.7, f"[{target_lang}] Assembling audio...")
                full_audio = time_sync.build_full_audio(
                    synced,
                    state.video_info['duration'],
                    use_real_positions=True
                )
            else:
                progress(base_progress + prog_step * 0.3, f"[{target_lang}] Pre-checking timing...")
                translated_for_lang, _ = time_sync.pre_check_and_shorten(
                    translated_for_lang, target_lang_code
                )
                
                progress(base_progress + prog_step * 0.5, f"[{target_lang}] Generating audio...")
                synced, _ = time_sync.sync_all(
                    translated_for_lang, iso, voice_mapping=None,
                    total_duration=state.video_info.get('duration'), gender=default_voice_gender
                )
                
                progress(base_progress + prog_step * 0.7, f"[{target_lang}] Assembling audio...")
                full_audio = time_sync.build_full_audio(
                    synced, 
                    state.video_info['duration']
                )

            mixed_audio = os.path.join(TEMP_DIR, f"mix_{iso}.wav")
            audio_mixer.mix(full_audio, state.video_info['background'], mixed_audio)
            
            final_audio_export = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
            shutil.copy2(mixed_audio, final_audio_export)
            output_files.append(final_audio_export)

            # --- VIDEO ASSEMBLY PHASE ---
            if output_type == "Video + Audio" and not state.video_info.get('is_audio_only', False):
                yield f"[{idx+1}/{total_langs}] Assembling Video {target_lang}...", output_files, metadata_display
                progress(base_progress + prog_step * 0.9, f"[{target_lang}] Mixed video assembly...")
                
                final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
                video_assembler.assemble(
                    state.video_info['video_path'], 
                    mixed_audio, 
                    final_video
                )
                output_files.append(final_video)

                # Automatic 3 Viral Shorts (9:16) generation if checked
                if generate_shorts and os.path.exists(final_video):
                    yield f"[{idx+1}/{total_langs}] 📱 Auto-generating 3 Viral Shorts (9:16) for {target_lang}...", output_files, metadata_display
                    progress(base_progress + prog_step * 0.95, f"[{target_lang}] Rendering vertical 9:16 shorts...")
                    try:
                        shorts_clips = shorts_studio.detect_viral_moments(
                            translated_for_lang,
                            llm_backend=None,
                            num_shorts=3,
                            source_lang=source_lang,
                            text_key="translated_text",
                            target_lang=target_lang_code
                        )
                        shorts_refined = shorts_studio.refine_boundaries(final_video, shorts_clips, translated_for_lang, text_key="translated_text")
                        for s_idx, s_clip in enumerate(shorts_refined, 1):
                            safe_t = re.sub(r'[^a-zA-Z0-9_-]', '_', s_clip.get("title", f"short_{s_idx}"))[:20]
                            short_out_name = f"short_{iso}_{s_idx}_{safe_t}.mp4"
                            short_out_path = os.path.join(OUTPUT_DIR, short_out_name)
                            ass_sub_path = os.path.join(TEMP_DIR, f"sub_bulk_{iso}_{s_idx}.ass")
                            
                            shorts_studio.generate_ass_subtitles(
                                translated_for_lang,
                                s_clip["start"],
                                s_clip["end"],
                                ass_sub_path,
                                style="tiktok_yellow",
                                text_key="translated_text"
                            )
                            shorts_studio.render_vertical_short(
                                video_path=final_video,
                                audio_path=None,
                                start_time=s_clip["start"],
                                end_time=s_clip["end"],
                                output_path=short_out_path,
                                crop_mode="blur_stack",
                                ass_subtitle_path=ass_sub_path
                            )
                            if os.path.exists(short_out_path):
                                output_files.append(short_out_path)
                    except Exception as s_err:
                        print(f"[SHORTS] Auto-generate bulk shorts failed for {target_lang}: {s_err}")
        except Exception as lang_err:
            print(f"[ERROR] Failed synthesis for {target_lang}: {lang_err}")
            yield f"⚠️ [{target_lang}] Synthesis failed ({lang_err}), continuing...", output_files, metadata_display
            
    if not state.keep_models:
        tts_engine.cleanup()
    
    meta_md_path = os.path.join(OUTPUT_DIR, "metadata_translations.md")
    if os.path.exists(meta_md_path) and meta_md_path not in output_files:
        output_files.append(meta_md_path)

    # Create a ZIP archive of all outputs for easy download
    import zipfile
    zip_path = os.path.join(OUTPUT_DIR, "bulk_export_all.zip")
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in output_files:
                if os.path.exists(f):
                    zipf.write(f, os.path.basename(f))
        output_files.insert(0, zip_path)
    except Exception as e:
        print(f"Failed to create ZIP: {e}")

    yield f"Completed! Processed {total_langs} languages.", output_files, metadata_display



def open_output_folder():
    """Open the output folder in the OS file explorer."""
    if os.path.exists(OUTPUT_DIR):
        if os.name == 'nt':
            os.startfile(OUTPUT_DIR)
        else:
            import subprocess
            subprocess.Popen(['xdg-open' if sys.platform.startswith('linux') else 'open', OUTPUT_DIR])

def step6_publish_youtube(progress=gr.Progress()):
    if not state.video_info or not state.video_info.get('youtube_id'):
        return "Error: No YouTube video ID found. Please import from a YouTube URL first."
    
    if not youtube_publisher.is_configured():
        return "Error: client_secret.json is missing. Please place it in the application folder."
        
    try:
        progress(0, "Authenticating with YouTube API...")
        youtube_publisher.authenticate()
        
        video_id = state.video_info['youtube_id']
        localizations = state.bulk_results.get('localizations', {})
        srts = state.bulk_results.get('srts', {})
        
        if localizations:
            progress(0.3, "Updating video metadata (Title & Description)...")
            youtube_publisher.update_metadata(video_id, localizations)
            
        progress_per_lang = 0.6 / max(1, len(srts))
        for idx, (lang_code, srt_path) in enumerate(srts.items()):
            progress(0.4 + idx * progress_per_lang, f"Uploading {lang_code} subtitles...")
            if os.path.exists(srt_path):
                # Name the caption track after the language code
                youtube_publisher.upload_caption(video_id, lang_code, f"{lang_code} Subtitles", srt_path)
                
        return "✅ Published Metadata and Subtitles successfully to YouTube!"
    except Exception as e:
        return f"❌ YouTube API Error: {str(e)}"


# --- VIRAL SHORTS STUDIO HANDLERS ---

def get_available_dubbed_languages():
    """Detect all available dubbed languages from state.bulk_translated_segments, state.video_info, or output folder."""
    langs = []
    # 1. From state.bulk_translated_segments
    if hasattr(state, "bulk_translated_segments") and state.bulk_translated_segments:
        for l in state.bulk_translated_segments.keys():
            disp_name = l
            for name, code in LANGUAGES.items():
                if code == l or name == l:
                    disp_name = name
                    break
            if disp_name not in langs:
                langs.append(disp_name)
    
    # 2. From state.video_info target_language
    if state.video_info and state.video_info.get("target_language"):
        t_lang = state.video_info.get("target_language")
        disp_name = t_lang
        for name, code in LANGUAGES.items():
            if code == t_lang or name == t_lang:
                disp_name = name
                break
        if disp_name not in langs:
            langs.append(disp_name)

    # 3. From output/final_video_*.mp4
    if os.path.exists(OUTPUT_DIR):
        for f in glob.glob(os.path.join(OUTPUT_DIR, "final_video_*.mp4")):
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split("_")
            if len(parts) >= 3:
                iso = parts[2].lower()
                for name, code in LANGUAGES.items():
                    if _get_iso_code(code).lower() == iso and name not in langs:
                        langs.append(name)
                        break

    return sorted(langs) if langs else ["English"]


def _resolve_dubbed_segments_and_video(dubbed_lang=None):
    """
    Resolve (active_segments, target_lang_code, video_path) for the chosen dubbed language.
    Works seamlessly for single dubbing (Tab 4) and bulk dubbing (Tab 5).
    """
    target_lang_code = "en"
    if dubbed_lang:
        target_lang_code = LANGUAGES.get(dubbed_lang, dubbed_lang)
    elif state.video_info and state.video_info.get("target_language"):
        target_lang_code = state.video_info.get("target_language")

    iso = _get_iso_code(target_lang_code).lower()

    # 1. Try finding segments in state.bulk_translated_segments
    active_segments = None
    if hasattr(state, "bulk_translated_segments") and state.bulk_translated_segments:
        active_segments = (
            state.bulk_translated_segments.get(dubbed_lang) or
            state.bulk_translated_segments.get(target_lang_code)
        )
        if not active_segments:
            for k, v in state.bulk_translated_segments.items():
                if _get_iso_code(k).lower() == iso:
                    active_segments = v
                    break

    # 2. Fallback to state.translated_segments
    if not active_segments and state.translated_segments:
        active_segments = state.translated_segments

    # 3. Fallback to parsing exported SRT files from output/
    if not active_segments and os.path.exists(OUTPUT_DIR):
        candidates = [
            os.path.join(OUTPUT_DIR, f"subtitles_fitted_{iso}.srt"),
            os.path.join(OUTPUT_DIR, f"subtitles_{iso}.srt"),
            os.path.join(OUTPUT_DIR, f"subtitles_fitted_{target_lang_code}.srt")
        ]
        for c_path in candidates:
            if os.path.exists(c_path):
                try:
                    parsed = srt_parser.parse_srt(c_path)
                    if parsed:
                        for p in parsed:
                            if not p.get("translated_text"):
                                p["translated_text"] = p.get("text", "")
                        active_segments = parsed
                        break
                except Exception as e:
                    print(f"[SHORTS] Failed to parse SRT fallback {c_path}: {e}")

    # 4. Resolve final dubbed video path
    video_path = None
    final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
    if os.path.exists(final_video):
        video_path = final_video
    else:
        for f in glob.glob(os.path.join(OUTPUT_DIR, f"final_video_{iso}*.mp4")):
            video_path = f
            break
        if not video_path:
            vids = glob.glob(os.path.join(OUTPUT_DIR, "final_video_*.mp4"))
            if vids:
                vids.sort(key=os.path.getmtime, reverse=True)
                video_path = vids[0]

    return active_segments, target_lang_code, video_path


def step7_detect_shorts(source_choice, crop_style, subtitle_style, num_shorts, dubbed_lang=None, progress=gr.Progress()):
    empty_cards = []
    for _ in range(5):
        empty_cards.extend([
            gr.update(visible=False), # card
            gr.update(value=False),   # include
            gr.update(value=""),      # title
            gr.update(value=0.0),     # start
            gr.update(value=0.0),     # end
            gr.update(value="")       # subtitles
        ])

    if not state.segments:
        return (
            "⚠️ **Prerequisite missing:** No transcription found. Please import a video in **Tab 1** and run Transcription in **Tab 2** (or import an SRT file) first.",
            [],
            gr.update(interactive=False),
            *empty_cards
        )

    num_shorts = max(1, min(5, int(num_shorts)))
    progress(0.2, f"Semantic analysis of viral moments (top {num_shorts})...")
    video_path = state.video_info.get("video_path") if state.video_info else None
    
    # Load LLM backend if available
    llm_backend = getattr(reformulator, "llm", None)
    if llm_backend is None:
        try:
            reformulator.load_model()
            llm_backend = getattr(reformulator, "llm", None)
        except Exception as e:
            print(f"[SHORTS] Failed to load LLM: {e}")

    # Determine source segments
    active_segments = state.segments
    text_key = "text"
    target_lang = state.video_info.get("target_language", "en") if state.video_info else "en"
    source_lang = state.video_info.get("source_language", "fr") if state.video_info else "fr"

    if "Dubbed" in source_choice:
        active_segments, target_lang, resolved_video = _resolve_dubbed_segments_and_video(dubbed_lang)
        if not active_segments:
            return (
                f"⚠️ **Dubbed Video selected, but no translated segments found for '{dubbed_lang or 'selected language'}'.** Please run Dubbing in Tab 4 or Bulk in Tab 5 first, or choose 'Original Video'.",
                [],
                gr.update(interactive=False),
                *empty_cards
            )
        text_key = "translated_text"
        if resolved_video:
            video_path = resolved_video

    progress(0.4, f"Detecting top {num_shorts} standalone viral moments...")
    raw_clips = shorts_studio.detect_viral_moments(
        active_segments,
        llm_backend=llm_backend,
        num_shorts=num_shorts,
        source_lang=source_lang,
        text_key=text_key,
        target_lang=target_lang if "Dubbed" in source_choice else None
    )

    progress(0.7, "Snapping to camera cuts (PySceneDetect) & speech boundaries...")
    refined_clips = shorts_studio.refine_boundaries(video_path, raw_clips, active_segments, text_key=text_key)
    
    # Ensure subtitles text is extracted for each clip with the requested language text_key
    for c in refined_clips:
        c["subtitles"] = shorts_studio.extract_clip_text(active_segments, c["start"], c["end"], text_key=text_key)

    state.detected_shorts = refined_clips

    table_data = []
    for i, c in enumerate(refined_clips, 1):
        table_data.append([
            f"#{i}",
            c["title"],
            round(c["start"], 2),
            round(c["end"], 2),
            f"{c['duration']:.1f}s",
            f"🔥 {c['score']}%",
            c.get("reason", "")
        ])

    card_updates = []
    for i in range(5):
        if i < len(refined_clips):
            c = refined_clips[i]
            card_updates.extend([
                gr.update(visible=True),                  # card
                gr.update(value=True),                    # include
                gr.update(value=c["title"]),              # title
                gr.update(value=round(c["start"], 2)),    # start
                gr.update(value=round(c["end"], 2)),      # end
                gr.update(value=c.get("subtitles", ""))   # subtitles
            ])
        else:
            card_updates.extend([
                gr.update(visible=False), # card
                gr.update(value=False),   # include
                gr.update(value=""),      # title
                gr.update(value=0.0),     # start
                gr.update(value=0.0),     # end
                gr.update(value="")       # subtitles
            ])

    status_msg = f"✅ **{len(refined_clips)} viral moments detected and calibrated!** You can preview each scene in the left player, fine-tune timestamps, select clips to render, and customize subtitles below."
    return (
        status_msg,
        table_data,
        gr.update(interactive=True),
        *card_updates
    )


def step7_render_shorts(
    source_choice,
    crop_style,
    subtitle_style,
    inc_1, title_1, st_1, en_1, subs_1,
    inc_2, title_2, st_2, en_2, subs_2,
    inc_3, title_3, st_3, en_3, subs_3,
    inc_4, title_4, st_4, en_4, subs_4,
    inc_5, title_5, st_5, en_5, subs_5,
    dubbed_lang=None,
    progress=gr.Progress()
):
    empty_gallery = []
    for _ in range(5):
        empty_gallery.extend([
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False)
        ])

    if not state.video_info or not state.video_info.get("video_path"):
        return (
            "⚠️ **No video loaded.** Please import a video in Tab 1 first.",
            *empty_gallery,
            None
        )

    raw_cards = [
        {"include": inc_1, "title": title_1, "start": st_1, "end": en_1, "subs": subs_1, "index": 1},
        {"include": inc_2, "title": title_2, "start": st_2, "end": en_2, "subs": subs_2, "index": 2},
        {"include": inc_3, "title": title_3, "start": st_3, "end": en_3, "subs": subs_3, "index": 3},
        {"include": inc_4, "title": title_4, "start": st_4, "end": en_4, "subs": subs_4, "index": 4},
        {"include": inc_5, "title": title_5, "start": st_5, "end": en_5, "subs": subs_5, "index": 5},
    ]

    clips_to_render = []
    for c in raw_cards:
        if not c["include"]:
            continue
        try:
            st = float(c["start"])
            en = float(c["end"])
            if en > st:
                clips_to_render.append({
                    "index": c["index"],
                    "title": str(c["title"]).strip() or f"Short #{c['index']}",
                    "start": st,
                    "end": en,
                    "duration": en - st,
                    "subtitles": str(c["subs"]).strip() if c["subs"] else ""
                })
        except Exception as e:
            print(f"[SHORTS] Parse card error: {e}")

    if not clips_to_render:
        return (
            "⚠️ **No shorts selected for rendering.** Check at least one short in the calibration section below.",
            *empty_gallery,
            None
        )

    video_path = state.video_info.get("video_path")
    audio_path = None
    active_segments = state.segments

    # Verify timestamps against actual video duration to prevent out-of-bounds render errors
    if video_path and os.path.exists(video_path):
        import subprocess
        try:
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            res_dur = subprocess.run(cmd_dur, capture_output=True, text=True)
            vid_dur = float(res_dur.stdout.strip())
        except Exception:
            vid_dur = 0.0

        if vid_dur > 0:
            for clip in clips_to_render:
                st = clip["start"]
                en = clip["end"]
                dur = clip.get("duration", en - st)
                if st >= vid_dur:
                    # Check if flat MMSS (e.g. 2027 for 20:27 -> 1227s)
                    m = int(st // 100)
                    s = int(st % 100)
                    if s < 60 and (m * 60 + s) <= vid_dur:
                        clip["start"] = m * 60.0 + s
                        clip["end"] = min(vid_dur, clip["start"] + dur)
                        print(f"[SHORTS] Corrected render start time from {st} to {clip['start']}s")
                    else:
                        clip["start"] = max(0.0, vid_dur - 35.0)
                        clip["end"] = vid_dur
                elif en > vid_dur:
                    clip["end"] = vid_dur

    if "Dubbed" in source_choice:
        resolved_segs, target_lang, resolved_vid = _resolve_dubbed_segments_and_video(dubbed_lang)
        if resolved_vid:
            video_path = resolved_vid
        else:
            mixed_audio = os.path.join(TEMP_DIR, "final_mix.wav")
            if os.path.exists(mixed_audio):
                audio_path = mixed_audio
        if resolved_segs:
            active_segments = resolved_segs

    crop_mode = "blur_stack"
    if "Center" in crop_style or "Crop" in crop_style or "Centré" in crop_style:
        crop_mode = "crop_center"

    sub_style = "tiktok_yellow"
    if "Vert" in subtitle_style or "Mint" in subtitle_style or "Green" in subtitle_style:
        sub_style = "tiktok_mint"
    elif "Cyan" in subtitle_style:
        sub_style = "tiktok_cyan"
    elif "Rose" in subtitle_style or "Fuchsia" in subtitle_style or "Pink" in subtitle_style:
        sub_style = "tiktok_magenta"
    elif "Blanc" in subtitle_style or "White" in subtitle_style:
        sub_style = "clean_white"
    elif "Sans" in subtitle_style or "No Subtitles" in subtitle_style:
        sub_style = None

    rendered_files = []
    total_clips = len(clips_to_render)
    for i, clip in enumerate(clips_to_render):
        progress((i / total_clips), f"Rendering Short #{clip['index']} ({clip['title']})...")
        
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', clip["title"])[:25]
        out_name = f"short_{clip['index']}_{safe_title}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        ass_path = None
        if sub_style:
            ass_name = f"short_{clip['index']}_{safe_title}.ass"
            ass_path = os.path.join(TEMP_DIR, ass_name)
            shorts_studio.generate_ass_subtitles(
                active_segments,
                clip["start"],
                clip["end"],
                ass_path,
                style=sub_style,
                text_key="translated_text" if "Dubbed" in source_choice else "text",
                custom_text=clip.get("subtitles")
            )

        success = shorts_studio.render_vertical_short(
            video_path=video_path,
            audio_path=audio_path,
            start_time=clip["start"],
            end_time=clip["end"],
            output_path=out_path,
            crop_mode=crop_mode,
            ass_subtitles_path=ass_path
        )

        if success and os.path.exists(out_path):
            rendered_files.append((clip["index"], out_path))

    zip_path = None
    if rendered_files:
        import zipfile
        zip_path = os.path.join(OUTPUT_DIR, "shorts_export_pack.zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for _, f in rendered_files:
                    zipf.write(f, os.path.basename(f))
        except Exception as e:
            print(f"[SHORTS] Failed to create ZIP pack: {e}")

    gallery_updates = []
    for slot_idx in range(5):
        if slot_idx < len(rendered_files):
            orig_idx, out_path = rendered_files[slot_idx]
            gallery_updates.extend([
                gr.update(visible=True),
                gr.update(value=out_path, label=f"Short #{orig_idx} Preview", visible=True),
                gr.update(value=out_path, label=f"Download Short #{orig_idx}", visible=True)
            ])
        else:
            gallery_updates.extend([
                gr.update(visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False)
            ])

    progress(1.0, "Shorts rendering complete!")
    status_msg = f"🎉 **{len(rendered_files)} vertical 9:16 Shorts generated successfully (1080x1920)!**"
    return (
        status_msg,
        *gallery_updates,
        zip_path
    )


# --- SEO BLOG & WORDPRESS STUDIO HANDLERS ---

def _get_or_recover_video_path(topic_hint: str = ""):
    """
    Get current video_path from state.video_info, or automatically recover
    the best matching video from temp/, output/ or user project folders if state was lost or if an audio-only file was loaded.
    """
    video_path = state.video_info.get("video_path") if (state.video_info and isinstance(state.video_info, dict)) else None
    
    # Check if existing video_path is a valid video file
    is_valid_video = False
    if video_path and os.path.exists(video_path):
        ext = os.path.splitext(video_path)[1].lower()
        if ext in [".mp4", ".mkv", ".mov", ".webm"]:
            is_valid_video = True
            
    if is_valid_video:
        return video_path
        
    # Auto-recovery: search temp/, output/, and project directories
    search_dirs = [
        TEMP_DIR,
        OUTPUT_DIR,
        r"D:\ps4\20260701\zaststranlsate\Tuto zasttranslate"
    ]
    ignore_prefixes = ('test_', 'preview_', 'short_', 'seg_')
    
    # Target duration from active segments or audio if known
    target_dur = 0.0
    if state.segments:
        target_dur = state.segments[-1].get("end", 0.0)
    elif state.video_info and state.video_info.get("duration"):
        target_dur = float(state.video_info["duration"])

    candidates = []
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        for f in os.listdir(sdir):
            f_lower = f.lower()
            if f_lower.endswith(('.mp4', '.mkv', '.mov', '.webm')) and not any(f_lower.startswith(p) for p in ignore_prefixes):
                full_p = os.path.join(sdir, f)
                if os.path.isfile(full_p):
                    score = 0
                    mtime = os.path.getmtime(full_p)
                    size_mb = os.path.getsize(full_p) / (1024 * 1024)
                    
                    # Duration match bonus
                    if target_dur > 10:
                        try:
                            cdur = blog_generator._get_video_duration(full_p)
                            if abs(cdur - target_dur) <= 4.0:
                                score += 200
                            elif abs(cdur - target_dur) <= 30.0:
                                score += 80
                        except Exception:
                            pass

                    # Recency bonus (files modified within the last 48h get a strong boost)
                    import time
                    age_hours = (time.time() - mtime) / 3600.0
                    if age_hours < 24:
                        score += 150
                    elif age_hours < 72:
                        score += 50

                    if 'zasttranslate' in f_lower: score += 40
                    if 'doublage' in f_lower: score += 25
                    if 'tuto' in f_lower: score += 20
                    if topic_hint and any(w in f_lower for w in topic_hint.lower().split() if len(w) > 3):
                        score += 30

                    candidates.append((score, mtime, size_mb, full_p))
                    
    if candidates:
        # Sort by score descending, then most recently modified, then size
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_recovered = candidates[0][3]
        print(f"[BLOG] [INFO] Auto-recovered video path for keyframe extraction: {best_recovered}")
        if state.video_info is None or not isinstance(state.video_info, dict):
            state.video_info = {}
        state.video_info["video_path"] = best_recovered
        return best_recovered
        
    return video_path

def step8_generate_blog_post(
    target_lang: str,
    style_choice: str,
    length_choice: str,
    include_meta: bool,
    extract_images_flag: bool,
    num_images: int = 6,
    keyframe_resolution: str = "1080p (Full HD - 1920x1080) [Recommandé Articles & Google SEO]",
    enhance_text_clarity: bool = True,
    progress=gr.Progress()
):
    """Step 8: Generate human-sounding SEO blog post and extract milestone video keyframes."""
    active_segments = state.translated_segments if (state.translated_segments and len(state.translated_segments) > 0) else state.segments
    if not active_segments or len(active_segments) == 0:
        from modules.srt_parser import SRTParser
        parser = SRTParser()
        for cand in ["temp/transcription_FR.srt", "output/transcription_FR.srt", "temp/transcription_AU.srt", getattr(state, "subtitles_path", "")]:
            if cand and os.path.exists(cand):
                parsed = parser.parse_srt(cand)
                if parsed:
                    active_segments = parsed
                    state.segments = parsed
                    print(f"[BLOG] Auto-recovered {len(parsed)} segments from {cand}")
                    break

    if not active_segments:
        return (
            "⚠️ **Prerequisite missing:** No transcription found. Please import a video in **Tab 1** and run Transcription in **Tab 2** (or import an SRT file) first.",
            "", "", "", "", "", "",
            "", "", "",
            None, None, None, None, None, None,
            "", "", "", "", "", "",
            None
        )

    # Dynamically reload blog_generator to ensure latest humanizer rules and prompts are immediately active
    import importlib
    import modules.blog_generator
    importlib.reload(modules.blog_generator)
    from modules.blog_generator import blog_generator

    progress(0.1, "Initializing AI backend and loading model...")
    # Load LLM backend (respect user selection, defaulting to Qwen3.5-9B)
    active_backend_name = user_config.get("llm_backend", "Qwen3.5-9B")
    if reformulator.backend_name != active_backend_name:
        if reformulator.llm is not None:
            reformulator.llm.unload()
            reformulator.llm = None
        reformulator.backend_name = active_backend_name

    llm_backend = getattr(reformulator, "llm", None)
    if llm_backend is None:
        try:
            reformulator.load_model()
            llm_backend = getattr(reformulator, "llm", None)
        except Exception as e:
            print(f"[BLOG] Warning loading LLM backend: {e}")
            llm_backend = None

    progress(0.3, f"Writing SEO Blog article in {target_lang} (Style: {style_choice})...")
    
    # Generate SEO article with full subtitle fidelity
    article_data = blog_generator.generate_article(
        segments=active_segments,
        video_info=state.video_info,
        target_lang=target_lang,
        style=style_choice,
        length=length_choice,
        include_meta=include_meta,
        llm_backend=llm_backend
    )

    if "error" in article_data:
        return (
            f"❌ Error: {article_data['error']}",
            "", "", "", "", "", "",
            "", "", "",
            None, None, None, None, None, None,
            "", "", "", "", "", "",
            None
        )

    progress(0.7, f"Extracting {num_images} milestone HD keyframes from video ({keyframe_resolution})...")
    extracted_images = []
    video_path = _get_or_recover_video_path(topic_hint=article_data.get("slug", "video"))
    
    if extract_images_flag and video_path and os.path.exists(video_path):
        blog_img_dir = os.path.join(OUTPUT_DIR, "blog_images")
        try:
            extracted_images = blog_generator.extract_article_keyframes(
                video_path=video_path,
                segments=active_segments,
                output_dir=blog_img_dir,
                num_images=int(num_images),
                topic_title=article_data.get("slug", "article"),
                target_resolution=keyframe_resolution,
                enhance_text_clarity=enhance_text_clarity
            )
        except Exception as img_err:
            print(f"[BLOG] Keyframe extraction error: {img_err}")
    elif extract_images_flag and not video_path:
        print("[BLOG] [WARN] Extract images was requested, but no video file was found in temp/ or loaded in state.")

    progress(0.9, "Generating WordPress ZIP package...")
    zip_output_path = os.path.join(OUTPUT_DIR, "blog_pack_wordpress.zip")
    packaged_zip = blog_generator.package_wordpress_zip(
        article_data=article_data,
        images_list=extracted_images,
        output_zip_path=zip_output_path
    )

    state.blog_package = {
        "article_data": article_data,
        "images": extracted_images,
        "zip_path": packaged_zip
    }

    progress(1.0, "SEO Blog article & WordPress package generated successfully!")

    # Format output fields
    title_val = article_data.get("title", "")
    meta_desc_val = article_data.get("meta_description", "")
    char_count = f"**{len(meta_desc_val)} characters** (Google recommended: 145-160)"
    slug_val = article_data.get("slug", "")
    focus_kw_val = article_data.get("focus_keyword", "")
    sec_kws_val = ", ".join(article_data.get("secondary_keywords", []))
    
    markdown_val = article_data.get("markdown_article", "")
    gutenberg_val = article_data.get("gutenberg_html", "")
    
    prompts_list = article_data.get("image_prompts", [])
    prompts_formatted = "\n\n".join([f"**Illustration Prompt #{i+1}:**\n```text\n{p}\n```" for i, p in enumerate(prompts_list)])

    # Images and captions (up to 6 slots)
    img_slots = [None] * 6
    caption_slots = [""] * 6
    for i, img in enumerate(extracted_images[:6]):
        img_slots[i] = img["path"]
        res_tag = f" • `{img.get('resolution_str', '1920x1080')}`" if img.get("resolution_str") else ""
        caption_slots[i] = f"📸 **{img['filename']}** (`{img['timestamp_display']}`{res_tag})\n\n*SEO ALT Text:*\n`{img['alt_text']}`"

    if extracted_images:
        res_summary = extracted_images[0].get('resolution_str', '1080p')
        visuals_line = f"- 🖼️ **Extracted Visuals:** {len(extracted_images)} HD keyframes ({res_summary}) with text/code clarity filter"
    else:
        visuals_line = "- ⚠️ **Extracted Visuals:** Aucune image extraite (aucun fichier vidéo .mp4 source trouvé dans temp/). Importez la vidéo en Onglet 1 pour générer les captures."

    status_summary = (
        f"🎉 **SEO Blog Article Generated Successfully!**\n\n"
        f"- 📊 **Length:** ~{article_data.get('word_count', 0)} words\n"
        f"- 🌍 **Language:** {target_lang}\n"
        f"- ✍️ **Style:** {style_choice}\n"
        f"{visuals_line}\n"
        f"- 📋 **Formats Ready:** WordPress Gutenberg Block HTML (1-Click Paste) & Standard Markdown!"
    )

    return (
        status_summary,
        title_val,
        meta_desc_val,
        char_count,
        slug_val,
        focus_kw_val,
        sec_kws_val,
        markdown_val,
        gutenberg_val,
        prompts_formatted,
        img_slots[0], img_slots[1], img_slots[2], img_slots[3], img_slots[4], img_slots[5],
        caption_slots[0], caption_slots[1], caption_slots[2], caption_slots[3], caption_slots[4], caption_slots[5],
        packaged_zip
    )

def step8_extract_images_only(
    num_images: int,
    keyframe_resolution: str = "1080p (Full HD - 1920x1080) [Recommandé Articles & Google SEO]",
    enhance_text_clarity: bool = True,
    progress=gr.Progress()
):
    """Extract or refresh only video keyframes without re-generating the article text."""
    video_path = _get_or_recover_video_path()
    if not video_path or not os.path.exists(video_path):
        return (
            "⚠️ **Prerequisite missing:** Please import a video first in **Tab 1** or place the MP4 video in `temp/`.",
            None, None, None, None, None, None,
            "", "", "", "", "", "",
            None
        )

    # Ensure segments exist
    active_segments = state.translated_segments if (state.translated_segments and len(state.translated_segments) > 0) else state.segments
    if not active_segments:
        from modules.srt_parser import SRTParser
        parser = SRTParser()
        for cand in ["temp/transcription_FR.srt", "output/transcription_FR.srt", "temp/transcription_AU.srt"]:
            if os.path.exists(cand):
                parsed = parser.parse_srt(cand)
                if parsed:
                    active_segments = parsed
                    state.segments = parsed
                    break

    progress(0.2, f"Extracting {num_images} milestone HD keyframes from video ({keyframe_resolution})...")
    blog_img_dir = os.path.join(OUTPUT_DIR, "blog_images")
    topic_slug = state.video_info.get("title", "video") if state.video_info else "zasttranslate-tuto"
    extracted_images = blog_generator.extract_article_keyframes(
        video_path=video_path,
        segments=active_segments,
        output_dir=blog_img_dir,
        num_images=int(num_images),
        topic_title=topic_slug,
        target_resolution=keyframe_resolution,
        enhance_text_clarity=enhance_text_clarity
    )

    progress(0.8, "Packaging images into WordPress media pack...")
    zip_output_path = os.path.join(OUTPUT_DIR, "blog_pack_wordpress.zip")
    if state.blog_package and "article_data" in state.blog_package:
        article_data = state.blog_package["article_data"]
    else:
        art_md_path = os.path.join(OUTPUT_DIR, "article_paradoxetemporel.md")
        art_html_path = os.path.join(OUTPUT_DIR, "article_gutenberg.html")
        art_seo_path = os.path.join(OUTPUT_DIR, "seo_metadata.json")
        if os.path.exists(art_md_path) and os.path.exists(art_html_path):
            with open(art_md_path, "r", encoding="utf-8") as f:
                saved_md = f.read()
            with open(art_html_path, "r", encoding="utf-8") as f:
                saved_html = f.read()
            saved_seo = {}
            if os.path.exists(art_seo_path):
                with open(art_seo_path, "r", encoding="utf-8") as f:
                    try:
                        saved_seo = json.load(f)
                    except Exception:
                        saved_seo = {}
            article_data = {
                "title": saved_seo.get("title", state.video_info.get("title", "ZastTranslate : traduire et doubler ses vidéos avec sa propre voix en local") if state.video_info else "ZastTranslate : traduire et doubler ses vidéos avec sa propre voix en local"),
                "slug": saved_seo.get("slug", "zasttranslate-doubler-video-propre-voix-local"),
                "meta_description": saved_seo.get("meta_description", ""),
                "focus_keyword": saved_seo.get("focus_keyword", "ZastTranslate doublage vidéo IA local"),
                "secondary_keywords": saved_seo.get("secondary_keywords", []),
                "markdown_article": saved_md,
                "gutenberg_html": saved_html,
                "image_prompts": saved_seo.get("image_prompts", [])
            }
        else:
            article_data = {
                "title": state.video_info.get("title", "Video Keyframes Pack") if state.video_info else "Video Keyframes Pack",
                "slug": "video-keyframes",
                "meta_description": "",
                "markdown_article": "# Video Keyframes Pack\n\nExtracted video milestones ready for WordPress Media Library.",
                "gutenberg_html": "<!-- wp:heading --><h2>Video Keyframes Pack</h2><!-- /wp:heading -->",
                "image_prompts": []
            }

    packaged_zip = blog_generator.package_wordpress_zip(
        article_data=article_data,
        images_list=extracted_images,
        output_zip_path=zip_output_path
    )

    state.blog_package = {
        "article_data": article_data,
        "images": extracted_images,
        "zip_path": packaged_zip
    }

    img_slots = [None] * 6
    caption_slots = [""] * 6
    for i, img in enumerate(extracted_images[:6]):
        img_slots[i] = img["path"]
        res_tag = f" • `{img.get('resolution_str', '1920x1080')}`" if img.get("resolution_str") else ""
        caption_slots[i] = f"📸 **{img['filename']}** (`{img['timestamp_display']}`{res_tag})\n\n*SEO ALT Text:*\n`{img['alt_text']}`"

    res_summary = extracted_images[0].get('resolution_str', '1080p') if extracted_images else '1080p'
    status_msg = f"✅ Extracted **{len(extracted_images)} HD keyframes ({res_summary})** successfully from video ({os.path.basename(video_path)}) with enhanced text/code sharpness!"
    return (
        status_msg,
        img_slots[0], img_slots[1], img_slots[2], img_slots[3], img_slots[4], img_slots[5],
        caption_slots[0], caption_slots[1], caption_slots[2], caption_slots[3], caption_slots[4], caption_slots[5],
        packaged_zip
    )

# --- GRADIO INTERFACE ---

def import_metadata_from_state():
    if state.video_info:
        return state.video_info.get('title', ''), state.video_info.get('description', '')
    return "", ""

def get_valid_languages(tts_backend_name, llm_backend_name):
    tts_engine_temp = get_tts_backend(tts_backend_name)
    llm_engine_temp = get_llm_backend(llm_backend_name)
    
    tts_langs = tts_engine_temp.capabilities.get("languages", [])
    llm_langs = llm_engine_temp.capabilities.get("languages", [])
    
    valid_lang_choices = []
    for display_name, iso in LANGUAGES.items():
        short_code = _get_iso_code(iso).lower()
        
        tts_ok = (tts_langs == "all") or (short_code in tts_langs)
        llm_ok = (llm_langs == "all") or (short_code in llm_langs)
        
        if tts_ok and llm_ok:
            valid_lang_choices.append(display_name)
            
    if not valid_lang_choices:
        valid_lang_choices = list(LANGUAGES.keys())
        
    return valid_lang_choices

INITIAL_VALID_LANGS = get_valid_languages(current_tts_backend, current_llm_backend)
INITIAL_LANG_VALUE = INITIAL_VALID_LANGS[0] if INITIAL_VALID_LANGS else None

BLOCKS_CSS = """
#segments_json_holder { display: none !important; }
html, body {
    overflow-y: auto !important;
    height: auto !important;
    min-height: 100vh !important;
}
.gradio-container {
    overflow-y: auto !important;
    height: auto !important;
    min-height: 100vh !important;
    max-width: 98% !important;
}
/* Ensure all multiline textboxes & textareas have visible, usable scrollbars and vertical resize */
textarea, .gradio-textbox textarea {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    resize: vertical !important;
    scrollbar-width: thin !important;
    scrollbar-color: #6366f1 rgba(0, 0, 0, 0.1) !important;
    line-height: 1.5 !important;
    font-family: inherit !important;
}

/* Universal scrollbar styling for textareas, dataframes, markdown previews and file lists */
.gradio-container *::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
    display: block !important;
}
.gradio-container *::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.06) !important;
    border-radius: 4px !important;
}
.gradio-container *::-webkit-scrollbar-thumb {
    background: #6366f1 !important;
    border-radius: 4px !important;
}
.gradio-container *::-webkit-scrollbar-thumb:hover {
    background: #4f46e5 !important;
}

/* --- RETRACTABLE PREVIEW PANEL --- */
#header_toolbar_row {
    margin-bottom: 6px !important;
    display: flex !important;
    justify-content: flex-start !important;
}

#btn_toggle_preview {
    max-width: 250px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    background: #1e1e38 !important;
    border: 1px solid #6366f1 !important;
    color: #e0e7ff !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}

#btn_toggle_preview:hover {
    background: #312e81 !important;
    border-color: #818cf8 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
    transform: translateY(-1px);
}

/* Collapsed layout logic */
#main_content_row.preview-collapsed #left_preview_col,
.preview-collapsed #left_preview_col {
    display: none !important;
}

#main_content_row.preview-collapsed #right_tabs_col,
.preview-collapsed #right_tabs_col {
    flex: 1 1 100% !important;
    width: 100% !important;
    max-width: 100% !important;
}

#right_tabs_col {
    transition: width 0.2s ease, max-width 0.2s ease, flex 0.2s ease !important;
}

/* AI Prompt Assistant Button - Prominent Visual Style */
#btn_ai_prompt_assistant {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    margin: 8px 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
#btn_ai_prompt_assistant:hover {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* --- TACTILE STUDIO DARK TAB NAVIGATION SYSTEM --- */

/* 1. Global Tab Navigation Bar (Main tabs & Sub-tabs) */
.gradio-container .tabs,
.gradio-container .tab-wrapper {
    overflow: visible !important;
    overflow-x: visible !important;
    max-height: none !important;
    height: auto !important;
}

.gradio-container .tab-wrapper > .tab-container,
.gradio-container div[role="tablist"],
.gradio-container .tab-nav {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    border-radius: 12px !important;
    padding: 6px 8px !important;
    gap: 6px !important;
    margin: 8px 0 16px 0 !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scrollbar-width: thin !important;
    align-items: center !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(12px) !important;
}

/* 2. Base Tab Button Style: Dark Slate Pills */
.gradio-container button[role="tab"],
.gradio-container .tab-nav > button {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    padding: 6px 11px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    margin: 0 !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}

/* Main Top-Level Tabs: Compact so all tabs fit seamlessly */
#right_tabs_col > .tabs > .tab-wrapper > .tab-container > button[role="tab"],
div:not(.tabitem) > .tabs > .tab-wrapper > .tab-container > button[role="tab"] {
    padding: 6px 10px !important;
    font-size: 0.82rem !important;
    gap: 4px !important;
    flex-shrink: 0 !important;
}

/* Hover state */
.gradio-container button[role="tab"]:hover,
.gradio-container .tab-nav > button:hover {
    background: #334155 !important;
    color: #ffffff !important;
    border-color: #818cf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
}

/* 3. Selected / Active Tab Button: Radiant Violet/Indigo Gradient */
.gradio-container button[role="tab"].selected,
.gradio-container .tab-nav > button.selected {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid #818cf8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* 4. Nested Sub-Tabs Accent (Tab 7, Tab 2, etc.) */
.gradio-container .tabitem .tab-wrapper > .tab-container,
.gradio-container .tabitem div[role="tablist"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(129, 140, 248, 0.25) !important;
    padding: 5px !important;
    margin: 12px 0 16px 0 !important;
}

.gradio-container .tabitem button[role="tab"] {
    background: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    color: #cbd5e1 !important;
}

.gradio-container .tabitem button[role="tab"].selected {
    background: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%) !important;
    color: #ffffff !important;
    border: 1px solid #60a5fa !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
}

/* --- INPUTS, DROPDOWNS, SLIDERS & FORM CONTROLS --- */
input[type="text"],
input[type="number"],
input[type="password"],
textarea,
select,
.gradio-textbox input,
.gradio-textbox textarea,
.gradio-dropdown input {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
}

input:focus,
textarea:focus,
select:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    outline: none !important;
}

/* Studio Glass Cards & Container Groups */
.zast-studio-card {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(12px) !important;
}

/* Labels */
.gradio-container label,
.gradio-container .block label,
.gradio-container .block span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Primary Action Buttons */
.gradio-container button.primary,
.gradio-container .btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.gradio-container button.primary:hover,
.gradio-container .btn-primary:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Distinct High-Visibility Emerald Validation Buttons */
.gradio-container button.zast-btn-validate,
.zast-btn-validate {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border: 1px solid #34d399 !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.45) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.gradio-container button.zast-btn-validate:hover,
.zast-btn-validate:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.7) !important;
    transform: translateY(-2px) scale(1.02) !important;
}

/* Secondary Buttons */
.gradio-container button.secondary,
.gradio-container .btn-secondary {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.secondary:hover,
.gradio-container .btn-secondary:hover {
    background: #334155 !important;
    color: #ffffff !important;
    border-color: #818cf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Dataframes / Tables */
.gradio-dataframe {
    background: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
}
.gradio-dataframe table {
    background: #0f172a !important;
    color: #f1f5f9 !important;
}
.gradio-dataframe th {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25) !important;
}
.gradio-dataframe td {
    background: #0f172a !important;
    color: #cbd5e1 !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
}
.gradio-dataframe tr:hover td {
    background: #1e293b !important;
}

/* --- FILE DOWNLOAD / UPLOAD CONTAINERS (Eliminates the giant empty bug-like square) --- */
.gradio-file,
div[data-testid="file"] {
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.gradio-file.empty,
div[data-testid="file"].empty,
.file-preview.empty {
    min-height: 48px !important;
    max-height: 56px !important;
    padding: 6px 12px !important;
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px dashed rgba(148, 163, 184, 0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.gradio-file.empty svg,
div[data-testid="file"].empty svg,
.file-preview.empty svg {
    width: 20px !important;
    height: 20px !important;
    opacity: 0.5 !important;
}
.gradio-file:not(.empty),
div[data-testid="file"]:not(.empty) {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
    border: 1px solid #6366f1 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
    border-radius: 10px !important;
}

/* --- TAB QUICK GUIDE & OPTIONS EXPLORER --- */
.zast-tab-guide-accordion {
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, rgba(23, 27, 44, 0.75) 0%, rgba(15, 19, 36, 0.85) 100%) !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
}
.zast-tab-guide-accordion:hover {
    border-color: rgba(129, 140, 248, 0.7) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.25) !important;
}
.zast-tab-guide-accordion > .label-wrap {
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #e0e7ff !important;
    padding: 8px 12px !important;
}
.zast-guide-container {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)) !important;
    gap: 14px !important;
    padding: 6px 4px 10px 4px !important;
    color: #f1f5f9 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.zast-guide-card {
    background: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
}
.zast-guide-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 3px 10px !important;
    border-radius: 16px !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    width: fit-content !important;
}
.zast-pill-goal {
    background: rgba(14, 165, 233, 0.25) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
}
.zast-pill-options {
    background: rgba(168, 85, 247, 0.25) !important;
    color: #c084fc !important;
    border: 1px solid rgba(192, 132, 252, 0.5) !important;
}
.zast-pill-click {
    background: rgba(16, 185, 129, 0.25) !important;
    color: #34d399 !important;
    border: 1px solid rgba(52, 211, 153, 0.5) !important;
}
.zast-guide-container,
.zast-tab-guide-accordion *,
.zast-guide-container *,
.zast-guide-card * {
    color: #e2e8f0 !important;
}
.zast-tab-guide-accordion b,
.zast-tab-guide-accordion strong,
.zast-guide-card b,
.zast-guide-card strong {
    color: #ffffff !important;
}
.zast-tab-guide-accordion code,
.zast-guide-card code {
    background: rgba(255, 255, 255, 0.15) !important;
    color: #a5b4fc !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}
.zast-tab-guide-accordion i,
.zast-tab-guide-accordion em,
.zast-guide-card i,
.zast-guide-card em {
    color: #94a3b8 !important;
}
.zast-pill-goal, .zast-pill-goal * {
    color: #38bdf8 !important;
}
.zast-pill-options, .zast-pill-options * {
    color: #c084fc !important;
}
.zast-pill-click, .zast-pill-click * {
    color: #34d399 !important;
}
.zast-guide-step-num, .zast-guide-step-num * {
    color: #ffffff !important;
    background: #4f46e5 !important;
}
.zast-guide-step-num {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    margin-right: 8px !important;
    flex-shrink: 0 !important;
}
.zast-card, .zast-short-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.zast-card:hover, .zast-short-card:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.15) !important;
}
/* Responsive Sticky Left Preview Column on Desktop */
@media (min-width: 992px) {
    #left_preview_col {
        position: sticky !important;
        top: 1rem !important;
        align-self: flex-start !important;
        max-height: calc(100vh - 2rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        scrollbar-color: #6366f1 rgba(15, 23, 42, 0.4) !important;
        padding-right: 6px !important;
    }
}

/* Card Status Badges (Pills) */
.zast-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    padding: 3px 10px !important;
    border-radius: 9999px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    float: right !important;
    margin-top: -2px !important;
}
.zast-pill-info {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
}
.zast-pill-success {
    background: rgba(34, 197, 94, 0.15) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(34, 197, 94, 0.35) !important;
}
.zast-pill-primary {
    background: rgba(99, 102, 241, 0.18) !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}
.zast-pill-purple {
    background: rgba(168, 85, 247, 0.18) !important;
    color: #c084fc !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
}
.zast-pill-warning {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.35) !important;
}
.zast-pill-cyan {
    background: rgba(6, 182, 212, 0.15) !important;
    color: #22d3ee !important;
    border: 1px solid rgba(6, 182, 212, 0.35) !important;
}

/* Subtitle Grid & Dataframe Enhancements */
.gradio-dataframe, [data-testid="dataframe"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #334155 !important;
}
.gradio-dataframe table, [data-testid="dataframe"] table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}
.gradio-dataframe th, [data-testid="dataframe"] th {
    background: #0f172a !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #334155 !important;
}
.gradio-dataframe tbody tr:nth-child(even), [data-testid="dataframe"] tbody tr:nth-child(even) {
    background-color: rgba(30, 41, 59, 0.45) !important;
}
.gradio-dataframe tbody tr:nth-child(odd), [data-testid="dataframe"] tbody tr:nth-child(odd) {
    background-color: rgba(15, 23, 42, 0.65) !important;
}
.gradio-dataframe tbody tr:hover, [data-testid="dataframe"] tbody tr:hover {
    background-color: rgba(99, 102, 241, 0.18) !important;
    transition: background-color 0.15s ease !important;
}
.gradio-dataframe td, [data-testid="dataframe"] td {
    padding: 10px 14px !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
    border-bottom: 1px solid rgba(51, 65, 85, 0.35) !important;
    color: #f1f5f9 !important;
}

/* Workflow Next Step Buttons */
.zast-next-step-row {
    margin-top: 18px !important;
    padding-top: 14px !important;
    border-top: 1px dashed rgba(148, 163, 184, 0.2) !important;
    display: flex !important;
    justify-content: flex-end !important;
    gap: 12px !important;
}
.zast-next-step-btn {
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.zast-next-step-btn:hover {
    transform: translateX(3px) translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.5) !important;
}

/* Quick Copy Buttons Hover & Active Glow */
button[aria-label="Copy"], button[title="Copy"] {
    transition: all 0.2s ease !important;
}
button[aria-label="Copy"]:hover, button[title="Copy"]:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    color: #818cf8 !important;
    transform: scale(1.08) !important;
}
button[aria-label="Copy"]:active, button[title="Copy"]:active {
    background: rgba(34, 197, 94, 0.3) !important;
    color: #4ade80 !important;
}

/* Interactive Information Tooltips */
.zast-tooltip {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: help !important;
    color: #94a3b8 !important;
    margin-left: 6px !important;
    font-size: 0.85em !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    background: rgba(148, 163, 184, 0.15) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    vertical-align: middle !important;
    position: relative !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
}
.zast-tooltip:hover {
    color: #ffffff !important;
    background: #6366f1 !important;
    border-color: #818cf8 !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.6) !important;
    transform: scale(1.15) !important;
}
/* Disable CSS pseudo-elements so only the single JS portal tooltip (#zast_floating_tooltip) renders */
.zast-tooltip::after,
.zast-tooltip::before {
    display: none !important;
}

/* Prevent parent container clipping for tooltips */
.zast-card,
.zast-studio-card,
.zast-card .gradio-markdown,
.zast-card .prose,
.zast-studio-card .gradio-markdown,
.zast-studio-card .prose,
.gradio-container .block:has(.zast-tooltip) {
    overflow: visible !important;
}

/* Floating Portal Tooltip (Document Body JS Driven) */
#zast_floating_tooltip {
    position: fixed !important;
    z-index: 2147483647 !important;
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #6366f1 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1.45 !important;
    max-width: 360px !important;
    white-space: pre-line !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(99, 102, 241, 0.35) !important;
    pointer-events: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    transition: opacity 0.15s ease, transform 0.15s ease !important;
    transform: translateY(4px) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
#zast_floating_tooltip.visible {
    opacity: 1 !important;
    visibility: visible !important;
    transform: translateY(0) !important;
}

#js_debug_log {
    display: none !important;
}
/* Ensure clean dropzone styling */
[data-testid="dropzone"] {
    border-color: rgba(99, 102, 241, 0.4) !important;
    background: rgba(30, 41, 59, 0.6) !important;
}
"""
BLOCKS_JS = """
(() => {
    // Client-side Tab Switcher with smooth scroll
    window.zastSwitchTab = (tabPrefix) => {
        try {
            // Find tab buttons within right_tabs_col or main_app_tabs or gradio-container
            const container = document.getElementById('main_app_tabs') || document.getElementById('right_tabs_col') || document.querySelector('.gradio-container') || document;
            
            // Only query buttons with role="tab" so we NEVER match content buttons like "Create Viral Shorts (Tab 6)"
            const tabButtons = Array.from(container.querySelectorAll('button[role="tab"]'));
            
            let targetBtn = null;
            if (tabButtons && tabButtons.length > 0) {
                targetBtn = tabButtons.find(b => {
                    const txt = (b.innerText || b.textContent || '').trim();
                    return txt.startsWith(tabPrefix);
                });
                
                if (!targetBtn) {
                    const keywords = {
                        '1.': 'Import',
                        '2.': 'Transcription',
                        '3.': 'Translation',
                        '4.': 'Dubbing',
                        '5.': 'Bulk',
                        '6.': 'Shorts',
                        '7.': 'Blog',
                        '8.': 'Help',
                        '9.': 'CPS'
                    };
                    const kw = keywords[tabPrefix] || tabPrefix;
                    targetBtn = tabButtons.find(b => {
                        const txt = (b.innerText || b.textContent || '').trim();
                        return txt.includes(kw);
                    });
                }
            }
            
            if (targetBtn) {
                targetBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                targetBtn.click();
                setTimeout(() => {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }, 60);
            } else {
                console.warn("zastSwitchTab: could not match target tab button for:", tabPrefix);
            }
        } catch(e) {
            console.error("Error switching tab:", e);
        }
    };

    // Global floating tooltip listener
    const initFloatingTooltips = () => {
        let tooltipEl = document.getElementById("zast_floating_tooltip");
        if (!tooltipEl) {
            tooltipEl = document.createElement("div");
            tooltipEl.id = "zast_floating_tooltip";
            document.body.appendChild(tooltipEl);
        }

        document.addEventListener("mouseover", (e) => {
            const target = e.target.closest(".zast-tooltip, [data-tooltip], [title]");
            if (target && (target.classList.contains("zast-tooltip") || target.hasAttribute("data-tooltip"))) {
                target.classList.add("tooltip-visible");
                const text = target.getAttribute("data-tooltip") || target.getAttribute("title");
                if (text) {
                    target.setAttribute("data-tooltip", text);
                    target.removeAttribute("title");
                    tooltipEl.textContent = text.replace(/<br\s*\/?>/gi, "\\n");
                    tooltipEl.style.whiteSpace = "pre-line";
                    tooltipEl.classList.add("visible");
                    
                    const rect = target.getBoundingClientRect();
                    const ttWidth = tooltipEl.offsetWidth || 260;
                    const ttHeight = tooltipEl.offsetHeight || 40;
                    
                    let left = rect.left + (rect.width / 2) - (ttWidth / 2);
                    if (left < 10) left = 10;
                    if (left + ttWidth > window.innerWidth - 10) left = window.innerWidth - ttWidth - 10;
                    
                    let top = rect.top - ttHeight - 8;
                    if (top < 10) {
                        top = rect.bottom + 8;
                    }
                    
                    tooltipEl.style.left = left + "px";
                    tooltipEl.style.top = top + "px";
                }
            }
        });

        document.addEventListener("mouseout", (e) => {
            const target = e.target.closest(".zast-tooltip, [data-tooltip]");
            if (target) {
                target.classList.remove("tooltip-visible");
                if (tooltipEl) tooltipEl.classList.remove("visible");
            }
        });
    };
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initFloatingTooltips);
    } else {
        initFloatingTooltips();
    }
    const debugLog = (msg) => {
        console.log("[ZastDebug] " + msg);
        window.zast_logs = window.zast_logs || [];
        window.zast_logs.push(new Date().toLocaleTimeString() + ": " + msg);
        const logEl = document.querySelector("#js_debug_log");
        if (logEl) {
            logEl.innerHTML = "[JS Debug Log Started]<br/>" + window.zast_logs.join("<br/>");
            logEl.scrollTop = logEl.scrollHeight;
        }
    };

    debugLog("Initializing JS Blocks...");
    try {
        document.documentElement.classList.add("dark");
        document.body.classList.add("dark");
    } catch(e) {}

    window.onerror = function(message, source, lineno, colno, error) {
        debugLog("GLOBAL ERROR: " + message + " at " + source + ":" + lineno);
        return false;
    };

    // --- PREVIEW PANEL TOGGLE LOGIC ---
    window.setZastPreviewCollapsed = function(collapsed) {
        const row = document.querySelector("#main_content_row");
        const container = document.querySelector(".gradio-container");
        const btn = document.querySelector("#btn_toggle_preview");
        
        if (row) {
            if (collapsed) {
                row.classList.add("preview-collapsed");
            } else {
                row.classList.remove("preview-collapsed");
            }
        }
        if (container) {
            if (collapsed) {
                container.classList.add("preview-collapsed");
            } else {
                container.classList.remove("preview-collapsed");
            }
        }
        
        const label = collapsed ? "▶ Show Preview & Subtitles" : "◀ Hide Preview & Subtitles";
        if (btn) {
            btn.innerHTML = label;
            btn.textContent = label;
        }
        
        try {
            localStorage.setItem("zast_preview_panel_collapsed", collapsed ? "true" : "false");
        } catch(e) {}
        
        // Trigger resize event so Gradio tables, wave surfers, and plots recalculate their layout
        window.dispatchEvent(new Event("resize"));
        setTimeout(() => { window.dispatchEvent(new Event("resize")); }, 50);
        setTimeout(() => { window.dispatchEvent(new Event("resize")); }, 250);
        debugLog("Preview collapsed state set to: " + collapsed);
    };

    window.toggleZastPreview = function() {
        const row = document.querySelector("#main_content_row");
        const isCollapsed = row ? row.classList.contains("preview-collapsed") : false;
        window.setZastPreviewCollapsed(!isCollapsed);
    };

    let hasRestoredPreviewState = false;
    const restoreSavedPreviewState = () => {
        if (hasRestoredPreviewState) return;
        const row = document.querySelector("#main_content_row");
        const btn = document.querySelector("#btn_toggle_preview");
        if (row && btn) {
            hasRestoredPreviewState = true;
            try {
                const saved = localStorage.getItem("zast_preview_panel_collapsed");
                if (saved === "true") {
                    window.setZastPreviewCollapsed(true);
                    debugLog("Restored collapsed preview state from localStorage on DOM mount.");
                }
            } catch(e) {}
        }
    };

    // Attach global click listener once for toggle preview button
    document.addEventListener("click", (e) => {
        const btn = e.target.closest("#btn_toggle_preview");
        if (btn) {
            e.preventDefault();
            e.stopPropagation();
            debugLog("Toggle preview button clicked via delegation.");
            window.toggleZastPreview();
        }
    });

    // Run initial checks
    restoreSavedPreviewState();

    setInterval(() => {
        restoreSavedPreviewState();
        const player = document.querySelector("#video_player video") || document.querySelector("video") || document.querySelector("#audio_player audio") || document.querySelector("audio");
        
        const logEl = document.querySelector("#js_debug_log");
        if (logEl && window.zast_logs && window.zast_logs.length > 0) {
            logEl.innerHTML = "[JS Debug Log Started]<br/>" + window.zast_logs.join("<br/>");
        }

        if (player) {
            if (!player.dataset.hasTimeupdate) {
                player.dataset.hasTimeupdate = "true";
                debugLog("Found player! Attaching timeupdate event listener. Tag: " + player.tagName);
                player.addEventListener("timeupdate", () => {
                    const currentTime = player.currentTime;
                    
                    const holder = document.querySelector("#segments_json_holder");
                    const overlay = document.querySelector("#subtitle_overlay");
                    
                    if (!holder || !overlay) {
                        if (Math.random() < 0.05) {
                            debugLog("Warning: holder or overlay missing from DOM");
                        }
                        return;
                    }
                    
                    let mode = window.ZastSubtitleMode;
                    if (!mode) {
                        const checked = document.querySelector("#subtitle_selection input:checked");
                        if (checked) {
                            const val = checked.value;
                            if (val === "None" || val === "Original" || val === "Translation (Fitted)" || val === "Translation (Normal)") {
                                mode = val;
                            } else {
                                const label = checked.closest("label");
                                if (label) {
                                    mode = label.textContent.trim();
                                } else {
                                    mode = "None";
                                }
                            }
                        } else {
                            mode = "None";
                        }
                    }
                    
                    if (mode === "None") {
                        overlay.innerHTML = "";
                        overlay.style.display = "none";
                        return;
                    }
                    
                    const jsonEl = holder.querySelector("#segments_json_data") || holder;
                    const b64 = jsonEl.textContent.trim();
                    if (!b64 || b64 === "W10=") {
                        if (Math.random() < 0.05) {
                            debugLog("Segments empty or W10= (default empty)");
                        }
                        overlay.innerHTML = "";
                        overlay.style.display = "none";
                        return;
                    }
                    
                    try {
                        const jsonStr = decodeURIComponent(escape(atob(b64)));
                        const segments = JSON.parse(jsonStr);
                        let activeText = "";
                        for (const seg of segments) {
                            if (currentTime >= seg.start && currentTime <= seg.end) {
                                if (mode === "Original") {
                                    activeText = seg.text || "";
                                } else if (mode === "Translation (Fitted)") {
                                    activeText = seg.translated_text || "";
                                } else if (mode === "Translation (Normal)") {
                                    activeText = seg.normal_text || seg.translated_text || "";
                                }
                                break;
                            }
                        }
                        
                        if (Math.random() < 0.05) {
                            debugLog("Time: " + currentTime.toFixed(2) + "s, Mode: " + mode + ", Segs: " + segments.length + ", Text: '" + activeText + "'");
                        }
                        
                        if (activeText) {
                            overlay.innerHTML = activeText;
                            overlay.style.display = "block";
                        } else {
                            overlay.innerHTML = "";
                            overlay.style.display = "none";
                        }
                    } catch (e) {
                        debugLog("Error parsing/decoding segments: " + e.message);
                    }
                });
            }
        } else {
            if (Math.random() < 0.05) {
                debugLog("No player element found in page yet.");
            }
        }
    }, 1000);

    // Monitor upload status of local_file_input and srt_file_input
    setInterval(() => {
        const checkUpload = (inputContainerId, buttonId, activeText, defaultText) => {
            const container = document.getElementById(inputContainerId);
            const button = document.getElementById(buttonId);
            if (!container || !button) return;
            
            const hasProgressBar = container.querySelector("[role='progressbar']") || 
                                   container.querySelector(".progress-bar") || 
                                   container.querySelector(".progress") || 
                                   container.querySelector(".progress-ring") || 
                                   container.querySelector(".uploading") || 
                                   container.textContent.includes("Uploading") ||
                                   container.textContent.includes("Chargement");
            
            if (hasProgressBar) {
                if (!button.disabled) {
                    button.disabled = true;
                    button.style.opacity = "0.5";
                    button.style.cursor = "not-allowed";
                    button.dataset.originalText = button.textContent;
                    button.textContent = activeText;
                }
            } else {
                if (button.disabled && (button.textContent === activeText)) {
                    button.disabled = false;
                    button.style.opacity = "1";
                    button.style.cursor = "pointer";
                    button.textContent = button.dataset.originalText || defaultText;
                }
            }
        };
        
        checkUpload("local_file_input", "btn_import_video", "Uploading...", "Import Video or Audio");
        checkUpload("srt_file_input", "btn_import_srt", "Uploading SRT...", "Import SRT");
    }, 300);

    window.zastSeekVideo = function(startTime, endTime) {
        debugLog("zastSeekVideo called with start: " + startTime);
        const player = document.querySelector("#video_player video") || document.querySelector("video") || document.querySelector("#audio_player audio") || document.querySelector("audio");
        if (player) {
            const st = parseFloat(startTime);
            if (!isNaN(st)) {
                if (window.setZastPreviewCollapsed) {
                    window.setZastPreviewCollapsed(false);
                }
                player.currentTime = Math.max(0, st);
                player.play().catch((err) => {
                    debugLog("Play error: " + err.message);
                });
            }
        } else {
            debugLog("Player element not found for seeking.");
        }
    };

    debugLog("Registering global dataframe click handler...");
    document.addEventListener("click", (e) => {
        const cell = e.target.closest("td");
        if (!cell) return;
        const row = cell.closest("tr");
        if (!row || !row.cells) return;
        
        const dfContainer = cell.closest("#transcription_df, #translation_df, #dubbing_segments_df, #shorts_table");
        if (!dfContainer) return;
        
        const id = dfContainer.id;
        debugLog("Table cell clicked in: " + id);
        let startTime = NaN;
        
        if (id === "transcription_df" || id === "translation_df") {
            if (row.cells.length > 0) {
                startTime = parseFloat(row.cells[0].innerText);
            }
        } else if (id === "dubbing_segments_df") {
            if (row.cells.length > 1) {
                // Cell index 0 is "Index", index 1 is "Start"
                startTime = parseFloat(row.cells[1].innerText);
            }
        } else if (id === "shorts_table") {
            // shorts_table columns: 0: #, 1: Hook, 2: Start (s), 3: End (s)
            if (row.cells.length > 2) {
                startTime = parseFloat(row.cells[2].innerText);
            }
        }
        
        if (!isNaN(startTime)) {
            debugLog("Parsed start time: " + startTime + "s. Seeking player...");
            if (window.zastSeekVideo) {
                window.zastSeekVideo(startTime);
            } else {
                const player = document.querySelector("#video_player video") || document.querySelector("video") || document.querySelector("#audio_player audio") || document.querySelector("audio");
                if (player) {
                    player.currentTime = startTime;
                    player.play().catch((err) => {
                        debugLog("Play error: " + err.message);
                    });
                } else {
                    debugLog("Player element not found for seeking.");
                }
            }
        } else {
            debugLog("Could not parse start time from cell.");
        }
    });
})();
"""

def _get_tab_guide_html(tab_num: int) -> str:
    guides = {
        1: {
            "goal": "Load any video or audio file into ZastTranslate, extract the audio track, and initialize your project timeline.",
            "options": [
                "<b style='color:#ffffff !important;'>YouTube URL</b>: <span style='color:#f1f5f9 !important;'>Paste any public YouTube link or Shorts URL.</span>",
                "<b style='color:#ffffff !important;'>Check URL</b>: <span style='color:#f1f5f9 !important;'>Queries YouTube to fetch video details, thumbnails, and available resolutions (1080p default).</span>",
                "<b style='color:#ffffff !important;'>TTS Model (Voice)</b>: <span style='color:#f1f5f9 !important;'>Selects the neural voice model used later in Tab 4 for dubbing. <i style='color:#cbd5e1 !important;'>Beginners: keep default VoxCPM 2.</i></span>",
                "<b style='color:#ffffff !important;'>Local File Upload</b>: <span style='color:#f1f5f9 !important;'>Drag and drop <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>.mp4</code>, <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>.mov</code>, <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>.mkv</code>, <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>.mp3</code>, or <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>.wav</code> directly.</span>"
            ],
            "steps": [
                "Paste your YouTube link OR drop a local video/audio file.",
                "Click <b style='color:#818cf8 !important;'>Import Video or Audio</b> to extract the vocal track."
            ]
        },
        2: {
            "goal": "Transcribe spoken audio into high-precision synchronized subtitles with exact word-level timestamps using Whisper AI.",
            "options": [
                "<b style='color:#ffffff !important;'>Source Language</b>: <span style='color:#f1f5f9 !important;'>Leave on <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>Auto</code> for automatic language detection, or pick your language for a speed boost.</span>",
                "<b style='color:#ffffff !important;'>Whisper Model</b>: <span style='color:#f1f5f9 !important;'><code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>base</code> is ultra-fast (~30s, 2GB VRAM). <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>large-v3</code> provides studio precision for slang, technical jargon, and accents.</span>",
                "<b style='color:#ffffff !important;'>Edit Subtitles</b>: <span style='color:#f1f5f9 !important;'>Click any row in the table to jump to that moment in the video. Double-click any cell to correct words.</span>",
                "<b style='color:#ffffff !important;'>Clean Fillers & Oral Tics</b>: <span style='color:#f1f5f9 !important;'>One-click AI cleanup to remove hesitations ('um', 'uh', 'euh', stuttering).</span>"
            ],
            "steps": [
                "Click <b style='color:#818cf8 !important;'>Run Transcription</b> and wait for completion.",
                "Review/edit subtitles in the interactive table below.",
                "Click <b style='color:#34d399 !important;'>Validate Transcription ✅</b>."
            ]
        },
        3: {
            "goal": "Translate your subtitles into 20+ languages with intelligent syllable adaptation so the synthesized voice matches original timing.",
            "options": [
                "<b style='color:#ffffff !important;'>Target Language</b>: <span style='color:#f1f5f9 !important;'>Select French, Spanish, German, Japanese, Chinese, Arabic, Portuguese, etc.</span>",
                "<b style='color:#ffffff !important;'>Normal vs. Fitted Translation</b>: <span style='color:#f1f5f9 !important;'><i style='color:#cbd5e1 !important;'>Normal</i> is a direct literal translation. <i style='color:#cbd5e1 !important;'>Fitted (Recommended)</i> rephrases sentences so the syllable count matches the original speaker's exact timeframe, preventing voice overflows!</span>",
                "<b style='color:#ffffff !important;'>Metadata Translation</b>: <span style='color:#f1f5f9 !important;'>Translates video Title & Description for international YouTube SEO.</span>"
            ],
            "steps": [
                "Choose your <b>Target Language</b>.",
                "Click <b style='color:#818cf8 !important;'>Run Translation</b>.",
                "Review the <i style='color:#cbd5e1 !important;'>Fitted</i> column and click <b style='color:#34d399 !important;'>Validate Translation ✅</b>."
            ]
        },
        4: {
            "goal": "Synthesize translated speech, duck original vocals/background music, and render the final 1080p dubbed MP4 video with direct 1-click video and audio downloads.",
            "options": [
                "<b style='color:#ffffff !important;'>Voice Mode</b>: <span style='color:#f1f5f9 !important;'><code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>Default voice</code> (Studio AI voice - Male/Female). <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>Clone from original</code> (Clones the speaker's vocal timbre from the video!). <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>Clone from file</code> (Upload a 10-30s clean sample).</span>",
                "<b style='color:#ffffff !important;'>Never Cut Vocal</b>: <span style='color:#f1f5f9 !important;'>Guarantees complete sentence delivery without rushing words (may slightly overflow video cuts).</span>",
                "<b style='color:#ffffff !important;'>Segment Fine-Tuning</b>: <span style='color:#f1f5f9 !important;'>Click any row to preview or re-generate individual sentences with custom timing.</span>",
                "<b style='color:#ffffff !important;'>🎬 Download Dubbed Video (MP4)</b>: <span style='color:#f1f5f9 !important;'>Dedicated 1-click download button and file card for the rendered 1080p dubbed video.</span>",
                "<b style='color:#ffffff !important;'>🎵 Download Audio Track (WAV)</b>: <span style='color:#f1f5f9 !important;'>Dedicated 1-click download button and file card for the isolated mixed audio track.</span>"
            ],
            "steps": [
                "Pick your <b>Voice Mode</b> (Default voice or Clone).",
                "Click <b style='color:#818cf8 !important;'>Assemble Final Video & Audio</b> to render your dubbed MP4.",
                "Download the dubbed video (MP4) or audio track (WAV) directly below each preview player."
            ]
        },
        5: {
            "goal": "Batch-translate and dub your video into multiple languages simultaneously in a single automated hands-free run.",
            "options": [
                "<b style='color:#ffffff !important;'>Target Languages (Multiselect)</b>: <span style='color:#f1f5f9 !important;'>Select multiple languages (e.g. Spanish + German + Japanese).</span>",
                "<b style='color:#ffffff !important;'>Output Generation</b>: <span style='color:#f1f5f9 !important;'><i style='color:#cbd5e1 !important;'>Video + Audio</i> (full MP4s), <i style='color:#cbd5e1 !important;'>Audio Only</i> (WAV tracks), or <i style='color:#cbd5e1 !important;'>Subtitles & Metadata Only</i> (generates translated SRTs & titles in seconds without TTS).</span>",
                "<b style='color:#ffffff !important;'>Generate 3 Viral Shorts</b>: <span style='color:#f1f5f9 !important;'>Auto-renders 3 vertical 9:16 clips for every selected target language.</span>"
            ],
            "steps": [
                "Select all target languages and voice settings.",
                "Click <b style='color:#818cf8 !important;'>Run Bulk Process</b>."
            ]
        },
        6: {
            "goal": "Auto-detect 1 to 5 high-impact viral moments, preview sequences in the player, customize burned subtitles, and reframe into vertical 9:16 (1080x1920) TikTok/Reels.",
            "options": [
                "<b style='color:#ffffff !important;'>Number of Shorts (1 to 5)</b>: <span style='color:#f1f5f9 !important;'>Slider to choose how many standalone viral moments to extract from the video (default 3, up to 5).</span>",
                "<b style='color:#ffffff !important;'>Scene Preview in Player</b>: <span style='color:#f1f5f9 !important;'>Click any table row or '▶️ Preview Short in Player' to inspect that exact scene in the left video player.</span>",
                "<b style='color:#ffffff !important;'>Selective Rendering</b>: <span style='color:#f1f5f9 !important;'>Uncheck individual short checkboxes to exclude clips you don't want to generate (e.g., skip Short #3).</span>",
                "<b style='color:#ffffff !important;'>Editable Karaoke Subtitles</b>: <span style='color:#f1f5f9 !important;'>Inspect and customize the word-by-word animated subtitles that will be burned into each vertical video.</span>",
                "<b style='color:#ffffff !important;'>Timecode Fine-Tuning</b>: <span style='color:#f1f5f9 !important;'>Adjust start and end seconds with automated camera cut snapping (PySceneDetect) and speech boundary alignment.</span>",
                "<b style='color:#ffffff !important;'>Track Source & Crop Style</b>: <span style='color:#f1f5f9 !important;'>Original or Dubbed video, Stacked Blur (Aesthetic) or Center Crop 9:16.</span>"
            ],
            "steps": [
                "Select number of shorts (1-5) and click <b style='color:#818cf8 !important;'>✨ 1. Detect Top Viral Moments</b>.",
                "Inspect each short in the editor cards, click <b style='color:#818cf8 !important;'>▶️ Preview Short in Player</b> to review, and tweak subtitles if desired.",
                "Uncheck any short you don't want to render.",
                "Click <b style='color:#34d399 !important;'>🎬 3. Render Selected 9:16 Shorts</b>."
            ]
        },
        7: {
            "goal": "Transform any transcribed video into a human-sounding, high-ranking SEO blog post, extract HD milestone keyframes, and create viral 4K YouTube thumbnails using FLUX.1-schnell.",
            "options": [
                "<b style='color:#ffffff !important;'>Target Language & 6 Writing Styles</b>: <span style='color:#f1f5f9 !important;'>Write in 12+ languages with tailored editorial tones (Step-by-Step Tutorial, Technical Deep-Dive, Storytelling Case Study, Journalistic Review, High-Converting Copywriting, or Beginner's Guide).</span>",
                "<b style='color:#ffffff !important;'>🛡️ Humanizer Engine (35 Anti-AI Rules)</b>: <span style='color:#f1f5f9 !important;'>Adheres strictly to Wikipedia's WikiProject AI Cleanup standards. Strips robotic clichés ('In this article', 'It is crucial to remember', 'In today's digital age', inflated buzzwords, shallow participle transitions) for authentic human burstiness and search engine trust.</span>",
                "<b style='color:#ffffff !important;'>🔍 Live Google Suggest Keywords</b>: <span style='color:#f1f5f9 !important;'>Queries real-time Google autocompletion to seamlessly weave high-volume keywords into your H1 title, meta description, and H2/H3 headings without keyword stuffing.</span>",
                "<b style='color:#ffffff !important;'>📋 1-Click WordPress Gutenberg & Markdown</b>: <span style='color:#f1f5f9 !important;'>Ready-to-paste native block markup (<code>&lt;!-- wp:heading --&gt;</code>, <code>&lt;!-- wp:paragraph --&gt;</code>, <code>&lt;!-- wp:list --&gt;</code>) or clean Markdown. Paste into WordPress Code Editor (<code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>Ctrl+Shift+Alt+M</code>) and switch back to Visual Editor. Download complete pack as <code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>blog_pack_wordpress.zip</code>.</span>",
                "<b style='color:#ffffff !important;'>📸 Video Keyframes & SEO ALT Tags</b>: <span style='color:#f1f5f9 !important;'>Extract 1 to 8 HD screenshots from milestone video timestamps, auto-generate descriptive ALT tags and captions, and get ready-to-use AI image prompts (Midjourney, DALL-E 3, Flux).</span>",
                "<b style='color:#ffffff !important;'>⚡ FLUX.1 AI YouTube Thumbnails & A/B Testing</b>: <span style='color:#f1f5f9 !important;'>12B Flow Transformer generating 4K thumbnails in 4 steps (~2s on GPU). Wrap text in single quotes (<code style='color:#a5b4fc !important; background:rgba(255,255,255,0.15); padding:1px 5px; border-radius:4px;'>'YOUR TITLE'</code>) for razor-sharp 3D typography. Run A/B testing to generate 3 distinct viral variants (Viral High-CTR, 3D Tech Glow, Cinematic Studio) with 100% stripped AI metadata ready for YouTube Studio's Test & Compare!</span>"
            ],
            "steps": [
                "Select your <b>Target Language</b>, <b>Writing Style</b>, and <b>Target Length</b>.",
                "Click <b style='color:#818cf8 !important;'>✨ Generate SEO Blog Article & WordPress Kit</b> to produce the complete post, metadata, and keyframes.",
                "Copy the native <b>Gutenberg Block HTML</b> directly into WordPress Code Editor (Ctrl+Shift+Alt+M), or download the complete ZIP pack.",
                "Switch to <b>Sub-Tab 3 (Thumbnail Studio)</b>, enter a prompt (put text in quotes: 'MY TITLE'), and click <b style='color:#818cf8 !important;'>⚡ Generate Single Thumbnail</b> or <b style='color:#818cf8 !important;'>🧪 Run A/B Testing</b>."
            ]
        },
        9: {
            "goal": "Calibrate the speaking rate (Characters Per Second - CPS) per target language to ensure fitted dubbing matches exact on-screen video cuts without speech overflows.",
            "options": [
                "<b style='color:#ffffff !important;'>Language & ISO Code</b>: <span style='color:#f1f5f9 !important;'>Standard ISO 639-1 language codes mapped to their calibrated speech duration.</span>",
                "<b style='color:#ffffff !important;'>Default CPS</b>: <span style='color:#f1f5f9 !important;'>Carefully benchmarked natural reading speeds for each language (e.g. 14.5 for French, 15.0 for English, 15.5 for Spanish).</span>",
                "<b style='color:#ffffff !important;'>Your CPS Override</b>: <span style='color:#f1f5f9 !important;'>Set a custom value if you prefer faster delivery (higher value = more words allowed per second) or slower delivery (lower value = shorter, more relaxed sentences).</span>"
            ],
            "steps": [
                "Locate your target language in the table.",
                "Enter your desired value in the <b>Your CPS</b> column (or leave empty to keep default).",
                "Click <b style='color:#818cf8 !important;'>💾 Save</b> to apply immediately across all dubbing operations without restarting."
            ]
        }
    }
    g = guides.get(tab_num, {})
    if not g:
        return ""

    opts_html = "".join([f"<li style='color:#f1f5f9 !important; margin-bottom:6px; font-size:0.88rem;'>{opt}</li>" for opt in g.get("options", [])])
    steps_html = "".join([
        f"<div style='display:flex; align-items:flex-start; margin-bottom:8px; line-height:1.5; color:#f1f5f9 !important;'>"
        f"<span class='zast-guide-step-num'>{i+1}</span><span style='color:#f1f5f9 !important; font-size:0.88rem;'>{s}</span></div>"
        for i, s in enumerate(g.get("steps", []))
    ])

    return f'''
    <div class="zast-guide-container" style="color:#f1f5f9 !important;">
        <div class="zast-guide-card" style="color:#f1f5f9 !important; background:#0f172a !important;">
            <div class="zast-guide-pill zast-pill-goal">🎯 Goal</div>
            <div class="zast-guide-text" style="color:#f1f5f9 !important; font-size:0.9rem !important; line-height:1.55 !important;">{g.get('goal', '')}</div>
        </div>
        <div class="zast-guide-card" style="color:#f1f5f9 !important; background:#0f172a !important;">
            <div class="zast-guide-pill zast-pill-options">⚙️ Key Options & Settings</div>
            <ul class="zast-guide-list" style="color:#f1f5f9 !important; padding-left:18px !important; margin:0 !important;">{opts_html}</ul>
        </div>
        <div class="zast-guide-card" style="color:#f1f5f9 !important; background:#0f172a !important;">
            <div class="zast-guide-pill zast-pill-click">👉 Where to Click</div>
            <div style="color:#f1f5f9 !important;">{steps_html}</div>
        </div>
    </div>
    '''

with gr.Blocks(title="ZastTranslate", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate")) as app:
    # Embed logo as base64 to avoid Gradio version compatibility issues
    import base64 as _b64
    _logo_path = os.path.join(BASE_DIR, "zastttranslate.png")
    _logo_html = ""
    if os.path.exists(_logo_path):
        with open(_logo_path, "rb") as _f:
            _logo_b64 = _b64.b64encode(_f.read()).decode()
        _logo_html = f"<center><img src='data:image/png;base64,{_logo_b64}' width='80' /></center>\n\n"
    gr.Markdown(f"{_logo_html}# 🎬 ZastTranslate — Beta {APP_VERSION}\n**Offline video translation & dubbing (No Lip-Sync)**")
    
    with gr.Row(elem_id="header_toolbar_row"):
        btn_toggle_preview = gr.Button(
            "◀ Hide Preview & Subtitles", 
            elem_id="btn_toggle_preview", 
            variant="secondary",
            size="sm"
        )
    
    with gr.Row(elem_id="main_content_row"):
        with gr.Column(scale=2, min_width=300, elem_id="left_preview_col"):
            gr.Markdown("### 📺 Preview & Subtitles")
            video_preview = gr.Video(label="Preview", height=300, elem_id="video_player")
            audio_preview = gr.Audio(label="Audio Preview", visible=False, elem_id="audio_player")
            
            # Subtitle Selection
            subtitle_selection = gr.Radio(
                choices=["None", "Original", "Translation (Fitted)", "Translation (Normal)"],
                value="None",
                label="Preview Subtitles",
                elem_id="subtitle_selection",
                interactive=True
            )
            
            # Hidden inputs for subtitle syncing
            segments_json_holder = gr.HTML(value=_get_empty_segments_html(), visible=True, elem_id="segments_json_holder")
            
            # Styled subtitle overlay box and debug log
            # Styled subtitle overlay box (debug log hidden in production)
            subtitle_overlay = gr.HTML(
                value="""
                <div style='text-align: center; font-size: 1.25em; padding: 12px; background: rgba(0,0,0,0.75); color: #00ffcc; border-radius: 6px; font-weight: bold; text-shadow: 1px 1px 2px black; display: none;' id='subtitle_overlay'></div>
                <div id='js_debug_log' style='display: none !important;'>[JS Debug Log Started]</div>
                """,
                elem_id="subtitle_overlay_container"
            )
            
        with gr.Column(scale=3, elem_id="right_tabs_col"):
            main_tabs = gr.Tabs(elem_id="main_app_tabs")
            main_tabs.__enter__()
            with gr.Tab("1. 📥 Import", id="tab_import") as tab1:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 1: Import)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(1))
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["zast-card"]):
                            gr.Markdown(f"### 🎥 1. Online Video (YouTube URL) {zast_tooltip('Download online video streams directly from YouTube with automatic resolution detection.')} <span class='zast-pill zast-pill-info'>Online Stream</span>")
                            gr.Markdown("Fetch video details, thumbnails, and highest available stream quality.")
                            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                            with gr.Row():
                                btn_check = gr.Button("🔍 Check URL", variant="secondary")
                                yt_resolution = gr.Dropdown(
                                    ["1080p"], label="Resolution", value="1080p",
                                    interactive=False, info="Click 'Check URL' to fetch resolutions"
                                )
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["zast-card"]):
                            gr.Markdown(f"### 📁 2. Local Media Upload {zast_tooltip('Upload local video (.mp4, .mkv, .mov) or audio (.mp3, .wav) for offline dubbing.')} <span class='zast-pill zast-pill-info'>Local File</span>")
                            gr.Markdown("Drag & drop your local video (`.mp4`, `.mov`, `.mkv`) or audio (`.mp3`, `.wav`).")
                            file_input = gr.File(
                                label="Upload local video or audio file", 
                                file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"],
                                elem_id="local_file_input"
                            )
                            local_title_input = gr.Textbox(
                                label="Video Title (Optional - for local files)",
                                placeholder="Give your video a descriptive title (used for SEO & metadata)...",
                                interactive=True,
                                info="Preliminary topic or title used to mine YouTube suggestions and seed the SEO package in Tab 2."
                            )

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### ⚙️ 3. Neural Voice Synthesis Engine & Import {zast_tooltip('VoxCPM 2 provides zero-shot voice cloning and speech synthesis across 30 languages.')} <span class='zast-pill zast-pill-success'>VoxCPM 2 Ready</span>")
                    with gr.Row():
                        tts_backend_dropdown = gr.Dropdown(
                            choices=list(available_tts_backends.keys()),
                            value=current_tts_backend,
                            label="🎙️ TTS Voice Synthesis Engine",
                            interactive=True,
                            scale=2,
                            info="Neural voice synthesis backend used for zero-shot voice cloning and speech generation."
                        )
                        btn_dl = gr.Button("📥 Import Video or Audio", variant="primary", scale=2, elem_id="btn_import_video")
                        btn_reset = gr.Button("🔄 New Project", variant="secondary", scale=1)
                    status_dl = gr.Textbox(label="Status", interactive=False)
                    with gr.Row(elem_classes=["zast-next-step-row"]):
                        btn_next_tab1 = gr.Button("➡️ Continue to Transcription & SEO (Tab 2)", variant="primary", elem_classes=["zast-next-step-btn"])
                
            with gr.Tab("2. 🎤 Transcription & SEO", id="tab_transcription") as tab2:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 2: Transcription & SEO)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(2))
                
                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🎙️ 1. Whisper AI Speech Recognition {zast_tooltip('WhisperX generates millisecond-accurate transcripts with phonetic word-level forced alignment.')} <span class='zast-pill zast-pill-primary'>WhisperX AI</span>")
                    with gr.Row():
                        lang_source = gr.Dropdown(
                            ["Auto", "French", "English", "Spanish", "German", "Italian", "Portuguese",
                             "Japanese", "Korean", "Chinese", "Russian", "Arabic", "Hindi",
                             "Dutch", "Polish", "Turkish", "Swedish", "Czech", "Romanian", "Hungarian"],
                            label="Source Language", value="Auto",
                            info="Source audio language. Leave on Auto or select manually for maximum speed."
                        )
                        model_size = gr.Dropdown(["base", "small", "medium", "large-v3"], label="Whisper Model", value="base", info="'base' is fast; 'large-v3' gives maximum transcription accuracy.")
                        llm_backend_dropdown = gr.Dropdown(
                            choices=list(available_llm_backends.keys()),
                            value=current_llm_backend,
                            label="LLM Model (Translation / SEO)",
                            interactive=True,
                            info="Local LLM used for syllable-fitted translation and Humanizer SEO writing."
                        )
                    with gr.Row():
                        btn_transcribe = gr.Button("▶️ Run Transcription", interactive=False, variant="primary", elem_id="btn_run_transcription")

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 📄 2. Subtitles Editor & SRT Import {zast_tooltip('Edit dialogue cues, clean conversational filler words, or import an external SRT.')} <span class='zast-pill zast-pill-info'>Interactive Grid</span>")
                    with gr.Row():
                        srt_file_input = gr.File(label="Upload existing SRT file (Optional)", file_types=[".srt"], elem_id="srt_file_input", scale=3)
                        btn_import_srt = gr.Button("📥 Import SRT", variant="secondary", elem_id="btn_import_srt", scale=1)
                    
                    transcription_status = gr.Textbox(label="Status", interactive=False)
                    transcription_df = gr.Dataframe(
                        headers=["Start", "End", "Text"],
                        label="Interactive Subtitles (Click row to seek left player, double-click cell to edit text)",
                        interactive=True,
                        wrap=True,
                        max_height=320,
                        elem_id="transcription_df"
                    )
                    _tab2_actions_tooltip = zast_tooltip("• Validate: Saves subtitles and unlocks Tab 3 (Translation).\n• Clean Fillers: Removes hesitation words (uh, um, like) with millisecond sync.\n• Export SRT: Refreshes the downloadable SRT file below.\n• Open Folder: Opens the output directory in Windows Explorer.")
                    gr.Markdown(f"**⚡ Quick Actions & Subtitle Export** {_tab2_actions_tooltip}")
                    with gr.Row():
                        btn_valid_transcription = gr.Button("Validate Transcription ✅", variant="primary", elem_classes=["zast-btn-validate"])
                        btn_clean_transcription = gr.Button("🧹 Clean Fillers & Oral Tics", variant="secondary")
                        btn_export_transcription = gr.Button("💾 Export / Refresh SRT", variant="secondary")
                        btn_open_output_tab2 = gr.Button("📂 Open Output Folder", variant="secondary")
                    gr.Markdown(
                        f"**⬇️ Direct Subtitle Download** {zast_tooltip('Populated automatically upon transcription completion. Click the file directly to download to your PC. If you edit text in the grid above, click Export / Refresh SRT to update the file.')}"
                    )
                    export_transcription_file = gr.File(label="Subtitles ready for download", interactive=False, show_label=False)

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🚀 3. YouTube SEO & Metadata Studio (Original Video Optimization) {zast_tooltip('Generates high-CTR titles, live YouTube Suggest queries, full chapters (00:00 to end), and clean descriptions without AI asterisks.')} <span class='zast-pill zast-pill-purple'>🛡️ Humanizer Certified</span>")
                    gr.Markdown(
                        "> 🛡️ **Humanizer Anti-AI Charter (35 Wikipedia AI Cleanup Patterns):** Same human-grade writing engine as Tab 7 (Blog Studio). Zero AI clichés ('Swiss Army knife', 'dive into', 'game changer', 'without further ado'), zero markdown asterisks `**`, live YouTube search suggest queries, and full timeline chapter landmarks (00:00 to end)."
                    )
                    with gr.Row():
                        seo_hashtag_pack = gr.Radio(
                            choices=[
                                "Pack 1: Subject & Specific",
                                "Pack 2: Review & Unboxing",
                                "Pack 3: Collector & Tech",
                                "Pack 4: Community & Trends"
                            ],
                            value="Pack 1: Subject & Specific",
                            label="🏷️ Recommended Hashtag Pack"
                        )
                        btn_generate_seo = gr.Button("✨ Generate Complete YouTube SEO Kit", variant="primary")
                    with gr.Row():
                        seo_title_out = gr.Textbox(label="Optimized YouTube Title (Front-Loaded & Natural Case)", lines=2, max_lines=4, interactive=True, buttons=["copy"])
                        seo_tags_out = gr.Textbox(label="YouTube Tags (Ready to paste in YouTube Studio)", lines=2, max_lines=6, interactive=True, buttons=["copy"])
                    with gr.Row():
                        seo_chapters_out = gr.Textbox(label="Full-Timeline YouTube Chapters (00:00 - End)", lines=8, max_lines=20, interactive=True, buttons=["copy"])
                        seo_desc_out = gr.Textbox(label="Full YouTube Description (Clean - No AI ** asterisks)", lines=12, max_lines=30, interactive=True, buttons=["copy"])
                    with gr.Row():
                        btn_apply_seo = gr.Button("📥 Apply as Original Video Metadata (Sync with Tabs 3 & 5)", variant="primary")
                    seo_status = gr.Markdown(value="")
                    with gr.Row(elem_classes=["zast-next-step-row"]):
                        btn_next_tab2 = gr.Button("➡️ Continue to Translation (Tab 3)", variant="primary", elem_classes=["zast-next-step-btn"])
        
            with gr.Tab("3. 🌍 Translation", id="tab_translation") as tab3:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 3: Translation)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(3))
                
                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🌐 1. Target Language & Video Metadata {zast_tooltip('Select your target translation language and preview localized title and description.')} <span class='zast-pill zast-pill-primary'>30 Languages</span>")
                    with gr.Row():
                        lang_target = gr.Dropdown(INITIAL_VALID_LANGS, label="Target Language", value=INITIAL_LANG_VALUE, scale=3, info="Target language for translation and dubbing, filtered by engine compatibility.")
                        btn_import_metadata_single = gr.Button("⬇️ Import from URL", variant="secondary", visible=False, scale=1)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            original_title_input = gr.Textbox(label="Original Video Title (From SEO Studio / YouTube)", placeholder="Title...")
                            original_desc_input = gr.Textbox(label="Original Video Description", placeholder="Description...", lines=4, max_lines=15)
                        with gr.Column(scale=1):
                            translated_title_input = gr.Textbox(label="Translated Video Title", placeholder="Translated Title...", buttons=["copy"])
                            translated_desc_input = gr.Textbox(label="Translated Video Description", placeholder="Translated Description...", lines=4, max_lines=15, buttons=["copy"])

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🎯 2. Translation & Syllable Timing Fitting {zast_tooltip('Fitted translation calibrates text length to the speaker exact speaking rate (CPS) to prevent speech overflows.')} <span class='zast-pill zast-pill-warning'>Calibrated CPS</span>")
                    gr.Markdown(
                        "**Fitted Translation (Recommended):** Rephrases sentences with LLM so translated syllables match the original speaker's exact timeframe, preventing speech overflows."
                    )
                    with gr.Row():
                        btn_translate = gr.Button("▶️ Run Translation", interactive=False, variant="primary", elem_id="btn_run_translation")
                    translation_status = gr.Textbox(label="Status", interactive=False)
                    translation_df = gr.Dataframe(
                        headers=["Start", "End", "Original", "Translation", "Fitted"],
                        label="Edit Translation",
                        interactive=True,
                        wrap=True,
                        max_height=350,
                        elem_id="translation_df"
                    )
                    _tab3_actions_tooltip = zast_tooltip("• Validate: Saves translation and unlocks Tab 4 (Dubbing).\n• Export Translation SRT: Downloads the full literal translated subtitles.\n• Export Fitted SRT: Downloads the concise dubbing-ready subtitles calibrated for vocal duration.\n• Open Folder: Opens the output directory in Windows Explorer.")
                    gr.Markdown(f"**⚡ Validation & Subtitle Export** {_tab3_actions_tooltip}")
                    with gr.Row():
                        btn_valid_translation = gr.Button("Validate Translation ✅", variant="primary", elem_classes=["zast-btn-validate"])
                        btn_export_translation = gr.Button("💾 Export Full Translation SRT", variant="secondary")
                        btn_export_fitted = gr.Button("💾 Export Fitted SRT (Dubbing)", variant="secondary")
                        btn_open_output_tab3 = gr.Button("📂 Open Output Folder", variant="secondary")
                    gr.Markdown(
                        f"**⬇️ Direct Translated Subtitle Download** {zast_tooltip('Populated automatically upon translation completion. Click the file directly to download to your PC. If you edit text in the grid above, click Export to refresh the file.')}"
                    )
                    export_translation_file = gr.File(label="Translated subtitles ready for download", interactive=False, show_label=False)
                    with gr.Row(elem_classes=["zast-next-step-row"]):
                        btn_next_tab3 = gr.Button("➡️ Continue to Dubbing & Export (Tab 4)", variant="primary", elem_classes=["zast-next-step-btn"])
                
            with gr.Tab("4. 🎬 Dubbing & Export", id="tab_dubbing") as tab4:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 4: Dubbing & Export)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(4))
                
                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🗣️ 1. Neural Voice Synthesis & Vocal Timbre {zast_tooltip('Replicate the original speaker voice timbre using zero-shot cloning, or use a persistent default voice.')} <span class='zast-pill zast-pill-purple'>Zero-Shot Cloning</span>")
                    with gr.Row():
                        voice_mode = gr.Radio(
                            ["Default voice", "Clone from original", "Clone from file"], 
                            label="Voice Mode", 
                            value="Default voice",
                            info="'Clone from original' isolates vocals. 'Clone from file' requires a clean 10-30s speech sample."
                        )
                        dubbing_text_source = gr.Radio(
                            ["Fitted Translation", "Normal Translation"],
                            label="Dubbing Text Source",
                            value="Fitted Translation",
                            info="Fitted translation is recommended to match speech duration with original cuts."
                        )
                    dubbing_warning_box = gr.Markdown(value="", visible=False)
                    
                    with gr.Row():
                        default_voice_gender = gr.Radio(
                            ["Man", "Woman"],
                            label="Default Voice Gender",
                            value="Woman",
                            visible=True
                        )
                        voice_file = gr.File(label="Voice sample file (WAV/MP3, 10-30s of clear speech)", visible=False)
                    
                    never_cut_mode = gr.Checkbox(
                        label="🔊 Never Cut Vocal",
                        value=False,
                        info="All text will be spoken in full. Prevents rushing words."
                    )
                    never_cut_warning = gr.Markdown(value="", visible=False)

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🎚️ 2. Dialogue Lines & Segment-Level Tuning {zast_tooltip('Click any dialogue row to seek the video player, adjust start/end times, or re-synthesize individual sentences.')} <span class='zast-pill zast-pill-info'>Fine-Tuning</span>")
                    dubbing_segments_df = gr.Dataframe(
                        headers=["Index", "Start", "End", "Text to Speak", "Status"],
                        label="Segments List (Click a row to edit text or timing)",
                        interactive=False,
                        wrap=True,
                        max_height=260,
                        elem_id="dubbing_segments_df"
                    )
                    
                    # Segment Editor Card
                    with gr.Column(visible=False, variant="panel") as segment_editor_card:
                        gr.Markdown("#### 🛠️ Edit Selected Segment")
                        with gr.Row():
                            edit_seg_idx = gr.Number(label="Segment Index", interactive=False, precision=0)
                            edit_seg_text = gr.Textbox(label="Text to Speak", lines=2)
                        with gr.Row():
                            edit_start_min = gr.Number(label="Start Minute", precision=0)
                            edit_start_sec = gr.Number(label="Start Second", precision=2)
                            edit_end_min = gr.Number(label="End Minute", precision=0)
                            edit_end_sec = gr.Number(label="End Second", precision=2)
                        with gr.Row():
                            btn_regen_seg = gr.Button("🔄 Regenerate Segment Audio", variant="primary")
                            edit_seg_audio = gr.Audio(label="Segment Audio Preview", interactive=False)
                        edit_seg_status = gr.Markdown("")

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🎬 3. Master Assembly & Video Render {zast_tooltip('Demucs separates background music and noise, mixes the synthesized voice, and renders the 1080p master MP4.')} <span class='zast-pill zast-pill-success'>Demucs Separation</span>")
                    with gr.Row():
                        btn_synth = gr.Button("🎬 Assemble Final Video & Audio", interactive=False, variant="primary", elem_id="btn_assemble_video")
                    synth_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            final_video_out = gr.Video(label="Final Dubbed Video (1080p)", height=320)
                            with gr.Row():
                                btn_export_video = gr.Button("🎬 Download Dubbed Video (MP4)", variant="primary")
                            export_video_file = gr.File(label="Download Dubbed Video (MP4)", interactive=False)
                        with gr.Column(scale=1):
                            final_audio_out = gr.Audio(label="Mixed Dubbed Audio Track")
                            with gr.Row():
                                btn_export_audio = gr.Button("🎵 Download Audio Track (WAV)", variant="secondary")
                            export_audio_file = gr.File(label="Download Audio Track (WAV)", interactive=False)
                    with gr.Row():
                        btn_open_output_tab4 = gr.Button("📂 Open Output Folder", variant="secondary")
                    with gr.Row(elem_classes=["zast-next-step-row"]):
                        btn_next_tab4_to_shorts = gr.Button("📱 Create Viral Shorts (Tab 6)", variant="secondary", elem_classes=["zast-next-step-btn"])
                        btn_next_tab4_to_blog = gr.Button("📝 Generate Blog Post (Tab 7)", variant="primary", elem_classes=["zast-next-step-btn"])
        
            with gr.Tab("5. ⚡ Bulk Mode", id="tab_bulk") as tab5:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 5: Bulk Mode)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(5))
                
                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 🌍 1. Multi-Language Batch Configuration {zast_tooltip('Batch translate and dub into multiple target languages simultaneously with a single click.')} <span class='zast-pill zast-pill-primary'>Batch Engine</span>")
                    bulk_target_langs = gr.Dropdown(
                        INITIAL_VALID_LANGS, 
                        label="Target Languages (Multi-select)", 
                        multiselect=True,
                        info="Select all languages you want to translate and dub simultaneously."
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=4):
                            bulk_title_input = gr.Textbox(label="Original Video Title (Optional)", placeholder="Title...")
                            bulk_desc_input = gr.Textbox(label="Original Video Description (Optional)", placeholder="Description...", lines=4, max_lines=15)
                        with gr.Column(scale=1):
                            btn_import_metadata = gr.Button("⬇️ Import from URL", variant="secondary", visible=False)
                    
                    with gr.Row():
                        bulk_voice_mode = gr.Radio(
                            ["Default voice", "Clone from original", "Clone from file"], 
                            label="Voice Mode", 
                            value="Default voice"
                        )
                        bulk_default_voice_gender = gr.Radio(
                            ["Man", "Woman"],
                            label="Default Voice Gender",
                            value="Woman",
                            visible=True
                        )
                        bulk_output_type = gr.Radio(
                            ["Video + Audio", "Audio Only", "Subtitles & Metadata Only"],
                            label="Output Generation",
                            value="Video + Audio",
                            info="'Video + Audio' renders MP4. 'Audio Only' outputs WAV. 'Subtitles & Metadata Only' outputs localized SRTs, titles, and descriptions in seconds without TTS."
                        )
                    bulk_voice_file = gr.File(label="Voice sample file (WAV/MP3, 10-30s of clear speech)", visible=False)

                with gr.Group(elem_classes=["zast-card"]):
                    gr.Markdown(f"### 📦 2. Batch Execution & Output {zast_tooltip('Launch the automated multi-language pipeline and download all dubbed videos, audios, and SRT packages in one ZIP.')} <span class='zast-pill zast-pill-warning'>Automated Pipeline</span>")
                    with gr.Row():
                        bulk_never_cut_mode = gr.Checkbox(
                            label="🔊 Never Cut Vocal",
                            value=False,
                            info="All text will be spoken in full."
                        )
                        bulk_generate_shorts = gr.Checkbox(
                            label="📱 Generate 3 Viral Shorts (9:16) per language",
                            value=False,
                            info="Automatically detects top moments and renders vertical 9:16 clips for every target language."
                        )
                    bulk_never_cut_warning = gr.Markdown(value="", visible=False)
                    
                    with gr.Row():
                        btn_bulk_run = gr.Button("🚀 Run Bulk Process", interactive=False, variant="primary")
                    bulk_status_output = gr.Textbox(label="Status", interactive=False)
                    bulk_files_output = gr.File(label="Generated Files Output", file_count="multiple")
                    with gr.Row():
                        btn_open_output = gr.Button("📂 Open Output Folder in Windows Explorer", variant="secondary")
                    bulk_metadata_output = gr.Markdown(label="Translated Metadata", height=400)
        
            with gr.Tab("6. 📱 Viral Shorts", id="tab_shorts") as tab6:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 6: Viral Shorts)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(6))
                gr.Markdown(f"### 📱 Viral Shorts Studio & 9:16 Auto-Clipper (YouTube Shorts, TikTok, Instagram Reels) {zast_tooltip('Detects viral moments, snaps to camera cuts via PySceneDetect, crops to vertical 9:16, and burns animated karaoke captions.')} <span class='zast-pill zast-pill-cyan'>9:16 Vertical Crop</span>")
                gr.Markdown(
                    "> 💡 **Viral Shorts Studio Workflow & Pro Tips:**\n"
                    "> 1. **Track & Crop**: Choose *Original Video* or *Dubbed Video*. *Stacked Blur* keeps the full landscape video visible with an aesthetic blurred top/bottom. *Center Crop* cuts directly to vertical 9:16.\n"
                    "> 2. **Subtitle Style**: Choose from 4 TikTok Neon Karaoke styles (Yellow, Mint, Cyan, Pink) with animated word-by-word highlights and 280px safe margins (won't overlap TikTok/Reels UI buttons).\n"
                    "> 3. **AI Moment Detection**: Click **✨ 1. Detect Top Viral Moments**. The AI finds peak engagement moments, and **PySceneDetect 0.7.1** snaps timestamps to nearby camera cuts to avoid chopping sentences mid-action.\n"
                    "> 4. **Preview & Fine-Tune**: Click **▶️ Preview Short in Player** on any card to review the scene in the left player. Edit dialogue text or timestamps if needed. Uncheck any clip you want to skip.\n"
                    "> 5. **Render & Export**: Click **🎬 3. Render Selected 9:16 Shorts** to generate vertical 1080x1920 MP4s and download individual files or the `.zip` pack."
                )

                with gr.Row():
                    shorts_source_choice = gr.Radio(
                        ["Original Video", "Dubbed Video (Tab 4 / Bulk)"],
                        label="🎙️ Video / Audio Source Track",
                        value="Original Video",
                        info="Original Video uses the source audio and language. Dubbed Video uses the synthesized translated track (Tab 4 or Bulk)."
                    )
                    shorts_dubbed_lang = gr.Dropdown(
                        label="🌐 Dubbed Language Track",
                        choices=[],
                        value=None,
                        visible=False,
                        info="Choose which dubbed language track to inspect, preview and render."
                    )
                    shorts_crop_style = gr.Radio(
                        ["Stacked Blur (Aesthetic)", "Center Crop (9:16)"],
                        label="📐 9:16 Crop Style",
                        value="Stacked Blur (Aesthetic)",
                        info="Stacked Blur centers the 16:9 video with aesthetic blurred top/bottom (ideal for landscape footage). Center Crop cuts directly to 1080x1920."
                    )
                    shorts_subtitle_style = gr.Radio(
                        [
                            "🔥 TikTok Karaoke Neon Yellow (Viral)",
                            "⚡ TikTok Karaoke Mint Green (Neon)",
                            "💎 TikTok Karaoke Cyber Cyan (Pop)",
                            "🌸 TikTok Karaoke Punchy Pink",
                            "⚪ Minimalist White (Clean Static)",
                            "🚫 No Subtitles"
                        ],
                        label="🎨 Subtitle Style (Dynamic Word-by-Word Karaoke)",
                        value="🔥 TikTok Karaoke Neon Yellow (Viral)",
                        info="Dynamic word-by-word neon karaoke synced to speech with 280px bottom safe margin (won't overlap TikTok/Reels overlay buttons)."
                    )

                with gr.Row():
                    shorts_num_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="🔢 Number of Viral Shorts to Detect (1 to 5)",
                        info="Number of viral highlights to detect from the video transcript (1 to 5 clips, default 3)."
                    )

                with gr.Row():
                    _detect_tooltip = zast_tooltip("Scans transcript with AI to identify viral hooks and snaps timecodes to camera cuts with PySceneDetect.")
                    btn_detect_shorts = gr.Button(
                        "✨ 1. Detect Top Viral Moments (AI + PySceneDetect)", 
                        variant="secondary"
                    )
                gr.Markdown(f"<small>💡 *Prerequisite: A transcribed video from Tab 2. PySceneDetect scans camera cuts to prevent chopping sentences mid-scene.* {_detect_tooltip}</small>")

                shorts_status = gr.Markdown(value="")

                _table_tooltip = zast_tooltip("Click any row to seek the left video player. Start and end points are automatically aligned with visual scene cuts.")
                gr.Markdown(f"#### 📊 Detected Viral Highlights (Click row to seek video player) {_table_tooltip}")
                shorts_table = gr.Dataframe(
                    headers=["#", "Hook / Topic", "Start (s)", "End (s)", "Duration", "Score", "Why it works"],
                    label="Detected Viral Moments (Click row to seek video player)",
                    interactive=True,
                    wrap=True,
                    max_height=200,
                    elem_id="shorts_table"
                )
                gr.Markdown("<small>💡 *Score (0-100) rates engagement potential based on speech energy and hook strength. 'Why it works' details the psychological retention trigger.*</small>")

                _cards_tooltip = zast_tooltip("Preview each scene in the player, adjust timecodes, edit burned karaoke subtitles, or uncheck clips to exclude from rendering.")
                gr.Markdown(f"#### 🛠️ 2. Fine-Tuning, Scene Preview & Subtitles Editor {_cards_tooltip}")
                gr.Markdown("Inspect each short: preview scenes in the left player, fine-tune timecodes, uncheck clips you don't want to render, and customize subtitles before burn-in.")

                short_cards = []
                short_includes = []
                short_titles = []
                short_starts = []
                short_ends = []
                short_previews = []
                short_subtitles = []

                for i in range(1, 6):
                    with gr.Group(visible=False, elem_classes=["zast-short-card"]) as scard:
                        with gr.Row():
                            inc = gr.Checkbox(label=f"🎬 Include Short #{i} in Render", value=True, scale=3, info="Check to include in final render; uncheck to skip.")
                            title = gr.Textbox(label=f"Hook / Title (Short #{i})", scale=5, info="Viral hook caption and social media title.")
                            st = gr.Number(label="Start (s)", scale=1, info="Start timestamp in seconds.")
                            en = gr.Number(label="End (s)", scale=1, info="End timestamp in seconds.")
                            prev_btn = gr.Button(f"▶️ Preview Short #{i} in Player", variant="secondary", scale=2)
                        subs = gr.Textbox(
                            label=f"📝 Burned Subtitles (Short #{i} - TikTok Karaoke - Edit text freely)",
                            lines=3,
                            placeholder="Subtitles that will be dynamically burned into the vertical video...",
                            info="Editable dialogue text burned as animated neon karaoke captions with 280px bottom safe margin."
                        )
                        short_cards.append(scard)
                        short_includes.append(inc)
                        short_titles.append(title)
                        short_starts.append(st)
                        short_ends.append(en)
                        short_previews.append(prev_btn)
                        short_subtitles.append(subs)

                _render_tooltip = zast_tooltip("Exports 1080x1920 vertical MP4s using GPU acceleration (NVENC/VAAPI) with burned animated karaoke captions.")
                gr.Markdown(f"#### 🎬 3. Video Rendering & Vertical 9:16 Export {_render_tooltip}")
                with gr.Row():
                    btn_render_shorts = gr.Button("🎬 3. Render Selected 9:16 Shorts (1080x1920 Vertical)", variant="primary", interactive=False)
                gr.Markdown("<small>💡 *Rendering utilizes GPU NVENC when available. Each vertical short is rendered at 1080x1920 (30fps) with dynamic neon karaoke highlights.*</small>")

                _gallery_tooltip = zast_tooltip("Download individual vertical MP4s or grab the all-in-one ZIP bundle containing all videos and subtitles.")
                gr.Markdown(f"#### 🎬 Generated Shorts Gallery {_gallery_tooltip}")
                with gr.Row(elem_id="shorts_gallery_row"):
                    short_cols = []
                    short_videos = []
                    short_files = []
                    for i in range(1, 6):
                        with gr.Column(scale=1, visible=(i <= 3)) as col:
                            gr.Markdown(f"**📱 Short #{i}**")
                            vid = gr.Video(label=f"Short #{i} Preview", height=380)
                            fil = gr.File(label=f"Download Short #{i} (1080x1920 MP4)")
                            short_cols.append(col)
                            short_videos.append(vid)
                            short_files.append(fil)

                # Aliases for backward compatibility
                short_video_1, short_video_2, short_video_3, short_video_4, short_video_5 = short_videos
                short_file_1, short_file_2, short_file_3, short_file_4, short_file_5 = short_files

                with gr.Row():
                    btn_open_shorts_folder = gr.Button("📂 Open Shorts Folder in Windows Explorer", variant="secondary")
                    shorts_zip_output = gr.File(label="📦 Download Complete Shorts Pack (.ZIP: All MP4s + Subtitles)")
                gr.Markdown("<small>💡 *The Complete Shorts Pack (.ZIP) includes all rendered vertical MP4s, individual SRT subtitle files, and metadata summary.*</small>")
        
            with gr.Tab("7. 📝 Blog Studio", id="tab_blog") as tab7:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 7: SEO Blog & Thumbnails)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(7))
                with gr.Group(elem_classes=["zast-studio-card"]):
                    gr.Markdown(f"### 📝 SEO Blog Post & Content Studio (WordPress Ready) {zast_tooltip('Generates complete high-ranking articles from transcriptions with 35 WikiProject anti-AI rules, HD keyframes, and WordPress Gutenberg blocks.')} <span class='zast-pill zast-pill-purple'>🛡️ 35-Pattern Humanizer</span>")
                    gr.Markdown(
                        "> ℹ️ **Prerequisite:** Please import a video (**Tab 1**) and run Transcription (**Tab 2**) or import an SRT file first.\n\n"
                        "Turn any video and its subtitle transcript into a **complete, human-sounding, high-ranking SEO blog post (without robotic AI clichés)**, "
                        "with calibrated SEO metadata, HD video keyframes, and copy-paste ready Gutenberg block HTML and Markdown."
                    )

                with gr.Group(elem_classes=["zast-studio-card"]):
                    gr.Markdown(
                        "> 🛡️ **Humanizer Anti-AI Charter & Live Keyword Discovery:**\n"
                        "> - **35 Wikipedia AI Cleanup Rules**: Strips robotic clichés (*'In this article'*, *'It is crucial to note'*, *'game-changer'*, *'couteau suisse'*, shallow participle transitions) for authentic human burstiness and search engine trust.\n"
                        "> - **Real-Time Google Autocomplete**: Queries Google Suggest in real-time (0 API keys needed) to discover high-volume search queries and naturally integrate them into your H1 title and H2/H3 subheadings.\n"
                        "> - **Sync Rules**: Click `🔄 Sync Anti-AI Rules (GitHub)` to update rule definitions directly from GitHub without restarting."
                    )
                    gr.Markdown("#### ⚙️ 1. Article Configuration & Style")
                    with gr.Row():
                        blog_target_lang = gr.Dropdown(
                            choices=["French", "English", "Spanish", "German", "Italian", "Portuguese", "Japanese", "Chinese", "Russian", "Arabic", "Dutch", "Polish"],
                            label="🌍 Target Language",
                            value="French",
                            scale=1,
                            info="Language in which the full SEO article and metadata will be written."
                        )
                        blog_style = gr.Dropdown(
                            choices=[
                                "Step-by-Step Tutorial (How-To Guide)",
                                "Expert & Technical Deep-Dive",
                                "Storytelling & Case Study",
                                "Journalistic & Objective Review",
                                "High-Converting Copywriting",
                                "Accessible Beginner's Guide"
                            ],
                            label="🎨 Writing Style & Tone",
                            value="Step-by-Step Tutorial (How-To Guide)",
                            scale=2,
                            info="Editorial voice adapted to your audience (Tutorial, Technical Deep-Dive, Storytelling, Review, etc.)."
                        )
                        blog_length = gr.Dropdown(
                            choices=[
                                "Short (600 - 800 words)",
                                "Medium (1000 - 1500 words)",
                                "Long (1800 - 2500 words)"
                            ],
                            label="📏 Target Length",
                            value="Medium (1000 - 1500 words)",
                            scale=1,
                            info="Target article word count. Medium (~1200w) is recommended for Google ranking."
                        )
                    with gr.Row():
                        with gr.Column(scale=2):
                            blog_include_meta = gr.Checkbox(
                                label="🎯 Generate Complete SEO Kit (H1 Title, Meta Description, URL Slug, Keywords)",
                                value=True,
                                info="Generates high-CTR H1 title, clean URL slug, 155-char Google snippet, and LSI keywords."
                            )
                        with gr.Column(scale=2):
                            blog_extract_images = gr.Checkbox(
                                label="🖼️ Extract Milestone HD Video Keyframes (with SEO ALT Tags & Captions)",
                                value=True,
                                info="Captures crisp milestone HD video keyframes at scene and speech boundaries."
                            )
                        with gr.Column(scale=2):
                            blog_num_images = gr.Slider(
                                minimum=2,
                                maximum=8,
                                step=1,
                                value=6,
                                label="📸 Number of Keyframes",
                                info="Number of HD video screenshots to capture and format with SEO ALT tags (2 to 8)."
                            )
                    with gr.Row():
                        with gr.Column(scale=3):
                            blog_keyframe_res = gr.Dropdown(
                                choices=[
                                    "1080p (Full HD - 1920x1080) [Recommandé Articles & Google SEO]",
                                    "1440p / 2K (2560x1440) [Ultra-Net / Lisibilité Code & Terminal]",
                                    "720p (HD - 1280x720) [Léger]",
                                    "Source Native"
                                ],
                                label="📐 Résolution des Keyframes Vidéo",
                                value="1080p (Full HD - 1920x1080) [Recommandé Articles & Google SEO]",
                                info="Résolution d'export des captures d'écran. 1080p garantit une netteté totale et dépasse les exigences Google Discover (>1200px)."
                            )
                        with gr.Column(scale=3):
                            blog_enhance_text = gr.Checkbox(
                                label="🔍 Amélioration Netteté Texte & Code (Lanczos + Unsharp Mask)",
                                value=True,
                                info="Rehausse la netteté et les contours du texte, code, fenêtres de terminal et diapositives pour une lisibilité parfaite."
                            )

                with gr.Group(elem_classes=["zast-studio-card"]):
                    _actions_tooltip = zast_tooltip("• Generate: Produces complete article, metadata, and keyframes.\n• Extract Only: Grabs keyframe images without LLM generation.\n• Sync: Fetches latest Wikipedia AI Cleanup rules from GitHub.")
                    gr.Markdown(f"#### 🚀 2. Generation Actions {_actions_tooltip}")
                    with gr.Row():
                        btn_generate_blog = gr.Button("✨ Generate SEO Blog Article & WordPress Kit", variant="primary", scale=3)
                        btn_extract_images_only = gr.Button("📸 Extract Video Keyframes Only", variant="secondary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_sync_humanizer = gr.Button("🔄 Sync Anti-AI Rules (GitHub)", variant="secondary", size="sm")
                        with gr.Column(scale=3):
                            humanizer_sync_status = gr.Markdown(value="*🛡️ Anti-AI style correction powered by [Humanizer (blader/humanizer)](https://github.com/blader/humanizer) & WikiProject AI Cleanup.*")
                    gr.Markdown("<small>💡 *'Generate' produces the complete WordPress Gutenberg article, SEO metadata, and milestone keyframes. 'Extract Video Keyframes Only' captures screenshots without LLM text generation.*</small>")
                    blog_status = gr.Markdown(value="")

                with gr.Tabs():
                    with gr.Tab("📄 1. Article & SEO Metadata"):
                        with gr.Group(elem_classes=["zast-studio-card"]):
                            gr.Markdown(
                                "> 💡 **WordPress Publishing Guide:**\n"
                                "> 1. Click **Copy** on the **Gutenberg Block HTML** box below.\n"
                                "> 2. In your WordPress post editor, press `Ctrl+Shift+Alt+M` (or click `⋮` > **Code Editor**).\n"
                                "> 3. Paste the Gutenberg code directly, then press `Ctrl+Shift+Alt+M` to return to **Visual Editor**.\n"
                                "> 4. Fill in the **Focus Keyword** and **Meta Description** in your SEO plugin (Yoast, Rank Math, SEOPress)."
                            )
                            gr.Markdown("#### 🎯 Calibrated SEO Metadata")
                            with gr.Row():
                                blog_title_out = gr.Textbox(label="H1 Article Title (SEO)", lines=2, interactive=True, buttons=["copy"], scale=3, info="High-CTR title crafted with live Google autocomplete query.")
                                blog_slug_out = gr.Textbox(label="WordPress URL Slug (Permalink)", lines=2, interactive=True, buttons=["copy"], scale=2, info="Clean, hyphenated URL permalink for WordPress.")
                            with gr.Row():
                                with gr.Column(scale=3):
                                    blog_meta_desc_out = gr.Textbox(label="Meta Description (Google Snippet)", lines=3, interactive=True, buttons=["copy"], info="Calibrated to 145-160 characters for high click-through rate in search engine snippets.")
                                    blog_char_count = gr.Markdown(value="")
                                with gr.Column(scale=2):
                                    blog_focus_kw_out = gr.Textbox(label="Focus Keyword", interactive=True, buttons=["copy"], info="Primary search query to configure in your SEO plugin.")
                                    blog_sec_kws_out = gr.Textbox(label="Secondary & LSI Keywords", lines=2, interactive=True, buttons=["copy"], info="Latent semantic indexing search terms woven into subheadings.")

                        with gr.Group(elem_classes=["zast-studio-card"]):
                            gr.Markdown("#### 📋 Complete Article (WordPress Gutenberg & Markdown)")
                            with gr.Tabs():
                                with gr.Tab("📋 Gutenberg Block HTML (1-Click Paste into WordPress Code Editor)"):
                                    blog_gutenberg_out = gr.Textbox(
                                        label="WordPress Gutenberg Block HTML",
                                        lines=20,
                                        max_lines=45,
                                        interactive=True,
                                        buttons=["copy"],
                                        elem_id="blog_gutenberg_code",
                                        info="Native Gutenberg block markup (<!-- wp:heading -->, <!-- wp:paragraph -->, <!-- wp:list -->). In WordPress, press Ctrl+Shift+Alt+M, paste, and press Ctrl+Shift+Alt+M again to edit visually."
                                    )
                                with gr.Tab("📝 Standard Markdown"):
                                    blog_markdown_out = gr.Textbox(
                                        label="Markdown Content",
                                        lines=20,
                                        max_lines=45,
                                        interactive=True,
                                        buttons=["copy"],
                                        info="Clean Markdown formatted with H1/H2/H3 headers, bullet points, and image placeholders. Ready for Ghost, Medium, Substack, Hugo, Jekyll, or Notion."
                                    )
                            gr.Markdown("<small>💡 *Tip: Click the 'copy' icon in the upper-right corner of either box to copy all formatted text to your clipboard in 1 click.*</small>")

                    with gr.Tab("📸 2. Extracted Video Keyframes"):
                        gr.Markdown(
                            "> 💡 **Keyframes & Image SEO Guide:**\n"
                            "> - **Milestone Keyframes**: Extracted at scene transitions and speech peaks using FFmpeg. Each image includes a contextual ALT tag and caption.\n"
                            "> - **Custom Image Override (Drag & Drop)**: You can **drag and drop your own custom screenshot or image** directly into any of the 6 boxes below to replace an extracted frame. Replaced images will be automatically packaged into the WordPress ZIP export!\n"
                            "> - **WordPress Export Pack**: Click `Download Complete WordPress Pack (.ZIP)` to get all keyframe images, Gutenberg HTML, and Markdown bundled together.\n"
                            "> - **AI Prompts**: Pre-formatted prompts ready to copy into Midjourney, DALL-E 3, or FLUX to create custom illustrative artwork."
                        )
                        _keyframes_tooltip = zast_tooltip("High-resolution frames captured at key scene changes. Drag and drop your own screenshots to override any frame.")
                        gr.Markdown(f"#### 📸 Milestone HD Video Keyframes (Extracted from Video) {_keyframes_tooltip}")
                        with gr.Row():
                            with gr.Column(scale=1):
                                blog_img_1 = gr.Image(label="Keyframe / Thumbnail #1", height=200, interactive=True)
                                blog_cap_1 = gr.Markdown(value="")
                            with gr.Column(scale=1):
                                blog_img_2 = gr.Image(label="Keyframe #2", height=200, interactive=True)
                                blog_cap_2 = gr.Markdown(value="")
                            with gr.Column(scale=1):
                                blog_img_3 = gr.Image(label="Keyframe #3", height=200, interactive=True)
                                blog_cap_3 = gr.Markdown(value="")
                        with gr.Row():
                            with gr.Column(scale=1):
                                blog_img_4 = gr.Image(label="Keyframe #4", height=200, interactive=True)
                                blog_cap_4 = gr.Markdown(value="")
                            with gr.Column(scale=1):
                                blog_img_5 = gr.Image(label="Keyframe #5", height=200, interactive=True)
                                blog_cap_5 = gr.Markdown(value="")
                            with gr.Column(scale=1):
                                blog_img_6 = gr.Image(label="Keyframe #6", height=200, interactive=True)
                                blog_cap_6 = gr.Markdown(value="")
                        gr.Markdown("<small>💡 *Tip: Drag & drop your own screenshots or diagrams directly into any image box above to override the automated capture. Replaced images are automatically bundled into the WordPress export pack.*</small>")

                        with gr.Row():
                            btn_open_blog_folder = gr.Button("📂 Open Images Folder in Windows Explorer", variant="secondary")
                            blog_zip_file = gr.File(label="📦 Download Complete WordPress Pack (.ZIP: Markdown + HTML + Images)")
                        gr.Markdown("<small>💡 *The Complete WordPress Pack (.ZIP) includes all HD images (with SEO-optimized filenames), the full Gutenberg HTML block code, and the clean Markdown article.*</small>")

                        _prompts_tooltip = zast_tooltip("Pre-engineered AI prompts for creating custom cover illustrations and concept diagrams in Midjourney, FLUX, or DALL-E.")
                        gr.Markdown(f"#### 🎨 AI Image Generation Prompts (Midjourney / DALL-E / FLUX) {_prompts_tooltip}")
                        gr.Markdown(
                            "> 💡 **How to Use AI Image Prompts:**\n"
                            "> - **Midjourney**: Copy the prompt, open Discord, and type `/imagine prompt:` followed by the copied text.\n"
                            "> - **FLUX.1-schnell**: Paste directly into **Sub-Tab 3 (Thumbnail Studio)** to render a 4K visual in ~2 seconds.\n"
                            "> - **DALL-E 3**: Paste directly into ChatGPT Plus / OpenAI DALL-E 3.\n"
                            "> All prompts are pre-engineered with professional 3D octane render, volumetric lighting, and aspect ratio tags."
                        )
                        blog_img_prompts_out = gr.Markdown(value="")

                    with gr.Tab("⚡ 3. YouTube Thumbnail Studio (FLUX.1-schnell)"):
                        gr.Markdown(f"### ⚡ AI YouTube Thumbnail Studio (FLUX.1-schnell) {zast_tooltip('12B flow transformer generating high-CTR 4K YouTube thumbnails in 4 steps (~2s) with custom 3D typography.')} <span class='zast-pill zast-pill-cyan'>12B Flow Transformer</span>")
                        gr.Markdown(
                            "> 💡 **FLUX.1-schnell Thumbnail Studio & A/B Testing Guide:**\n"
                            "> - **⚡ Ultra-Fast 12B Flow Transformer**: Distilled 4-step generation (~2 seconds on RTX GPUs) producing ultra-clean 4K visuals.\n"
                            "> - **🔤 3D Typography Syntax**: Put any title text in **single quotes** (e.g. `'AI TUTORIAL'` or `'HERMES AGENT'`) in your prompt. FLUX renders crisp, readable, styled 3D typography!\n"
                            "> - **🎨 5 Curated Visual Presets**: Choose *Viral High-CTR* for YouTube thumbnails, *3D Isometric* for tech/software, or *Cinematic Studio* for premium courses.\n"
                            "> - **🧪 1-Click A/B Testing**: Click `🧪 Generate 3 A/B Test Variants` to get 3 diverse angles ready for YouTube Studio's *'Test & Compare'* feature.\n"
                            "> - **🛡️ 100% Anti-AI Sanitizer**: All images are automatically stripped of AI metadata, prompts, C2PA manifests, and EXIF tags before export.\n"
                            "> - **⭐ 1-Click Blog Cover**: Click `⭐ Use for Blog Thumbnail (#1)` under any variant to set it as your featured article header in WordPress!"
                        )

                        _flux_mgmt_tooltip = zast_tooltip("• Install: Downloads the ~12 GB model weights into local models/flux cache.\n• Free VRAM: Unloads model from GPU back to RAM so other AI engines have full GPU memory.\n• Free Disk Space: Deletes weights folder from disk to recover ~12 GB storage.")
                        gr.Markdown(f"#### 💾 Model Management & VRAM Control {_flux_mgmt_tooltip}")
                        with gr.Group():
                            flux_model_status = gr.Markdown(
                                value=flux_studio.get_model_status()["status_text"]
                            )
                            with gr.Row():
                                btn_install_flux = gr.Button("📥 Install / Download FLUX.1-schnell (~12 GB)", variant="secondary", scale=2)
                                btn_unload_flux = gr.Button("🧹 Free VRAM (Unload)", variant="secondary", scale=1)
                                btn_delete_flux = gr.Button("🗑️ Free Disk Space (Delete Weights)", variant="stop", scale=1)
                            gr.Markdown("<small>💡 *FLUX.1-schnell requires ~12 GB VRAM on GPU. Click 'Free VRAM (Unload)' when finished so Whisper or TTS voice cloning have full GPU headroom.*</small>")

                        with gr.Row():
                            with gr.Column(scale=5):
                                with gr.Group():
                                    flux_style_preset = gr.Dropdown(
                                        choices=[
                                            "YouTube Viral High-CTR",
                                            "3D Isometric & Tech Glow",
                                            "Cyberpunk & Bold Neon",
                                            "Minimalist & Clean SaaS",
                                            "Photorealistic Studio Shot"
                                        ],
                                        value="YouTube Viral High-CTR",
                                        label="🎨 Thumbnail Visual Style Preset",
                                        info="Curated aesthetic presets tuned for high click-through rate (CTR)."
                                    )
                                    flux_prompt = gr.Textbox(
                                        label="🎨 FLUX.1-schnell Prompt (Put text in quotes for bold 3D typography: e.g. 'YOUR TITLE')",
                                        lines=4,
                                        placeholder="E.g.: Viral YouTube thumbnail for Hermes Agent, bold 3D glowing neon typography with text 'TUTO WINDOWS', vibrant electric cyan and warm amber lighting, 8k render...",
                                        interactive=True,
                                        info="Visual scene description. Put words in single quotes (e.g. 'SECRET REVEALED') for 3D typography."
                                    )
                                    btn_enhance_prompt = gr.Button("✨ AI Prompt Assistant (Expand & Add Catchy 3D Typography)", elem_id="btn_ai_prompt_assistant", variant="primary")
                                    gr.Markdown("<small>💡 *Click AI Prompt Assistant to automatically expand your idea into a viral high-CTR prompt with 3D text syntax (single quotes '...').*</small>")

                                with gr.Group():
                                    with gr.Row():
                                        flux_aspect = gr.Radio(
                                            choices=["16:9 (YouTube & Blog)", "9:16 (Shorts & Reels)", "1:1 (Square)"],
                                            value="16:9 (YouTube & Blog)",
                                            label="📐 Aspect Ratio",
                                            info="16:9 for YouTube & Blog; 9:16 for Shorts & Reels; 1:1 for Square."
                                        )
                                        flux_steps = gr.Radio(
                                            choices=["4 steps (Fast / ~2s)", "6 steps (Balanced)", "8 steps (High Detail)"],
                                            value="4 steps (Fast / ~2s)",
                                            label="⚡ Diffusion Steps",
                                            info="4 steps is distilled for ultra-fast generation (~2s)."
                                        )
                                    with gr.Row():
                                        flux_seed = gr.Number(
                                            value=-1,
                                            label="🎲 Generation Seed (-1 = Random)",
                                            precision=0,
                                            scale=2,
                                            info="Leave at -1 for a unique image on every run, or set a fixed number to reproduce."
                                        )
                                        btn_random_seed = gr.Button("🎲 Reset to Random (-1)", variant="secondary", scale=1)

                                with gr.Group():
                                    flux_ref_img = gr.Image(
                                        label="👤 Optional Reference Photo (Speaker Face / Product / Keyframe)",
                                        type="filepath",
                                        height=190,
                                        interactive=True
                                    )
                                    gr.Markdown("<small>💡 *Leave empty for pure Text-to-Image, or upload a photo to preserve face identity or product composition.*</small>")

                                _run_flux_ab_tooltip = zast_tooltip("• Generate Single: Renders 1 thumbnail with the active preset in ~2-4s.\n• Generate 3 A/B Variants: Generates 3 distinct high-CTR styles (Viral, 3D Tech, Cinematic) for YouTube Studio's Test & Compare.")
                                gr.Markdown(f"#### 🚀 Generation Actions {_run_flux_ab_tooltip}")
                                with gr.Row():
                                    btn_run_flux = gr.Button("🚀 Generate Single Thumbnail", variant="secondary", scale=2)
                                    btn_run_flux_ab = gr.Button("🧪 Generate 3 A/B Test Variants (1-Click)", variant="primary", scale=3)
                                gr.Markdown("<small>💡 *Fast 4-step generation (~2-4s on RTX GPU). AI metadata, EXIF, and C2PA are automatically 100% stripped from all outputs.*</small>")
                                
                                flux_status = gr.Markdown(value="")

                            with gr.Column(scale=7):
                                with gr.Group():
                                    _ab_tooltip = zast_tooltip("3 distinct visual styles ready for YouTube Studio Test & Compare. All metadata is stripped.")
                                    gr.Markdown(f"### 🧪 YouTube A/B Test Studio (3 Diverse Variants for High CTR) {_ab_tooltip}")
                                    with gr.Tabs(elem_id="flux_variant_tabs") as flux_variant_tabs:
                                        with gr.Tab("🅰️ Variant A : Viral High-CTR", id="tab_variant_a") as tab_variant_a:
                                            flux_thumb_a = gr.Image(label="Variant A (Viral)", height=300, interactive=False)
                                            with gr.Row():
                                                btn_apply_a = gr.Button("⭐ Use for Blog Thumbnail (#1)", variant="secondary", scale=2)
                                                file_down_a = gr.File(label="⬇️ Download Sanitized PNG A", scale=2, interactive=False)
                                            gr.Markdown("<small>💡 *Click 'Use for Blog Thumbnail' to assign this image as Keyframe #1 in the WordPress export pack.*</small>")
                                        with gr.Tab("🅱️ Variant B : 3D Tech Glow", id="tab_variant_b") as tab_variant_b:
                                            flux_thumb_b = gr.Image(label="Variant B (Tech)", height=300, interactive=False)
                                            with gr.Row():
                                                btn_apply_b = gr.Button("⭐ Use for Blog Thumbnail (#1)", variant="secondary", scale=2)
                                                file_down_b = gr.File(label="⬇️ Download Sanitized PNG B", scale=2, interactive=False)
                                            gr.Markdown("<small>💡 *Click 'Use for Blog Thumbnail' to assign this image as Keyframe #1 in the WordPress export pack.*</small>")
                                        with gr.Tab("🅲 Variant C : Cinematic Studio", id="tab_variant_c") as tab_variant_c:
                                            flux_thumb_c = gr.Image(label="Variant C (Studio)", height=300, interactive=False)
                                            with gr.Row():
                                                btn_apply_c = gr.Button("⭐ Use for Blog Thumbnail (#1)", variant="secondary", scale=2)
                                                file_down_c = gr.File(label="⬇️ Download Sanitized PNG C", scale=2, interactive=False)
                                            gr.Markdown("<small>💡 *Click 'Use for Blog Thumbnail' to assign this image as Keyframe #1 in the WordPress export pack.*</small>")

                                    with gr.Row():
                                        flux_ab_zip = gr.File(label="📦 Download Complete A/B Testing ZIP Pack (3 PNGs + Instructions)", scale=3, interactive=False)
                                        btn_open_flux_folder = gr.Button("📂 Open Folder", variant="secondary", scale=1)
                                    gr.Markdown("<small>💡 *The Complete A/B Testing ZIP Pack contains all 3 sanitized PNGs and a guide for uploading directly into YouTube Studio's 'Test & Compare' feature.*</small>")

                                    gr.Markdown(
                                        "🛡️ **Anti-AI Detection Sanitizer**: All 3 thumbnails have their AI metadata (EXIF, prompts, C2PA, PNG text chunks) "
                                        "**100% stripped** on save. Ready to drag & drop directly into YouTube Studio's *'Test & Compare'* A/B testing tool!"
                                    )
        
            with gr.Tab("ℹ️ Help", id="tab_help") as tab8:
                gr.Markdown("## How to use ZastTranslate")
                
                with gr.Accordion("📺 Preview & Subtitles (Left Column)", open=True):
                    gr.Markdown(
                        "The left column provides a real-time, interactive preview of your project:\n\n"
                        "- **Player Preview** — Displays the active video or audio player.\n"
                        "- **Preview Subtitles** — Select which subtitles to overlay during playback:\n"
                        "  - `None` — Hide subtitles.\n"
                        "  - `Original` — Show original transcribed speech.\n"
                        "  - `Translation (Fitted)` — Show concise fitted translation (used for dubbing).\n"
                        "  - `Translation (Normal)` — Show full natural translation.\n"
                        "- **Subtitle Overlay Box** — Subtitles are rendered dynamically and synchronized in real-time with the player's playhead.\n"
                        "- **JS Debug Log** — Displays real-time playback updates and timing logs for testing."
                    )
                
                with gr.Accordion("📥 Tab 1 — Import", open=False):
                    gr.Markdown(
                        "Load your video or audio file from one of these sources:\n\n"
                        "- **YouTube URL** — Paste any YouTube link. The video is downloaded automatically via yt-dlp.\n"
                        "- **Local file** — Upload a video (MP4, MKV, AVI, MOV, WebM) or audio (MP3, WAV, M4A, FLAC, OGG, AAC) file from your computer.\n\n"
                        "Click **Import Video or Audio** to start. A video or audio player preview will appear below depending on the file type.\n"
                        "Use **New Project** to clear everything and start over.\n\n"
                        "**YouTube resolution:** Click **🔍 Check URL** to see available resolutions before downloading. "
                        "Select the desired quality and click **Import Video or Audio**.\n\n"
                        "🗑️ **New Project** clears all data and deletes temporary files (downloads, audio, separated tracks) to free disk space.\n\n"
                        "💡 iPhone videos (.MOV with HEVC codec) are supported — they're automatically converted for browser playback.\n"
                        "💡 Audio files are automatically processed without video packaging (audio-only outputs in Dubbing and Bulk Mode)."
                    )
                
                with gr.Accordion("🎤 Tab 2 — Transcription & SEO Studio", open=False):
                    gr.Markdown(
                        "This step separates vocals from background music (Demucs), transcribes speech (WhisperX), and generates a complete YouTube SEO publication kit.\n\n"
                        "**Options:**\n"
                        "- **Source Language** — Select the spoken language from 20+ languages, or leave on *Auto* for auto-detection. "
                        "Setting it manually improves accuracy.\n"
                        "- **Whisper Model** — Choose the model size:\n"
                        "  - `base` — Fast, lower accuracy (good for testing)\n"
                        "  - `small` / `medium` — Balanced\n"
                        "  - `large-v3` — Best accuracy, uses more VRAM (~3 GB)\n\n"
                        "**Transcription Tools:**\n"
                        "- **🧹 Clean Fillers & Oral Tics** — Automatically filters out speech hesitations (*euh*, *um*, *ben*, *you know*, *like*) and false starts while maintaining strict millisecond synchronization.\n"
                        "- **Review & Edit** — Directly edit the table (Start, End, Text). Split or merge rows as needed.\n"
                        "- **Export SRT 💾** — Download subtitles locally in UTF-8.\n\n"
                        "**🚀 YouTube SEO & Description Studio:**\n"
                        "- Click **✨ Generate Metadata, Chapters & SEO** to create:\n"
                        "  - **High-CTR Video Title** — Formatted in natural sentence case with front-loaded search intent and brand normalization (*Hermès Agent*, *Windows*, *IA*, *API*, *ChatGPT*).\n"
                        "  - **⏱️ Full Timeline Chapters & Landmarks** — Samples 100% of the video duration (from 00:00 to the end) and automatically detects major tools (*Ollama*, *Qwen Local LLM*, *Telegram*, *Smartphone Remote Control*, *Jobs*).\n"
                        "  - **📝 300+ Word Description** — Clean plain text without broken markdown asterisks (`**`), including hook, feature overview, bullet points, and calls to action.\n"
                        "  - **🏷️ 4 Strategic Hashtag Packs** — Displayed directly on interactive radio buttons (Subject, Tutorial, Tech Stack, Trends) with instant 1-click description updating.\n"
                        "  - **🔍 Live YouTube Search Suggestion Mining** — Queries Google's live autocompletion endpoint in real time to rank for what users are actually searching right now on YouTube, with automatic fashion/homonym disambiguation.\n"
                        "- Click **📋 Apply to Translation & Bulk** to forward the generated title and description to Tab 3 (Single Translation) and Tab 5 (Bulk Mode).\n\n"
                        "⚠️ **You MUST click 'Validate Transcription ✅' before going to the Translation tab.** "
                        "Without validation, downstream translation steps will not have data to work with.\n\n"
                        "**Alternative:** You can skip transcription entirely by importing an existing **SRT file**."
                    )
                
                with gr.Accordion("🌍 Tab 3 — Translation", open=False):
                    gr.Markdown(
                        "Select the target language and click **Run Translation**.\n\n"
                        "The app generates **two versions** of each segment:\n"
                        "- **Translation** — A natural, full translation (faithful to the original meaning).\n"
                        "- **Fitted** — A concise version shortened to fit the original segment duration for dubbing. "
                        "Marked with ✅ if it fits, ⚠️ if it may overflow.\n\n"
                        "**You can edit both columns** before validating. The Fitted column is what will be spoken during dubbing.\n\n"
                        "⚠️ **You MUST click 'Validate Translation ✅' before going to the Dubbing tab.** "
                        "Without validation, dubbing will not work.\n\n"
                        "**Export options:**\n"
                        "- **Export Translation SRT** — Full natural translation as subtitles\n"
                        "- **Export Fitted SRT** — Concise dubbing-ready subtitles\n\n"
                        "**Supported languages:** The dropdown dynamically updates based on the intersection of the selected **TTS Backend** and **LLM Backend**.\n"
                        "- **VoxCPM 2** supports 30 languages (Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese).\n"
                        "- **Qwen2.5/3.5 LLM** support all languages. **EuroLLM** supports only European languages.\n\n"
                        "The available target languages are always the intersection of the TTS engine + LLM capabilities.\n\n"
                        "**Fitted text length** is controlled by the CPS (chars/sec) calibration. "
                        "You can adjust per-language values in the **⚙️ Config CPS** tab.\n\n"
                        "💡 **Copy Buttons**: Use the copy icons on the 'Translated Video Title' and 'Translated Video Description' fields to quickly copy the translated texts to your system clipboard."
                    )
                
                with gr.Accordion("🎬 Tab 4 — Dubbing & Export", open=False):
                    gr.Markdown(
                        "Generate the dubbed video with synthesized speech.\n\n"
                        "**Interactive Segment Editor & Timeline Adjustments:**\n"
                        "- **Dialogue Lines List** — A complete list of all dialogue segments.\n"
                        "- **Click to Seek** — Click on any row in the Transcription, Translation, or Dialogue lists to automatically move the video player's playhead to that segment's exact start time.\n"
                        "- **Segment Editor Card** — When a row is clicked, an editor opens below allowing fine-tuning of the text to speak, and precise timing adjustments (Minutes and Seconds) for both Start and End times.\n"
                        "- **🔄 Regenerate Segment Audio** — Re-synthesize the voice for the selected segment only, updating the local segment cache immediately without needing to regenerate the entire project.\n"
                        "- **⚠️ Reformulation Warning** — If a segment is shortened or reformulated by the translation LLM during synthesis to fit timing constraints, the segment editor card will display a warning icon ⚠️ along with the exact shortened text, and the main synthesis status will report the number of reformulations.\n\n"
                        "**Voice Mode:**\n\n"
                        "| Mode | Description | When to use |\n"
                        "|---|---|---|\n"
                        "| **Default voice** | TTS preset voice | Quick dubbing, no reference needed |\n"
                        "| **Clone from original** | Clones the speaker's voice from the extracted vocals | Best result — sounds like the original speaker |\n"
                        "| **Clone from file** | Uses an uploaded WAV/MP3 file as voice reference | When you want a specific voice |\n\n"
                        "💡 Voice cloning uses **VoxCPM 2**, installed automatically during setup.\n\n"
                        "**Options:**\n"
                        "- **Voice sample file** — Only needed for *Clone from file* mode. Use 10-30s of clear speech (WAV or MP3).\n"
                        "- **🔊 Never Cut Vocal** — Speaks all text in full without truncation. Produces more natural speech "
                        "but the dubbing may drift out of sync with the video.\n\n"
                        "**Output:**\n"
                        "- **🎬 Final Video (MP4)** — 1080p dubbed video player and direct 1-click download button/card\n"
                        "- **🎵 Mixed Audio Track (WAV)** — Dubbed waveform player and direct 1-click download button/card\n\n"
                        "**⚠️ Current limitations:**\n"
                        "- **No lip-sync** — The audio is replaced but the video is not modified (no face/lip adaptation)\n"
                        "- **Single voice only** — All segments use the same voice. Multi-speaker dubbing is not supported yet."
                    )
                
        
                
                with gr.Accordion("📚 Tab 5 — Bulk Mode", open=False):
                    gr.Markdown(
                        "Automate translation and dubbing for multiple languages simultaneously.\n\n"
                        "**How it works:**\n"
                        "1. Select all target languages from the dropdown.\n"
                        "2. Optionally, provide the **Original Video Title** and **Description**, or click **⬇️ Import from URL** / **📥 Apply to Translation & Bulk**.\n"
                        "3. Select your **Output Generation** mode:\n"
                        "   - `Video + Audio`: Generates dubbed MP4 videos, WAV tracks, and translated SRTs.\n"
                        "   - `Audio Only`: Generates WAV tracks and translated SRTs without video rendering.\n"
                        "   - `Subtitles & Metadata Only`: Bypasses voice synthesis completely to produce localized SRT subtitles (Natural & Fitted), titles, and descriptions across all languages in under 15 seconds!\n"
                        "4. Click **Run Bulk Process**. The system executes the requested pipeline and packages all results into `bulk_export_all.zip`."
                    )

                with gr.Accordion("📱 Tab 6 — Viral Shorts Studio (9:16)", open=False):
                    gr.Markdown(
                        "Automatically extract high-impact vertical shorts from your video for YouTube Shorts, TikTok, and Instagram Reels:\n\n"
                        "- **Source Selection** — Choose whether to extract clips from the **Original Video** or the **Dubbed Video (Tab 4)**.\n"
                        "- **🔢 1 to 5 Shorts Slider** — Choose exactly how many standalone viral moments to detect and render (from 1 up to 5).\n"
                        "- **✨ AI Viral Detection** — The local LLM scans the speech transcript to detect the top standalone viral moments, assigning catchy titles and impact scores.\n"
                        "- **Visual & Speech Boundary Snapping** — Uses **PySceneDetect 0.7.1** and WhisperX silence bounds to snap start and end timestamps to natural scene cuts and speech pauses.\n"
                        "- **👁️ Instant Scene Preview in Player** — Click any table row or the `▶️ Preview Short in Player` button to seek the main left video player directly to that segment.\n"
                        "- **🎬 Selective Rendering Checkboxes** — Check or uncheck individual shorts to render only the clips you want (e.g., skip Short #3).\n"
                        "- **📝 Editable TikTok Karaoke Subtitles** — Inspect and customize the word-by-word animated subtitles that will be burned into each vertical clip.\n"
                        "- **📐 Recropping Modes**:\n"
                        "  - `Stacked Blur (Aesthetic)`: Keeps the full 16:9 frame centered with aesthetically blurred top and bottom backgrounds.\n"
                        "  - `Center Crop (9:16)`: Cuts a direct 1080x1920 central crop.\n"
                        "- **🎨 Stylized Subtitles**:\n"
                        "  - `Neon Yellow (Viral Style)`: Ultra-readable yellow text with pure black border and drop shadow.\n"
                        "  - `Mint Green` / `Cyber Cyan` / `Punchy Pink` / `Minimalist White`: Modern aesthetic color schemes.\n"
                        "  - Safe-area bottom margins (280px) to prevent overlap with TikTok and YouTube Shorts UI controls.\n"
                        "- **📦 Export & ZIP** — Preview generated shorts in individual vertical players, download them individually, or grab the complete `shorts_export_pack.zip`."
                    )

                with gr.Accordion("📝 Tab 7 — SEO Blog Post Studio (WordPress Ready)", open=False):
                    gr.Markdown(
                        "Transform any transcribed video into a **complete, human-like, SEO-optimized blog post ready for WordPress**:\n\n"
                        "- **Target Language Selection** — Write the blog article in French, English, Spanish, German, Italian, or any chosen language.\n"
                        "- **🎨 Writing Style & Tone Selector**:\n"
                        "  - `Step-by-Step Tutorial (How-To Guide)`: Practical, actionable step-by-step how-to guide.\n"
                        "  - `Expert & Technical Deep-Dive`: In-depth architectural, technical, and benchmark analysis.\n"
                        "  - `Storytelling & Case Study`: Engaging problem-solution narrative and real-world journey.\n"
                        "  - `Journalistic & Objective Review`: Objective, structured reporting with executive summary.\n"
                        "  - `High-Converting Copywriting`: High-converting benefit-focused copy with strong hooks.\n"
                        "  - `Accessible Beginner's Guide`: Friendly, everyday analogies for beginners.\n"
                        "- **🛡️ Humanizer Engine (35 Anti-AI Detection Rules)** — Adheres strictly to Wikipedia's *WikiProject AI Cleanup* and [blader/humanizer](https://github.com/blader/humanizer) standards. Eliminates robotic AI clichés (*'In this article...'*, *'It is crucial to note...'*, *'In conclusion...'*, *'In today's digital landscape...'*, inflated legacy hype, and shallow participle clauses) while maximizing burstiness and natural sentence-case phrasing.\n"
                        "- **🔄 Live 1-Click Humanizer Rule Sync** — Click `🔄 Sync Anti-AI Rules (Humanizer GitHub)` to fetch and cache the latest rules and watch words directly from GitHub.\n"
                        "- **🔍 Live Google & YouTube Keyword Discovery** — Queries Google Autocomplete in real-time without API keys, weaving high-intent search terms into H1, intro paragraphs, and subheadings.\n"
                        "- **🎯 Full SEO Pack & Meta Description** — Generates High-CTR H1 Title, calibrated Meta Description (145-160 chars), clean URL slug, focus keyword, and LSI secondary keywords.\n"
                        "- **📸 Interactive Keyframes & Custom Thumbnail Studio** — Captures HD screenshots from key video timestamps (FFmpeg), allows **drag-and-drop custom thumbnail uploads**, generates contextual SEO ALT tags, provides ready-to-use **AI Image Generation Prompts** (Midjourney, DALL-E, Flux, SenseNova), and opens the image folder in 1-click (`📂 Open Images Folder in Windows Explorer`).\n"
                        "- **📋 1-Click WordPress Gutenberg Export** — Outputs both Markdown and native WordPress Gutenberg block comments (`<!-- wp:heading -->`, `<!-- wp:paragraph -->`, `<!-- wp:list -->`, `<!-- wp:quote -->`, `<!-- wp:code -->`) ready to paste directly into the WordPress Code Editor, and packages everything into `blog_pack_wordpress.zip`."
                    )

                with gr.Accordion("⚡ Tab 7 — YouTube Thumbnail Studio (FLUX.1-schnell & A/B Testing)", open=False):
                    gr.Markdown(
                        "### ⚡ AI-Powered 4K YouTube Thumbnail Studio (FLUX.1-schnell)\n\n"
                        "ZastTranslate integrates the state-of-the-art **FLUX.1-schnell** 12-billion parameter Flow Transformer model by Black Forest Labs to generate eye-catching, high-converting 16:9 YouTube thumbnails in ~2 seconds.\n\n"
                        "#### 🔑 Key Features & Capabilities:\n"
                        "- **⚡ Ultra-Fast 4-Step Distilled Diffusion** — Optimized for consumer NVIDIA GPUs (RTX 3060/3080/4090). Generates full 1280x720 (16:9), 720x1280 (9:16), or 1024x1024 (1:1) visuals in ~2 seconds.\n"
                        "- **🔤 Flawless 3D Typography** — Unlike older diffusion models that generate garbled text, FLUX.1 has native text-rendering capabilities. Simply surround your text with single quotes (e.g., `'HERMES AGENT'` or `'PYTHON TUTORIAL'`) to generate crisp, styled 3D typography embedded in the scene.\n"
                        "- **🎨 5 Visual Style Presets**:\n"
                        "  - `YouTube Viral High-CTR`: Vibrant contrasting colors (electric cyan and warm amber), expressive lighting, bold dramatic depth.\n"
                        "  - `3D Isometric & Tech Glow`: Clean isometric 3D models with neon accents, soft shadows, and modern tech aesthetic.\n"
                        "  - `Cyberpunk & Bold Neon`: High-contrast dark backgrounds with vivid neon glow, volumetric smoke, and futuristic elements.\n"
                        "  - `Minimalist & Clean SaaS`: Elegant editorial design, subtle gradients, lots of clean negative space.\n"
                        "  - `Photorealistic Studio Shot`: Cinematic depth of field (f/1.8), realistic skin tones, professional 3-point studio lighting.\n"
                        "- **🧪 1-Click YouTube A/B Testing Suite** — Click `🧪 Generate 3 A/B Test Variants` to produce 3 fundamentally distinct thumbnail angles (Variant A: Viral High-CTR, Variant B: 3D Tech Glow, Variant C: Cinematic Studio). Download the complete pack as a ZIP ready to upload directly into YouTube Studio's *'Test & Compare'* A/B testing tool.\n"
                        "- **🛡️ Automatic Anti-AI Metadata Stripping** — Automatically removes all generative AI metadata (EXIF tags, PNG tEXt/iTXt prompt chunks, C2PA manifests) so files are 100% clean and compliant.\n"
                        "- **⭐ 1-Click Blog Cover Integration** — Click `⭐ Use for Blog Thumbnail (#1)` under any variant to immediately assign it as the featured cover image in the WordPress export pack.\n"
                        "- **🧹 Dynamic VRAM Management** — Click `🧹 Free VRAM (Unload)` at any time to flush model weights from GPU memory before assembling videos in Tab 4."
                    )
        
                with gr.Accordion("🔧 Troubleshooting", open=False):
                    gr.Markdown(
                        "- **Models download on first run** — WhisperX, Qwen3-8B, Demucs, and TTS models are cached automatically (~8 GB total)\n"
                        "- **Out of VRAM** — Models are loaded and unloaded sequentially to minimize GPU memory. "
                        "Try a smaller Whisper model (base or small) if you run out\n"
                        "- **Clean install** — Click **Reset** then **Install** in the Pinokio launcher\n\n"
                        "**Harmless terminal warnings (can be safely ignored):**\n"
                        "- **`Could not load libtorchcodec`** — Long error traceback about FFmpeg DLLs. "
                        "This is a TorchCodec compatibility message — it does NOT affect functionality.\n"
                        "- **`Video does not have browser-compatible container or codec`** — "
                        "Gradio auto-converts iPhone MOV/HEVC videos to MP4 for browser playback. This is normal.\n"
                        "- **`ConnectionResetError [WinError 10054]`** — A harmless Windows networking warning "
                        "from the Gradio server. Does not affect the application."
                    )
                
                with gr.Accordion("🧠 System & VRAM Settings", open=False):
                    keep_models_ui = gr.Checkbox(
                        label="Keep models in memory (Fast Mode, requires >16GB VRAM)", 
                        value=False,
                        info="If checked, AI models are not unloaded between steps. Dramatically speeds up bulk processing, but requires high VRAM."
                    )
                    
                with gr.Accordion("⚙️ System requirements", open=False):
                    gr.Markdown(
                        "- **GPU**: NVIDIA GPU with 4+ GB VRAM recommended (CUDA)\n"
                        "- **CPU**: Works on CPU but significantly slower\n"
                        "- **Disk**: ~8 GB for AI models (downloaded on first use)\n"
                        "- **OS**: **Tested on Windows only**. May work on Linux/macOS but untested.\n"
                        f"\n**Current system**: {GPU_NAME} ({GPU_VRAM}) — {'CUDA ✅' if DEVICE == 'cuda' else 'CPU mode ⚠️'}"
                    )
                
                with gr.Accordion("⚙️ Config CPS tab", open=False):
                    gr.Markdown(
                        "Customize the **characters-per-second (CPS)** speaking rate used to compute the maximum Fitted text length per segment.\n\n"
                        "| Column | Description |\n"
                        "|---|---|\n"
                        "| **Language** | Display name of the language |\n"
                        "| **ISO** | ISO 639-1 code used internally |\n"
                        "| **Default CPS** | Calibrated default from the built-in table |\n"
                        "| **Your CPS** | Your override — leave empty to use the default |\n\n"
                        "Click **Save** to apply changes immediately (no restart needed). "
                        "Click **Reset to defaults** to clear all overrides."
                    )
        
                with gr.Accordion("🔗 About / Links & Credits", open=False):
                    gr.Markdown(
                        "**ZastTranslate** is made by Zast.\n\n"
                        "- 🌐 [zast57.com](https://zast57.com) — Website\n"
                        "- 🤓 [paradoxetemporel.fr](https://paradoxetemporel.fr) — Tech & Geek blog\n"
                        "- 🎬 [zast.fr](https://zast.fr) — YouTube channel\n\n"
                        "**Open Source Credits & Acknowledgements:**\n"
                        "- 🛡️ [Humanizer (blader/humanizer)](https://github.com/blader/humanizer) — Anti-AI writing style rules & patterns based on Wikipedia's *WikiProject AI Cleanup* (MIT License).\n"
                        "- 🎙️ [WhisperX](https://github.com/m-bain/whisperX) — High-accuracy speech recognition & word-level alignment.\n"
                        "- 🎵 [Demucs](https://github.com/facebookresearch/demucs) — State-of-the-art vocal isolation by Meta AI."
                    )
        
            with gr.Tab("⚙️ CPS", id="tab_cps") as tab9:
                with gr.Accordion("💡 Quick Guide & Options Explained (Tab 9: Voice Speed CPS)", open=False, elem_classes=["zast-tab-guide-accordion"]):
                    gr.HTML(_get_tab_guide_html(9))
                gr.Markdown(
                    "### Voice Speed Calibration (Chars/sec) per Language\n\n"
                    "These values control the maximum text length in the **Fitted** column for each dubbing segment.\n"
                    "Fill in **Your CPS** only for the languages you want to customize — leave empty to use the default value.\n\n"
                    "> 💡 **Higher value** = longer Fitted text allowed (TTS speaks faster). "
                    "**Lower value** = shorter text (TTS speaks slower). Changes apply immediately after saving."
                )
                cps_table = gr.DataFrame(
                    value=_build_cps_dataframe(),
                    label="CPS table per language",
                    interactive=True,
                    wrap=True,
                )
                with gr.Row():
                    btn_save_cps = gr.Button("💾 Save", variant="primary", scale=1)
                    btn_reset_cps = gr.Button("↩️ Reset to defaults", variant="secondary", scale=1)
                cps_status = gr.Markdown("")
            main_tabs.__exit__(None, None, None)

    JS_DOWNLOAD_WAIT = """async (...args) => {
        const container = document.getElementById("local_file_input");
        if (!container) return args;
        const isUploading = () => {
            return container.querySelector("[role='progressbar']") || 
                   container.querySelector(".progress-bar") || 
                   container.querySelector(".progress") || 
                   container.querySelector(".progress-ring") || 
                   container.querySelector(".uploading") || 
                   container.textContent.includes("Uploading") ||
                   container.textContent.includes("Chargement");
        };
        if (isUploading()) {
            await new Promise((resolve) => {
                const interval = setInterval(() => {
                    if (!isUploading()) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 100);
            });
        }
        return args;
    }"""

    JS_SRT_WAIT = """async (...args) => {
        const container = document.getElementById("srt_file_input");
        if (!container) return args;
        const isUploading = () => {
            return container.querySelector("[role='progressbar']") || 
                   container.querySelector(".progress-bar") || 
                   container.querySelector(".progress") || 
                   container.querySelector(".progress-ring") || 
                   container.querySelector(".uploading") || 
                   container.textContent.includes("Uploading") ||
                   container.textContent.includes("Chargement");
        };
        if (isUploading()) {
            await new Promise((resolve) => {
                const interval = setInterval(() => {
                    if (!isUploading()) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 100);
            });
        }
        return args;
    }"""

    # EVENTS
    btn_check.click(step0_check_url, [url_input], [status_dl, yt_resolution, btn_dl])
    btn_dl.click(step1_download, [url_input, file_input, yt_resolution, local_title_input], [status_dl, video_preview, audio_preview, btn_transcribe, btn_import_metadata, bulk_output_type, final_video_out, btn_import_metadata_single], show_progress="full", js=JS_DOWNLOAD_WAIT)
    btn_reset.click(
        reset_project,
        [],
        [
            url_input, file_input, local_title_input, status_dl, video_preview, audio_preview,
            btn_transcribe, btn_translate, btn_synth, btn_bulk_run,
            btn_import_metadata,
            bulk_output_type, final_video_out, segments_json_holder, dubbing_segments_df,
            original_title_input, original_desc_input, translated_title_input, translated_desc_input,
            btn_import_metadata_single,
            seo_title_out, seo_tags_out, seo_chapters_out, seo_desc_out, seo_status
        ]
    )
    
    def on_llm_backend_change(selected_backend):
        global reformulator, current_llm_backend
        if selected_backend in available_llm_backends:
            current_llm_backend = selected_backend
            user_config["llm_backend"] = selected_backend
            save_config(user_config)
            if reformulator.backend_name != selected_backend:
                if reformulator.llm is not None:
                    reformulator.llm.unload()
                    reformulator.llm = None
                reformulator.backend_name = selected_backend
                print(f"[LLM] Switched active LLM backend to {selected_backend}")

    llm_backend_dropdown.change(on_llm_backend_change, inputs=[llm_backend_dropdown], outputs=[])

    def on_tts_backend_change(selected_backend):
        global tts_engine, current_tts_backend
        if selected_backend in available_tts_backends:
            current_tts_backend = selected_backend
            user_config["tts_backend"] = selected_backend
            save_config(user_config)
            tts_engine = get_tts_backend(selected_backend)
            time_sync.tts_engine = tts_engine
            print(f"[TTS] Switched active TTS backend to {selected_backend}")

    tts_backend_dropdown.change(on_tts_backend_change, inputs=[tts_backend_dropdown], outputs=[])

    btn_transcribe.click(step2_transcribe, [lang_source, model_size], [transcription_status, transcription_df, segments_json_holder, export_transcription_file], show_progress="full")
    btn_import_srt.click(step2b_import_srt, [srt_file_input, lang_source], [transcription_status, transcription_df, segments_json_holder, export_transcription_file], show_progress="full", js=JS_SRT_WAIT)
    
    btn_valid_transcription.click(step3_save_transcription, [transcription_df], [transcription_status, btn_translate, btn_bulk_run, segments_json_holder, export_transcription_file])
    btn_clean_transcription.click(step2_clean_transcription, [transcription_df, lang_source], [transcription_status, transcription_df, segments_json_holder, export_transcription_file], show_progress="full")
    btn_export_transcription.click(export_transcription_srt, [transcription_df], [transcription_status, export_transcription_file])
    btn_open_output_tab2.click(open_output_folder, [], [])
    
    # SEO Studio Events
    btn_generate_seo.click(
        step2_generate_seo_metadata,
        [transcription_df, seo_hashtag_pack, lang_source],
        [seo_title_out, seo_tags_out, seo_chapters_out, seo_desc_out, seo_hashtag_pack, seo_status],
        show_progress="full"
    )
    seo_hashtag_pack.change(
        step2_change_hashtag_pack,
        [seo_hashtag_pack, seo_desc_out],
        [seo_desc_out]
    )
    btn_apply_seo.click(
        step2_apply_seo_metadata,
        [seo_title_out, seo_desc_out],
        [original_title_input, original_desc_input, bulk_title_input, bulk_desc_input, seo_status]
    )
    
    btn_import_metadata_single.click(import_metadata_from_state, [], [original_title_input, original_desc_input])
    
    btn_translate.click(
        step4_translate,
        [lang_target, original_title_input, original_desc_input],
        [translation_status, translation_df, segments_json_holder, translated_title_input, translated_desc_input, export_translation_file],
        show_progress="full"
    )
    
    btn_valid_translation.click(
        step5_save_translation,
        [translation_df, dubbing_text_source, translated_title_input, translated_desc_input],
        [translation_status, btn_synth, segments_json_holder, dubbing_segments_df]
    )
    btn_export_translation.click(export_translation_srt, [translation_df], [translation_status, export_translation_file])
    btn_export_fitted.click(export_fitted_srt, [translation_df], [translation_status, export_translation_file])
    btn_open_output_tab3.click(open_output_folder, [], [])
    
    # Python-only editor loader for Dubbing segments table click.
    # Note: client-side seeking for transcription_df, translation_df, and dubbing_segments_df is handled globally in BLOCKS_JS.
    dubbing_segments_df.select(
        fn=load_segment_to_editor,
        inputs=[dubbing_text_source, voice_mode, voice_file, never_cut_mode, default_voice_gender],
        outputs=[
            segment_editor_card, edit_seg_idx, edit_seg_text, edit_start_min, edit_start_sec,
            edit_end_min, edit_end_sec, edit_seg_audio, edit_seg_status,
            segments_json_holder
        ]
    )
    
    # Listen to subtitle mode selection changes to update client state reliably
    subtitle_selection.change(
        fn=None,
        inputs=[subtitle_selection],
        outputs=None,
        js="(val) => { window.ZastSubtitleMode = val; }"
    )
    
    # Text source change warning & table data updates
    def on_dubbing_text_source_change(choice):
        if state.translated_segments:
            for seg in state.translated_segments:
                if 'fitted_text' not in seg:
                    seg['fitted_text'] = seg.get('translated_text', '')
                if choice == "Normal Translation":
                    seg['translated_text'] = seg.get('normal_text', seg.get('fitted_text', ''))
                else:
                    seg['translated_text'] = seg.get('fitted_text', '')
        
        warning_box = toggle_dubbing_text_source_warning(choice)
        df_data = _build_dubbing_df_data(choice)
        return warning_box, df_data, _get_segments_json_html()

    dubbing_text_source.change(
        on_dubbing_text_source_change,
        [dubbing_text_source],
        [dubbing_warning_box, dubbing_segments_df, segments_json_holder]
    )
    
    btn_regen_seg.click(
        step6_regenerate_segment,
        [
            edit_seg_idx, edit_seg_text, edit_start_min, edit_start_sec, edit_end_min, edit_end_sec,
            voice_mode, voice_file, never_cut_mode, default_voice_gender, dubbing_text_source
        ],
        [
            edit_seg_audio, edit_seg_status, dubbing_segments_df, segments_json_holder,
            video_preview, audio_preview, final_video_out, final_audio_out,
            export_video_file, export_audio_file
        ]
    )
    
    def toggle_never_cut_warning(enabled):
        if enabled:
            return gr.update(value=NEVER_CUT_WARNING, visible=True)
        return gr.update(value="", visible=False)
    
    never_cut_mode.change(toggle_never_cut_warning, [never_cut_mode], [never_cut_warning])
    bulk_never_cut_mode.change(toggle_never_cut_warning, [bulk_never_cut_mode], [bulk_never_cut_warning])
    
    def toggle_voice_inputs(mode):
        return gr.update(visible=(mode == "Clone from file")), gr.update(visible=(mode == "Default voice"))
        
    voice_mode.change(toggle_voice_inputs, inputs=[voice_mode], outputs=[voice_file, default_voice_gender])
    bulk_voice_mode.change(toggle_voice_inputs, inputs=[bulk_voice_mode], outputs=[bulk_voice_file, bulk_default_voice_gender])
    
    def on_bulk_output_type_change(choice, mode):
        is_sub_only = (choice == "Subtitles & Metadata Only")
        return (
            gr.update(visible=not is_sub_only), # bulk_voice_mode
            gr.update(visible=(mode == "Clone from file" and not is_sub_only)), # bulk_voice_file
            gr.update(visible=(mode == "Default voice" and not is_sub_only)), # bulk_default_voice_gender
            gr.update(visible=not is_sub_only), # bulk_never_cut_mode
            gr.update(visible=False) # bulk_never_cut_warning
        )
    
    bulk_output_type.change(
        on_bulk_output_type_change,
        inputs=[bulk_output_type, bulk_voice_mode],
        outputs=[bulk_voice_mode, bulk_voice_file, bulk_default_voice_gender, bulk_never_cut_mode, bulk_never_cut_warning]
    )
    
    keep_models_ui.change(lambda x: setattr(state, 'keep_models', x), inputs=[keep_models_ui], outputs=[])
    btn_import_metadata.click(import_metadata_from_state, [], [bulk_title_input, bulk_desc_input])
    
    btn_synth.click(
        step6_synthesize, 
        [voice_mode, voice_file, never_cut_mode, default_voice_gender, dubbing_text_source], 
        [
            synth_status, final_video_out, final_audio_out, video_preview, audio_preview,
            dubbing_segments_df, segments_json_holder, export_video_file, export_audio_file
        ],
        show_progress="full"
    )
    btn_export_video.click(export_video, [], [synth_status, export_video_file])
    btn_export_audio.click(export_audio, [], [synth_status, export_audio_file])
    btn_open_output_tab4.click(open_output_folder, [], [])
    
    btn_bulk_run.click(
        step5_bulk_run, 
        [bulk_target_langs, bulk_voice_mode, bulk_voice_file, bulk_never_cut_mode, bulk_output_type, bulk_title_input, bulk_desc_input, bulk_default_voice_gender, bulk_generate_shorts], 
        [bulk_status_output, bulk_files_output, bulk_metadata_output],
        show_progress="full"
    )
    btn_open_output.click(open_output_folder, [], [])

    # Tab 6: Viral Shorts Handlers
    for i in range(5):
        short_previews[i].click(
            fn=None,
            inputs=[short_starts[i]],
            js="(st) => { if (window.zastSeekVideo) { window.zastSeekVideo(st); } }"
        )

    btn_detect_shorts.click(
        step7_detect_shorts,
        inputs=[shorts_source_choice, shorts_crop_style, shorts_subtitle_style, shorts_num_slider, shorts_dubbed_lang],
        outputs=[
            shorts_status,
            shorts_table,
            btn_render_shorts,
            short_cards[0], short_includes[0], short_titles[0], short_starts[0], short_ends[0], short_subtitles[0],
            short_cards[1], short_includes[1], short_titles[1], short_starts[1], short_ends[1], short_subtitles[1],
            short_cards[2], short_includes[2], short_titles[2], short_starts[2], short_ends[2], short_subtitles[2],
            short_cards[3], short_includes[3], short_titles[3], short_starts[3], short_ends[3], short_subtitles[3],
            short_cards[4], short_includes[4], short_titles[4], short_starts[4], short_ends[4], short_subtitles[4],
        ]
    )

    btn_render_shorts.click(
        step7_render_shorts,
        inputs=[
            shorts_source_choice, shorts_crop_style, shorts_subtitle_style,
            short_includes[0], short_titles[0], short_starts[0], short_ends[0], short_subtitles[0],
            short_includes[1], short_titles[1], short_starts[1], short_ends[1], short_subtitles[1],
            short_includes[2], short_titles[2], short_starts[2], short_ends[2], short_subtitles[2],
            short_includes[3], short_titles[3], short_starts[3], short_ends[3], short_subtitles[3],
            short_includes[4], short_titles[4], short_starts[4], short_ends[4], short_subtitles[4],
            shorts_dubbed_lang
        ],
        outputs=[
            shorts_status,
            short_cols[0], short_videos[0], short_files[0],
            short_cols[1], short_videos[1], short_files[1],
            short_cols[2], short_videos[2], short_files[2],
            short_cols[3], short_videos[3], short_files[3],
            short_cols[4], short_videos[4], short_files[4],
            shorts_zip_output
        ],
        show_progress="full"
    )

    def on_shorts_source_choice_change(source_choice, current_dubbed_lang, st_1, en_1, st_2, en_2, st_3, en_3, st_4, en_4, st_5, en_5):
        if "Dubbed" in source_choice:
            avail = get_available_dubbed_languages()
            cur_lang = current_dubbed_lang if (current_dubbed_lang and current_dubbed_lang in avail) else (avail[0] if avail else "English")
            active_segs, _, _ = _resolve_dubbed_segments_and_video(cur_lang)
            t_key = "translated_text"
            dropdown_update = gr.update(choices=avail, value=cur_lang, visible=True)
        else:
            active_segs = state.segments
            t_key = "text"
            dropdown_update = gr.update(visible=False)

        time_pairs = [(st_1, en_1), (st_2, en_2), (st_3, en_3), (st_4, en_4), (st_5, en_5)]
        updates = []
        for st, en in time_pairs:
            try:
                st_f = float(st)
                en_f = float(en)
                if en_f > st_f and active_segs:
                    extracted = shorts_studio.extract_clip_text(active_segs, st_f, en_f, text_key=t_key)
                    updates.append(gr.update(value=extracted))
                else:
                    updates.append(gr.update())
            except Exception:
                updates.append(gr.update())
        return (*updates, dropdown_update)

    def on_shorts_dubbed_lang_change(dubbed_lang, st_1, en_1, st_2, en_2, st_3, en_3, st_4, en_4, st_5, en_5):
        active_segs, _, _ = _resolve_dubbed_segments_and_video(dubbed_lang)
        time_pairs = [(st_1, en_1), (st_2, en_2), (st_3, en_3), (st_4, en_4), (st_5, en_5)]
        updates = []
        for st, en in time_pairs:
            try:
                st_f = float(st)
                en_f = float(en)
                if en_f > st_f and active_segs:
                    extracted = shorts_studio.extract_clip_text(active_segs, st_f, en_f, text_key="translated_text")
                    updates.append(gr.update(value=extracted))
                else:
                    updates.append(gr.update())
            except Exception:
                updates.append(gr.update())
        return tuple(updates)

    shorts_source_choice.change(
        on_shorts_source_choice_change,
        inputs=[
            shorts_source_choice,
            shorts_dubbed_lang,
            short_starts[0], short_ends[0],
            short_starts[1], short_ends[1],
            short_starts[2], short_ends[2],
            short_starts[3], short_ends[3],
            short_starts[4], short_ends[4]
        ],
        outputs=[
            short_subtitles[0],
            short_subtitles[1],
            short_subtitles[2],
            short_subtitles[3],
            short_subtitles[4],
            shorts_dubbed_lang
        ]
    )

    shorts_dubbed_lang.change(
        on_shorts_dubbed_lang_change,
        inputs=[
            shorts_dubbed_lang,
            short_starts[0], short_ends[0],
            short_starts[1], short_ends[1],
            short_starts[2], short_ends[2],
            short_starts[3], short_ends[3],
            short_starts[4], short_ends[4]
        ],
        outputs=[
            short_subtitles[0],
            short_subtitles[1],
            short_subtitles[2],
            short_subtitles[3],
            short_subtitles[4]
        ]
    )

    tab6.select(
        fn=lambda: gr.update(choices=get_available_dubbed_languages(), value=get_available_dubbed_languages()[0] if get_available_dubbed_languages() else None),
        inputs=[],
        outputs=[shorts_dubbed_lang]
    )

    btn_open_shorts_folder.click(open_output_folder, [], [])

    # Tab 7: SEO Blog & WordPress Studio Handlers
    btn_generate_blog.click(
        step8_generate_blog_post,
        inputs=[blog_target_lang, blog_style, blog_length, blog_include_meta, blog_extract_images, blog_num_images, blog_keyframe_res, blog_enhance_text],
        outputs=[
            blog_status,
            blog_title_out,
            blog_meta_desc_out,
            blog_char_count,
            blog_slug_out,
            blog_focus_kw_out,
            blog_sec_kws_out,
            blog_markdown_out,
            blog_gutenberg_out,
            blog_img_prompts_out,
            blog_img_1, blog_img_2, blog_img_3, blog_img_4, blog_img_5, blog_img_6,
            blog_cap_1, blog_cap_2, blog_cap_3, blog_cap_4, blog_cap_5, blog_cap_6,
            blog_zip_file
        ],
        show_progress="full"
    )

    btn_extract_images_only.click(
        step8_extract_images_only,
        inputs=[blog_num_images, blog_keyframe_res, blog_enhance_text],
        outputs=[
            blog_status,
            blog_img_1, blog_img_2, blog_img_3, blog_img_4, blog_img_5, blog_img_6,
            blog_cap_1, blog_cap_2, blog_cap_3, blog_cap_4, blog_cap_5, blog_cap_6,
            blog_zip_file
        ],
        show_progress="full"
    )

    btn_open_blog_folder.click(open_output_folder, [], [])

    def on_sync_humanizer():
        res = sync_humanizer_rules_from_github()
        return res.get("message", "Synced.")

    btn_sync_humanizer.click(on_sync_humanizer, inputs=[], outputs=[humanizer_sync_status])

    # FLUX.1-schnell Generative Handlers
    def on_install_flux(progress=gr.Progress()):
        progress(0.1, "Installing dependencies & downloading FLUX.1-schnell weights...")
        res = flux_studio.download_model_weights()
        return res.get("status_text")

    btn_install_flux.click(
        on_install_flux,
        inputs=[],
        outputs=[flux_model_status],
        show_progress="full"
    )

    def on_delete_flux():
        res = flux_studio.delete_model_weights()
        return res.get("status_text")

    btn_delete_flux.click(
        on_delete_flux,
        inputs=[],
        outputs=[flux_model_status]
    )

    def on_unload_flux():
        flux_studio.unload()
        return "🧹 FLUX model unloaded from VRAM successfully! (GPU memory freed)"

    btn_unload_flux.click(
        on_unload_flux,
        inputs=[],
        outputs=[flux_status]
    )

    def on_enhance_prompt(current_prompt, style_choice):
        video_title = state.video_info.get("title", "") if state.video_info else ""
        enhanced = flux_studio.enhance_prompt(
            user_input=current_prompt,
            video_title=video_title,
            style_preset=style_choice
        )
        return enhanced

    btn_enhance_prompt.click(
        on_enhance_prompt,
        inputs=[flux_prompt, flux_style_preset],
        outputs=[flux_prompt]
    )

    def free_vram_for_flux():
        try:
            from modules.llm_backends.factory import get_backend as get_llm_backend
            for b in ["qwen3.5-9b", "local_causal"]:
                try:
                    backend = get_llm_backend(b)
                    if backend and hasattr(backend, "unload"):
                        backend.unload()
                except Exception:
                    pass
        except Exception:
            pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_run_flux(prompt, ref_img, aspect_ratio, steps_choice, style_choice, seed_choice):
        if not prompt or not prompt.strip():
            return (
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                "⚠️ Please enter a prompt or topic for FLUX.1-schnell."
            )
        
        free_vram_for_flux()
        steps_val = 4
        if "6" in str(steps_choice):
            steps_val = 6
        elif "8" in str(steps_choice):
            steps_val = 8
        aspect_val = "16:9"
        if "9:16" in str(aspect_ratio):
            aspect_val = "9:16"
        elif "1:1" in str(aspect_ratio):
            aspect_val = "1:1"

        seed_val = None
        try:
            if seed_choice is not None and int(seed_choice) >= 0:
                seed_val = int(seed_choice)
        except (ValueError, TypeError):
            seed_val = None

        video_title = state.video_info.get("title", "") if state.video_info else ""
        raw_prompt = prompt.strip()

        # Check if user entered short words or a full custom prompt
        quotes_found = re.findall(r"['\"]([^'\"]+)['\"]", raw_prompt)
        words_count = len(raw_prompt.split())

        if words_count <= 3 and not quotes_found:
            # 1-3 words without quotes: enhance into a high-CTR scene with 3D text
            tailored_prompt = flux_studio.enhance_prompt(
                user_input=raw_prompt,
                video_title=video_title,
                style_preset=style_choice
            )
        else:
            # User typed custom prompt or quotes: preserve user's prompt
            # Standardize quotes to double quotes for FLUX T5 tokenizer and strip accents
            formatted = re.sub(r"'(.*?)'", r'"\1"', raw_prompt)
            tailored_prompt = flux_studio.clean_ascii_typography(formatted)

        res = flux_studio.generate_thumbnail(
            prompt=tailored_prompt,
            reference_image_path=ref_img,
            aspect_ratio=aspect_val,
            steps=steps_val,
            seed=seed_val
        )

        if res.get("success"):
            img_path = res.get("image_path")
            used_seed = res.get("seed", "")
            status_msg = res.get("message", f"✅ Thumbnail generated! (Seed: {used_seed})")

            # Route to appropriate tab and keep other tabs intact (never None)
            style_str = str(style_choice).lower()
            if "3d" in style_str or "tech" in style_str:
                return (
                    gr.update(selected="tab_variant_b"),
                    gr.update(), gr.update(),
                    img_path, img_path,
                    gr.update(), gr.update(),
                    gr.update(),
                    status_msg
                )
            elif "photorealistic" in style_str or "studio" in style_str or "cinematic" in style_str:
                return (
                    gr.update(selected="tab_variant_c"),
                    gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    img_path, img_path,
                    gr.update(),
                    status_msg
                )
            else:
                return (
                    gr.update(selected="tab_variant_a"),
                    img_path, img_path,
                    gr.update(), gr.update(),
                    gr.update(), gr.update(),
                    gr.update(),
                    status_msg
                )
        else:
            return (
                gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(),
                res.get("message", "❌ Error generating FLUX image.")
            )

    def on_run_flux_ab(prompt, ref_img, aspect_ratio, steps_choice, seed_choice, progress=gr.Progress()):
        video_title = state.video_info.get("title", "") if state.video_info else ""
        base_text = prompt.strip() if prompt and prompt.strip() else video_title
        if not base_text:
            return (
                gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(),
                "⚠️ Please enter a prompt or import a video first."
            )

        free_vram_for_flux()
        steps_val = 4
        if "6" in str(steps_choice):
            steps_val = 6
        elif "8" in str(steps_choice):
            steps_val = 8
        aspect_val = "16:9"
        if "9:16" in str(aspect_ratio):
            aspect_val = "9:16"
        elif "1:1" in str(aspect_ratio):
            aspect_val = "1:1"

        seed_val = None
        try:
            if seed_choice is not None and int(seed_choice) >= 0:
                seed_val = int(seed_choice)
        except (ValueError, TypeError):
            seed_val = None

        def prog_cb(pct, msg):
            progress(pct, msg)

        res = flux_studio.generate_ab_thumbnails(
            base_prompt=base_text,
            video_title=video_title,
            reference_image_path=ref_img,
            aspect_ratio=aspect_val,
            steps=steps_val,
            base_seed=seed_val,
            progress_callback=prog_cb
        )

        if res.get("success"):
            variants = res.get("variants", [])
            img_a = variants[0]["image_path"] if len(variants) > 0 else None
            img_b = variants[1]["image_path"] if len(variants) > 1 else None
            img_c = variants[2]["image_path"] if len(variants) > 2 else None
            zip_path = res.get("zip_path")
            return (
                gr.update(selected="tab_variant_a"),
                img_a, img_a,
                img_b, img_b,
                img_c, img_c,
                zip_path,
                res.get("message")
            )
        else:
            return (
                gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                gr.update(),
                res.get("message", "❌ Error generating A/B thumbnails.")
            )

    btn_run_flux.click(
        on_run_flux,
        inputs=[flux_prompt, flux_ref_img, flux_aspect, flux_steps, flux_style_preset, flux_seed],
        outputs=[flux_variant_tabs, flux_thumb_a, file_down_a, flux_thumb_b, file_down_b, flux_thumb_c, file_down_c, flux_ab_zip, flux_status],
        show_progress="full"
    )

    btn_run_flux_ab.click(
        on_run_flux_ab,
        inputs=[flux_prompt, flux_ref_img, flux_aspect, flux_steps, flux_seed],
        outputs=[flux_variant_tabs, flux_thumb_a, file_down_a, flux_thumb_b, file_down_b, flux_thumb_c, file_down_c, flux_ab_zip, flux_status],
        show_progress="full"
    )

    btn_random_seed.click(fn=lambda: -1, inputs=[], outputs=[flux_seed])

    tab_variant_a.select(fn=lambda: "YouTube Viral High-CTR", inputs=[], outputs=[flux_style_preset])
    tab_variant_b.select(fn=lambda: "3D Isometric & Tech Glow", inputs=[], outputs=[flux_style_preset])
    tab_variant_c.select(fn=lambda: "Photorealistic Studio Shot", inputs=[], outputs=[flux_style_preset])

    def on_style_preset_change(style):
        style_str = str(style).lower()
        if "3d" in style_str or "tech" in style_str:
            return gr.update(selected="tab_variant_b")
        elif "photorealistic" in style_str or "studio" in style_str or "cinematic" in style_str:
            return gr.update(selected="tab_variant_c")
        else:
            return gr.update(selected="tab_variant_a")

    flux_style_preset.change(
        on_style_preset_change,
        inputs=[flux_style_preset],
        outputs=[flux_variant_tabs]
    )

    btn_open_flux_folder.click(open_output_folder, [], [])

    def on_apply_flux_thumb(gen_img):
        if gen_img is None:
            return gr.update()
        return gen_img

    btn_apply_a.click(on_apply_flux_thumb, inputs=[flux_thumb_a], outputs=[blog_img_1])
    btn_apply_b.click(on_apply_flux_thumb, inputs=[flux_thumb_b], outputs=[blog_img_1])
    btn_apply_c.click(on_apply_flux_thumb, inputs=[flux_thumb_c], outputs=[blog_img_1])

    # CPS Config tab handlers
    def save_cps_table(df):
        global _user_cps_overrides
        overrides = {}
        for _, row in df.iterrows():
            iso = str(row["ISO"]).lower().strip()
            val = row["Your CPS"]
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                try:
                    fval = float(val)
                    if fval > 0:
                        overrides[iso] = fval
                except (ValueError, TypeError):
                    pass
        _user_cps_overrides = overrides
        save_user_cps(overrides)
        n = len(overrides)
        return f"✅ Saved — {n} active override(s)."

    def reset_cps_table():
        global _user_cps_overrides
        _user_cps_overrides = {}
        save_user_cps({})
        return _build_cps_dataframe(), "↩️ All values reset to defaults."

    btn_save_cps.click(save_cps_table, [cps_table], [cps_status])
    btn_reset_cps.click(reset_cps_table, [], [cps_table, cps_status])

    # Backend Change Logic
    def update_language_dropdowns():
        valid_lang_choices = get_valid_languages(current_tts_backend, current_llm_backend)
        new_lang_value = valid_lang_choices[0] if valid_lang_choices else None
        
        return (
            gr.update(choices=valid_lang_choices, value=new_lang_value),
            gr.update(choices=valid_lang_choices, value=[new_lang_value] if new_lang_value else [])
        )
        
    def on_tts_backend_change(selected_name):
        global tts_engine, current_tts_backend
        current_tts_backend = selected_name
        user_config["tts_backend"] = current_tts_backend
        save_config(user_config)
        
        # Switch backend
        tts_engine = get_tts_backend(current_tts_backend)
        time_sync.tts = tts_engine
        
        return update_language_dropdowns()
        
    def on_llm_backend_change(selected_name):
        global current_llm_backend
        current_llm_backend = selected_name
        user_config["llm_backend"] = current_llm_backend
        save_config(user_config)
        
        # Switch backend for reformulator
        reformulator.backend_name = current_llm_backend
        
        return update_language_dropdowns()
        
    tts_backend_dropdown.change(
        on_tts_backend_change,
        [tts_backend_dropdown],
        [lang_target, bulk_target_langs]
    )
    
    llm_backend_dropdown.change(
        on_llm_backend_change,
        [llm_backend_dropdown],
        [lang_target, bulk_target_langs]
    )

    def on_app_load():
        """Restore UI state on page load/reload if the backend session has active video/transcriptions."""
        has_video = state.video_info is not None
        if not has_video:
            # Return static default values instead of bare gr.update() to avoid Svelte re-render loops in Gradio 5/6
            return (
                None, None,
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
                gr.update(value=None), _get_empty_segments_html(),
                gr.update(value=None), gr.update(value=None),
                "", "", "", ""
            )
            
        has_segments = len(state.segments) > 0
        has_translated = len(state.translated_segments) > 0
        
        video_val = state.video_info.get('video_path') if 'video_path' in state.video_info else None
        is_audio = state.video_info.get('is_audio_only', False)
        
        # Transcription dataframe
        trans_update = gr.update(value=None)
        if has_segments:
            trans_data = []
            for seg in state.segments:
                trans_data.append([round(seg['start'], 2), round(seg['end'], 2), seg['text']])
            trans_update = gr.update(value=trans_data)
                
        # Translation dataframe
        trans_df_update = gr.update(value=None)
        if has_translated:
            trans_data_df = []
            for seg in state.translated_segments:
                normal_text = seg.get("normal_text", "")
                status = "Ready" if find_segment_audio_path(seg.get('start', 0.0), lang=state.video_info.get('target_language')) else "Not Generated"
                trans_data_df.append([round(seg.get('start', 0.0), 2), round(seg.get('end', 0.0), 2), seg.get('text', ''), normal_text, status])
            trans_df_update = gr.update(value=trans_data_df)
                
        # Dubbing segments dataframe
        dubbing_update = gr.update(value=None)
        if has_translated:
            dubbing_data = _build_dubbing_df_data("Fitted Translation")
            dubbing_update = gr.update(value=dubbing_data)
            
        # Metadata
        orig_title = state.video_info.get('title', '')
        orig_desc = state.video_info.get('description', '')
        trans_title = state.video_info.get('translated_title', '')
        trans_desc = state.video_info.get('translated_description', '')
        
        return (
            # video_preview, audio_preview
            gr.update(visible=not is_audio, value=video_val if not is_audio else None),
            gr.update(visible=is_audio, value=video_val if is_audio else None),
            # Interactive buttons
            gr.update(interactive=has_video),  # btn_transcribe
            gr.update(interactive=has_segments),  # btn_translate
            gr.update(interactive=has_translated),  # btn_synth
            gr.update(interactive=has_segments),  # btn_bulk_run
            # transcription_df, segments_json_holder
            trans_update,
            _get_segments_json_html() if (has_segments or has_translated) else _get_empty_segments_html(),
            # translation_df, dubbing_segments_df
            trans_df_update,
            dubbing_update,
            # metadata inputs
            gr.update(value=orig_title),
            gr.update(value=orig_desc),
            gr.update(value=trans_title),
            gr.update(value=trans_desc)
        )

    def on_tab_select(*args):
        try:
            has_video = state.video_info is not None
            has_segments = len(state.segments) > 0 if state.segments else False
            has_translated = len(state.translated_segments) > 0 if state.translated_segments else False
            return (
                gr.update(interactive=has_video),      # btn_transcribe
                gr.update(interactive=has_segments),   # btn_translate
                gr.update(interactive=has_translated), # btn_synth
                gr.update(interactive=has_segments)    # btn_bulk_run
            )
        except Exception:
            return gr.update(), gr.update(), gr.update(), gr.update()
        
    for tab in [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9]:
        tab.select(
            fn=on_tab_select,
            inputs=[],
            outputs=[btn_transcribe, btn_translate, btn_synth, btn_bulk_run]
        )

    # Workflow Next-Step Action Navigation Handlers (Fail-Safe: auto-saves state & unlocks next step even if user didn't click manual validate)
    btn_next_tab1.click(
        fn=lambda: gr.Tabs(selected="tab_transcription"),
        inputs=None,
        outputs=[main_tabs],
        js="() => { window.zastSwitchTab('2.'); }"
    )
    btn_next_tab2.click(
        step3_save_transcription,
        [transcription_df],
        [transcription_status, btn_translate, btn_bulk_run, segments_json_holder, export_transcription_file],
        js="() => { window.zastSwitchTab('3.'); }"
    )
    btn_next_tab3.click(
        step5_save_translation,
        [translation_df, dubbing_text_source, translated_title_input, translated_desc_input],
        [translation_status, btn_synth, segments_json_holder, dubbing_segments_df],
        js="() => { window.zastSwitchTab('4.'); }"
    )
    btn_next_tab4_to_shorts.click(
        fn=lambda: gr.Tabs(selected="tab_shorts"),
        inputs=None,
        outputs=[main_tabs],
        js="() => { window.zastSwitchTab('6.'); }"
    )
    btn_next_tab4_to_blog.click(
        fn=lambda: gr.Tabs(selected="tab_blog"),
        inputs=None,
        outputs=[main_tabs],
        js="() => { window.zastSwitchTab('7.'); }"
    )

    app.load(
        fn=on_app_load,
        inputs=[],
        outputs=[
            video_preview, audio_preview,
            btn_transcribe, btn_translate, btn_synth, btn_bulk_run,
            transcription_df, segments_json_holder,
            translation_df, dubbing_segments_df,
            original_title_input, original_desc_input, translated_title_input, translated_desc_input
        ]
    )

if __name__ == "__main__":
    import os
    import sys
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    app.queue() # Enable websocket queue to prevent GPU process thread deadlocks
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            theme=gr.themes.Soft(),
            allowed_paths=[BASE_DIR],
            js=BLOCKS_JS,
            css=BLOCKS_CSS,
        )
    except OSError:
        # Port is already bound -> automatically pick next available open port
        app.launch(
            server_name="127.0.0.1",
            theme=gr.themes.Soft(),
            allowed_paths=[BASE_DIR],
            js=BLOCKS_JS,
            css=BLOCKS_CSS,
        )
