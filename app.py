import warnings
# Suppress noisy 3rd-party startup warnings
warnings.filterwarnings("ignore", message="rope_config_validation", category=FutureWarning)
warnings.filterwarnings("ignore", message="In 2.9, this function", category=UserWarning)
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", message="\ntorchcodec is not installed")  # pyannote wraps torchcodec RuntimeError
warnings.filterwarnings("ignore", message="fast path is not available")
warnings.filterwarnings("ignore", message="You are using a Python version", category=FutureWarning)  # google-api-core Python EOL warning

import gradio as gr
import os
import shutil
import time
import json
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
from modules.video_assembler import VideoAssembler
from modules.srt_parser import SRTParser
from modules.youtube_publisher import YouTubePublisher
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

# --- GLOBAL STATE ---
class AppState:
    def __init__(self):
        self.video_info = None
        self.segments = []  # Transcriptions
        self.translated_segments = []
        self.synced_segments = []
        self.temp_dir = TEMP_DIR
        self.keep_models = False
        
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
current_llm_backend = user_config.get("llm_backend", "Qwen2.5-7B-Instruct")

available_tts_backends = get_available_tts_backends()
available_llm_backends = get_available_llm_backends()

if current_tts_backend not in available_tts_backends:
    current_tts_backend = "VoxCPM 2"
if current_llm_backend not in available_llm_backends:
    current_llm_backend = "Qwen2.5-7B-Instruct"

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


# --- UI FUNCTIONS ---

def reset_project():
    """Reset all state for a new project."""
    state.video_info = None
    state.segments = []
    state.translated_segments = []
    state.synced_segments = []
    # Clean temp directory
    from config import TEMP_DIR
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
    return (
        "",           # url_input
        None,         # file_input
        "Ready for a new project.",  # status_dl
        gr.update(visible=True, value=None),  # video_preview
        gr.update(visible=False, value=None), # audio_preview
        gr.Button(interactive=False),  # btn_transcribe
        gr.Button(interactive=False),  # btn_translate
        gr.Button(interactive=False),  # btn_synth
        gr.Button(interactive=False),  # btn_bulk_run
        gr.update(visible=False),      # btn_import_metadata
        gr.update(visible=False),      # btn_youtube_publish
        gr.update(visible=False, value=""), # bulk_publish_status
        gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only"]), # bulk_output_type
        gr.update(visible=True, value=None) # final_video_out
    )


def step0_check_url(url):
    """Check YouTube URL and return available resolutions."""
    if not url:
        return "Please enter a YouTube URL.", gr.Dropdown(choices=["1080p"], value="1080p"), gr.Button(interactive=False)
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
        return status, gr.Dropdown(choices=resolutions, value=default, interactive=True), gr.Button(interactive=True)
    except Exception as e:
        return f"Error: {str(e)}", gr.Dropdown(choices=["1080p"], value="1080p"), gr.Button(interactive=False)


def step1_download(url, local_file, resolution, progress=gr.Progress()):
    progress(0, "Downloading...")
    try:
        is_audio = False
        if url:
            info = downloader.download(url, resolution=resolution)
            show_btn = gr.update(visible=True)
        elif local_file:
            filepath = local_file.name if hasattr(local_file, 'name') else local_file
            ext = os.path.splitext(filepath)[1].lower()
            if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
                is_audio = True
            info = downloader.import_local(filepath)
            show_btn = gr.update(visible=False)
        else:
            raise ValueError("Please provide a YouTube URL or a local file.")
        
        info['is_audio_only'] = is_audio
        state.video_info = info
        
        if info.get('youtube_id'):
            show_publish_btn = gr.update(visible=True)
            show_publish_status = gr.update(visible=True, value="")
        else:
            show_publish_btn = gr.update(visible=False)
            show_publish_status = gr.update(visible=False)

        if is_audio:
            return (
                f"Audio loaded: {info['title']}",
                gr.update(visible=False, value=None),
                gr.update(visible=True, value=info['video_path']),
                gr.update(interactive=True),
                show_btn,
                show_publish_btn,
                show_publish_status,
                gr.update(value="Audio Only", choices=["Audio Only"]),
                gr.update(visible=False, value=None)
            )
        else:
            return (
                f"Video loaded: {info['title']}",
                gr.update(visible=True, value=info['video_path']),
                gr.update(visible=False, value=None),
                gr.update(interactive=True),
                show_btn,
                show_publish_btn,
                show_publish_status,
                gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only"]),
                gr.update(visible=True, value=None)
            )
    except Exception as e:
        return (
            f"Error: {str(e)}",
            None,
            None,
            gr.Button(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="Video + Audio", choices=["Video + Audio", "Audio Only"]),
            gr.update(visible=True)
        )



def step2_transcribe(lang_source, model_size, progress=gr.Progress()):
    if not state.video_info:
        return "Error: No video loaded.", None
    
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
    
    res = transcriber.transcribe(
        state.video_info['audio_16k'], 
        language=lang_code,
        enable_diarization=False
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
    
    return f"Transcription complete ({len(data)} segments). Review below, then click 'Validate Transcription'.", gr.Dataframe(value=data)

def step2b_import_srt(srt_file, lang_source):
    """Import an SRT file as transcription."""
    if srt_file is None:
        return "Error: No SRT file selected.", None
    
    try:
        segments, errors = srt_parser.convert_user_srt_to_segments(srt_file)
        
        if not segments:
            return "Error: No segments found in SRT file.", None
        
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
        
        return f"SRT imported ({len(data)} segments).{warning} Review below, then click 'Validate Transcription'.", gr.Dataframe(value=data)
    except Exception as e:
        return f"Error importing SRT: {str(e)}", None

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
        return "⚠️ No segments found. Make sure the transcription table has data.", gr.Button(interactive=False), gr.Button(interactive=False)
    return f"✅ Transcription validated ({len(new_segments)} segments). Go to the 'Translation' tab.", gr.Button(interactive=True), gr.Button(interactive=True)

def step4_translate(target_lang, progress=gr.Progress()):
    if not state.segments:
        return "Error: No transcription available.", None

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
    
    # PHASE 3: Normal/full translation (no length constraint)
    progress(0.6, "Phase 3/3: Natural full translation...")
    reformulator.translate_normal(state.translated_segments, source_lang, target_lang_code)
    
    if not state.keep_models:
        reformulator.cleanup()
    
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
        
    return status_msg, data

def step5_save_translation(data):
    rows = _dataframe_to_rows(data)
    count = 0
    import re
    for i, row in enumerate(rows):
        if i < len(state.translated_segments):
            # Column 3 = Translation (normal/full)
            normal_text = str(row[3]).strip()
            if normal_text:
                state.translated_segments[i]['normal_text'] = normal_text
            
            # Column 4 = Fitted (with status emoji prefix — strip it)
            text = str(row[4])
            text = re.sub(r'^[✅⚠️\ufe0f]+\s*', '', text)
            state.translated_segments[i]['translated_text'] = text
            count += 1
    
    if count == 0:
        return "⚠️ No translation data found.", gr.Button(interactive=False)
    return f"✅ Translation validated ({count} segments). Go to the 'Dubbing & Export' tab.", gr.Button(interactive=True)

def export_transcription_srt():
    """Export current transcription as SRT file with source language ISO code."""
    if not state.segments:
        return "No transcription to export.", None
    src_lang = state.video_info.get('detected_language', '') if state.video_info else ''
    iso = _get_iso_code(src_lang)
    srt_path = os.path.join(TEMP_DIR, f"transcription_{iso}.srt")
    srt_parser.segments_to_srt(state.segments, srt_path)
    return f"Exported {len(state.segments)} segments.", srt_path

def export_translation_srt():
    """Export normal/full translation as SRT file."""
    if not state.translated_segments:
        return "No translation to export.", None
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    srt_path = os.path.join(TEMP_DIR, f"translation_{iso}.srt")
    srt_parser.segments_to_srt(state.translated_segments, srt_path, text_key="normal_text")
    return f"Exported {len(state.translated_segments)} segments (full translation).", srt_path

def export_fitted_srt():
    """Export fitted/concise translation as SRT file (used for dubbing)."""
    if not state.translated_segments:
        return "No translation to export.", None
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    srt_path = os.path.join(TEMP_DIR, f"fitted_{iso}.srt")
    srt_parser.segments_to_srt(state.translated_segments, srt_path, text_key="translated_text")
    return f"Exported {len(state.translated_segments)} segments (fitted for dubbing).", srt_path

def step6_synthesize(voice_mode, voice_file, never_cut, default_voice_gender="Woman", progress=gr.Progress()):
    if not state.translated_segments:
        return "Error: No translation available.", None, None
    
    progress(0.05, "Initializing TTS & Sync...")
    
    # Check if we need to separate background audio (e.g. if skipped via SRT import)
    if state.video_info and 'background' not in state.video_info:
        progress(0.05, "Separating background audio (was skipped)...")
        stems = separator.separate(state.video_info['audio_44k'])
        state.video_info['vocals'] = stems['vocals']
        state.video_info['background'] = stems['background']
        if not state.keep_models:
            separator.cleanup()
            
        # Try to extract clean reference audio since we just got the vocals
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
            if best_seg:
                data, sr = sf.read(state.video_info['vocals'])
                start_sample = int(best_seg['start'] * sr)
                end_sample = int(best_seg['end'] * sr)
                extracted = data[start_sample:end_sample]
                out_path = os.path.join(TEMP_DIR, "ref_audio_extracted.wav")
                sf.write(out_path, extracted, sr)
                state.video_info['ref_audio_extracted'] = out_path
        except Exception as e:
            print(f"Failed to extract clean reference audio after delayed separation: {e}")

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
        return status, gr.update(visible=False, value=None), mixed_audio
    else:
        progress(0.9, "Assembling final video...")
        final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
        video_assembler.assemble(
            state.video_info['video_path'], 
            mixed_audio, 
            final_video
        )
        return status, gr.update(visible=True, value=final_video), mixed_audio


def export_audio():
    """Export the mixed audio as a downloadable file."""
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    audio_path = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
    if os.path.exists(audio_path):
        return f"Audio exported ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB).", audio_path
    return "No audio available. Run Dubbing first.", None


def step5_bulk_run(target_langs, voice_mode, voice_file, never_cut, output_type, bulk_title, bulk_desc, default_voice_gender="Woman", progress=gr.Progress()):
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
    
    # Detect source language from transcription
    source_lang = state.video_info.get('detected_language', 'en') if state.video_info else 'en'
    
    # Check if we need to separate background audio (e.g. if skipped via SRT import)
    if state.video_info and 'background' not in state.video_info:
        yield "Separating background audio (was skipped)...", output_files, metadata_display
        stems = separator.separate(state.video_info['audio_44k'])
        state.video_info['vocals'] = stems['vocals']
        state.video_info['background'] = stems['background']
        if not state.keep_models:
            separator.cleanup()
            
        # Try to extract clean reference audio since we just got the vocals
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
            if best_seg:
                data, sr = sf.read(state.video_info['vocals'])
                start_sample = int(best_seg['start'] * sr)
                end_sample = int(best_seg['end'] * sr)
                extracted = data[start_sample:end_sample]
                out_path = os.path.join(TEMP_DIR, "ref_audio_extracted.wav")
                sf.write(out_path, extracted, sr)
                state.video_info['ref_audio_extracted'] = out_path
        except Exception as e:
            print(f"Failed to extract clean reference audio after delayed separation: {e}")
    
    # Phase 1: Translate ALL languages first to prevent VRAM fragmentation
    all_translated_segments = {}
    for idx, target_lang in enumerate(target_langs):
        # Progress math setup
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
        srt_parser.segments_to_srt(translated, trans_srt, text_key="normal_text")
        output_files.append(trans_srt)
        
        fitted_srt = os.path.join(TEMP_DIR, f"fitted_{iso}.srt")
        srt_parser.segments_to_srt(translated, fitted_srt, text_key="translated_text")
        output_files.append(fitted_srt)
        
        # Store SRT path for YouTube Publishing (We prefer natural translation for subtitles)
        state.bulk_results['srts'][target_lang_code] = trans_srt

        # Store for the synthesis phase
        import copy
        all_translated_segments[target_lang] = copy.deepcopy(translated)

    # CRITICAL: Clean up LLM from VRAM completely before loading TTS
    if not state.keep_models:
        reformulator.cleanup()
    
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
            
    if not state.keep_models:
        tts_engine.cleanup()
    
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

# --- GRADIO INTERFACE ---

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

with gr.Blocks(title="ZastTranslate") as app:
    # Embed logo as base64 to avoid Gradio version compatibility issues
    import base64 as _b64
    _logo_path = os.path.join(BASE_DIR, "zastttranslate.png")
    _logo_html = ""
    if os.path.exists(_logo_path):
        with open(_logo_path, "rb") as _f:
            _logo_b64 = _b64.b64encode(_f.read()).decode()
        _logo_html = f"<center><img src='data:image/png;base64,{_logo_b64}' width='80' /></center>\n\n"
    gr.Markdown(f"{_logo_html}# 🎬 ZastTranslate — Beta 1.05\n**Offline video translation & dubbing (No Lip-Sync)**")
    
    with gr.Tab("1. Import"):
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        with gr.Row():
            btn_check = gr.Button("🔍 Check URL", variant="secondary")
            yt_resolution = gr.Dropdown(
                ["1080p"], label="Resolution", value="1080p",
                interactive=False, info="Click 'Check URL' to see available resolutions"
            )
            tts_backend_dropdown = gr.Dropdown(
                choices=list(available_tts_backends.keys()),
                value=current_tts_backend,
                label="TTS Model (Voice)",
                interactive=True
            )
        file_input = gr.File(
            label="Or upload a local video or audio file", 
            file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
        )
        with gr.Row():
            btn_dl = gr.Button("Import Video or Audio", variant="primary")
            btn_reset = gr.Button("New Project", variant="secondary")
        status_dl = gr.Textbox(label="Status", interactive=False)
        video_preview = gr.Video(label="Preview", height=300)
        audio_preview = gr.Audio(label="Audio Preview", visible=False)
        
    with gr.Tab("2. Transcription"):
        with gr.Row():
            lang_source = gr.Dropdown(
                ["Auto", "French", "English", "Spanish", "German", "Italian", "Portuguese",
                 "Japanese", "Korean", "Chinese", "Russian", "Arabic", "Hindi",
                 "Dutch", "Polish", "Turkish", "Swedish", "Czech", "Romanian", "Hungarian"],
                label="Source Language", value="Auto"
            )
            model_size = gr.Dropdown(["base", "small", "medium", "large-v3"], label="Whisper Model", value="base")
            llm_backend_dropdown = gr.Dropdown(
                choices=list(available_llm_backends.keys()),
                value=current_llm_backend,
                label="LLM Model (Translation)",
                interactive=True
            )

        
        with gr.Row():
            btn_transcribe = gr.Button("Run Transcription", interactive=False, variant="primary")
        
        gr.Markdown("**Or import an existing SRT file:**")
        with gr.Row():
            srt_file_input = gr.File(label="Upload SRT file", file_types=[".srt"])
            btn_import_srt = gr.Button("Import SRT", variant="secondary")
        
        transcription_status = gr.Textbox(label="Status", interactive=False)
        transcription_df = gr.Dataframe(
            headers=["Start", "End", "Text"],
            label="Edit Transcription",
            interactive=True,
            wrap=True,
            max_height=400
        )
        with gr.Row():
            btn_valid_transcription = gr.Button("Validate Transcription ✅", variant="primary")
            btn_export_transcription = gr.Button("Export SRT 💾", variant="secondary")
        export_transcription_file = gr.File(label="Download SRT")

    with gr.Tab("3. Translation"):
        lang_target = gr.Dropdown(INITIAL_VALID_LANGS, label="Target Language", value=INITIAL_LANG_VALUE)
        btn_translate = gr.Button("Run Translation", interactive=False, variant="primary")
        translation_status = gr.Textbox(label="Status", interactive=False)
        translation_df = gr.Dataframe(
            headers=["Start", "End", "Original", "Translation", "Fitted"],
            label="Edit Translation",
            interactive=True,
            wrap=True,
            max_height=400
        )
        with gr.Row():
            btn_valid_translation = gr.Button("Validate Translation ✅", variant="primary")
            btn_export_translation = gr.Button("Export Translation SRT 💾", variant="secondary")
            btn_export_fitted = gr.Button("Export Fitted SRT 💾", variant="secondary")
        export_translation_file = gr.File(label="Download SRT")
        
    with gr.Tab("4. Dubbing & Export"):
        voice_mode = gr.Radio(
            ["Default voice", "Clone from original", "Clone from file"], 
            label="Voice Mode", 
            value="Default voice",
            info="'Clone from original' uses the separated vocals from step 2. 'Clone from file' requires uploading a voice sample."
        )
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
            info="All text will be spoken in full. May cause desync with on-screen actions."
        )
        never_cut_warning = gr.Markdown(value="", visible=False)
        
        btn_synth = gr.Button("Run Synthesis & Export", interactive=False, variant="primary")
        synth_status = gr.Textbox(label="Status", interactive=False)
        final_video_out = gr.Video(label="Final Video", height=300)
        final_audio_out = gr.Audio(label="Mixed Audio")
        with gr.Row():
            btn_export_audio = gr.Button("Export Audio 🎵", variant="secondary")
        export_audio_file = gr.File(label="Download Audio (WAV)")

    with gr.Tab("5. Bulk Mode"):
        gr.Markdown("### Automate translation and dubbing for multiple languages at once")
        
        bulk_target_langs = gr.Dropdown(
            INITIAL_VALID_LANGS, 
            label="Target Languages", 
            multiselect=True,
            info="Select all the languages you want to translate and dub."
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                bulk_title_input = gr.Textbox(label="Original Video Title (Optional)", placeholder="Title...")
                bulk_desc_input = gr.Textbox(label="Original Video Description (Optional)", placeholder="Description...", lines=3)
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
                ["Video + Audio", "Audio Only"],
                label="Output Generation",
                value="Video + Audio",
                info="'Video + Audio' will render the final MP4. 'Audio Only' will just output the WAV track."
            )
            
        bulk_voice_file = gr.File(label="Voice sample file (WAV/MP3, 10-30s of clear speech)", visible=False)
        
        bulk_never_cut_mode = gr.Checkbox(
            label="🔊 Never Cut Vocal",
            value=False,
            info="All text will be spoken in full. May cause desync with on-screen actions."
        )
        bulk_never_cut_warning = gr.Markdown(value="", visible=False)
        
        btn_bulk_run = gr.Button("Run Bulk Process", interactive=False, variant="primary")
        bulk_status_output = gr.Textbox(label="Status", interactive=False)
        bulk_files_output = gr.File(label="Generated Files Output", file_count="multiple")
        bulk_metadata_output = gr.Markdown(label="Translated Metadata", height=600)
        
        with gr.Row():
            btn_youtube_publish = gr.Button("🔴 Publish Metadata & Subtitles to YouTube", variant="primary", visible=False)
            bulk_publish_status = gr.Textbox(label="Publish Status", interactive=False, visible=False)

    with gr.Tab("ℹ️ Help"):
        gr.Markdown("## How to use ZastTranslate")
        
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
        
        with gr.Accordion("🎤 Tab 2 — Transcription", open=False):
            gr.Markdown(
                "This step separates vocals from background music (Demucs), then transcribes the speech (WhisperX).\n\n"
                "**Options:**\n"
                "- **Source Language** — Select the spoken language from 20+ languages, or leave on *Auto* for auto-detection. "
                "Setting it manually improves accuracy.\n"
                "- **Whisper Model** — Choose the model size:\n"
                "  - `base` — Fast, lower accuracy (good for testing)\n"
                "  - `small` / `medium` — Balanced\n"
                "  - `large-v3` — Best accuracy, uses more VRAM (~3 GB)\n\n"
                "**After transcription:**\n"
                "- Review and edit the table (Start, End, Text). You can fix mistakes, split/merge segments.\n"
                "- Click **Export SRT 💾** to download subtitles.\n\n"
                "⚠️ **You MUST click 'Validate Transcription ✅' before going to the Translation tab.** "
                "Without validation, the next step will not have any data to work with.\n\n"
                "**Alternative:** You can skip transcription entirely by importing an existing **SRT file** instead."
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
                "You can adjust per-language values in the **⚙️ Config CPS** tab."
            )
        
        with gr.Accordion("🎬 Tab 4 — Dubbing & Export", open=False):
            gr.Markdown(
                "Generate the dubbed video with synthesized speech.\n\n"
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
                "- **Final Video** — Dubbed MP4 video ready to share\n"
                "- **Mixed Audio** — Listen to the voice + background mix\n"
                "- **Export Audio 🎵** — Download the audio track separately as WAV\n\n"
                "**⚠️ Current limitations:**\n"
                "- **No lip-sync** — The audio is replaced but the video is not modified (no face/lip adaptation)\n"
                "- **Single voice only** — All segments use the same voice. Multi-speaker dubbing is not supported yet."
            )
        

        
        with gr.Accordion("📚 Tab 5 — Bulk Mode", open=False):
            gr.Markdown(
                "Automate translation and dubbing for multiple languages simultaneously.\n\n"
                "**How it works:**\n"
                "1. Select all the target languages from the dropdown.\n"
                "2. Optionally, provide the **Original Video Title** and **Description**, or click **⬇️ Import from URL** to fetch them automatically if you used a YouTube link.\n"
                "3. Configure your **Voice Mode** and **Output Generation** preferences (Video+Audio or Audio Only).\n"
                "4. Click **Run Bulk Process**. The system handles all translations first, then all audio synthesis, and packages everything in a ZIP file.\n\n"
                "**YouTube Publishing (🔴 Publish Metadata & Subtitles to YouTube):**\n"
                "- This button appears if you imported the video via a YouTube URL.\n"
                "- It allows you to automatically upload the translated title, description, and subtitles (SRT) directly to your YouTube video.\n"
                "- **Prerequisites**: You must have a Google Cloud API `client_secret.json` file (with YouTube Data API v3 enabled) placed in the application folder (`ZastTranslate/client_secret.json`). You must also own the YouTube channel.\n"
                "- **Usage**: After the Bulk Process completes, click this button to upload the localized data to YouTube."
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

        with gr.Accordion("🔗 About / Links", open=False):
            gr.Markdown(
                "**ZastTranslate** is made by Zast.\n\n"
                "- 🌐 [zast57.com](https://zast57.com) — Website\n"
                "- 🤓 [paradoxetemporel.fr](https://paradoxetemporel.fr) — Tech & Geek blog\n"
                "- 🎬 [zast.fr](https://zast.fr) — YouTube channel"
            )

    with gr.Tab("⚙️ Config CPS"):
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

    # EVENTS
    btn_check.click(step0_check_url, [url_input], [status_dl, yt_resolution, btn_dl])
    btn_dl.click(step1_download, [url_input, file_input, yt_resolution], [status_dl, video_preview, audio_preview, btn_transcribe, btn_import_metadata, btn_youtube_publish, bulk_publish_status, bulk_output_type, final_video_out])
    btn_reset.click(reset_project, [], [url_input, file_input, status_dl, video_preview, audio_preview, btn_transcribe, btn_translate, btn_synth, btn_bulk_run, btn_import_metadata, btn_youtube_publish, bulk_publish_status, bulk_output_type, final_video_out])
    
    btn_transcribe.click(step2_transcribe, [lang_source, model_size], [transcription_status, transcription_df])
    btn_import_srt.click(step2b_import_srt, [srt_file_input, lang_source], [transcription_status, transcription_df])
    
    btn_valid_transcription.click(step3_save_transcription, [transcription_df], [transcription_status, btn_translate, btn_bulk_run])
    btn_export_transcription.click(export_transcription_srt, [], [transcription_status, export_transcription_file])
    
    btn_translate.click(step4_translate, [lang_target], [translation_status, translation_df])
    
    btn_valid_translation.click(step5_save_translation, [translation_df], [translation_status, btn_synth])
    btn_export_translation.click(export_translation_srt, [], [translation_status, export_translation_file])
    btn_export_fitted.click(export_fitted_srt, [], [translation_status, export_translation_file])
    
    def toggle_never_cut_warning(enabled):
        if enabled:
            return gr.Markdown(value=NEVER_CUT_WARNING, visible=True)
        return gr.Markdown(value="", visible=False)
    
    never_cut_mode.change(toggle_never_cut_warning, [never_cut_mode], [never_cut_warning])
    bulk_never_cut_mode.change(toggle_never_cut_warning, [bulk_never_cut_mode], [bulk_never_cut_warning])
    
    def toggle_voice_inputs(mode):
        return gr.update(visible=(mode == "Clone from file")), gr.update(visible=(mode == "Default voice"))
        
    voice_mode.change(toggle_voice_inputs, inputs=[voice_mode], outputs=[voice_file, default_voice_gender])
    bulk_voice_mode.change(toggle_voice_inputs, inputs=[bulk_voice_mode], outputs=[bulk_voice_file, bulk_default_voice_gender])
    
    keep_models_ui.change(lambda x: setattr(state, 'keep_models', x), inputs=[keep_models_ui], outputs=[])
    
    def import_metadata_from_state():
        if state.video_info:
            return state.video_info.get('title', ''), state.video_info.get('description', '')
        return "", ""
    btn_import_metadata.click(import_metadata_from_state, [], [bulk_title_input, bulk_desc_input])
    
    btn_synth.click(step6_synthesize, [voice_mode, voice_file, never_cut_mode, default_voice_gender], [synth_status, final_video_out, final_audio_out])
    btn_export_audio.click(export_audio, [], [synth_status, export_audio_file])
    
    btn_bulk_run.click(
        step5_bulk_run, 
        [bulk_target_langs, bulk_voice_mode, bulk_voice_file, bulk_never_cut_mode, bulk_output_type, bulk_title_input, bulk_desc_input, bulk_default_voice_gender], 
        [bulk_status_output, bulk_files_output, bulk_metadata_output]
    )
    
    btn_youtube_publish.click(step6_publish_youtube, [], [bulk_publish_status])

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

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        allowed_paths=[BASE_DIR],
    )
