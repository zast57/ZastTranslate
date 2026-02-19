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
from modules.tts_engine import TTSEngine
from modules.time_sync import TimeSync
from modules.audio_mixer import AudioMixer
from modules.video_assembler import VideoAssembler
from modules.srt_parser import SRTParser

# --- HELPERS ---

# NLLB code → ISO 639-1 uppercase
_NLLB_TO_ISO = {
    "fra_Latn": "FR", "eng_Latn": "EN", "spa_Latn": "ES",
    "deu_Latn": "DE", "ita_Latn": "IT", "por_Latn": "PT",
    "jpn_Jpan": "JA", "kor_Hang": "KO", "zho_Hans": "ZH",
    "rus_Cyrl": "RU",
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
        
state = AppState()

# --- MODULE INSTANCES (lazy loading) ---
downloader = VideoDownloader()
separator = VocalSeparator()
transcriber = Transcriber()
translator = Translator()
reformulator = Reformulator()
tts_engine = TTSEngine()
time_sync = TimeSync(tts_engine, reformulator)
audio_mixer = AudioMixer()
video_assembler = VideoAssembler()
srt_parser = SRTParser()

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
        None,         # video_preview
        gr.Button(interactive=False),  # btn_transcribe
        gr.Button(interactive=False),  # btn_translate
        gr.Button(interactive=False),  # btn_synth
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
        if url:
            info = downloader.download(url, resolution=resolution)
        elif local_file:
            info = downloader.import_local(local_file)
        else:
            raise ValueError("Please provide a YouTube URL or a local file.")
        
        state.video_info = info
        return f"Video loaded: {info['title']}", info['video_path'], gr.Button(interactive=True)
    except Exception as e:
        return f"Error: {str(e)}", None, gr.Button(interactive=False)



def step2_transcribe(lang_source, model_size, progress=gr.Progress()):
    if not state.video_info:
        return "Error: No video loaded.", None
    
    progress(0.1, "Separating vocals...")
    stems = separator.separate(state.video_info['audio_44k'])
    state.video_info['vocals'] = stems['vocals']
    state.video_info['background'] = stems['background']
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
    transcriber.cleanup()
    
    # Prepare dataframe for editor
    data = []
    for seg in state.segments:
        data.append([
            round(seg['start'], 2), 
            round(seg['end'], 2), 
            seg['text']
        ])
    
    return f"Transcription complete ({len(data)} segments). Review below, then click 'Validate Transcription'.", gr.Dataframe(value=data)

def step2b_import_srt(srt_file):
    """Import an SRT file as transcription."""
    if srt_file is None:
        return "Error: No SRT file selected.", None
    
    try:
        segments, errors = srt_parser.convert_user_srt_to_segments(srt_file)
        
        if not segments:
            return "Error: No segments found in SRT file.", None
        
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
        return "⚠️ No segments found. Make sure the transcription table has data.", gr.Button(interactive=False)
    return f"✅ Transcription validated ({len(new_segments)} segments). Go to the 'Translation' tab.", gr.Button(interactive=True)

def step4_translate(target_lang, progress=gr.Progress()):
    if not state.segments:
        return "Error: No transcription available.", None

    progress(0, f"Translating to {target_lang}...")
    
    # Detect source language from transcription
    source_lang = state.video_info.get('detected_language', 'en') if state.video_info else 'en'
    target_lang_code = LANGUAGES.get(target_lang, target_lang)
    lang_family = time_sync.get_language_family(target_lang_code)
    cps = CHARS_PER_SECOND.get(lang_family, CHARS_PER_SECOND["default"])

    if state.video_info:
        state.video_info['target_language'] = target_lang_code
    
    # PHASE 1: LLM fitted translation (Qwen3-8B) — aggressive concision for timing
    progress(0.1, "Phase 1/3: Fitted translation (time-constrained)...")
    translated = reformulator.translate_segments(
        state.segments, source_lang, target_lang, target_lang_code,
        cps=cps, speed_factor=MAX_SPEED_FACTOR
    )
    state.translated_segments = translated
    
    # PHASE 2: LLM reformulation for segments STILL too long (safety net)
    progress(0.4, "Phase 2/3: Reformulating remaining long segments...")
    reformulated_count = 0
    for seg in state.translated_segments:
        text = seg.get("translated_text", "")
        duration = seg["end"] - seg["start"]
        max_chars = int(duration * cps * MAX_SPEED_FACTOR)
        
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
        
    return status_msg, gr.Dataframe(value=data)

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

def step6_synthesize(voice_mode, voice_file, never_cut, progress=gr.Progress()):
    if not state.translated_segments:
        return "Error: No translation available.", None, None
    
    progress(0.05, "Initializing TTS & Sync...")
    
    # Determine voice path based on mode
    voice_path = None
    if voice_mode == "Clone from original" and state.video_info and 'vocals' in state.video_info:
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
    tts_engine.load_model(voice_path=voice_path)
    if voice_path:
        tts_engine.set_reference_audio(voice_path)
    
    if never_cut:
        # ---- NEVER CUT VOCAL MODE ----
        print(NEVER_CUT_WARNING)
        
        progress(0.2, "[Never Cut] Generating all audio at natural speed...")
        synced, drift_info = time_sync.sync_all_never_cut(
            state.translated_segments, target_lang,
            state.video_info['duration'], voice_mapping=None
        )
        state.synced_segments = synced
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
            total_duration=state.video_info.get('duration')
        )
        
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
    
    progress(0.9, "Assembling final video...")
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    
    final_video = os.path.join(OUTPUT_DIR, f"final_video_{iso}.mp4")
    video_assembler.assemble(
        state.video_info['video_path'], 
        mixed_audio, 
        final_video
    )
    
    # Copy final audio to output dir for easy export
    final_audio_export = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
    shutil.copy2(mixed_audio, final_audio_export)
    
    return status, final_video, mixed_audio


def export_audio():
    """Export the mixed audio as a downloadable file."""
    tgt_lang = state.video_info.get('target_language', '') if state.video_info else ''
    iso = _get_iso_code(tgt_lang)
    audio_path = os.path.join(OUTPUT_DIR, f"final_audio_{iso}.wav")
    if os.path.exists(audio_path):
        return f"Audio exported ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB).", audio_path
    return "No audio available. Run Dubbing first.", None


# --- GRADIO INTERFACE ---

with gr.Blocks(title="ZastTranslate") as app:
    # Embed logo as base64 to avoid Gradio version compatibility issues
    import base64 as _b64
    _logo_path = os.path.join(BASE_DIR, "zastttranslate.png")
    _logo_html = ""
    if os.path.exists(_logo_path):
        with open(_logo_path, "rb") as _f:
            _logo_b64 = _b64.b64encode(_f.read()).decode()
        _logo_html = f"<center><img src='data:image/png;base64,{_logo_b64}' width='80' /></center>\n\n"
    gr.Markdown(f"{_logo_html}# 🎬 ZastTranslate — Beta 0.9\n**Offline video translation & dubbing (No Lip-Sync)**")
    
    with gr.Tab("1. Import"):
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        with gr.Row():
            btn_check = gr.Button("🔍 Check URL", variant="secondary")
            yt_resolution = gr.Dropdown(
                ["1080p"], label="Resolution", value="1080p",
                interactive=False, info="Click 'Check URL' to see available resolutions"
            )
        file_input = gr.File(label="Or upload a local video file", file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm"])
        with gr.Row():
            btn_dl = gr.Button("Import Video", variant="primary")
            btn_reset = gr.Button("New Project", variant="secondary")
        status_dl = gr.Textbox(label="Status", interactive=False)
        video_preview = gr.Video(label="Preview", height=300)
        
    with gr.Tab("2. Transcription"):
        with gr.Row():
            lang_source = gr.Dropdown(
                ["Auto", "French", "English", "Spanish", "German", "Italian", "Portuguese",
                 "Japanese", "Korean", "Chinese", "Russian", "Arabic", "Hindi",
                 "Dutch", "Polish", "Turkish", "Swedish", "Czech", "Romanian", "Hungarian"],
                label="Source Language", value="Auto"
            )
            model_size = gr.Dropdown(["base", "small", "medium", "large-v3"], label="Whisper Model", value="base")

        
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
        lang_target = gr.Dropdown(list(LANGUAGES.keys()), label="Target Language", value="English")
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
        voice_file = gr.File(label="Voice sample file (WAV/MP3, 10-30s of clear speech)", visible=True)
        
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

    with gr.Tab("ℹ️ Help"):
        gr.Markdown("## How to use ZastTranslate")
        
        with gr.Accordion("📥 Tab 1 — Import", open=False):
            gr.Markdown(
                "Load your video from one of two sources:\n\n"
                "- **YouTube URL** — Paste any YouTube link. The video is downloaded automatically via yt-dlp.\n"
                "- **Local file** — Upload a video from your computer. Supported formats: **MP4, MKV, AVI, MOV, WebM**.\n\n"
                "Click **Import Video** to start. A preview will appear below.\n"
                "Use **New Project** to clear everything and start over.\n\n"
                "**YouTube resolution:** Click **🔍 Check URL** to see available resolutions before downloading. "
                "Select the desired quality and click **Import Video**.\n\n"
                "🗑️ **New Project** clears all data and deletes temporary files (downloads, audio, separated tracks) to free disk space.\n\n"
                "💡 iPhone videos (.MOV with HEVC codec) are supported — they're automatically converted for browser playback."
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
                "**Supported languages** (limited to languages supported by Qwen3-TTS for dubbing):\n\n"
                "| Language | Language |\n"
                "|---|---|\n"
                "| 🇫🇷 Français | 🇯🇵 Japanese |\n"
                "| 🇬🇧 English | 🇰🇷 Korean |\n"
                "| 🇪🇸 Español | 🇨🇳 Chinese (Simplified) |\n"
                "| 🇩🇪 Deutsch | 🇷🇺 Russian |\n"
                "| 🇮🇹 Italiano | 🇧🇷 Português |"
            )
        
        with gr.Accordion("🎬 Tab 4 — Dubbing & Export", open=False):
            gr.Markdown(
                "Generate the dubbed video with synthesized speech.\n\n"
                "**Voice Mode:**\n\n"
                "| Mode | Description | When to use |\n"
                "|---|---|---|\n"
                "| **Default voice** | Qwen3-TTS preset voice | Quick dubbing, no reference needed |\n"
                "| **Clone from original** | Clones the speaker's voice from the extracted vocals | Best result — sounds like the original speaker |\n"
                "| **Clone from file** | Uses an uploaded WAV/MP3 file as voice reference | When you want a specific voice |\n\n"
                "💡 Voice cloning uses the Qwen3-TTS model, installed automatically during setup.\n\n"
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
        
        with gr.Accordion("⚙️ System requirements", open=False):
            gr.Markdown(
                "- **GPU**: NVIDIA GPU with 4+ GB VRAM recommended (CUDA)\n"
                "- **CPU**: Works on CPU but significantly slower\n"
                "- **Disk**: ~8 GB for AI models (downloaded on first use)\n"
                "- **OS**: **Tested on Windows only**. May work on Linux/macOS but untested.\n"
                f"\n**Current system**: {GPU_NAME} ({GPU_VRAM}) — {'CUDA ✅' if DEVICE == 'cuda' else 'CPU mode ⚠️'}"
            )
        
        with gr.Accordion("🔗 About / Links", open=False):
            gr.Markdown(
                "**ZastTranslate** is made by Zast.\n\n"
                "- 🌐 [zast57.com](https://zast57.com) — Website\n"
                "- 🤓 [paradoxetemporel.fr](https://paradoxetemporel.fr) — Tech & Geek blog\n"
                "- 🎬 [zast.fr](https://zast.fr) — YouTube channel"
            )

    # EVENTS
    btn_check.click(step0_check_url, [url_input], [status_dl, yt_resolution, btn_dl])
    btn_dl.click(step1_download, [url_input, file_input, yt_resolution], [status_dl, video_preview, btn_transcribe])
    btn_reset.click(reset_project, [], [url_input, file_input, status_dl, video_preview, btn_transcribe, btn_translate, btn_synth])
    
    btn_transcribe.click(step2_transcribe, [lang_source, model_size], [transcription_status, transcription_df])
    btn_import_srt.click(step2b_import_srt, [srt_file_input], [transcription_status, transcription_df])
    
    btn_valid_transcription.click(step3_save_transcription, [transcription_df], [transcription_status, btn_translate])
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
    
    btn_synth.click(step6_synthesize, [voice_mode, voice_file, never_cut_mode], [synth_status, final_video_out, final_audio_out])
    btn_export_audio.click(export_audio, [], [synth_status, export_audio_file])

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        allowed_paths=[BASE_DIR],
    )
