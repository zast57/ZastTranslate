import yt_dlp
import os
import shutil
import subprocess
from modules.utils import get_exact_duration, convert_sample_rate
from config import TEMP_DIR

class VideoDownloader:
    def __init__(self):
        os.makedirs(TEMP_DIR, exist_ok=True)

    def check_url(self, url):
        """
        Fetch video info without downloading.
        Returns {"title": str, "duration": float, "resolutions": list[str]}
        """
        ydl_opts = {'quiet': True, 'noplaylist': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                # Collect unique video heights
                heights = sorted(set(
                    f['height'] for f in formats
                    if f.get('height') and f.get('vcodec', 'none') != 'none'
                ), reverse=True)
                resolutions = [f"{h}p" for h in heights]
                return {
                    "title": info.get('title', 'Unknown'),
                    "duration": info.get('duration', 0),
                    "resolutions": resolutions,
                }
        except Exception as e:
            print(f"yt-dlp check error: {e}")
            raise

    def download(self, url, resolution="1080p"):
        """
        Download video via yt-dlp.
        Returns {"video_path": str, "audio_16k": str, "audio_44k": str, "duration": float, "title": str}
        """
        # Build yt-dlp format string based on resolution
        if resolution == "Best":
            fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        else:
            height = resolution.replace("p", "")
            fmt = f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}][ext=mp4]/best"

        ydl_opts = {
            'format': fmt,
            'outtmpl': os.path.join(TEMP_DIR, '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'quiet': True,
            'restrictfilenames': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_filename = ydl.prepare_filename(info)
                if 'merge_output_format' in ydl_opts and ydl_opts['merge_output_format']:
                    base, _ = os.path.splitext(video_filename)
                    video_filename = base + '.' + ydl_opts['merge_output_format']
                
                return self._process_video(video_filename, info.get('title', 'video'))
        except Exception as e:
            print(f"yt-dlp download error: {e}")
            raise

    def import_local(self, filepath):
        """
        Copy local file to TEMP_DIR.
        """
        filename = os.path.basename(filepath)
        dest_path = os.path.join(TEMP_DIR, filename)
        shutil.copy2(filepath, dest_path)
        return self._process_video(dest_path, os.path.splitext(filename)[0])

    def _process_video(self, video_path, title):
        """
        Extract audio and return file info.
        """
        audio_paths = self.extract_audio(video_path)
        duration = get_exact_duration(video_path)
        
        return {
            "video_path": video_path,
            "audio_16k": audio_paths["audio_16k"],
            "audio_44k": audio_paths["audio_44k"],
            "duration": duration,
            "title": title
        }

    def extract_audio(self, video_path):
        """
        Extract two audio versions:
        - WAV 16kHz mono (for WhisperX)
        - WAV 44.1kHz stereo (for Demucs)
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_16k = os.path.join(TEMP_DIR, f"{base_name}_16k.wav")
        audio_44k = os.path.join(TEMP_DIR, f"{base_name}_44k.wav")

        # Extract 16k mono
        convert_sample_rate(video_path, audio_16k, 16000, 1)
        
        # Extract 44.1k stereo
        convert_sample_rate(video_path, audio_44k, 44100, 2)

        return {"audio_16k": audio_16k, "audio_44k": audio_44k}

if __name__ == "__main__":
    import sys
    dl = VideoDownloader()
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        print(f"Processing: {arg}")
        if arg.startswith("http"):
            res = dl.download(arg)
        else:
            res = dl.import_local(arg)
        print(f"Result: {res}")
    else:
        print("Usage: python downloader.py [URL or LOCAL_PATH]")
