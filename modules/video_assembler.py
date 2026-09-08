import subprocess
import os
from modules.utils import get_exact_duration
from config import DEVICE

class VideoAssembler:
    def __init__(self):
        self._has_nvenc = None

    def _check_nvenc_available(self):
        if self._has_nvenc is not None:
            return self._has_nvenc
        if DEVICE != "cuda":
            self._has_nvenc = False
            return False
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
            self._has_nvenc = 'h264_nvenc' in res.stdout
        except Exception:
            self._has_nvenc = False
        return self._has_nvenc

    def assemble(self, video_path, audio_path, output_path, srt_path=None, hardsub=False):
        """
        Assemble video + audio (+ optional SRT).
        Uses -c:v copy for speed when no hardsub.
        Uses NVENC hardware acceleration when hardsubbing with subtitles filter.
        """
        cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path]
        
        if srt_path and hardsub:
            # Hardsub requires video re-encoding with subtitle filter
            srt_path_esc = srt_path.replace('\\', '/').replace(':', '\\:')
            cmd.extend(['-vf', f"subtitles='{srt_path_esc}'"])
            
            if self._check_nvenc_available():
                print("Video Assembler: NVIDIA NVENC Hardware Video Encoding enabled (h264_nvenc, preset p6, cq 20).")
                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p6', '-cq', '20'])
            else:
                print("Video Assembler: Software Video Encoding (libx264, preset fast, crf 20).")
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '20'])
                
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        elif srt_path:
            # Softsub aka Stream mapping (stream copy - instant)
            cmd.extend(['-i', srt_path])
            cmd.extend(['-map', '0:v', '-map', '1:a', '-map', '2:0'])
            cmd.extend(['-c:v', 'copy', '-c:a', 'aac', '-c:s', 'mov_text'])
            cmd.extend(['-metadata:s:s:0', 'language=fre'])
        else:
            # Audio replacement only (stream copy - instant)
            cmd.extend(['-map', '0:v', '-map', '1:a'])
            cmd.extend(['-c:v', 'copy', '-c:a', 'aac'])

        cmd.append(output_path)
        
        print(f"Video assembly: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # If NVENC fails for any GPU reason, transparent fallback to CPU libx264
            if srt_path and hardsub and 'h264_nvenc' in cmd:
                print("NVENC encoding failed, retrying with CPU libx264 fallback...")
                fallback_cmd = [c if c != 'h264_nvenc' else 'libx264' for c in cmd]
                for idx, arg in enumerate(fallback_cmd):
                    if arg == '-preset' and idx + 1 < len(fallback_cmd) and fallback_cmd[idx + 1] == 'p6':
                        fallback_cmd[idx + 1] = 'fast'
                    if arg == '-cq':
                        fallback_cmd[idx] = '-crf'
                subprocess.run(fallback_cmd, check=True)
            else:
                raise e
        
        return output_path

    def verify_duration(self, original, output):
        d_orig = get_exact_duration(original)
        d_out = get_exact_duration(output)
        diff = abs(d_orig - d_out)
        return {
            "match": diff < 0.1, # Tolerance 100ms
            "diff": diff,
            "original": d_orig,
            "output": d_out
        }
