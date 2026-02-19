import subprocess
import os
from modules.utils import get_exact_duration

class VideoAssembler:
    def assemble(self, video_path, audio_path, output_path, srt_path=None, hardsub=False):
        """
        Assemble video + audio (+ optional SRT).
        Uses -c:v copy for speed when no hardsub.
        """
        cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path]
        
        if srt_path and hardsub:
            # Hardsub necessite re-encodage video
            # Echapment des backslashes pour filter_complex sous Windows
            srt_path_esc = srt_path.replace('\\', '/').replace(':', '\\:')
            cmd.extend(['-vf', f"subtitles='{srt_path_esc}'"])
            cmd.extend(['-c:a', 'aac', '-b:a', '192k']) # Re-encode audio aussi souvent necessaire
        elif srt_path:
            # Softsub aka Stream mapping
            cmd.extend(['-i', srt_path])
            cmd.extend(['-map', '0:v', '-map', '1:a', '-map', '2:0'])
            cmd.extend(['-c:v', 'copy', '-c:a', 'aac', '-c:s', 'mov_text'])
            cmd.extend(['-metadata:s:s:0', 'language=fre']) # Exemple, a parametrer
        else:
            # Juste remplacement audio
            cmd.extend(['-map', '0:v', '-map', '1:a'])
            cmd.extend(['-c:v', 'copy', '-c:a', 'aac'])

        cmd.append(output_path)
        
        print(f"Video assembly: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
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
