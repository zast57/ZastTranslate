"""
ZastTranslate — Qwen-Image-2.1 Generative Thumbnail & Visuals Backend
Based on Qwen/Qwen-Image-2.1 (7B Single-Stream DiT Architecture)
Unified Text-to-Image and Image-to-Image (Editing / Reference) with Native RGBA Transparency.

100% Optional module: the application starts and functions normally without Qwen-Image.
Qwen-Image is only loaded if explicitly installed and requested by the user.
"""

import os
import sys
import time
import re
import gc
from typing import Optional, Dict, Any, List
from config import OUTPUT_DIR, MODELS_DIR, DEVICE

class QwenImageGenerator:
    """
    On-demand AI Image & YouTube Thumbnail generator using Qwen-Image-2.1.
    Features:
    - 7B Single-Stream DiT (Diffusion Transformer) with Qwen3-VL encoder
    - Unified pipeline for Text-to-Image and Reference Image editing
    - Native RGBA transparent background generation (stickers, graphics, cutouts)
    - World-class typography and photorealistic portrait lighting
    - Anti-AI metadata purging (100% clean EXIF / C2PA / PNG text chunks)
    """
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        self.model_id = "Qwen/Qwen-Image-2.1"
        self.models_dir = os.path.join(MODELS_DIR, "qwen_image_2_1")
        os.makedirs(self.models_dir, exist_ok=True)

    def is_diffusers_available(self) -> bool:
        """Check if diffusers and accelerate libraries are installed and support QwenImage21Pipeline."""
        try:
            import diffusers
            import accelerate
            # Check for QwenImage21Pipeline directly in diffusers
            if hasattr(diffusers, "QwenImage21Pipeline"):
                return True
            # Also check if it can be dynamically imported
            from diffusers import QwenImage21Pipeline
            return True
        except Exception:
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Check whether diffusers and Qwen-Image-2.1 model weights are installed."""
        diffusers_ok = self.is_diffusers_available()

        # Check local weights size if downloaded
        total_size = 0
        file_count = 0
        has_index = False
        if os.path.exists(self.models_dir):
            has_index = os.path.exists(os.path.join(self.models_dir, "model_index.json"))
            for root, _, files in os.walk(self.models_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                        file_count += 1
                    except Exception:
                        pass

        size_gb = total_size / (1024 ** 3)
        has_weights = size_gb > 10.0 and has_index

        # Hardware detection & advisory
        vram_advisory = ""
        try:
            import torch
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_vram < 7.9:
                    vram_advisory = (
                        f"\n\n⚠️ **Low-VRAM Hardware Advisory**: Detected GPU with **{total_vram:.1f} GB VRAM**. "
                        "Qwen-Image-2.1 is a large model (~33 GB) that performs best with 8 to 12 GB VRAM and 32 GB RAM. "
                        "On low-spec setups, generation may be slow or encounter Out Of Memory (OOM) errors. "
                        "Note: This module is 100% optional; core app features run without it."
                    )
                else:
                    vram_advisory = f"\n\n💡 *Detected GPU: {total_vram:.1f} GB VRAM (Compatible with model CPU offloading).* "
        except Exception:
            pass

        if not diffusers_ok:
            status_text = (
                "⚪ **Qwen-Image-2.1 Dependencies Not Installed (100% Optional — ~33 GB)** — "
                "Requires diffusers from git (`QwenImage21Pipeline`). "
                "Click **📥 Install Dependencies & Download Qwen-Image-2.1** below or run `Install Qwen-Image` in Pinokio."
                + vram_advisory
            )
        elif not has_weights:
            status_text = (
                f"⚠️ **Dependencies Ready — Model Weights Not Downloaded (~33 GB)** — "
                f"Diffusers is installed. Run **Install Qwen-Image** in Pinokio (recommended, live progress bar in terminal) or click **📥 Download Qwen-Image-2.1 (~33 GB)** below to download model weights."
                + vram_advisory
            )
        else:
            status_text = (
                f"✅ **Qwen-Image-2.1 Installed & Ready** ({size_gb:.1f} GB on disk) — "
                "Unified 7B DiT model for high-CTR thumbnails, reference editing, and native RGBA transparency."
                + vram_advisory
            )

        return {
            "diffusers_installed": diffusers_ok,
            "weights_installed": has_weights,
            "installed": diffusers_ok and has_weights,
            "size_gb": round(size_gb, 2),
            "file_count": file_count,
            "status_text": status_text,
            "models_dir": self.models_dir
        }

    def install_dependencies(self) -> Dict[str, Any]:
        """Install diffusers from git, accelerate, and helpers into virtual environment."""
        if self.is_diffusers_available():
            return {
                "success": True,
                "message": "✅ Qwen-Image-2.1 dependencies (diffusers git, accelerate) are already installed!"
            }
        import subprocess
        try:
            print("[QWEN-IMAGE] Installing diffusers from git, accelerate, transformers, pillow...")
            python_bin = sys.executable
            cmd = [
                "uv", "pip", "install", "--python", python_bin,
                "git+https://github.com/huggingface/diffusers.git",
                "git+https://github.com/huggingface/transformers.git",
                "accelerate", "sentencepiece", "protobuf", "pillow", "huggingface_hub"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                cmd = [
                    python_bin, "-m", "pip", "install",
                    "git+https://github.com/huggingface/diffusers.git",
                    "git+https://github.com/huggingface/transformers.git",
                    "accelerate", "sentencepiece", "protobuf", "pillow", "huggingface_hub", "pip-system-certs"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)

            if self.is_diffusers_available():
                return {
                    "success": True,
                    "message": "✅ Qwen-Image-2.1 dependencies (diffusers git, accelerate) installed successfully!"
                }
            else:
                return {
                    "success": False,
                    "message": f"⚠️ Diffusers installed but QwenImage21Pipeline not detected: {res.stderr[:300]}"
                }
        except Exception as e:
            return {"success": False, "message": f"❌ Error: {e}"}

    def download_model_weights(self) -> Dict[str, Any]:
        """Download Qwen-Image-2.1 model weights to local directory (~33 GB)."""
        if not self.is_diffusers_available():
            dep_res = self.install_dependencies()
            if not dep_res.get("success"):
                return {
                    "success": False,
                    "message": f"Failed to install dependencies: {dep_res.get('message')}",
                    "status_text": self.get_model_status()["status_text"]
                }

        curr_status = self.get_model_status()
        if curr_status.get("weights_installed"):
            return {
                "success": True,
                "message": f"✅ Qwen-Image-2.1 weights already downloaded ({curr_status['size_gb']} GB) in `{self.models_dir}`.",
                "status_text": curr_status["status_text"]
            }

        try:
            try:
                import pip_system_certs
            except Exception:
                pass

            from huggingface_hub import snapshot_download
            token = os.environ.get("HF_TOKEN") or None
            print(f"[QWEN-IMAGE] Downloading {self.model_id} (~33 GB) to {self.models_dir}...")
            print("[QWEN-IMAGE] Note: You can track download progress in this terminal window.")

            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.models_dir,
                resume_download=True,
                token=token
            )
            status = self.get_model_status()
            return {
                "success": True,
                "message": f"✅ Qwen-Image-2.1 downloaded successfully ({status['size_gb']} GB) in `{self.models_dir}`.",
                "status_text": status["status_text"]
            }
        except Exception as e:
            status = self.get_model_status()
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ Download error: {e}",
                "status_text": f"⚠️ Download error: {e}"
            }

    def delete_model_weights(self) -> Dict[str, Any]:
        """Delete local Qwen-Image model weights to reclaim disk space (~30 GB)."""
        try:
            self.unload()
            if os.path.exists(self.models_dir):
                import shutil
                shutil.rmtree(self.models_dir, ignore_errors=True)
                os.makedirs(self.models_dir, exist_ok=True)
            status = self.get_model_status()
            return {
                "success": True,
                "message": "🗑️ Qwen-Image-2.1 weights deleted. ~30 GB disk space reclaimed!",
                "status_text": status["status_text"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ Delete error: {e}",
                "status_text": f"⚠️ Delete error: {e}"
            }

    @staticmethod
    def clean_ascii_typography(text: str) -> str:
        """
        Cleans text and formats quotes properly for Qwen3-VL tokenizer.
        Preserves characters while eliminating broken diacritics that cause font artifacts.
        """
        import unicodedata
        nfkd = unicodedata.normalize('NFKD', text)
        ascii_text = nfkd.encode('ASCII', 'ignore').decode('utf-8')
        # Keep letters, numbers, spaces, and safe punctuation including the period (.) for version numbers like 1.21
        ascii_clean = re.sub(r'[^a-zA-Z0-9\s\-_:!,\'\"\.]', '', ascii_text)
        return re.sub(r'\s+', ' ', ascii_clean).strip()

    def enhance_prompt(
        self,
        user_input: str = "",
        video_title: str = "",
        style_preset: str = "YouTube Viral High-CTR",
        reference_image_path: Optional[str] = None
    ) -> str:
        """
        AI Prompt Assistant tuned for Qwen-Image-2.1:
        Generates natural descriptive prompts with double quotes for typography,
        weaves the uploaded reference photo (speaker face / creator / product) into the composition,
        and provides specific trigger keywords for transparent RGBA if selected.
        """
        if isinstance(style_preset, list):
            style_preset = style_preset[0] if style_preset else "YouTube Viral High-CTR"
        elif not style_preset:
            style_preset = "YouTube Viral High-CTR"

        user_raw = user_input.strip() if user_input else ""
        user_raw = re.sub(r'[\r\n]+', ' ', user_raw)

        explicit_quotes = re.findall(r"['\"]([^'\"]+)['\"]", user_raw)
        if explicit_quotes:
            combined = " ".join(q.strip() for q in explicit_quotes if q.strip())
            words = combined.split()
            short_badge = self.clean_ascii_typography(" ".join(words[:5])).upper()
        elif user_raw:
            # 1. Strip conversational meta-request prefixes (e.g. "Fun image for my new software update...", "Make a thumbnail for...")
            prefix_pattern = (
                r"^(?:"
                r"(?:can\s+you\s+)?(?:please\s+)?(?:create|generate|make|design|build|draw|render|produce|give\s+me)\s+(?:me\s+)?(?:a|an|the)?|"
                r"(?:peux-tu\s+)?(?:crée|créer|génère|générer|fais|faire|dessine|dessiner|donne-moi)\s+(?:moi\s+)?(?:un|une|des)?|"
                r"(?:a|an|the|un|une|le|la|des)\s+"
                r")?\s*"
                r"(?:(?:fun|cool|epic|awesome|great|new|super|belle|jolie|nouvelle)\s+)*"
                r"(?:image|picture|thumbnail|photo|poster|banner|visual|illustration|miniature|affiche|bannière|visuel)\s*"
                r"(?:of|for|about|featuring|showing|themed\s+around|depicting|pour|sur|de|avec)?\s*"
                r"(?:my|our|a|an|the|new|mon|ma|mes|notre|nos|le|la|les|un|une|nouvelle|nouveau)?\s*"
            )
            cleaned_text = re.sub(prefix_pattern, "", user_raw, flags=re.IGNORECASE).strip()
            if not cleaned_text:
                cleaned_text = user_raw

            # 2. Look for prominent uppercase blocks or brand names with version numbers (e.g. "ZAST TRANSLATE 1.21")
            caps_matches = re.findall(r"\b([A-Z]{2,}(?:\s+[A-Z0-9\.\-]+)+)\b", cleaned_text)
            title_version_matches = re.findall(r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+\d+(?:\.\d+)?)\b", cleaned_text)

            if caps_matches:
                caps_matches.sort(key=lambda s: len(s), reverse=True)
                words = caps_matches[0].strip().split()
                short_badge = self.clean_ascii_typography(" ".join(words[:4])).upper()
            elif title_version_matches:
                title_version_matches.sort(key=lambda s: len(s), reverse=True)
                words = title_version_matches[0].strip().split()
                short_badge = self.clean_ascii_typography(" ".join(words[:4])).upper()
            else:
                stopwords = {
                    'dans', 'pour', 'avec', 'les', 'des', 'une', 'qui', 'que', 'the', 'and', 'for', 'with', 'sur', 'par', 'un', 'le', 'la',
                    'my', 'new', 'our', 'your', 'this', 'that', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'software', 'update', 'mise', 'jour',
                    'inside', 'in', 'out', 'app', 'application'
                }
                raw_words = cleaned_text.split()
                filtered = [w for w in raw_words if w.lower() not in stopwords]
                if not filtered:
                    filtered = raw_words
                short_badge = self.clean_ascii_typography(" ".join(filtered[:4])).upper()
        else:
            topic = video_title.strip() or "AI TUTORIAL"
            topic = re.sub(r'[\r\n]+', ' ', topic)
            topic = re.sub(r'\.(mp4|mkv|mov|avi)$', '', topic, flags=re.IGNORECASE)
            stopwords = {
                'tutoriel', 'tuto', 'guide', 'complet', 'installation', 'configuration', 'comment', 'faire',
                'sur', 'dans', 'pour', 'avec', 'les', 'des', 'une', 'qui', 'que', 'the', 'how', 'to', 'and',
                'for', 'with', 'full', 'setup', 'tutorial', 'video', 'windows', 'cours', 'apprendre'
            }
            raw_words = [w for w in re.sub(r'[^\w\s\.]', ' ', topic).split() if len(w) >= 2]
            filtered = [w for w in raw_words if w.lower() not in stopwords]
            if not filtered:
                filtered = raw_words
            short_badge = self.clean_ascii_typography(" ".join(filtered[:4])).upper() if filtered else "AI TUTORIAL"

        if not short_badge:
            short_badge = "AI TUTORIAL"

        # Refined topic description: clean meta-prompt prefixes from user_raw
        topic_desc = user_raw.strip()
        if user_raw:
            topic_desc = re.sub(
                r"^(?:(?:can\s+you\s+)?(?:please\s+)?(?:create|generate|make|design|build|give\s+me)\s+(?:me\s+)?(?:a|an|the)?\s*)?"
                r"(?:(?:fun|cool|epic|awesome|great|new|super|belle)\s+)*"
                r"(?:image|picture|thumbnail|photo|poster|banner|visual|illustration|miniature|affiche)\s+"
                r"(?:of|for|about|featuring|showing|themed\s+around|pour|sur|de)?\s*",
                "",
                user_raw,
                flags=re.IGNORECASE
            ).strip()
            if not topic_desc:
                topic_desc = user_raw.strip()
        elif video_title:
            topic_desc = video_title.strip()
        else:
            topic_desc = "creative technology"

        topic_desc = re.sub(r'\.(mp4|mkv|mov|avi)$', '', topic_desc, flags=re.IGNORECASE).strip()

        # Tone & mood detection (e.g. fun, funny, celebratory, exciting)
        is_fun = any(k in user_raw.lower() for k in [
            'fun', 'amusant', 'humour', 'humor', 'funny', 'drole', 'drôle', 'excited', 'celebrat', 'party', 'fete', 'fête', 'hype', 'joyful', 'cool'
        ])

        # Secondary feature extraction (e.g. "with Qwen-image 2.1 inside" -> sub-badge: "QWEN-IMAGE 2.1")
        sub_badge = ""
        sub_match = re.search(
            r'(?:with|avec|featuring|incluant|intégrant)\s+([A-Za-z0-9_\-\.\s]{2,35}?)(?:\s+(?:inside|inclus|intégré|on board|dedans))?(?:\s*$|[!,;])',
            user_raw,
            flags=re.IGNORECASE
        )
        if sub_match:
            candidate_sub = self.clean_ascii_typography(sub_match.group(1).strip()).upper()
            if candidate_sub and candidate_sub != short_badge and candidate_sub not in short_badge:
                sub_badge = candidate_sub

        reaction_desc = (
            "an energetic, joyful, and amazed celebratory expression with an enthusiastic smile, excitedly gesturing toward the new software release"
            if is_fun else
            "an engaging, expressive reaction"
        )

        extra_typography = (
            f' Directly beneath it, an illuminated high-tech pill badge reads "{sub_badge}" with glowing neon accents.'
            if sub_badge else ""
        )

        extra_atmosphere = (
            "Dynamic celebratory light sparks, colorful floating holographic cards, and glowing soundwaves surround the scene, creating a lively, festive mood."
            if is_fun else
            f"Floating translucent glass interface cards display glowing digital frequency waveforms and visual badges themed around {topic_desc}."
        )

        # Strict check: only activate reference photo conditioning if an actual file exists on disk
        has_ref = bool(
            reference_image_path 
            and str(reference_image_path).strip() 
            and os.path.exists(str(reference_image_path).strip())
        )

        # Presets strictly aligned with Alibaba Qwen-Image-2.1 official 8-step observer guidelines:
        # Step 3 (Opening medium/style/subject/background) -> Step 4 & 5 (Walk regions with spatial anchors) ->
        # Step 6 (Exact straight double quotes for text with materials) -> Step 7 (Dedicated lighting sentence) ->
        # Step 8 (Observer composition closing sentence). No banned quality boosters (8K, masterpiece, award-winning).
        if has_ref:
            # Presets specifically conditioned on the uploaded reference photo (speaker face / creator identity / product)
            styles = {
                "YouTube Viral High-CTR": (
                    f'The image is a high-impact, wide cinematic YouTube thumbnail poster focused on {topic_desc}. '
                    f'On the right side of the frame, the person from the reference photo looks toward the viewer with {reaction_desc}. '
                    f'The subject preserves the exact facial features, hairstyle, and appearance from the reference image, accented by dramatic electric cyan key lighting and warm golden amber rim lighting on the hair and shoulders. '
                    f'In the center and left of the frame, prominent embossed 3D brushed titanium typography reads "{short_badge}" with sharp bevelled edges and glowing neon borders.{extra_typography} '
                    f'{extra_atmosphere} '
                    f'The background is a vibrant modern tech studio with dark acoustic slats and soft atmospheric haze. The overall composition is punchy, high-contrast, and viral-ready.'
                ),
                "Photorealistic Studio Shot": (
                    f'The image is a wide commercial studio photograph of a contemporary workspace and creator studio desk themed around {topic_desc}. '
                    f'In the center, the creator from the reference photo sits at the desk, looking toward the camera with a confident, welcoming expression. '
                    f'The person preserves the exact facial features, skin tones, and hairstyle from the reference image, dressed in tasteful modern attire. '
                    f'In front on the dark polished wood desk surface, physical custom-milled 3D acrylic letters stand reading "{short_badge}" with sharp bevels and subtle warm internal illumination.{extra_typography} '
                    f'On the desk sits a sleek modern silver laptop displaying content related to {topic_desc}, alongside a professional broadcast microphone on a black boom arm. '
                    f'The background is a softly blurred studio interior with warm ambient accent lighting and dark acoustic slat wall panels. '
                    f'The lighting is soft and directional, provided by a large overhead softbox creating clean, gentle shadows and delicate highlights. '
                    f'Shot with an 85mm prime lens at f/1.8 with shallow depth of field and soft circular bokeh.'
                ),
                "Travel & Scenic Adventure": (
                    f'The image is an expansive, cinematic travel documentary photograph capturing the breathtaking scenic atmosphere of {topic_desc} during golden hour. '
                    f'In the foreground, the traveler from the reference photo stands admiring the sweeping landscape. '
                    f'The person preserves the exact facial features, appearance, and identity from the reference image, wearing outdoor adventure gear and smiling toward the horizon. '
                    f'In the lower third of the frame, prominent embossed natural stone typography reads "{short_badge}" with clean, organic bevels. '
                    f'In the midground and background, dramatic panoramic vistas and natural landmarks of {topic_desc} unfold under warm late-afternoon sunlight casting long luminous amber shadows. '
                    f'Shot on a 24mm wide-angle prime lens at f/8 with edge-to-edge optical sharpness and rich natural earth tones.'
                ),
                "Editorial Portrait & Creator": (
                    f'The image is a wide medium-format editorial creator portrait featuring the person from the reference photo at a modern studio desk dedicated to {topic_desc}. '
                    f'The subject preserves the exact facial features, natural skin micro-textures, and hairstyle from the reference image, with a calm, authoritative expression looking toward the camera. '
                    f'In the foreground on the desk, bespoke carved wooden typography reads "{short_badge}" with clean bevelled lettering. '
                    f'The creator is surrounded by professional production tools and warm ambient lighting. '
                    f'The lighting is soft morning sidelight with delicate golden rim highlights. '
                    f'Shot with an 85mm prime lens at f/1.8 with shallow depth of field and subtle film grain.'
                ),
                "Culinary & Food Lifestyle": (
                    f'The image is an inviting, mouthwatering commercial culinary photograph capturing an artisanal gourmet preparation of {topic_desc}. '
                    f'In the center, the chef from the reference photo presents a masterfully plated culinary creation on a dark rustic reclaimed wood table, smiling warmly toward the camera. '
                    f'The subject preserves their recognizable facial features, hairstyle, and appearance from the reference image. '
                    f'In the lower center, delicate carved ivory typography reads "{short_badge}". '
                    f'Surrounding the main dish are fresh organic ingredients, vibrant culinary garnishes, and delicate wisps of steam rising into the air, highlighted by soft natural morning window light. '
                    f'Shot with a 50mm macro lens at f/2.2 with creamy shallow depth of field.'
                ),
                "Business, Finance & News": (
                    f'The image is an authoritative business documentary editorial focused on {topic_desc}. '
                    f'In the center, the professional from the reference photo stands in a sleek corporate boardroom overlooking a metropolitan skyline at twilight, dressed in sharp business attire with a confident posture. '
                    f'The subject preserves the exact facial features, appearance, and identity from the reference image. '
                    f'In the center of the frame, bold modern typography in brushed gold and deep charcoal reads "{short_badge}" with sharp geometric letterforms. '
                    f'Floating translucent glass cards display analytical metrics and trend lines relevant to {topic_desc}. '
                    f'The lighting combines architectural blue dusk light with warm tungsten interior accent illumination. '
                    f'Shot on a 35mm documentary prime lens with clean lines and deep focus.'
                ),
                "Gaming & Epic Cinematic": (
                    f'The image is a high-energy cinematic action gaming visual set in an epic battlefront themed around {topic_desc}. '
                    f'In the center of the frame, a heroic champion inspired by the person in the reference photo stands battle-ready, surrounded by swirling volumetric mist and drifting sparks. '
                    f'The subject preserves the facial structure, facial features, and recognizable identity from the reference image, reimagined in heroically detailed armor. '
                    f'In the center, massive battle-worn metallic 3D typography reads "{short_badge}" with glowing fiery embers, molten cracks, and heavy chiseled bevels. '
                    f'In the sky above, dynamic celestial energy beams illuminate the dramatic arena. '
                    f'The lighting is intensely cinematic with high-contrast rim lighting from fiery energy.'
                ),
                "3D Isometric & Tech Glow": (
                    f'The image is a wide 3D isometric digital rendering of an advanced modular tech station designed for {topic_desc}. '
                    f'In the center of the platform, the subject from the reference photo is integrated into the scene, maintaining their recognizable facial features, hairstyle, and appearance from the reference image. '
                    f'In the center of the platform, vibrant glowing extruded 3D typography reads "{short_badge}" with bevelled metallic edges and an electric blue neon backlight. '
                    f'Across the platform surface run glowing fiber-optic data channels and holographic interactive modules illustrating {topic_desc}, illuminated by turquoise and violet point lights casting soft ambient occlusion shadows.'
                ),
                "Cyberpunk & Bold Neon": (
                    f'The image is a wide cinematic cyberpunk photograph set in a futuristic metropolis at night themed around {topic_desc}. '
                    f'On the rain-slicked city street, the person from the reference photo stands looking forward, illuminated by electric magenta and cyan neon rim lights. '
                    f'The subject maintains the exact facial features, hairstyle, and appearance from the reference image. '
                    f'In the upper center, an illuminated neon sign mounted to an industrial facade reads "{short_badge}" in glowing neon glass tubes, reflected in shimmering pools of water on the dark wet asphalt below with atmospheric night haze.'
                ),
                "Minimalist & Clean SaaS": (
                    f'The image is a wide modern editorial graphic representing premium digital solutions for {topic_desc}. '
                    f'In the composition, the professional from the reference photo stands in a bright, contemporary Scandinavian studio environment, looking confidently toward the viewer while preserving their natural facial features and identity from the reference image. '
                    f'Beside them, a crisp floating frosted-glass card displays bold modern typography reading "{short_badge}" in deep charcoal and electric indigo lettering, surrounded by clean UI elements and soft contact shadows against an airy gradient background.'
                ),
                "Refined Typography Poster": (
                    f'Design a refined editorial typography poster celebrating {topic_desc}. '
                    f'In the composition, an artistic portrait silhouette inspired by the person in the reference photo is tastefully integrated into the Art Deco geometry, preserving their recognizable facial profile. '
                    f'Headline typography reads exactly "{short_badge}" in elegant embossed ivory and champagne gold lettering with clear typographic hierarchy. '
                    f'Muted brass linear wave accents and deep charcoal navy backgrounds create generous, balanced negative margins with print-ready composition.'
                ),
                "Transparent RGBA (Sticker / Cutout)": (
                    f'This is an RGBA image with transparency. The image has alpha channel and the background is completely transparent. '
                    f'Featuring the subject from the reference photo as a clean cutout sticker with a bold white contour border and subtle contact drop shadow. '
                    f'The subject preserves the exact facial features, hairstyle, and appearance from the reference image, accompanied by bold glossy embossed 3D typography reading "{short_badge}" themed around {topic_desc} with vibrant gradient fill and crisp edge highlights.'
                ),
                "Transparent RGBA (Clean Cutout)": (
                    f'This is an RGBA image with transparency. The image has alpha channel and the background is completely transparent. '
                    f'Featuring the subject from the reference photo as a clean cutout sticker with a bold white contour border and subtle contact drop shadow. '
                    f'The subject preserves the exact facial features, hairstyle, and appearance from the reference image, accompanied by bold glossy embossed 3D typography reading "{short_badge}" themed around {topic_desc} with vibrant gradient fill and crisp edge highlights.'
                ),
                "Hyper-Realistic Photo 8K": (
                    f'The image is an ultra-detailed, photorealistic 85mm medium-format studio photograph capturing the world of {topic_desc}. '
                    f'In the center, the creator from the reference photo is featured in a modern creative environment, preserving their exact facial features, natural skin micro-textures with visible pores, believable skin tones, and hairstyle from the reference image, looking confidently toward the viewer. '
                    f'Beside them, pristine physical 3D typography reads "{short_badge}" with tactile bevels and natural reflections. Balanced softbox lighting creates clean, gentle shadows with deep natural contrast.'
                )
            }
        else:
            styles = {
                "YouTube Viral High-CTR": (
                    f'The image is a wide cinematic high-CTR YouTube thumbnail poster focused on {topic_desc}, '
                    f'set against a sleek tech studio environment with deep charcoal and dark navy acoustic paneling. '
                    f'In the center of the frame, prominent embossed 3D brushed titanium typography reads "{short_badge}", '
                    f'with sharp bevelled edges and glowing neon borders.{extra_typography} '
                    f'{extra_atmosphere} '
                    f'In the lower third, a sleek matte dark slate surface reflects '
                    f'the glowing lettering with soft, realistic specular highlights. The lighting is dramatic and directional, featuring a cool electric cyan '
                    f'key light from the left and a warm golden amber rim light outlining the central lettering, creating deep volumetric depth with subtle atmospheric haze. '
                    f'The overall composition is punchy, high-contrast, and impeccably balanced, drawing immediate focus to the central typography with rich depth and modern tech aesthetics.'
                ),
                "3D Isometric & Tech Glow": (
                    f'The image is a wide 3D isometric digital rendering of an advanced modular tech station designed for {topic_desc}. '
                    f'In the center of the platform, vibrant glowing extruded 3D typography reads "{short_badge}" with bevelled metallic edges and an electric blue neon backlight. '
                    f'Across the platform surface run glowing fiber-optic data channels and intricate microchip circuits. On the left side, floating translucent glass interface tiles '
                    f'display graphical data and real-time interactive modules illustrating {topic_desc}. On the right side, a miniature holographic globe with interconnected arcs '
                    f'rotates gently above a cylindrical emitter. The background is deep navy and dark grey with subtle grid lines fading into the distance. '
                    f'The lighting is crisp and multi-colored, with turquoise and violet point lights illuminating the circuit paths and casting soft ambient occlusion shadows beneath the platform. '
                    f'The overall composition is clean, highly structured, and visually engaging, showcasing cutting-edge digital craftsmanship.'
                ),
                "Cyberpunk & Bold Neon": (
                    f'The image is a wide cinematic photograph of a rain-slicked city street in a futuristic metropolis at night themed around {topic_desc}. '
                    f'In the upper center of the frame, a large illuminated neon sign mounted to an industrial brick facade reads "{short_badge}" in vivid electric magenta and cyan glowing neon glass tubes. '
                    f'Below the sign on the street level, wet dark asphalt reflects the bright neon colors in shimmering pools of water, with rising steam escaping from an iron manhole cover. '
                    f'In the background on the left and right, towering skyscrapers with glowing windows and futuristic commercial architecture themed around {topic_desc} fade into a dark, foggy night sky with subtle atmospheric rain haze. '
                    f'The lighting is high-contrast and atmospheric, dominated by the vivid magenta and cyan neon glow casting saturated reflections across wet surfaces and deep dark shadows in the alleys. '
                    f'The overall composition is moody and cinematic, capturing a classic dystopian cyberpunk atmosphere with strong atmospheric depth and rich reflective textures.'
                ),
                "Minimalist & Clean SaaS": (
                    f'The image is a wide modern editorial graphic representing premium digital solutions for {topic_desc}. '
                    f'In the center, a crisp, floating frosted-glass card displays bold modern typography reading "{short_badge}" in deep charcoal and electric indigo lettering. '
                    f'Surrounding the central card are subtle floating geometric UI badges and visual elements illustrating {topic_desc} with delicate drop shadows. '
                    f'The background is an expansive, clean gradient transitioning smoothly from pale off-white to soft dusty lavender, offering generous negative space. '
                    f'The lighting is bright, even, and diffused, reminiscent of daylight in a Scandinavian design studio, casting soft, faint contact shadows beneath the floating elements. '
                    f'The overall composition is elegant, airy, and meticulously balanced, emphasizing clarity, modern digital craftsmanship, and premium aesthetics.'
                ),
                "Photorealistic Studio Shot": (
                    f'The image is a wide commercial studio photograph of a contemporary workspace and creator desk themed around {topic_desc}. '
                    f'In the center, physical custom-milled 3D acrylic letters stand on a dark polished wood desk surface, reading "{short_badge}" with sharp bevels and subtle internal warm LED illumination. '
                    f'Behind the lettering on the left sits a modern silver laptop with software visible on the display related to {topic_desc}, flanked by studio reference monitors with yellow Kevlar cones. '
                    f'On the right, a professional broadcast microphone on a black boom arm is angled toward the scene. '
                    f'The background is a gently blurred studio interior with warm ambient accent lighting and dark acoustic slat wall panels. '
                    f'The lighting is soft and directional, provided by a large overhead softbox creating clean, gentle shadows and delicate highlights on the lettering surfaces. '
                    f'Shot with an 85mm prime lens at f/1.8 with shallow depth of field and soft circular bokeh. '
                    f'The overall composition is professional, tactile, and natural, evoking a premium modern media production studio.'
                ),
                "Travel & Scenic Adventure": (
                    f'The image is an expansive, cinematic travel documentary photograph capturing the breathtaking scenic atmosphere of {topic_desc} during golden hour. '
                    f'In the lower third of the frame, prominent embossed natural stone or weathered teak typography reads "{short_badge}" with clean, organic bevels. '
                    f'In the midground, a scenic panoramic viewpoint overlooks majestic landscape landmarks of {topic_desc}. '
                    f'To the left, an adventurous traveler wearing outdoor trekking gear stands admiring the sweeping horizon. '
                    f'The lighting is breathtaking warm late-afternoon sunlight from a low-angle sun on the right, illuminating mist in the valleys and casting long, luminous amber shadows across the terrain. '
                    f'Shot on a 24mm wide-angle prime lens at f/8 with edge-to-edge optical sharpness, rich atmospheric depth, and vibrant, natural earth tones. '
                    f'The overall composition is wanderlust-inducing, majestic, and impeccably balanced, evoking a world-class travel documentary.'
                ),
                "Culinary & Food Lifestyle": (
                    f'The image is an inviting, mouthwatering commercial food photography scene of an artisanal gourmet preparation of {topic_desc}. '
                    f'In the lower center of the frame, delicate carved ivory or handwritten slate typography reads "{short_badge}" with crisp, elegant lettering. '
                    f'In the center, a masterfully plated culinary creation rests on a dark rustic reclaimed wood table, garnished with vibrant fresh herbs, glistening sauces, and delicate wisps of steam rising into the air. '
                    f'Surrounding the main dish are raw organic ingredients, a small brass oil cruet, and a pinch bowl of coarse sea salt. '
                    f'The lighting is soft, natural morning window light entering from the left, highlighting the succulent textures, glossy glazes, and rich food colors with gentle, realistic contact shadows. '
                    f'Shot with a 50mm macro lens at f/2.2, producing a creamy, shallow depth of field with soft out-of-focus copper kitchenware in the warm background. '
                    f'The overall composition is warm, authentic, and deeply appetizing.'
                ),
                "Business, Finance & News": (
                    f'The image is an authoritative, high-impact editorial photograph for a business and finance documentary focused on {topic_desc}. '
                    f'In the center of the frame, bold, clean modern typography in brushed gold and deep charcoal reads "{short_badge}" with sharp, geometric letterforms. '
                    f'In the background, a sleek modern corporate boardroom with expansive floor-to-ceiling windows overlooks a bustling metropolitan skyline at twilight. '
                    f'To the left and right, subtle translucent floating glass cards display upward analytical trend lines and metrics focused on {topic_desc}. '
                    f'The lighting is crisp and balanced, combining cool architectural blue dusk light from outside with warm tungsten interior accent lamps and sharp key light on the central lettering. '
                    f'Shot on a 35mm documentary prime lens with clean lines and deep focus. '
                    f'The overall composition is credible, high-stakes, and sophisticated, commanding instant viewer authority.'
                ),
                "Gaming & Epic Cinematic": (
                    f'The image is a high-energy cinematic action gaming illustration set in an epic battlefront themed around {topic_desc}. '
                    f'In the center of the frame, massive battle-worn metallic 3D typography reads "{short_badge}" with glowing fiery embers, molten cracks, and heavy chiseled bevels. '
                    f'In the midground, a heroic armored warrior stands poised with a glowing weapon, overlooking a dramatic battlefield shrouded in volumetric mist and drifting sparks. '
                    f'In the sky above, swirling dark storm clouds are pierced by radiant beams of ethereal energy from a cosmic rift. '
                    f'The lighting is dramatic and intensely cinematic, featuring high-contrast rim lighting from the fiery glow and deep, moody ambient shadows across the craggy terrain. '
                    f'The overall composition is adrenaline-fueled, epic in scale, and visually commanding, built for maximum viewer curiosity.'
                ),
                "Editorial Portrait & Creator": (
                    f'The image is a wide editorial portrait photograph of a focused media creator at a modern studio workspace dedicated to {topic_desc}. '
                    f'In the lower center of the frame, custom carved walnut typography resting on the desk reads "{short_badge}" in elegant bevelled lettering. '
                    f'To the left, a professional creator with believable skin tones, natural skin texture with visible micro-pores, and tasteful clothing works attentively amidst creative tools. '
                    f'Behind them, subtle acoustic foam diffusers and warm filament accent lamps create a calm, sophisticated workspace ambiance. '
                    f'The lighting is cinematic and natural, featuring soft morning window sidelight from the right and subtle golden fill light, casting gentle, natural contact shadows. '
                    f'Shot with a medium-format camera and an 85mm prime lens at f/1.8, producing subtle film grain and smooth circular background bokeh with calm, lifelike expression.'
                ),
                "Refined Typography Poster": (
                    f'Design a refined editorial typography poster celebrating {topic_desc}. '
                    f'In the center of the composition, render the headline typography exactly as "{short_badge}" in elegant embossed ivory and champagne gold lettering with clear typographic hierarchy. '
                    f'Below the headline, a crisp subtitle provides context for {topic_desc} in a clean geometric sans-serif font. '
                    f'Surrounding the text are subtle Art Deco linear wave accents and sound frequency curves in muted brass tones against a deep charcoal and navy background. '
                    f'Generous, balanced negative margins frame the composition, with soft ambient drop shadows beneath the lettering. '
                    f'The overall composition is print-ready, refined, and impeccably balanced, conveying luxury craftsmanship and cutting-edge software engineering.'
                ),
                "Transparent RGBA (Sticker / Cutout)": (
                    f'This is an RGBA image with transparency. The image has alpha channel and the background is completely transparent. '
                    f'In the center of the frame is a standalone 3D graphic emblem featuring bold, glossy embossed typography that reads "{short_badge}" themed around {topic_desc}. '
                    f'The lettering has bevelled metallic chrome borders, a vibrant cyan-to-amber gradient fill, and crisp specular highlights along the edges. '
                    f'A subtle translucent drop shadow is cast beneath the emblem, fully contained within the alpha channel. '
                    f'The lighting is bright studio rim lighting that accentuates the curvature and 3D depth of each character against the transparent void. '
                    f'The overall design is punchy, clean, and ready to be used as a high-resolution sticker or overlay.'
                ),
                "Transparent RGBA (Clean Cutout)": (
                    f'This is an RGBA image with transparency. The image has alpha channel and the background is completely transparent. '
                    f'In the center of the frame is a standalone 3D graphic emblem featuring bold, glossy embossed typography that reads "{short_badge}" themed around {topic_desc}. '
                    f'The lettering has bevelled metallic chrome borders, a vibrant cyan-to-amber gradient fill, and crisp specular highlights along the edges. '
                    f'A subtle translucent drop shadow is cast beneath the emblem, fully contained within the alpha channel. '
                    f'The lighting is bright studio rim lighting that accentuates the curvature and 3D depth of each character against the transparent void. '
                    f'The overall design is punchy, clean, and ready to be used as a high-resolution sticker or overlay.'
                ),
                "Hyper-Realistic Photo 8K": (
                    f'The image is an ultra-detailed, photorealistic 85mm medium-format studio photograph capturing the world of {topic_desc}. '
                    f'In the center of the frame, pristine physical 3D typography reads "{short_badge}" with tactile bevels and natural reflections on a polished dark desk. '
                    f'Surrounding the lettering are professional tools and subtle ambient studio lighting. '
                    f'Balanced softbox lighting creates clean, gentle shadows with deep natural contrast and rich textures.'
                )
            }

        return styles.get(style_preset, styles["YouTube Viral High-CTR"])

    @staticmethod
    def sanitize_image(image) -> Any:
        """
        Completely purges all AI generation metadata, EXIF tags, PNG text chunks
        (prompts, models, steps, diffusers signatures, C2PA manifests) to ensure 100%
        human-looking image files that appear hand-crafted or raster-exported.
        Supports both RGB and RGBA transparent images.
        """
        from PIL import Image
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        clean = Image.frombytes(mode, image.size, image.convert(mode).tobytes())
        clean.info = {}
        return clean

    def generate_thumbnail(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        steps: int = 40,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
        transparent: bool = False,
        cfg_scale: float = 1.0,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate AI Visual / YouTube Thumbnail using Qwen-Image-2.1.
        Supports Text-to-Image and unified Reference Image editing.
        Supports RGBA transparent output and optional Classifier-Free Guidance (CFG).
        All generated images are automatically sanitized to strip 100% of AI metadata.
        """
        status = self.get_model_status()
        if not status.get("diffusers_installed"):
            return {
                "success": False,
                "error": "diffusers/QwenImage21Pipeline not installed",
                "message": (
                    "⚠️ Qwen-Image-2.1 dependencies are not installed yet. "
                    "Run 'Install Qwen-Image' from the Pinokio menu or click 'Install / Download Qwen-Image-2.1' above (Optional)."
                )
            }

        if not status.get("weights_installed"):
            return {
                "success": False,
                "error": "weights not downloaded",
                "message": (
                    "⚠️ Qwen-Image-2.1 model weights (~33 GB) are not downloaded yet! "
                    "Please run 'Install Qwen-Image' from the Pinokio menu to download the ~33 GB model weights with live progress before generating."
                )
            }

        if not prompt or not prompt.strip():
            return {"success": False, "error": "Prompt cannot be empty."}

        clean_prompt = prompt.strip()
        # If transparency is requested and prompt doesn't mention RGBA, add trigger keywords
        if transparent and "rgba" not in clean_prompt.lower():
            clean_prompt = f"This is an RGBA image with transparency. {clean_prompt} The image has alpha channel and the background is transparent."

        timestamp = int(time.time())
        if not output_path:
            output_dir = os.path.join(OUTPUT_DIR, "qwen_generated")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"qwen_thumb_{timestamp}.png")

        # Dimensions: official Qwen-Image-2.1 aspect ratios
        # Scaled to balanced resolution for GPU speed while preserving aspect
        if "9:16" in aspect_ratio:
            width, height = 768, 1376  # (official max: 1536x2752)
        elif "1:1" in aspect_ratio:
            width, height = 1024, 1024  # (official max: 2048x2048)
        elif "4:3" in aspect_ratio:
            width, height = 1200, 896   # (official max: 2400x1792)
        else: # 16:9 standard YouTube Thumbnail
            width, height = 1376, 768   # (official max: 2752x1536)

        # Steps: standard 40 steps for Qwen-Image-2.1 DiT (minimum 20 for quality)
        actual_steps = max(20, min(steps, 60))

        if seed is None or seed < 0:
            import random
            seed = random.randint(1, 2147483647)

        try:
            import torch
            from PIL import Image
            from diffusers import QwenImage21Pipeline

            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Lazy load pipeline if not in memory
            if self.pipeline is None:
                model_source = self.models_dir if os.path.exists(self.models_dir) and any(os.scandir(self.models_dir)) else self.model_id
                print(f"[QWEN-IMAGE] Loading Qwen-Image-2.1 from {model_source} (Seed: {seed})...")
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                print(f"[QWEN-IMAGE] Initializing QwenImage21Pipeline with {dtype}...")
                self.pipeline = QwenImage21Pipeline.from_pretrained(
                    model_source,
                    torch_dtype=dtype
                )
                if torch.cuda.is_available():
                    print("[QWEN-IMAGE] Enabling model CPU offload for optimal VRAM efficiency (~8-12 GB)...")
                    self.pipeline.enable_model_cpu_offload()
            elif torch.cuda.is_available() and not getattr(self.pipeline, "_is_offloaded", False):
                try:
                    print("[QWEN-IMAGE] Enabling model CPU offload for optimal VRAM efficiency (~8-12 GB)...")
                    self.pipeline.enable_model_cpu_offload()
                    self.pipeline._is_offloaded = True
                except Exception:
                    pass

            is_editing = bool(reference_image_path and os.path.exists(reference_image_path))
            t0 = time.time()

            pipe_kwargs = {
                "prompt": clean_prompt,
                "num_inference_steps": actual_steps,
                "generator": generator,
                "width": width,
                "height": height
            }

            if cfg_scale is not None and float(cfg_scale) > 1.0:
                pipe_kwargs["true_cfg_scale"] = float(cfg_scale)
                if negative_prompt and str(negative_prompt).strip():
                    pipe_kwargs["negative_prompt"] = str(negative_prompt).strip()

            if is_editing:
                print(f"[QWEN-IMAGE] Running image-conditioned generation with reference: {reference_image_path}")
                ref_img = Image.open(reference_image_path).convert("RGB")
                # Downscale oversized reference photos (e.g. 3088x2316 or 4K) to max 1280px to prevent visual token explosion
                max_ref_dim = 1280
                if max(ref_img.width, ref_img.height) > max_ref_dim:
                    ref_img.thumbnail((max_ref_dim, max_ref_dim), Image.Resampling.LANCZOS)
                    print(f"[QWEN-IMAGE] Downscaled reference image to {ref_img.width}x{ref_img.height} for rapid visual encoding.")
                pipe_kwargs["image"] = ref_img
                # If the prompt doesn't already reference the conditioning subject, anchor it so DiT prioritizes the reference face
                lower_clean = clean_prompt.lower()
                if not any(k in lower_clean for k in ["reference", "photo", "person", "face", "speaker", "creator", "subject"]):
                    clean_prompt = f"Featuring the person from the reference photo, preserving their exact facial features, hairstyle, and appearance from the reference image. {clean_prompt}"
                    pipe_kwargs["prompt"] = clean_prompt
                pipe_kwargs["width"] = width
                pipe_kwargs["height"] = height

            print(f"[QWEN-IMAGE] Generating ({actual_steps} steps, CFG: {cfg_scale}, Seed: {seed})...")
            result = self.pipeline(**pipe_kwargs)

            gen_img = result.images[0]
            clean_img = self.sanitize_image(gen_img)
            clean_img.save(output_path, format="PNG", optimize=True)
            elapsed = time.time() - t0
            res_str = f"{clean_img.width}x{clean_img.height}"
            mode_str = "RGBA Transparent" if clean_img.mode == "RGBA" else "RGB"
            print(f"[QWEN-IMAGE] Generated & sanitized successfully in {elapsed:.2f}s ({res_str}, {mode_str}, 100% clean metadata)!")

            return {
                "success": True,
                "image_path": output_path,
                "prompt": clean_prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": res_str,
                "mode": mode_str,
                "seed": seed,
                "elapsed_seconds": round(elapsed, 1),
                "message": f"✅ Visual generated with Qwen-Image-2.1 in {elapsed:.1f}s ({res_str}, {mode_str}, Seed: {seed})!"
            }

        except Exception as e:
            import traceback
            print(f"[QWEN-IMAGE] Generation error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ Qwen-Image generation error: {e}"
            }

    def generate_ab_thumbnails(
        self,
        base_prompt: str,
        video_title: str = "",
        reference_image_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        steps: int = 40,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        cfg_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        styles_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate diverse, high-CTR YouTube thumbnail variants for YouTube 'Test & Compare' (A/B testing).
        Accepts custom selected style presets (up to 3 variants).
        All generated images are sanitized (metadata purged) and packaged into a ready-to-upload ZIP archive.
        """
        if not output_dir:
            output_dir = os.path.join(OUTPUT_DIR, "qwen_generated")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())

        topic = base_prompt.strip() or video_title.strip() or "YouTube Video"

        if base_seed is None or base_seed <= 0:
            import random
            base_seed = random.randint(1, 2147480000)

        # Process styles_list
        if not styles_list:
            styles_list = [
                "YouTube Viral High-CTR",
                "3D Isometric & Tech Glow",
                "Photorealistic Studio Shot"
            ]
        elif isinstance(styles_list, str):
            styles_list = [styles_list]
        
        # If user picked 1 style, expand to 3 diverse styles for A/B testing
        if len(styles_list) == 1:
            chosen = styles_list[0]
            pool = [
                "YouTube Viral High-CTR",
                "3D Isometric & Tech Glow",
                "Photorealistic Studio Shot",
                "Travel & Scenic Adventure",
                "Editorial Portrait & Creator"
            ]
            additional = [s for s in pool if s.lower() != chosen.lower()][:2]
            styles_list = [chosen] + additional

        # Limit to max 3 variants (YouTube Studio limit)
        styles_list = styles_list[:3]

        variants_config = []
        letters = ["A", "B", "C"]
        for idx, st in enumerate(styles_list):
            letter = letters[idx] if idx < len(letters) else chr(65 + idx)
            clean_tag = re.sub(r'[^a-zA-Z0-9]+', '_', st).strip('_').lower()[:15]
            variants_config.append({
                "id": f"variant_{letter.lower()}",
                "label": f"Variant {letter} ({st})",
                "style": st,
                "filename": f"youtube_thumb_{letter}_{clean_tag}_{timestamp}.png",
                "seed": base_seed + (idx * 101),
            })

        total_variants = len(variants_config)
        results = []
        for i, var in enumerate(variants_config):
            if progress_callback:
                progress_callback((i / float(total_variants)), f"Generating {var['label']} ({i+1}/{total_variants})...")

            tailored_prompt = self.enhance_prompt(
                user_input=topic,
                video_title=video_title,
                style_preset=var["style"],
                reference_image_path=reference_image_path
            )
            out_file = os.path.join(output_dir, var["filename"])
            res = self.generate_thumbnail(
                prompt=tailored_prompt,
                reference_image_path=reference_image_path,
                aspect_ratio=aspect_ratio,
                steps=steps,
                output_path=out_file,
                seed=var["seed"],
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt
            )
            if res.get("success"):
                results.append({
                    "id": var["id"],
                    "label": var["label"],
                    "style": var["style"],
                    "prompt": tailored_prompt,
                    "image_path": out_file,
                    "filename": var["filename"],
                    "seed": var["seed"],
                    "elapsed_seconds": res.get("elapsed_seconds", 0.0)
                })
            else:
                return {
                    "success": False,
                    "error": res.get("error", "Error generating variant"),
                    "message": f"❌ Error on {var['label']}: {res.get('message')}"
                }

        import zipfile
        zip_filename = f"youtube_ab_testing_pack_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        pack_contents_lines = []
        for r in results:
            pack_contents_lines.append(f"- {r['label']}: {r['filename']}")
        pack_contents_str = "\n".join(pack_contents_lines)

        readme_content = (
            "==============================================================\n"
            f"🎯 YOUTUBE A/B TESTING PACK — {len(results)} QWEN-IMAGE-2.1 VARIANTS\n"
            "==============================================================\n\n"
            f"This pack contains {len(results)} distinct high-CTR thumbnail variants generated with Qwen-Image-2.1.\n"
            "All generative AI metadata (EXIF, C2PA manifests, prompts) has been 100% stripped.\n\n"
            "PACK CONTENTS:\n"
            f"{pack_contents_str}\n\n"
            "HOW TO RUN A/B TESTING IN YOUTUBE STUDIO:\n"
            "1. Open YouTube Studio -> Content -> Select your video.\n"
            "2. Under the 'Thumbnail' section, click the 3-dots menu (...) -> 'Test & compare'.\n"
            f"3. Upload the {len(results)} variants.\n"
            "4. YouTube will automatically test them with your audience and pick the winner!\n"
            "==============================================================\n"
        )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README_YOUTUBE_AB_TEST.txt", readme_content)
            for r in results:
                if os.path.exists(r["image_path"]):
                    zf.write(r["image_path"], arcname=r["filename"])

        if progress_callback:
            progress_callback(1.0, f"A/B Testing {len(results)}-Variant Pack generated successfully!")

        return {
            "success": True,
            "variants": results,
            "zip_path": zip_path,
            "zip_filename": zip_filename,
            "message": f"🎉 **{len(results)} A/B Test Variants Generated Successfully!** (Ready-to-upload ZIP pack)"
        }

    def unload(self):
        """Free GPU VRAM and release Qwen-Image pipeline."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.model_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
        print("[QWEN-IMAGE] VRAM cleared and model unloaded.")

qwen_image_studio = QwenImageGenerator()
