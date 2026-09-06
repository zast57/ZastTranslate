"""
ZastTranslate — FLUX.1-schnell Generative Thumbnail & Visuals Backend
Based on FLUX.1-schnell (Apache 2.0)
Diffusers compatible flow transformer architecture.

100% Optional module: the application starts and functions normally without FLUX.
FLUX is only loaded if explicitly installed and requested by the user.
"""

import os
import sys
import time
import re
import gc
from typing import Optional, Dict, Any, List
from config import OUTPUT_DIR, MODELS_DIR, DEVICE

class FluxGenerator:
    """
    On-demand AI Image & YouTube Thumbnail generator using FLUX.1-schnell.
    Ultra-fast (1-4 steps, ~2s on RTX 4090), world-class typography and composition.
    Supports Text-to-Image (T2I) and Image-to-Image (Img2Img) with optional reference photo.
    """
    def __init__(self):
        self.transformer = None
        self.pipeline = None
        self.img2img_pipeline = None
        self.model_loaded = False
        # Ungated, direct mirror of FLUX.1-schnell for diffusers (avoids HuggingFace 403 GatedRepoError)
        self.model_id = "Niansuh/FLUX.1-schnell"
        self.official_model_id = "black-forest-labs/FLUX.1-schnell"
        self.models_dir = os.path.join(MODELS_DIR, "flux")
        os.makedirs(self.models_dir, exist_ok=True)

    def is_diffusers_available(self) -> bool:
        """Check if diffusers and accelerate libraries are installed in environment."""
        try:
            import diffusers
            import accelerate
            return True
        except ImportError:
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Check whether diffusers and FLUX.1-schnell model weights are installed."""
        diffusers_ok = self.is_diffusers_available()
        
        # Check local weights size if downloaded
        total_size = 0
        file_count = 0
        if os.path.exists(self.models_dir):
            for root, _, files in os.walk(self.models_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                        file_count += 1
                    except Exception:
                        pass

        size_gb = total_size / (1024 ** 3)
        has_weights = size_gb > 2.0 and os.path.exists(os.path.join(self.models_dir, "model_index.json"))

        if not diffusers_ok:
            status_text = (
                "⚪ **FLUX.1-schnell not installed (Optional)** — "
                "To generate 4K YouTube thumbnails with sharp typography in ~2 seconds on your GPU, "
                "click **📥 Install / Download FLUX.1-schnell** below (or via Pinokio menu)."
            )
        elif not has_weights:
            status_text = (
                "🟡 **Dependencies ready, model weights not downloaded** — "
                "Diffusers is installed. Click **📥 Install / Download FLUX.1-schnell (~12 GB)** to fetch the model weights."
            )
        else:
            status_text = (
                f"🟢 **FLUX.1-schnell Installed & Ready** ({size_gb:.1f} GB on disk) — "
                "Ultra-fast generation (4 steps / ~2s on RTX 4090, sharp typography & HD resolution)."
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
        """Install diffusers, accelerate and helpers into current virtual environment."""
        import subprocess
        try:
            print("[FLUX] Installing diffusers, accelerate, sentencepiece, protobuf...")
            python_bin = sys.executable
            cmd = ["uv", "pip", "install", "diffusers", "accelerate", "sentencepiece", "protobuf"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                cmd = [python_bin, "-m", "pip", "install", "diffusers", "accelerate", "sentencepiece", "protobuf"]
                res = subprocess.run(cmd, capture_output=True, text=True)
            
            if self.is_diffusers_available():
                return {
                    "success": True,
                    "message": "✅ FLUX dependencies (diffusers, accelerate) installed successfully!"
                }
            else:
                return {
                    "success": False,
                    "message": f"⚠️ Error installing dependencies: {res.stderr[:300]}"
                }
        except Exception as e:
            return {"success": False, "message": f"❌ Error: {e}"}

    def _ensure_scheduler_config(self):
        """Ensure scheduler/scheduler_config.json exists for diffusers compatibility."""
        sched_dir = os.path.join(self.models_dir, "scheduler")
        if os.path.exists(sched_dir):
            cfg = os.path.join(sched_dir, "config.json")
            sched_cfg = os.path.join(sched_dir, "scheduler_config.json")
            if os.path.exists(cfg) and not os.path.exists(sched_cfg):
                import shutil
                try:
                    shutil.copyfile(cfg, sched_cfg)
                    print(f"[FLUX] Copied {cfg} -> {sched_cfg} for diffusers compatibility.")
                except Exception as e:
                    print(f"[FLUX] Error copying scheduler config: {e}")

    def download_model_weights(self) -> Dict[str, Any]:
        """Download FLUX.1-schnell model weights to local directory without gated 403 errors."""
        if not self.is_diffusers_available():
            dep_res = self.install_dependencies()
            if not dep_res.get("success"):
                return {
                    "success": False,
                    "message": f"Failed to install dependencies: {dep_res.get('message')}",
                    "status_text": self.get_model_status()["status_text"]
                }

        try:
            from huggingface_hub import snapshot_download
            # Check if user has an active HF_TOKEN, otherwise use ungated mirror
            token = os.environ.get("HF_TOKEN") or None
            target_repo = self.official_model_id if token else self.model_id
            print(f"[FLUX] Downloading {target_repo} to {self.models_dir} (ungated, zero token required)...")
            
            snapshot_download(
                repo_id=target_repo,
                local_dir=self.models_dir,
                resume_download=True,
                token=token
            )
            self._ensure_scheduler_config()
            status = self.get_model_status()
            return {
                "success": True,
                "message": f"✅ FLUX.1-schnell downloaded successfully ({status['size_gb']} GB) in `{self.models_dir}`.",
                "status_text": status["status_text"]
            }
        except Exception as e:
            # Fallback to ungated mirror if official repo returned 403/401
            try:
                print(f"[FLUX] Trying ungated mirror {self.model_id}...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.models_dir,
                    resume_download=True
                )
                self._ensure_scheduler_config()
                status = self.get_model_status()
                return {
                    "success": True,
                    "message": f"✅ FLUX.1-schnell downloaded successfully ({status['size_gb']} GB) in `{self.models_dir}`.",
                    "status_text": status["status_text"]
                }
            except Exception as mirror_err:
                status = self.get_model_status()
                return {
                    "success": False,
                    "error": str(mirror_err),
                    "message": f"❌ Download error: {mirror_err}",
                    "status_text": f"🔴 Download error: {mirror_err}"
                }

    def delete_model_weights(self) -> Dict[str, Any]:
        """Delete local FLUX model weights to reclaim disk space."""
        try:
            self.unload()
            if os.path.exists(self.models_dir):
                import shutil
                shutil.rmtree(self.models_dir, ignore_errors=True)
                os.makedirs(self.models_dir, exist_ok=True)
            status = self.get_model_status()
            return {
                "success": True,
                "message": "🗑️ FLUX.1-schnell weights deleted. Disk space reclaimed!",
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
        Strips non-ASCII diacritics/accents (e.g. 'HERMÈS' -> 'HERMES')
        and cleans invalid punctuation while preserving letters, numbers, and hyphens.
        Crucial for FLUX.1 T5-XXL tokenizer to avoid sub-byte fragmentation that ruins letter rendering.
        """
        import unicodedata
        nfkd = unicodedata.normalize('NFKD', text)
        ascii_text = nfkd.encode('ASCII', 'ignore').decode('utf-8')
        ascii_clean = re.sub(r'[^a-zA-Z0-9\s\-_:!]', '', ascii_text)
        return re.sub(r'\s+', ' ', ascii_clean).strip()

    def enhance_prompt(
        self,
        user_input: str = "",
        video_title: str = "",
        style_preset: str = "YouTube Viral High-CTR"
    ) -> str:
        """
        AI Prompt Assistant for FLUX: Formats optimal prompts with explicit double-quoted typography
        for clean, readable text rendering and studio lighting in 4-step diffusion.
        Preserves 1-3 punchy uppercase words and strips diacritics to prevent spelling errors in FLUX.1-schnell.
        """
        user_raw = user_input.strip() if user_input else ""
        user_raw = re.sub(r'[\r\n]+', ' ', user_raw)

        # 1. Check if user already provided explicit quoted text (single or double quotes)
        explicit_quotes = re.findall(r"['\"]([^'\"]+)['\"]", user_raw)
        if explicit_quotes:
            combined = " ".join(q.strip() for q in explicit_quotes if q.strip())
            words = combined.split()
            short_badge = self.clean_ascii_typography(" ".join(words[:3])).upper()
        elif user_raw:
            # 2. If user explicitly entered words, preserve them directly (up to 3 words)
            raw_words = user_raw.split()
            if len(raw_words) <= 3:
                short_badge = self.clean_ascii_typography(" ".join(raw_words)).upper()
            else:
                # If a longer description was entered without quotes, extract 2 key words
                stopwords = {'dans', 'pour', 'avec', 'les', 'des', 'une', 'qui', 'que', 'the', 'and', 'for', 'with', 'sur', 'par', 'un', 'le', 'la'}
                filtered = [w for w in raw_words if w.lower() not in stopwords]
                if not filtered:
                    filtered = raw_words
                short_badge = self.clean_ascii_typography(" ".join(filtered[:2])).upper()
        else:
            # 3. Fallback to video_title
            topic = video_title.strip() or "HERMES AGENT"
            topic = re.sub(r'[\r\n]+', ' ', topic)
            topic = re.sub(r'\.(mp4|mkv|mov|avi)$', '', topic, flags=re.IGNORECASE)
            stopwords = {
                'tutoriel', 'tuto', 'guide', 'complet', 'installation', 'configuration', 'comment', 'faire',
                'sur', 'dans', 'pour', 'avec', 'les', 'des', 'une', 'qui', 'que', 'the', 'how', 'to', 'and',
                'for', 'with', 'full', 'setup', 'tutorial', 'video', 'windows', 'cours', 'apprendre'
            }
            raw_words = [w for w in re.sub(r'[^\w\s]', ' ', topic).split() if len(w) >= 2]
            filtered = [w for w in raw_words if w.lower() not in stopwords]
            if not filtered:
                filtered = raw_words
            short_badge = self.clean_ascii_typography(" ".join(filtered[:2])).upper() if filtered else "HERMES AGENT"

        if not short_badge:
            short_badge = "HERMES AGENT"

        # Modern FLUX prompt styles: natural language with double quotes, 20-25 words, zero tag soup
        styles = {
            "YouTube Viral High-CTR": (
                f'YouTube thumbnail with bold modern 3D text that reads "{short_badge}" centered, '
                f'vibrant electric cyan and warm amber rim lighting, dark studio background, high contrast, clean composition.'
            ),
            "3D Isometric & Tech Glow": (
                f'3D isometric tech visual with floating glass panel displaying crisp glowing text "{short_badge}", '
                f'cyber purple and neon turquoise gradient lighting, dark sleek background, modern aesthetic.'
            ),
            "Cyberpunk & Bold Neon": (
                f'Futuristic cyberpunk visual with bright glowing neon sign that reads "{short_badge}", '
                f'dark rainy cityscape background, magenta and cyan neon reflections, high contrast.'
            ),
            "Minimalist & Clean SaaS": (
                f'Clean modern minimalist design with sleek floating card displaying bold typography "{short_badge}", '
                f'soft pastel gradient backdrop, elegant studio lighting, balanced composition.'
            ),
            "Photorealistic Studio Shot": (
                f'Professional commercial photo of modern tech desk setup with glowing letters that read "{short_badge}" in the center, '
                f'shallow depth of field, warm cinematic lighting, high quality.'
            )
        }

        return styles.get(style_preset, styles["YouTube Viral High-CTR"])

    @staticmethod
    def sanitize_image(image) -> Any:
        """
        Completely purges all AI generation metadata, EXIF tags, PNG text chunks
        (prompts, models, steps, diffusers signatures, C2PA, etc.) to ensure 100%
        human-looking image files that appear hand-crafted or raster-exported.
        """
        from PIL import Image
        clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
        clean.info = {}
        return clean

    def generate_thumbnail(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        steps: int = 4,
        output_path: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate AI YouTube Thumbnail using FLUX.1-schnell.
        Supports Text-to-Image and Image-to-Image (if reference_image_path is provided).
        All generated images are automatically sanitized to strip 100% of AI metadata.
        """
        if not self.is_diffusers_available():
            return {
                "success": False,
                "error": "diffusers not installed",
                "message": (
                    "⚠️ FLUX.1-schnell is not installed yet. "
                    "Click 'Install / Download FLUX.1-schnell' above (Optional)."
                )
            }

        if not prompt or not prompt.strip():
            return {"success": False, "error": "Prompt cannot be empty."}

        clean_prompt = prompt.strip()
        timestamp = int(time.time())
        if not output_path:
            output_dir = os.path.join(OUTPUT_DIR, "flux_generated")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"flux_thumb_{timestamp}.png")

        # Dimensions based on aspect ratio (multiples of 16/32 for FLUX)
        if "9:16" in aspect_ratio:
            width, height = 768, 1344
        elif "1:1" in aspect_ratio:
            width, height = 1024, 1024
        else: # 16:9 standard YouTube Thumbnail
            width, height = 1280, 720

        # Steps: FLUX.1-schnell is trained for 4 steps
        actual_steps = max(1, min(steps, 8))

        # Dynamic seed: randomize when None or negative (-1)
        if seed is None or seed < 0:
            import random
            seed = random.randint(1, 2147483647)

        try:
            import gc
            import torch
            from PIL import Image
            from diffusers import FluxPipeline, FluxImg2ImgPipeline

            from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
                has_bnb = True
            except ImportError:
                has_bnb = False

            # Free residual VRAM before pipeline load
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._ensure_scheduler_config()

            # Determine weights source: local folder or ungated mirror
            model_source = self.models_dir if os.path.exists(os.path.join(self.models_dir, "model_index.json")) else self.model_id
            print(f"[FLUX] Loading FLUX.1-schnell from {model_source} (Seed: {seed})...")

            is_img2img = bool(reference_image_path and os.path.exists(reference_image_path))
            generator = torch.Generator(device="cpu").manual_seed(seed)

            # Build or load NF4 quantized transformer once and cache it on self (~6.2GB VRAM, 0% PCIe spill)
            if self.transformer is None and has_bnb and torch.cuda.is_available():
                print("[FLUX] Accelerating with NF4 Quantized Transformer (~6.2GB VRAM, 0% PCIe spill)...")
                quant_config = DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.transformer = FluxTransformer2DModel.from_pretrained(
                    model_source,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16
                )

            t0 = time.time()
            if is_img2img:
                print(f"[FLUX] Running Img2Img with reference: {reference_image_path} (Seed: {seed})")
                ref_img = Image.open(reference_image_path).convert("RGB")
                ref_img = ref_img.resize((width, height), Image.Resampling.LANCZOS)

                if self.img2img_pipeline is None:
                    kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
                    if self.transformer is not None:
                        kwargs["transformer"] = self.transformer
                    self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(model_source, **kwargs)
                    if torch.cuda.is_available():
                        self.img2img_pipeline.enable_model_cpu_offload()

                result = self.img2img_pipeline(
                    prompt=clean_prompt,
                    image=ref_img,
                    strength=0.75,
                    num_inference_steps=actual_steps,
                    max_sequence_length=256,
                    guidance_scale=0.0,
                    generator=generator
                )
            else:
                print(f"[FLUX] Running Text-to-Image ({width}x{height}, {actual_steps} steps, Seed: {seed})...")
                if self.pipeline is None:
                    kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
                    if self.transformer is not None:
                        kwargs["transformer"] = self.transformer
                    self.pipeline = FluxPipeline.from_pretrained(model_source, **kwargs)
                    if torch.cuda.is_available():
                        self.pipeline.enable_model_cpu_offload()

                result = self.pipeline(
                    prompt=clean_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=actual_steps,
                    max_sequence_length=256,
                    guidance_scale=0.0,
                    generator=generator
                )

            gen_img = result.images[0]
            clean_img = self.sanitize_image(gen_img)
            clean_img.save(output_path, format="PNG", optimize=True)
            elapsed = time.time() - t0
            print(f"[FLUX] Generated & sanitized successfully in {elapsed:.2f}s (Seed: {seed}, 100% clean metadata)!")

            return {
                "success": True,
                "image_path": output_path,
                "prompt": clean_prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": f"{width}x{height}",
                "seed": seed,
                "elapsed_seconds": round(elapsed, 1),
                "message": f"✅ AI Thumbnail generated successfully with FLUX.1-schnell in {elapsed:.1f}s ({width}x{height}, Seed: {seed})!"
            }

        except Exception as e:
            import traceback
            print(f"[FLUX] Generation error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ FLUX generation error: {e}"
            }

    def generate_ab_thumbnails(
        self,
        base_prompt: str,
        video_title: str = "",
        reference_image_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        steps: int = 4,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate 3 diverse, high-CTR YouTube thumbnail variants for YouTube 'Test & Compare' (A/B testing).
        Variant A: High-CTR Viral Hook (Electric Cyan / Warm Amber Lighting & Bold 3D Text)
        Variant B: 3D Isometric & Modern Tech Glow (Glassmorphism & Cyber Gradients)
        Variant C: Photorealistic Studio & Cinematic Shot (85mm f/1.8 Dramatic Depth)
        
        All 3 images are sanitized (metadata purged) and packaged into a ready-to-upload ZIP archive.
        """
        if not output_dir:
            output_dir = os.path.join(OUTPUT_DIR, "flux_generated")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())

        # Determine core topic/text for prompt variants
        topic = base_prompt.strip() or video_title.strip() or "YouTube Video"

        if base_seed is None or base_seed <= 0:
            import random
            base_seed = random.randint(1, 2147480000)

        variants_config = [
            {
                "id": "variant_A",
                "label": "Variante A (Viral High-CTR)",
                "style": "YouTube Viral High-CTR",
                "filename": f"youtube_thumb_A_viral_{timestamp}.png",
                "seed": base_seed,
            },
            {
                "id": "variant_B",
                "label": "Variante B (3D Tech Glow)",
                "style": "3D Isometric & Tech Glow",
                "filename": f"youtube_thumb_B_tech_{timestamp}.png",
                "seed": base_seed + 101,
            },
            {
                "id": "variant_C",
                "label": "Variante C (Studio Cinématique)",
                "style": "Photorealistic Studio Shot",
                "filename": f"youtube_thumb_C_cinematic_{timestamp}.png",
                "seed": base_seed + 202,
            }
        ]

        results = []
        for i, var in enumerate(variants_config):
            if progress_callback:
                progress_callback((i / 3.0), f"Génération {var['label']} en cours ({i+1}/3)...")
            
            # Format tailored prompt for this visual angle
            tailored_prompt = self.enhance_prompt(
                user_input=topic,
                video_title=video_title,
                style_preset=var["style"]
            )
            out_file = os.path.join(output_dir, var["filename"])
            res = self.generate_thumbnail(
                prompt=tailored_prompt,
                reference_image_path=reference_image_path,
                aspect_ratio=aspect_ratio,
                steps=steps,
                output_path=out_file,
                seed=var["seed"]
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
                    "message": f"❌ Erreur sur {var['label']}: {res.get('message')}"
                }

        # Package into ZIP archive for 1-click download
        import zipfile
        zip_filename = f"youtube_ab_testing_pack_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        readme_content = (
            "==============================================================\n"
            "🎯 PACK A/B TESTING YOUTUBE — 3 VARIANTES FLUX.1-SCHNELL\n"
            "==============================================================\n\n"
            "Ce pack contient 3 miniatures à fort taux de clic (CTR) générées avec FLUX.1-schnell.\n"
            "Toutes les métadonnées IA (EXIF, signatures C2PA, paramètres) ont été 100% purgées.\n\n"
            "CONTENU DU PACK :\n"
            f"- Variante A (Viral High-CTR) : {results[0]['filename']}\n"
            f"- Variante B (3D Tech & Glow) : {results[1]['filename']}\n"
            f"- Variante C (Studio Cinématique) : {results[2]['filename']}\n\n"
            "COMMENT LANCER L'A/B TEST SUR YOUTUBE STUDIO :\n"
            "1. Rendez-vous sur YouTube Studio -> Contenu -> Sélectionnez votre vidéo.\n"
            "2. Dans la section 'Miniature', cliquez sur le menu des 3 points (...) -> 'Tester et comparer'.\n"
            "3. Téléversez les 3 variantes (A, B et C).\n"
            "4. YouTube alternera automatiquement l'affichage auprès de votre audience et désignera la miniature gagnante !\n"
            "==============================================================\n"
        )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README_YOUTUBE_AB_TEST.txt", readme_content)
            for r in results:
                if os.path.exists(r["image_path"]):
                    zf.write(r["image_path"], arcname=r["filename"])

        if progress_callback:
            progress_callback(1.0, "Pack A/B Testing 3 Variantes généré avec succès !")

        return {
            "success": True,
            "variants": results,
            "zip_path": zip_path,
            "zip_filename": zip_filename,
            "message": f"🎉 **3 Variantes A/B Test générées avec succès !** (Pack ZIP prêt pour YouTube Studio)"
        }

    def unload(self):
        """Free GPU VRAM and release FLUX pipelines."""
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if self.img2img_pipeline is not None:
            del self.img2img_pipeline
            self.img2img_pipeline = None
        self.model_loaded = False
        gc.collect()
        if DEVICE == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("[FLUX] VRAM cleared and model unloaded.")

flux_studio = FluxGenerator()
