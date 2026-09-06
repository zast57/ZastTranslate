"""
ZastTranslate — Blog Generator & SEO Studio
Anti-AI Humanizer Engine adapted from:
- blader/humanizer (https://github.com/blader/humanizer) — MIT License
- Wikipedia: Signs of AI writing (WikiProject AI Cleanup)
"""

import os
import re
import sys
import json
import shutil
import zipfile
import subprocess
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple

HUMANIZER_GITHUB_SKILL_URL = "https://raw.githubusercontent.com/blader/humanizer/main/SKILL.md"
_CACHED_HUMANIZER_RULES: Dict[str, Any] = {
    "version": "2.11.2",
    "watch_words": []
}

def sync_humanizer_rules_from_github() -> Dict[str, Any]:
    """
    Fetch and sync the latest Humanizer rules and AI watch words directly from blader/humanizer GitHub repository.
    """
    global _CACHED_HUMANIZER_RULES
    try:
        headers = {
            "User-Agent": "ZastTranslate-HumanizerSync/1.0"
        }
        req = urllib.request.Request(HUMANIZER_GITHUB_SKILL_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            
        version_match = re.search(r'version:\s*["\']?([^"\'\n]+)', content)
        version = version_match.group(1).strip() if version_match else "latest"
        
        watch_words = []
        for line in content.split("\n"):
            if "Words to watch:" in line or "High-frequency AI words:" in line:
                raw = line.split(":", 1)[1].strip()
                words = [w.strip().strip('*_`"') for w in raw.split(",") if w.strip()]
                for w in words:
                    if w and w not in watch_words:
                        watch_words.append(w)
                        
        _CACHED_HUMANIZER_RULES["version"] = version
        _CACHED_HUMANIZER_RULES["watch_words"] = watch_words
        print(f"[Humanizer] Synced from blader/humanizer GitHub (version: {version}, {len(watch_words)} AI watch patterns loaded)")
        return {
            "success": True,
            "version": version,
            "watch_words_count": len(watch_words),
            "message": f"✅ Successfully synced with blader/humanizer v{version} ({len(watch_words)} AI patterns loaded from GitHub)."
        }
    except Exception as e:
        print(f"[Humanizer] GitHub sync note (using built-in v{_CACHED_HUMANIZER_RULES.get('version', '2.11.2')}): {e}")
        return {
            "success": False,
            "version": _CACHED_HUMANIZER_RULES.get("version", "2.11.2"),
            "message": f"ℹ️ Using built-in Humanizer rules v{_CACHED_HUMANIZER_RULES.get('version', '2.11.2')} (Offline / local fallback)."
        }

def fetch_live_seo_keywords(query: str, lang_code: str = "fr", max_keywords: int = 12) -> List[str]:
    """
    Query Google Autocomplete & YouTube Suggest endpoints in real-time
    to discover real human search queries and high-intent LSI keywords (0 API keys required).
    """
    if not query or not query.strip():
        return []
        
    clean_q = re.sub(r'[\(\)\[\]_:\-]+', ' ', query).strip()
    words = [w for w in clean_q.split() if len(w) > 2][:4]
    search_terms = [" ".join(words)]
    if len(words) >= 2:
        search_terms.append(f"{words[0]} {words[1]}")
    
    discovered = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    for term in search_terms:
        encoded = urllib.parse.quote(term)
        endpoints = [
            f"https://suggestqueries.google.com/complete/search?client=chrome&hl={lang_code}&q={encoded}",
            f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&hl={lang_code}&q={encoded}"
        ]
        for url in endpoints:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                        for item in data[1]:
                            if isinstance(item, str):
                                item_clean = item.strip().lower()
                                if item_clean and item_clean not in discovered:
                                    discovered.append(item_clean)
            except Exception:
                pass
                
    return discovered[:max_keywords]

def format_seo_title(title: str, is_fr: bool = True) -> str:
    """Format and polish titles: sentence case (no English Title Case in French), brand capitalization, no buzzwords, 50-65 chars calibration."""
    if not title:
        return ""
    t = title.strip().strip('"\'«»#*')
    
    # Strip site name or slug suffixes appended with pipes or double dashes (e.g. "Title | Slug")
    if "|" in t:
        t = t.split("|")[0].strip()
    
    # Remove AI buzzword clichés
    cliches = [
        r"\b(?:le\s+)?guide\s+ultime\b",
        r"\b(?:the\s+)?ultimate\s+guide\b",
        r"\brévolutionnaire\b",
        r"\brevolutionary\b",
        r"\bsecret(?:s)?\s+dévoilé(?:s)?\b",
        r"\btout\s+ce\s+que\s+vous\s+devez\s+savoir\b",
        r"\beverything\s+you\s+need\s+to\s+know\b",
        r"\brévélation\s+choc\b",
        r"\ble\s+twist\s+final\b",
        r"\bla\s+preuve\s+par\s+la\s+démo\b"
    ]
    for c in cliches:
        t = re.sub(c, "", t, flags=re.IGNORECASE)
    
    # Clean redundant spaces or trailing colons/dashes
    t = re.sub(r'[\s\-:_]+$', '', t).strip()
    t = re.sub(r'^[\s\-:_]+', '', t).strip()

    if is_fr:
        # In French, only the first letter and words after a single colon are capitalized, plus proper nouns
        parts = [p.strip() for p in t.split(":") if p.strip()]
        # If multiple colons were present, keep at most 2 parts (Title : Subtitle)
        if len(parts) > 2:
            parts = [parts[0], " - ".join(parts[1:])]
        formatted_parts = []
        for part_idx, p_strip in enumerate(parts):
            words = p_strip.split()
            fixed_words = []
            for idx, w in enumerate(words):
                w_lower = w.lower()
                # Check if brand or proper noun
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
                elif part_idx == 0 and idx == 0:
                    fixed_words.append(w.capitalize())
                else:
                    fixed_words.append(w.lower())
            formatted_parts.append(" ".join(fixed_words))
        t = " : ".join(formatted_parts)
    else:
        if t:
            t = t[0].upper() + t[1:]

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

    # SEO character calibration (ideally 50-65 chars for Google SERP)
    if len(t) > 65:
        # If there is a colon, check if the main part alone or a trimmed subtitle fits
        if " : " in t:
            main_p, sub_p = t.split(" : ", 1)
            if 45 <= len(main_p) <= 65:
                # If main part is strong and fits sweet spot, keep main part
                t = main_p
            else:
                # Trim cleanly at last space before 65
                trimmed = t[:65].rsplit(" ", 1)[0]
                t = re.sub(r'[\s\-:_]+$', '', trimmed).strip()
        else:
            trimmed = t[:65].rsplit(" ", 1)[0]
            t = re.sub(r'[\s\-:_]+$', '', trimmed).strip()

    return t

class SEOArticleGenerator:
    """
    Elite SEO Blog Post & WordPress Content Studio Engine.
    - Ground-truth knowledge extraction from Whisper transcripts & video metadata.
    - Anti-AI detection heuristics & natural human voice styling.
    - Full SEO Kit: Title H1, Meta Description (145-160 chars), Slug, Focus Keywords.
    - Double Output: Standard Markdown + Ready-to-paste WordPress Gutenberg HTML.
    - Automatic video keyframe capture (FFmpeg) with SEO ALT tags & captions.
    - AI Image Generation Prompts for Featured Image and section headers.
    """

    STYLES = {
        "Step-by-Step Tutorial (How-To Guide)": {
            "fr": "Rédige sous la forme d'un guide pratique et concret, étape par étape (How-to). Utilise des verbes d'action, des instructions précises, des encadrés d'astuces pratiques et des prérequis clairs.",
            "en": "Write as a hands-on, step-by-step tutorial (How-to guide). Use actionable verbs, clear instructions, practical tip callouts, and explicit prerequisites.",
            "es": "Escribe como una guía práctica paso a paso (Tutorial). Usa verbos de acción, instrucciones claras, consejos prácticos y requisitos previos.",
            "de": "Schreibe als praktische Schritt-für-Schritt-Anleitung (How-to). Verwende Handlungsverben, klare Anweisungen, Praxistipps und Voraussetzungen."
        },
        "Expert & Technical Deep-Dive": {
            "fr": "Rédige avec une posture d'expert chevronné et d'architecte technique. Analyse les fonctionnalités en profondeur, les choix d'architecture, les performances, la sécurité des données et les cas d'usage avancés.",
            "en": "Write with the tone of a seasoned technical expert and software architect. Deep-dive into features, architectural decisions, performance, data security, and advanced use cases.",
            "es": "Escribe con la postura de un experto técnico y arquitecto de software. Analiza en profundidad las funciones, la arquitectura, el rendimiento y la seguridad.",
            "de": "Schreibe mit der Expertise eines erfahrenen Software-Architekten. Analysiere Funktionen, Architektur, Performance und Datensicherheit fundiert."
        },
        "Storytelling & Case Study": {
            "fr": "Rédige avec une approche narrative immersive et incarnée. Pars d'un problème réel ou d'une frustration commune, raconte le cheminement et démontre comment les solutions présentées transforment l'expérience.",
            "en": "Write with an engaging, narrative storytelling approach. Start from a real-world frustration or problem, share the journey, and illustrate how the solutions transform the outcome.",
            "es": "Escribe con un enfoque narrativo y testimonial. Comienza con un problema real y muestra cómo la solución transforma los resultados.",
            "de": "Schreibe mit einem mitreißenden Storytelling-Ansatz. Gehe von einem konkreten Problem aus und zeige, wie die Lösung echte Ergebnisse liefert."
        },
        "Journalistic & Objective Review": {
            "fr": "Rédige dans un style journalistique rigoureux et objectif. Présente un résumé exécutif, décortique les nouveautés, compare les avantages et inconvénients sans parti pris commercial.",
            "en": "Write in a rigorous, objective journalistic style. Provide an executive summary, break down new capabilities, and weigh pros and cons objectively.",
            "es": "Escribe en un estilo periodístico riguroso y objective, evaluando pros y contras con un resumen ejecutivo.",
            "de": "Schreibe in einem sachlichen, journalistischen Stil mit Management-Zusammenfassung und neutraler Vor- und Nachteile-Analyse."
        },
        "High-Converting Copywriting": {
            "fr": "Rédige avec les techniques du copywriting à fort taux de conversion : accroche magnétique (AIDA/PAS), focalisation sur les bénéfices concrets plutôt que les fonctionnalités, et appels à l'action percutants.",
            "en": "Write with high-converting copywriting frameworks (AIDA/PAS): magnetic hook, strong focus on tangible user benefits over raw specs, and compelling calls to action.",
            "es": "Escribe con técnicas de copywriting persuasivo enfocadas en beneficios y llamadas a la acción irresistibles.",
            "de": "Schreibe mit verkaufsstarkem Copywriting, Fokus auf echten Nutzerwert und klaren Call-to-Actions."
        },
        "Accessible Beginner's Guide": {
            "fr": "Rédige dans un langage simple, chaleureux et ultra-accessible. Définis tous les termes techniques à l'aide d'analogies du quotidien et réponds aux questions fréquentes des débutants.",
            "en": "Write in a friendly, warm, and highly accessible tone. Explain technical concepts using simple everyday analogies and address common beginner questions.",
            "es": "Escribe en un lenguaje sencillo, cercano y muy accesible, explicando conceptos técnicos con analogías sencillas.",
            "de": "Schreibe in einer verständlichen, sympathischen Sprache und erkläre Fachbegriffe mit einfachen Analogien."
        }
    }

    STYLE_ALIASES = {
        "Guide Tutoriel (Pas à pas)": "Step-by-Step Tutorial (How-To Guide)",
        "Expert & Analyse Technique": "Expert & Technical Deep-Dive",
        "Storytelling & Cas Pratique": "Storytelling & Case Study",
        "Journalistique & Revue Neutre": "Journalistic & Objective Review",
        "Copywriting & Vendeur / Persuasif": "High-Converting Copywriting",
        "Vulgarisation & Grand Public": "Accessible Beginner's Guide",
        "Court (600 - 800 mots)": "Short (600 - 800 words)",
        "Moyen (1000 - 1500 mots)": "Medium (1000 - 1500 words)",
        "Long (1800 - 2500 mots)": "Long (1800 - 2500 words)"
    }

    LENGTH_PRESETS = {
        "Short (600 - 800 words)": 800,
        "Medium (1000 - 1500 words)": 1500,
        "Long (1800 - 2500 words)": 2500,
        "Court (600 - 800 mots)": 800,
        "Moyen (1000 - 1500 mots)": 1500,
        "Long (1800 - 2500 mots)": 2500
    }

    def __init__(self):
        pass

    def _get_lang_instructions(self, target_lang: str) -> Tuple[str, str]:
        """Normalize target language into code and standard name."""
        lang_lower = target_lang.lower()
        if "fr" in lang_lower or "french" in lang_lower or "français" in lang_lower:
            return "fr", "Français"
        elif "en" in lang_lower or "english" in lang_lower or "anglais" in lang_lower:
            return "en", "English"
        elif "es" in lang_lower or "spanish" in lang_lower or "espagnol" in lang_lower:
            return "es", "Español"
        elif "de" in lang_lower or "german" in lang_lower or "allemand" in lang_lower:
            return "de", "Deutsch"
        elif "it" in lang_lower or "italian" in lang_lower or "italien" in lang_lower:
            return "it", "Italiano"
        elif "pt" in lang_lower or "portuguese" in lang_lower or "portugais" in lang_lower:
            return "pt", "Português"
        elif "ja" in lang_lower or "japanese" in lang_lower or "japonais" in lang_lower:
            return "ja", "日本語"
        elif "zh" in lang_lower or "chinese" in lang_lower or "chinois" in lang_lower:
            return "zh", "中文"
        elif "ru" in lang_lower or "russian" in lang_lower or "russe" in lang_lower:
            return "ru", "Русский"
        elif "ar" in lang_lower or "arabic" in lang_lower or "arabe" in lang_lower:
            return "ar", "العربية"
        return "en", target_lang

    def _clean_semicolons(self, text: str) -> str:
        """
        Eliminate semicolons (;) in prose, which are an unmistakable sign of AI writing.
        Preserves semicolons inside code blocks (```...```), inline code (`...`),
        and HTML entities (&amp;, &nbsp;, etc.).
        """
        if not text or ";" not in text:
            return text

        # Protect code blocks
        code_blocks = []
        def save_block(m):
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(r'```.*?```', save_block, text, flags=re.DOTALL)
        text = re.sub(r'`[^`\n]+`', save_block, text)

        # Protect HTML entities
        entities = []
        def save_entity(m):
            entities.append(m.group(0))
            return f"__HTML_ENTITY_{len(entities)-1}__"
        
        text = re.sub(r'&[a-zA-Z0-9#]+;', save_entity, text)

        # In prose: replace semicolons with a period and capitalize the next character
        def repl_semi(m):
            char = m.group(1)
            return ". " + char.upper()
        
        # Semicolon followed by a letter (Latin, accented French, etc.)
        text = re.sub(r'\s*;\s*([a-zA-Z\u00C0-\u00FF])', repl_semi, text)
        # Semicolon before newline or at end of sentence
        text = re.sub(r'\s*;\s*$', '.', text, flags=re.MULTILINE)
        # Any remaining semicolon before quotes, digits or symbols
        text = re.sub(r'\s*;\s*', '. ', text)
        # Clean double dots resulting from replacements
        text = re.sub(r'\.\s*\.', '.', text)

        # Restore HTML entities and code blocks
        for i, ent in enumerate(entities):
            text = text.replace(f"__HTML_ENTITY_{i}__", ent)
        for i, cb in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", cb)

        return text

    def _clean_heading_emojis(self, text: str) -> str:
        """Strip decorative emojis from Markdown headings (Humanizer Pattern #18)."""
        if not text:
            return ""
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27ff\u2b50\u200d\ufe0f]')
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            m = re.match(r'^(#{1,6}\s+)(.*)$', line)
            if m:
                h_hashes, h_content = m.groups()
                h_no_emoji = emoji_pattern.sub('', h_content).strip()
                cleaned.append(f"{h_hashes}{h_no_emoji}")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def clean_ai_artifacts(self, text: str, is_fr: bool = True) -> str:
        """
        Strip AI reasoning blocks, markdown chatter, robotic phrases, and cliché formulas,
        fully inspired by the Humanizer rulebook (WikiProject AI Cleanup 35 patterns).
        Eliminates third-person video commentaries, robotic semicolons (;), and heading emojis.
        """
        if not text:
            return ""
        # Remove thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = text.replace('<think>', '').replace('</think>', '').strip()
        text = re.sub(r'Thinking Process:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        # Remove LLM introductory / concluding chatter (Pattern #20)
        meta_chatter = [
            r"^Here is (the|an)?.*?(article|blog post|guide).*?:\s*",
            r"^Voici (l'|un|le)?.*?(article|guide|tutoriel).*?:\s*",
            r"^Certainly!.*?\n+",
            r"^Bien sûr!.*?\n+",
            r"^Sure, here is.*?\n+",
            r"Hope this helps!.*?\Z",
            r"J'espère que cet article vous plaira.*?\Z",
            r"N'hésitez pas à me poser des questions.*?\Z",
            r"Feel free to ask.*?\Z"
        ]
        for pat in meta_chatter:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Normalization: Em dashes / En dashes -> spaced hyphen (Humanizer Pattern #14)
        text = re.sub(r'\s*[—–]\s*', ' - ', text)
        # Curly quotes to plain ASCII quotes (Humanizer Pattern #19)
        text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")

        # Anti-AI Cliché replacements inspired by WikiProject AI Cleanup (French)
        if is_fr:
            fr_cliches = [
                # Meta-Video & Third-Party Commentary (Strip all references to "la vidéo", "l'auteur", etc.)
                (r"\bComme\s+(?:vu|montré|expliqué|indiqué|démontré|précisé)\s+(?:dans\s+la\s+vidéo|dans\s+ce\s+tuto|dans\s+ce\s+tutoriel\s+vidéo|à\s+l'écran),?\s*", "Comme nous allons le voir, "),
                (r"\bcomme\s+on\s+le\s+voit\s+à\s+l'écran,?\s*", "en pratique, "),
                (r"\bDans\s+cette\s+vidéo,?\s*", "Dans ce guide, "),
                (r"\bdans\s+cette\s+vidéo,?\s*", "dans ce guide, "),
                (r"\bDans\s+la\s+vidéo,?\s*", "Dans ce guide, "),
                (r"\bdans\s+la\s+vidéo,?\s*", "dans ce guide, "),
                (r"\bAu\s+cours\s+de\s+la\s+vidéo,?\s*", "Au fil de ce guide, "),
                (r"\bau\s+cours\s+de\s+la\s+vidéo,?\s*", "au fil de ce guide, "),
                (r"\bTout\s+au\s+long\s+de\s+la\s+vidéo,?\s*", "Tout au long de ce guide, "),
                (r"\btout\s+au\s+long\s+de\s+la\s+vidéo,?\s*", "tout au long de ce guide, "),
                (r"\bÀ\s+la\s+fin\s+de\s+la\s+vidéo,?\s*", "Pour clore ce tutoriel, "),
                (r"\bEn\s+regardant\s+la\s+vidéo,?\s*", "En pratique, "),
                (r"\bdans\s+l'extrait,?\s*", "dans cette étape, "),
                (r"\b(?:la|cette)\s+vidéo\s+(?:précise|indique|explique|montre|démontre|souligne|recommande|conseille)\s+(?:également\s+)?que\b", "À noter que "),
                (r"\b(?:la|cette)\s+vidéo\s+(?:précise|indique|explique|montre|démontre|souligne|recommande|conseille)\b", "Ce guide précise"),
                (r"\bL'auteur\s+(?:de\s+la\s+vidéo\s+)?mentionne\s+personnellement\s+(?:que\s+)?\b", "Côté configuration, "),
                (r"\bL'auteur\s+(?:de\s+la\s+vidéo\s+)?(?:nous\s+)?(?:montre|explique|précise|indique|démontre|partage|souligne|conseille|recommande|mentionne)\s*(?:que)?\b", "À noter : "),
                (r"\bL'auteur\s+(?:mentionne|explique|précise|indique|démontre|souligne|montre)\b", "On constate"),
                (r"\bl'auteur\b", "le guide"),
                (r"\bLe\s+(?:créateur|présentateur|formateur|vidéaste|youtubeur|narrateur)\s+(?:de\s+la\s+vidéo\s+)?(?:nous\s+)?(?:montre|explique|précise|indique|démontre|partage|souligne)\s*(?:que)?\b", "Ce guide détaille "),
                (r"\bLe\s+(?:créateur|présentateur|formateur|vidéaste|youtubeur)\s+de\s+la\s+vidéo\b", "Le guide"),
                (r"\bL'intervenant\s+(?:de\s+la\s+vidéo\s+)?(?:nous\s+)?(?:montre|explique|précise|indique)\s*(?:que)?\b", "Ce guide détaille "),
                # Phonetic correction for Whisper transcript hallucinations
                (r"\bMorpheus\b", "un jeu de morpion"),

                # Oral video screen narration chatter (Must never appear in a written blog post)
                (r"\bLà,?\s+(?:je\s+(?:vous\s+)?(?:le\s+)?montre|il\s+cherche|c'est\s+en\s+accéléré|j'ai\s+cliqué|j'ai\s+stoppé|je\s+vais\s+attendre)[^.!?\n]*[.!?\n]?", ""),
                (r"\b(?:En haut )?[Jj]'ai cliqué,?\s*(?:vu qu'il m'a.*?j'ai pu faire Close\.?|sur Close\.?)\s*", ""),
                (r"\bcliqué sur Close\b", ""),
                (r"\bLà,\s+j'ai\s+stoppé\s+le\s+prompt[^.!?\n]*[.!?\n]?", ""),
                (r"\bVu\s+que\s+c'est\s+une\s+démo\s+que\s+je\s+vous\s+montre[^.!?\n]*[.!?\n]?", ""),
                (r"\bAprès,?\s+c'est\s+du\s+blabla\s+technique[^.!?\n]*[.!?\n]?", ""),
                (r"\b(?:Hop\.?\s*)+", ""),
                (r"\bDans cet article,\s*(?:nous allons|nous verrons|découvrez|plongeons dans)\b[^.!?\n]*[.!?\n]?", "\n"),
                (r"\bPlongeons (?:sans plus attendre )?dans (?:le vif du sujet|les détails)[^.!?\n]*[.!?\n]?", "\n"),
                (r"\bSans plus attendre,?\s*", ""),
                (r"\bVoyons sans plus tarder,?\s*", ""),
                (r"\bEn conclusion,?\s*", "\n## Ce qu'il faut retenir\n"),
                (r"\bEn résumé,?\s*", "\n## Synthèse des points clés\n"),
                (r"\bPour conclure,?\s*", "\n## Prochaines étapes\n"),
                (r"\bEn définitive,?\s*", "\n## Ce qu'il faut retenir\n"),

                # YouTube Outros & Video Call-to-Actions (Never belong in a written blog post)
                (r"\b(?:N'hésitez pas à|Pensez à|N'oubliez pas de)\s+(?:liker|partager|commenter|vous abonner|mettre un pouce bleu)[^.!?\n]*[.!?\n]?", ""),
                (r"\b(?:Abonnez-vous|Abonne-toi)\s+(?:à la chaîne|pour ne rien rater|pour plus de guides|pour plus de conseils)[^.!?\n]*[.!?\n]?", ""),
                (r"\bLaissez\s+(?:un commentaire|un pouce bleu|un like)[^.!?\n]*[.!?\n]?", ""),
                (r"\bÀ la prochaine\s*!\s*", ""),
                (r"\bRetrouvez (?:tous )?les liens (?:en|dans la) description[^.!?\n]*[.!?\n]?", ""),
                (r"\bSi cette vidéo vous a plu[^.!?\n]*[.!?\n]?", ""),
                (r"\bMerci d'avoir (?:regardé|suivi) cette vidéo[^.!?\n]*[.!?\n]?", ""),

                # Inflated Importance & Legacy (Pattern #1)
                (r"\bmarque un tournant (?:décisif|majeur|historique)\b", "constitue une étape"),
                (r"\btémoigne (?:avec éclat )?de\b", "montre"),
                (r"\bjoue un rôle (?:charnière|crucial|pivot|incontournable|fondamental)\b", "intervient directement"),
                (r"\bune avancée (?:majeure|révolutionnaire|sans précédent)\b", "une amélioration pratique"),
                (r"\bun véritable tournant\b", "un changement"),
                (r"\bfaçonnant l'avenir de\b", "faisant évoluer"),
                (r"\bune empreinte indélébile\b", "un impact durable"),
                (r"\bchangement de paradigme\b", "nouvelle approche"),
                (r"\bpierre angulaire\b", "base"),

                # Shallow -ing Participle Analysis (Pattern #3)
                (r"\bpermettant de souligner\b", "et souligne"),
                (r"\bmettant en lumière\b", "montrant"),
                (r"\bfavorisant ainsi\b", "et facilite"),
                (r"\billustrant parfaitement\b", "comme le montre"),
                (r"\bassurant ainsi\b", "garantissant"),
                (r"\bouvrant la voie à\b", "permettant"),

                # Sales Language & Hype (Pattern #4)
                (r"\bUn véritable couteau suisse\b", "Un outil polyvalent"),
                (r"\bune solution révolutionnaire\b", "une solution performante"),
                (r"\bun outil incontournable\b", "un outil adapté"),
                (r"\bdes performances époustouflantes\b", "de bonnes performances"),
                (r"\bniché au cœur de\b", "situé dans"),
                (r"\bLe Guide Ultime\b", "Le guide pratique"),
                (r"\bvéritable bijou\b", "outil remarquable"),
                (r"\bbluffant\b", "efficace"),

                # Vague Authority & Unbacked Attribution (Pattern #5)
                (r"\bLes experts s'accordent à dire que\b", "Il apparaît que"),
                (r"\bCertains observateurs notent que\b", "On constate que"),
                (r"\bIl est (?:crucial|primordial|essentiel|impératif|important) de (?:noter|souligner|garder à l'esprit|comprendre) que\b", "À noter :"),
                (r"\bIl convient de (?:noter|souligner|rappeler) que\b", "À retenir :"),

                # Overused AI Words & Formulations (Patterns #7, #27)
                (r"\bÀ l'ère (?:du numérique|de l'intelligence artificielle|de la transformation digitale)[^,\n]*,?", ""),
                (r"\bDans un monde (?:en constante évolution|de plus en plus connecté|numérique)[^,\n]*,?", ""),
                (r"\bDans le paysage (?:actuel|technologique|numérique)[^,\n]*,?", ""),
                (r"\bNaviguer dans les méandres de\b", "Configurer"),
                (r"\bForce est de constater que\b", ""),
                (r"\bAu fond, ce qui compte vraiment,?\s*", ""),
                (r"\bAu cœur de\b", "Dans"),

                # Avoiding is and are (Pattern #8)
                (r"\bse positionne comme\b", "est"),
                (r"\bfait office de\b", "sert de"),
                (r"\bs'érige en\b", "constitue"),

                # False ranges & fillers (Patterns #12, #23)
                (r"\bdes débutants aux experts(?: les plus chevronnés)?\b", "pour tous les profils"),
                (r"\bafin de pouvoir\b", "pour"),
                (r"\bdans le but de\b", "pour"),

                # Fake-candid openings & objections (Patterns #33, #34, #35)
                (r"\bHonnêtement,?\s*", ""),
                (r"\bSoyons clairs,?\s*", ""),
                (r"\bIl ne s'agit pas ici de\b", "L'objectif est de"),
                (r"\bOn pourrait (?:penser|croire|être tenté de penser) que\b", "En pratique,")
            ]
            for pat, rep in fr_cliches:
                text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        else: # English Humanizer Patterns
            en_cliches = [
                # Meta-Video & Third-Party Commentary (Strip all references to "the video", "the author", etc.)
                (r"\bAs\s+(?:seen|shown|explained|demonstrated)\s+in\s+the\s+video,?\s*", "As shown below, "),
                (r"\bThroughout\s+the\s+video,?\s*", "Throughout this tutorial, "),
                (r"\bIn\s+this\s+video,?\s*", "In this guide, "),
                (r"\bIn\s+the\s+video,?\s*", "In this guide, "),
                (r"\bThe\s+video\s+(?:explains|shows|demonstrates|points\s+out|highlights|mentions|states)\s+(?:that\s+)?\b", "Note that "),
                (r"\bThe\s+author\s+(?:personally\s+)?(?:mentions|explains|states|points\s+out|recommends)\s+(?:that\s+)?\b", "It is recommended that "),
                (r"\bThe\s+(?:creator|presenter|host|youtuber)\s+(?:of\s+the\s+video\s+)?(?:explains|shows|points\s+out|mentions)\b", "The method consists of"),
                (r"\bAs\s+seen\s+on\s+screen,?\s*", "In practice, "),

                # Intro / Outro Chatter & Meta Announcements (Patterns #20, #28, #33)
                (r"\bIn this article,?\s*(?:we will|we'll|let's)\s*(?:explore|dive into|discover)\b[^.!?\n]*[.!?\n]?", "\n"),
                (r"\bLet's dive (?:right )?into[^.!?\n]*[.!?\n]?", "\n"),
                (r"\bWithout further ado,?\s*", ""),
                (r"\bIn conclusion,?\s*", "\n## Key Takeaways\n"),
                (r"\bIn summary,?\s*", "\n## Summary & Next Steps\n"),
                (r"\bTo wrap up,?\s*", "\n## Next Steps\n"),

                # YouTube Outros & Video Call-to-Actions (Never belong in a written blog post)
                (r"\b(?:Don't forget to|Remember to)\s+(?:like, share, and subscribe|hit the bell|leave a comment|subscribe to the channel)[^.!?\n]*[.!?\n]?", ""),
                (r"\b(?:Please )?(?:like, share, and subscribe|subscribe to the channel)[^.!?\n]*[.!?\n]?", ""),
                (r"\bSee you in the next one!?", ""),
                (r"\bThanks for watching!?", ""),
                (r"\bLinks are in the description[^.!?\n]*[.!?\n]?", ""),

                # Inflated Importance & Legacy (Pattern #1)
                (r"\bmarks a pivotal moment in\b", "is part of"),
                (r"\bstands as a testament to\b", "shows"),
                (r"\bplays a (?:vital|crucial|pivotal|key) role in\b", "is used for"),
                (r"\bunderscores the significance of\b", "highlights"),
                (r"\bshaping the future of\b", "changing"),
                (r"\ban indelible mark\b", "a lasting impact"),

                # Shallow -ing Participles (Pattern #3)
                (r"\bhighlighting the need for\b", "and requires"),
                (r"\bshowcasing how\b", "showing how"),
                (r"\bfostering a sense of\b", "encouraging"),

                # Sales Language (Pattern #4)
                (r"\bA game changer\b", "An effective solution"),
                (r"\bboasts a (?:wide range|variety)\b", "offers a range"),
                (r"\bnestled in the heart of\b", "located in"),
                (r"\bgroundbreaking\b", "notable"),
                (r"\bbreathtaking\b", "impressive"),

                # Vague Sources (Pattern #5)
                (r"\bExperts believe (?:that)?\b", "Evidence suggests"),
                (r"\bIndustry observers note (?:that)?\b", "It is observed that"),
                (r"\bIt is (?:crucial|essential|imperative|important) to (?:note|remember|keep in mind) that\b", "Note:"),

                # Stock Era & Abstract Landscape (Patterns #7, #27)
                (r"\bIn (?:today's )?(?:fast-paced )?digital (?:world|age|landscape)[^,\n]*,?", ""),
                (r"\bIn an ever-evolving (?:world|landscape)[^,\n]*,?", ""),
                (r"\bAt its core, what (?:really )?matters is,?\s*", "")
            ]
            for pat, rep in en_cliches:
                text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        # Eliminate semicolons (;) in prose outside of code blocks
        text = self._clean_semicolons(text)

        # Strip decorative emojis from headings (Pattern #18)
        text = self._clean_heading_emojis(text)

        # Clean redundant double spaces and multi-line gaps
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def generate_article(
        self,
        segments: List[Dict[str, Any]],
        video_info: Optional[Dict[str, Any]] = None,
        target_lang: str = "French",
        style: str = "Step-by-Step Tutorial (How-To Guide)",
        length: str = "Medium (1000 - 1500 words)",
        include_meta: bool = True,
        llm_backend=None
    ) -> Dict[str, Any]:
        """
        Generate a complete, high-ranking, human-styled SEO article from transcript.
        Returns a dict containing SEO metadata, Markdown article, Gutenberg HTML, and Image Prompts.
        """
        lang_code, lang_name = self._get_lang_instructions(target_lang)
        
        # Build comprehensive knowledge base from segments
        if not segments:
            return {
                "error": "Aucune transcription disponible. Veuillez d'abord importer une vidéo ou un fichier SRT."
            }

        video_title = ""
        video_desc = ""
        if video_info:
            video_title = video_info.get("title", "")
            video_desc = video_info.get("description", "")

        # Aggregate segments into cohesive chronological paragraphs respecting sentence boundaries
        # Preserves 100% of contextual nuance, complete sentences, and step-by-step logic
        transcript_paragraphs = []
        curr_p = []
        curr_start = 0.0
        curr_words = 0

        for s in segments:
            txt = s.get("text", "").strip().replace("\n", " ")
            if not txt:
                continue
            if not curr_p:
                curr_start = s.get("start", 0.0)
            curr_p.append(txt)
            words = len(txt.split())
            curr_words += words

            # Group on natural sentence endings (~60-80 words), keeping thoughts complete
            ends_sentence = txt.endswith((".", "!", "?", "...", ":"))
            if (curr_words >= 60 and ends_sentence) or curr_words >= 90:
                transcript_paragraphs.append(" ".join(curr_p))
                curr_p = []
                curr_words = 0

        if curr_p:
            transcript_paragraphs.append(" ".join(curr_p))

        # Preserve 100% of the subtitles. Only apply high-water limit if transcript exceeds 350 paragraphs (~25,000 words)
        if len(transcript_paragraphs) > 350:
            step = len(transcript_paragraphs) / 350
            transcript_paragraphs = [transcript_paragraphs[int(i * step)] for i in range(350)]

        transcript_text = "\n\n".join(transcript_paragraphs)

        # Fetch live Google & YouTube search queries (LSI Keywords)
        kw_seed = video_title or (segments[0].get("text", "")[:50] if segments else "Tutoriel Guide")
        live_keywords = fetch_live_seo_keywords(kw_seed, lang_code=lang_code, max_keywords=10)
        kw_instruction_fr = f"Mots-clés réels recherchés sur Google à intégrer naturellement : {', '.join(live_keywords[:8]) if live_keywords else 'tutoriel pas à pas, configuration, astuces pratiques, erreurs fréquentes'}"
        kw_instruction_en = f"Live high-intent search queries from Google to weave naturally: {', '.join(live_keywords[:8]) if live_keywords else 'step-by-step tutorial, configuration, best practices, troubleshooting'}"

        # Style instruction
        resolved_style = self.STYLE_ALIASES.get(style, style)
        style_info = self.STYLES.get(resolved_style, self.STYLES["Step-by-Step Tutorial (How-To Guide)"])
        style_prompt = style_info.get(lang_code, style_info.get("en", list(style_info.values())[0]))

        # Word count guideline
        target_words = self.LENGTH_PRESETS.get(length, 1200)

        # Anti-AI Detection Rules & Guidelines in target language (Humanizer 35-Pattern Benchmark)
        if lang_code == "fr":
            system_prompt = (
                "Tu es un journaliste tech de référence et rédacteur web d'élite francophone (style Korben, Frandroid, Ars Technica).\n"
                "Ta mission est de rédiger un article de blog complet, percutant, riche en substance technique et optimisé SEO à partir des faits réels tirés de la transcription.\n\n"
                "POSTURE ÉDITORIALE INCARNÉE (RÈGLE ESSENTIELLE) :\n"
                "- Tu incarnes le testeur qui a installé, configuré et éprouvé l'outil sur son propre PC. Tu emploies une voix incarnée vivante : le 'Je' de retour d'expérience pratique ('Je l'ai installé de bout en bout sur mon PC...', 'Dans mes tests, je lui ai demandé...') et le 'Vous' pour guider et conseiller le lecteur.\n"
                "- Tu rédiges un article de blog natif pour des lecteurs du web. Tu ne résumes JAMAIS une vidéo et tu ne commentes JAMAIS ce qu'un tiers fait à l'écran !\n"
                "- INTERDICTION STRICTE DU STYLE ORAL ET DE LA PARAPHRASE MINUTE PAR MINUTE DES SOUS-TITRES :\n"
                "  * Ne décris JAMAIS les actions orales de la vidéo : BANNIS ABSOLUMENT 'Là je vous montre', 'Là il cherche en accéléré', 'Là j'ai cliqué sur Close', 'Là j'ai stoppé le prompt', 'Sur la droite j'ai mon graphique', 'En bas à droite je peux changer mon modèle', 'Moi je n'ai pas envie de le faire'.\n"
                "  * Tu rédiges une SYNTHÈSE THÉMATIQUE STRUCTURÉE, pas un commentaire audio de ce qui se passe sur un écran !\n"
                "- INTERDICTION FORMELLE d'employer : 'la vidéo précise', 'la vidéo montre', 'dans cette vidéo', 'dans la vidéo', 'au cours de la vidéo', 'l'auteur mentionne', 'l'auteur explique', 'l'auteur précise', 'le créateur', 'le présentateur', 'l'intervenant', 'comme vu dans la vidéo', 'comme montré à l'écran'.\n"
                "- INTERDICTION FORMELLE DES FORMULES D'APPEL À L'ACTION VIDÉO : Ne termine JAMAIS par 'N'hésitez pas à liker, partager, commenter et vous abonner à la chaîne', 'laissez un pouce bleu', 'activez la cloche', 'à la prochaine' ou 'liens en description'. Tu rédiges un article de blog écrit !\n\n"
                "RÈGLES FORMELLES STRICTES ANTI-DÉTECTION IA (HUMANIZER BENCHMARK) :\n"
                "1. INTERDICTION ABSOLUE DU POINT-VIRGULE (;) : N'utilise AUCUN point-virgule (;) dans toute la prose. Le point-virgule trahit immédiatement un texte généré par IA. Rédige des phrases courtes, directes et rythmées, séparées par des points simples (.).\n"
                "2. INTERDICTION DES TIRETS CADRATINS (— ou –) dans la prose : Utilise des virgules, des parenthèses ou des deux-points.\n"
                "3. ZÉRO EMOJI DANS LES TITRES DE SECTIONS : Les titres H1, H2, H3 doivent être sobres, professionnels et sans aucun emoji.\n"
                "4. TITRE SEO : Calibré strictement entre 50 et 65 caractères, sans guillemets, en casse de phrase (ex: 'Hermes Agent Windows : installation et prise en main pas à pas').\n"
                "5. AUCUN SUPERLATIF CREUX NI HYPE : Bannis 'révolutionnaire', 'couteau suisse', 'époustouflant', 'incontournable', 'game changer', 'tournant décisif'. Reste factuel, précis et mesuré.\n"
                "6. AUCUNE INTRO NI CONCLUSION BANALE : Pas de 'Dans cet article nous allons explorer' ni 'À l'ère du numérique'. Pas de 'En conclusion' ni 'Pour conclure'. Utilise un titre d'action comme 'Mon avis : pour qui et pour quoi faire ?' ou 'Ce qu'il faut retenir'.\n\n"
                "FIDÉLITÉ TECHNIQUE CHIRURGICALE AUX FAITS (NE RIEN INVENTER) :\n"
                "- Correction des erreurs phonétiques Whisper : Attention, les sous-titres Whisper comportent des transcriptions phonétiques approximatives. Exemple crucial : quand la transcription mentionne 'Morpheus', il s'agit en réalité d'un **jeu de morpion** (tic-tac-toe) interactif codé et lancé en direct !\n"
                "- Prérequis matériels réels : L'application elle-même est légère et tourne dès 4 Go de RAM avec une API distante. La machine montrée dans l'inspection système (128 Go de RAM, RTX 4090 + puce graphique intégrée) était simplement la configuration personnelle du testeur, pas les prérequis de l'outil !\n"
                "- Modes d'exécution du cerveau LLM : Clarifie les deux options réelles :\n"
                "  * Option 1 (API distante cloud : NVIDIA NIM, Claude, OpenAI...) : zéro charge sur le PC local, tourne sur n'importe quelle machine sans carte graphique dédiée, mais attention aux quotas d'API gratuites qui bloquent vite.\n"
                "  * Option 2 (Modèle en local avec Ollama) : 100% gratuit et privé, aucune donnée ne sort du PC, mais nécessite au moins 16 Go de RAM et un GPU 8 Go VRAM recommandé. Attention au piège du contexte : Hermes exige un contexte d'au moins 64 000 tokens (65 536 dans Ollama), sinon il refuse de démarrer.\n"
                "- Installation Windows isolée : L'installeur officiel s'installe dans %LOCALAPPDATA%\\hermes, installe ses propres versions isolées de Python, Node.js et Git sans droits administrateur et sans toucher aux variables d'environnement globales de Windows.\n"
                "- Cas d'usage réels et démonstrations concrètes : Développe en détail les vrais cas testés :\n"
                "  * Coder et lancer une application de zéro (ex: jeu de morpion interactif codé et exécuté en une commande).\n"
                "  * Automatiser des tâches récurrentes avec cron (sorties cinéma le mercredi à 6h du matin avec résumé et filtre de genres ; surveillance des prix de billets d'avion Paris-Tokyo toutes les 5h avec alerte sous les 600€).\n"
                "  * Manipuler et analyser les données locales (inspection de l'espace restant sur le disque C et détection des dossiers les plus lourds ; génération d'un graphique météo interactif HTML sauvegardé sur le disque).\n"
                "  * Pilotage à distance depuis son smartphone avec un bot Telegram (connecteur Telegram intégré).\n"
                "  * Dictée vocale avec le micro directement intégré dans l'interface.\n"
                "- La contrepartie / Sécurité : Comme l'agent a un accès direct au système de fichiers et au shell pour exécuter du code, il faut le manipuler avec discernement."
            )

            user_prompt = f"""Rédige un article de blog SEO complet et de référence sur le sujet suivant.

INFORMATIONS SUR LE SUJET DU GUIDE :
- Sujet principal : {video_title or 'Guide pratique et tutoriel pas à pas'}
- Thématique : Tutoriel technique d'installation et cas d'usage réels
- Style rédactionnel souhaité : {style} -> {style_prompt}
- Longueur cible : environ {target_words} mots
- Langue de rédaction : Français (naturel, soigné, sans fioritures, avec le ton incarné de l'expérimentateur terrain)
- {kw_instruction_fr}

EXEMPLAIRE COMPLET DU NIVEAU D'EXCELLENCE ATTENDU (MODÈLE ET STRUCTURE DE RÉFÉRENCE) :
Voici exactement le plan thématique, la densité technique, le ton incarné et le style attendus :
'''
# Tutoriel Hermès Agent : installation et configuration sur Windows

On vous propose aujourd'hui d'installer Hermes Agent sur Windows. C'est un agent IA open source de Nous Research. Il est gratuit et il tourne sur un ordinateur normal ou encore sur un VPS. La différence avec un ChatGPT classique, c'est qu'il ne se contente pas de répondre : il agit, il garde en mémoire ce que vous lui avez dit et il peut travailler pendant que vous dormez. Cela correspond au mode agent des LLM que l'on peut activer mais en beaucoup plus précis.

Je l'ai installé de bout en bout sur mon PC, avec une API distante d'abord, puis avec Ollama et un LLM en local.

## Hermes Agent, qu'est-ce que c'est ?

Hermes Agent est un agent autonome développé par Nous Research, publié sous licence MIT. Il existe en version terminal et en application de bureau, Hermes Desktop, disponible pour Windows 10 et 11, macOS et Linux.

Ce qui le distingue d'un simple LLM sans agent :
- Il peut accéder à votre système de fichiers, il lit, il écrit, il exécute.
- Il peut naviguer sur le web tout seul pour aller chercher une donnée.
- Il peut mémoriser le contexte d'une session à l'autre.
- Il installe et crée des compétences, appelées skills (comme sur ChatGPT ou Claude).
- Il tourne en permanence tant que le serveur est lancé (donc c'est vous qui lancez et fermez votre serveur).
- Il se pilote en ligne de commande mais aussi avec Telegram, Discord, Slack ou par mail.

## Télécharger et installer Hermes Agent sur Windows

Il faut aller sur le site officiel hermes-agent.nousresearch.com. On appuie sur le bouton Download for Windows.

Vous lancez l'exécutable, vous cliquez sur Install et c'est terminé. Il n'y a aucune ligne de commande à taper et aucune dépendance à installer à la main. L'installeur récupère lui-même Git, Node.js et Python, clone le dépôt dans %LOCALAPPDATA%\\hermes et crée l'environnement virtuel. Il n'y a pas besoin de droits administrateur.

Une fois l'installation finie, vous cliquez sur Launch.

## Connecter un modèle à Hermes Agent

Au premier lancement, Hermes propose son propre portail, le Nous Portal avec ses modèles maison. C'est pratique mais payant avec un abonnement. Si vous avez déjà une clé API ailleurs (ChatGPT, Claude, Gemini), ce n'est pas nécessaire.

Vous cliquez sur I have an API key. Vous avez alors le choix entre OpenAI, Anthropic pour Claude, Qwen et d'autres fournisseurs. De mon côté j'ai utilisé la clé API NVIDIA, qui donne accès gratuitement à un certain nombre de modèles.

Vous collez la clé, vous faites Connect, et vous choisissez votre modèle dans la liste.

Les modèles gratuits fonctionnent mal avec l'API gratuite de NVIDIA. J'ai testé Kimi via l'offre gratuite et l'agent se prend des erreurs de requête. Ce n'est pas un problème d'Hermes, c'est le quota et la limitation des offres gratuites qui coincent. Pour un usage sérieux, il faut soit un modèle payant, soit un modèle en local.

Hermes exige un modèle avec au moins 64 000 tokens de contexte. En dessous, il refuse de démarrer, parce qu'un agent qui enchaîne les appels d'outils a besoin de mémoire de travail. C'est un point important à retenir quand on utilisera Ollama pour faire tourner notre LLM en local.

## Régler l'interface et découvrir les skills

La roue crantée en haut à droite ouvre les préférences. Première chose que j'ai faite : passer en thème sombre, parce que l'interface claire pique un peu les yeux. Vous pouvez aussi changer la disposition des panneaux.

L'onglet Capabilities liste les skills. Un skill, c'est une compétence déjà écrite avec sa procédure étape par étape. L'agent lit les descriptions courtes en permanence et il ne charge le contenu complet que quand la tâche le demande, ce qui évite d'alourdir chaque requête. Certains sont actifs par défaut, les autres s'activent d'un clic.

Dans le catalogue, on trouve par exemple un skill Home Assistant pour la maison connectée. On pourrait demander à Hermes d'allumer une lampe ou de programmer un allumage.

## Premiers tests concrets

Le prompt se trouve en bas de l'écran. On tape en langage naturel sans syntaxe particulière.

- Créer un jeu : Je lui ai demandé un jeu de morpion. Il l'a codé et il a exécuté directement. C'était jouable.
- Chercher un billet d'avion : Je lui ai demandé les prix Paris-Tokyo pour septembre 2026. Il ouvre un navigateur, il va chercher les données, il se plante, il corrige et il relance. Il explique à chaque étape ce qu'il fait. À un moment il a eu un souci avec Chrome et il a contourné tout seul. Au bout du compte il permet de trouver des opportunités de vols les moins chers avec et sans escale et il pose des questions pour affiner ses recherches.
- Faire un graphique : J'ai demandé un graphique des températures à Paris pour août 2026. Il est allé chercher les données via une API météo (sans que je lui indique) et il a généré un fichier HTML. Il l'a enregistré sur mon disque sans me demander l'autorisation. Le rendu est propre, exportable en PNG ou en PDF.
- Interroger la machine : « Combien j'ai de place sur mon disque C ? » Il répond avec la taille totale, l'espace utilisé, l'espace libre et le taux d'occupation. Il s'est trompé de méthode, il s'en est aperçu et il a changé d'approche sans que j'intervienne. Au niveau hardware, il a détecté mes 128 Go de RAM et mes deux GPU, la RTX 4090 plus le GPU intégré à la carte mère.

Il écrit sur le disque et il peut aussi effacer. Je ne lui ai jamais demandé de trier mes photos et je ne le ferai pas. Le niveau de permission se règle dans les paramètres, prenez cinq minutes pour le faire avant de le laisser tourner.

## Créer des jobs automatiques

C'est la fonction la plus intéressante. Hermes intègre un planificateur qui se configure en langage naturel et on n'a pas besoin de faire soi-même un crontab.

Je lui dis : tous les matins à 6h, tu me fais un résumé des derniers articles parus sur mon site paradoxetemporel.fr. Il crée le job, il le confirme et il le déclenche chaque jour tant que le serveur tourne.

J'en ai fait un autre pour la veille tarifaire : toutes les cinq heures, tu vérifies les prix Paris-Tokyo et tu me préviens si tu trouves un aller-retour sous 600 euros.

La condition, c'est que la machine reste allumée et avoir accès à un LLM. Un job planifié sur un PC éteint ne se déclenche pas. Il faut faire tourner Hermes sur une machine qui ne s'arrête jamais comme un vieux PC, un mini-PC ou un VPS que vous louez. Il faut 4 Go de RAM au minimum.

## Recevoir les résultats sur Telegram

Un résumé qui s'affiche dans une console que personne ne regarde ne sert à rien. Hermes propose une passerelle de messagerie qui couvre Telegram, Discord, Slack, WhatsApp, Signal et le mail.

Je l'ai branché à Telegram. Tous les mercredis je reçois sur mon téléphone les sorties ciné de la semaine avec le titre, le genre et un résumé. On peut faire la même chose pour les relevés de température ou les alertes de prix.

Cela fonctionne dans les deux sens. Depuis Telegram, j'envoie la phrase Dis-moi combien j'ai de place restante sur mon disque. Le message part du smartphone, il arrive sur le PC où Hermes tourne, la commande s'exécute, et j'ai la réponse sur Telegram.

## Passer Hermes Agent en local avec Ollama

Jusque-là, j'avais fait des tests par une API distante. Vos données sortent de votre machine et sont donc lisibles par une société. Si cela vous gêne et cela devrait, l'alternative c'est le modèle local mais qui nécessite une machine puissante.

Dans Provider puis Account, vous choisissez le fournisseur auto-hébergé, pas Ollama Cloud mais Ollama local. Vous collez l'URL de l'endpoint compatible OpenAI :
http://localhost:11434/v1

La clé API se laisse vide. Vous faites Connect et Hermes récupère la liste des modèles que vous avez téléchargés en local sur Ollama. Pour installer Ollama, il suffit d'aller sur le site officiel.

À partir de là, il n'y a plus rien qui ne sort de votre PC. Le changement de modèle se fait ensuite en bas à droite, sans rien reconfigurer.

Il faut un GPU récent comme une RTX 4090 ou RTX 5090 pour faire tourner un modèle capable d'appeler des outils rapidement. Il faut aussi changer le contexte à 65536 tokens dans Ollama, sinon Hermes refuse le modèle.

## Le mode vocal

Hermes intègre un mode vocal. Vous activez le micro, vous parlez, il transcrit et il exécute. J'ai testé avec la phrase Donne-moi la température à Paris aujourd'hui. La transcription est passée sans problème. On peut même dicter des ordres pour qu'il programme.

## Mon avis sur Hermes Agent

Je trouve que Hermes Agent tient ses promesses sur trois points : l'installation sous Windows se fait rapidement, les jobs planifiés fonctionnent et la passerelle Telegram transforme l'outil en assistant réellement utilisable au quotidien.

Il faut payer une API ou avoir un bon GPU. La machine où l'on a installé Hermes Agent doit rester allumée en permanence. C'est pareil pour la machine où vous avez installé le LLM local. L'agent a la main sur votre système ce qui impose de régler les permissions avant de l'utiliser.

## La vidéo d'installation et l'utilisation de Hermes Agent
'''

DONNÉES TECHNIQUES ET FAITS EXTRAITS DE L'EXPÉRIMENTATION :
[RÈGLE CRUCIALE : Ces données constituent ta base factuelle. Ne raconte PAS la vidéo et ne commente pas l'écran ! Rédige un guide de synthèse thématique en suivant rigoureusement la structure ci-dessous.]
{transcript_text}

STRUCTURE DU RÉSULTAT DEMANDÉ (RESPECTE SCRUPULEUSEMENT CES DÉLIMITEURS) :

---SEO_METADATA---
TITLE: [Titre H1 optimisé pour le SEO, 50 à 65 caractères, sans emoji, en casse de phrase]
SLUG: [slug-url-optimise-sans-accents-separes-par-des-tirets]
META_DESCRIPTION: [Meta description percutante de 145 à 160 caractères avec mot-clé principal, sans point-virgule]
FOCUS_KEYWORD: [Mot-clé principal extrait de l'intention de recherche]
SECONDARY_KEYWORDS: [3 à 5 mots-clés secondaires LSI séparés par des virgules]
---END_SEO_METADATA---

---ARTICLE_CONTENT---
# [Titre H1 de l'article en casse de phrase, 50-65 caractères, sans emoji]

[Introduction directe et incarnée : annonce claire du sujet et retour d'expérience vécu ("Je l'ai installé de bout en bout sur mon PC..."). 2 paragraphes denses et rythmés.]

## Hermes Agent, qu'est-ce que c'est ?
[Présentation de l'agent développé par Nous Research, licence MIT, versions terminal et desktop pour Windows 10/11, macOS, Linux.]

## Ce qui le distingue d'un simple LLM sans agent
[Les points clés sous forme de liste à tirets : accès système de fichiers, navigation web autonome, persistance du contexte, compétences skills, serveur permanent, pilotage en ligne de commande et messageries.]

## Télécharger et installer Hermes Agent sur Windows
[Étapes claires : téléchargement sur hermes-agent.nousresearch.com, installation silencieuse dans %LOCALAPPDATA%\\hermes sans droits admin (Git, Node.js, Python isolés et venv automatique), bouton Launch.]

## Connecter un modèle à Hermes Agent
[Nous Portal payant vs option I have an API key (OpenAI, Anthropic Claude, Qwen, NVIDIA gratuite), retour d'expérience sur les quotas de requêtes qui bloquent les offres gratuites comme Kimi, et impératif des 64 000 tokens de contexte.]

## Régler l'interface et découvrir les skills
[Roue crantée des préférences : passage au thème sombre, disposition des panneaux. Catalogue Capabilities : descriptions courtes en mémoire et chargement complet à la demande, exemple du skill Home Assistant.]

## Premiers tests concrets
[Sous-points denses et précis : Créer un jeu (morpion codé et exécuté), Chercher un billet d'avion (Paris-Tokyo septembre 2026, contournement d'erreur Chrome), Faire un graphique (températures Paris août 2026 en HTML sauvegardé sur disque, export PNG/PDF), Interroger la machine (espace disque C, 128 Go RAM et deux GPU : RTX 4090 + puce intégrée carte mère), avertissement et réglage des permissions.]

## Créer des jobs automatiques
[Planificateur en langage naturel sans crontab : résumé quotidien à 6h des articles du site paradoxetemporel.fr, veille vols Paris-Tokyo < 600€ toutes les 5h, prérequis machine allumée (PC, mini-PC, VPS avec 4 Go RAM min).]

## Recevoir les résultats sur Telegram
[Passerelle multi-messagerie (Telegram, Discord, Slack, etc.) : sorties ciné du mercredi avec titre, genre, résumé sur smartphone, et communication bidirectionnelle (commande envoyée depuis le smartphone avec réponse directe sur Telegram).]

## Passer Hermes Agent en local avec Ollama
[Confidentialité totale des données : Provider > Account > auto-hébergé Ollama local sur http://localhost:11434/v1, clé vide, GPU récent, réglage impératif du contexte à 65536 tokens dans Ollama.]

## Le mode vocal
[Micro activé, dictée vocale sans syntaxe particulière ("Donne-moi la température à Paris aujourd'hui"), transcription et exécution d'ordres pour programmer.]

## Mon avis sur Hermes Agent
[Bilan franc sur trois points forts (installation rapide, jobs planifiés, passerelle Telegram) et contraintes (machine allumée en continu, bon GPU ou API payante, permissions à paramétrer).]

## La vidéo d'installation et l'utilisation de Hermes Agent
[Courte présentation invitant à visionner la vidéo complète ci-dessous pour voir toutes les étapes en direct.]
---END_ARTICLE_CONTENT---

---IMAGE_PROMPTS---
IMAGE_1: [Prompt en anglais pour l'image à la une / Featured Image, style moderne photoréaliste ou illustration tech]
IMAGE_2: [Prompt en anglais pour illustrer la section configuration technique]
IMAGE_3: [Prompt en anglais pour illustrer les fonctionnalités d'automatisation avancée]
---END_IMAGE_PROMPTS---
"""
        else: # English & generic (Humanizer 35-Pattern Benchmark)
            system_prompt = (
                f"You are a seasoned tech journalist and senior technical writer writing in {lang_name} (style of Ars Technica, The Verge).\n"
                "Your objective is to craft an authoritative, highly engaging, and hands-on SEO blog post based on real demonstration facts.\n\n"
                "FIRST-PERSON INCARNATED TESTER VOICE (MANDATORY RULE):\n"
                "- You are the hands-on practitioner who tested and deployed this tool on your own PC. Use an authentic first-person voice ('I tested it on my Windows workstation...', 'In my tests, I asked it to...') combined with 'You' to guide the reader step-by-step.\n"
                "- Write a native written web tutorial. You NEVER comment on a video or third-party presenter!\n"
                "- STRICTLY FORBIDDEN: Oral screen-commentary phrases like 'Now I show you', 'Here it is searching in fast-forward', 'I clicked on Close', 'I stopped the prompt'. Write a clean, thematic written guide!\n"
                "- STRICTLY FORBIDDEN phrases: 'the video explains', 'the video shows', 'in this video', 'the author mentions', 'the creator explains', 'the presenter', 'as seen in the video', 'as shown on screen'.\n"
                "- STRICTLY FORBIDDEN: Calls to like, subscribe, share, ring the bell, leave a comment, or check the description. You are writing an authoritative blog post, NOT a YouTube script.\n\n"
                "STRICT HUMANIZER ANTI-AI RULES:\n"
                "1. ABSOLUTE BAN ON SEMICOLONS (;): Do NOT use semicolons in prose. Write crisp, punchy sentences separated by simple periods (.).\n"
                "2. NO EM DASHES (— or –) in prose: Use commas, colons, or parentheses.\n"
                "3. NO HEADING EMOJIS: Headings H1, H2, H3 must be clean and professional, with zero emojis.\n"
                "4. SEO TITLE CALIBRATION: Strictly 50 to 65 characters in sentence case.\n"
                "5. ZERO HYPE OR BUZZWORDS: Avoid 'groundbreaking', 'game changer', 'pivotal moment', 'testament to'. Keep it measured and factual.\n"
                "6. ZERO GENERIC INTROS OR OUTROS: No 'In this article we will explore'. No 'In conclusion'. Use an action-driven heading like 'My Verdict: Who Is This For?'."
            )

            user_prompt = f"""Write an exceptional, comprehensive SEO blog post based on the following technical context.

TOPIC INFORMATION:
- Primary Topic: {video_title or 'Hands-on Technical Guide & Tutorial'}
- Scope: In-depth technical walkthrough and real-world test cases
- Writing Style: {style} -> {style_prompt}
- Target Length: ~{target_words} words
- Target Language: {lang_name}
- {kw_instruction_en}

TECHNICAL CONTEXT & RECORDED FACTS:
[Crucial note: You are the hands-on practitioner writing your own tutorial. Use these technical facts, but NEVER narrate a video or refer to 'the video', 'in this video', or 'the author'. Synthesize into the 5 thematic sections below.]
{transcript_text}

OUTPUT STRUCTURE REQUIRED:
You must strictly provide the response using these delimiters:

---SEO_METADATA---
TITLE: [High-CTR SEO Title H1, 50-65 characters, sentence case, no emoji]
SLUG: [optimized-hyphenated-url-slug]
META_DESCRIPTION: [Compelling Meta Description between 145 and 160 characters with focus keyword and clear CTA, no semicolons]
FOCUS_KEYWORD: [Primary keyword]
SECONDARY_KEYWORDS: [3 to 5 secondary keywords separated by commas]
---END_SEO_METADATA---

---ARTICLE_CONTENT---
# [Title H1, 50-65 chars, no emoji]

[Engaging first-person hook delivering instant value without generic preamble. 2 dense paragraphs.]

## What It Is and How It Outperforms Web Chatbots
[In-depth breakdown of persistent memory, file system access, modular skills, and background cron execution.]

## System Requirements: What You Actually Need
[Clear realistic prerequisites (4 GB RAM minimum for remote APIs vs local Ollama requiring 16 GB RAM and 64,000 token context window).]

## Step-by-Step Installation on Windows
[Step by step guidance: installer in %LOCALAPPDATA%\\hermes, zero admin rights needed, API and Ollama configuration.]

## Real-World Automations: Games, Cron Jobs & Telegram
[In-depth real test cases: interactive tic-tac-toe game, flight alerts under 600€, Wednesday movie summaries, drive C inspection, and Telegram bot mobile control.]

## My Verdict: Who Is This For?
[Impactful takeaway and hardware recommendations (mini-PC vs always-on desktop) without writing 'In Conclusion' and without any YouTube outro formulas.]
---END_ARTICLE_CONTENT---

---IMAGE_PROMPTS---
IMAGE_1: [AI Image prompt for Featured Header image in English, photorealistic or modern tech vector]
IMAGE_2: [AI Image prompt for Section 2 in English]
IMAGE_3: [AI Image prompt for Section 4 in English]
---END_IMAGE_PROMPTS---
"""

        # Generate via LLM backend if available
        raw_response = ""
        if llm_backend is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                # Use large max_new_tokens for long-form article generation
                raw_response = llm_backend.generate(
                    messages,
                    max_new_tokens=3000,
                    temperature=0.35,
                    do_sample=True,
                    repetition_penalty=1.05,
                    multiline=True
                )
            except Exception as e:
                print(f"[SEOArticleGenerator] LLM Generation Error: {e}")
                raw_response = ""

        # If LLM generation failed or was empty, provide a clean structured fallback
        if not raw_response or len(raw_response.strip()) < 100:
            raw_response = self._build_deterministic_fallback(
                video_title, transcript_paragraphs, lang_code, style
            )

        raw_response = self.clean_ai_artifacts(raw_response, is_fr=(lang_code == "fr"))

        # Parse sections
        parsed = self._parse_generated_output(raw_response, lang_code, video_title, live_keywords=live_keywords)
        
        # Convert Markdown article into Gutenberg WordPress Blocks
        gutenberg_html = self.markdown_to_gutenberg(parsed["markdown_article"])
        parsed["gutenberg_html"] = gutenberg_html

        return parsed

    def _parse_generated_output(
        self,
        raw_text: str,
        lang_code: str,
        fallback_title: str,
        live_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract metadata, article content, and image prompts from delimited text."""
        # Defaults
        title = fallback_title or ("Guide Complet et Tutoriel Vidéo" if lang_code == "fr" else "Complete Video Guide & Tutorial")
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')[:50]
        meta_desc = (
            f"Découvrez notre guide complet et tutoriel pas à pas : étapes clés, conseils d'experts et démonstration détaillée pour réussir facilement."
            if lang_code == "fr"
            else f"Discover our comprehensive step-by-step guide: key highlights, expert tips, and in-depth walkthrough to get started quickly."
        )
        focus_kw = "Tutoriel Guide" if lang_code == "fr" else "Complete Guide"
        sec_kws = (live_keywords[:5] if live_keywords else (["tutoriel", "guide complet", "astuces", "démonstration"] if lang_code == "fr" else ["tutorial", "guide", "tips", "walkthrough"]))
        markdown_article = ""
        image_prompts = []

        # Extract SEO METADATA block
        meta_match = re.search(r'---SEO_METADATA---(.*?)---END_SEO_METADATA---', raw_text, re.DOTALL | re.IGNORECASE)
        if meta_match:
            meta_chunk = meta_match.group(1).strip()
            t_m = re.search(r'^TITLE:\s*(.*)$', meta_chunk, re.MULTILINE | re.IGNORECASE)
            if t_m and t_m.group(1).strip():
                title = t_m.group(1).strip().strip('"\'')

            s_m = re.search(r'^SLUG:\s*(.*)$', meta_chunk, re.MULTILINE | re.IGNORECASE)
            if s_m and s_m.group(1).strip():
                slug = s_m.group(1).strip().strip('"\'').lower()

            d_m = re.search(r'^META_DESCRIPTION:\s*(.*)$', meta_chunk, re.MULTILINE | re.IGNORECASE)
            if d_m and d_m.group(1).strip():
                meta_desc = d_m.group(1).strip().strip('"\'')

            f_m = re.search(r'^FOCUS_KEYWORD:\s*(.*)$', meta_chunk, re.MULTILINE | re.IGNORECASE)
            if f_m and f_m.group(1).strip():
                focus_kw = f_m.group(1).strip().strip('"\'')

            sk_m = re.search(r'^SECONDARY_KEYWORDS:\s*(.*)$', meta_chunk, re.MULTILINE | re.IGNORECASE)
            if sk_m and sk_m.group(1).strip():
                extracted_sec = [k.strip() for k in sk_m.group(1).split(",") if k.strip()]
                if extracted_sec:
                    sec_kws = extracted_sec

        # Extract ARTICLE CONTENT block
        art_match = re.search(r'---ARTICLE_CONTENT---(.*?)(?:---END_ARTICLE_CONTENT---|---IMAGE_PROMPTS---|\Z)', raw_text, re.DOTALL | re.IGNORECASE)
        if art_match:
            markdown_article = art_match.group(1).strip()
        else:
            # If delimiters were omitted by LLM, strip metadata and prompts
            clean = re.sub(r'---SEO_METADATA---.*?---END_SEO_METADATA---', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'---IMAGE_PROMPTS---.*', '', clean, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'---(?:ARTICLE_CONTENT|END_ARTICLE_CONTENT)---', '', clean, flags=re.IGNORECASE)
            markdown_article = clean.strip()

        # Guarantee no rogue boundary markers remain in the article text
        markdown_article = re.sub(r'^\s*---(?:ARTICLE_CONTENT|END_ARTICLE_CONTENT|SEO_METADATA|END_SEO_METADATA|IMAGE_PROMPTS|END_IMAGE_PROMPTS)---\s*', '', markdown_article, flags=re.MULTILINE).strip()

        # Clean all textual fields against AI artifacts, semicolons, and heading emojis
        is_fr = (lang_code == "fr")
        title = format_seo_title(self._clean_semicolons(self.clean_ai_artifacts(title, is_fr=is_fr)), is_fr=is_fr)
        meta_desc = self._clean_semicolons(self.clean_ai_artifacts(meta_desc, is_fr=is_fr))
        focus_kw = self._clean_semicolons(self.clean_ai_artifacts(focus_kw, is_fr=is_fr))
        sec_kws = [self._clean_semicolons(self.clean_ai_artifacts(k, is_fr=is_fr)) for k in sec_kws]
        markdown_article = self.clean_ai_artifacts(markdown_article, is_fr=is_fr)

        # Extract IMAGE PROMPTS block
        img_match = re.search(r'---IMAGE_PROMPTS---(.*?)---END_IMAGE_PROMPTS---', raw_text, re.DOTALL | re.IGNORECASE)
        if img_match:
            img_chunk = img_match.group(1).strip()
            lines = img_chunk.split("\n")
            for line in lines:
                p_m = re.search(r'^IMAGE_\d+:\s*(.*)$', line.strip(), re.IGNORECASE)
                if p_m and p_m.group(1).strip():
                    image_prompts.append(p_m.group(1).strip().strip('"\''))

        if not image_prompts:
            image_prompts = [
                f"Modern 3D digital illustration for blog header about {title}, vibrant lighting, clean composition, 8k resolution, photorealistic",
                f"Close-up technical workstation mockup showing step-by-step workflow for {title}, soft studio lighting, ultra-detailed",
                f"Futuristic infographic layout illustrating key benefits and performance architecture of {title}, sleek UI elements"
            ]

        # Ensure Meta Description length is well-calibrated (145-165 chars)
        if len(meta_desc) > 165:
            meta_desc = meta_desc[:162].rsplit(' ', 1)[0] + '...'

        return {
            "title": title,
            "slug": slug,
            "meta_description": meta_desc,
            "meta_desc_length": len(meta_desc),
            "focus_keyword": focus_kw,
            "secondary_keywords": sec_kws,
            "markdown_article": markdown_article,
            "image_prompts": image_prompts,
            "word_count": len(markdown_article.split())
        }

    def _build_deterministic_fallback(
        self,
        video_title: str,
        transcript_snippets: List[str],
        lang_code: str,
        style: str
    ) -> str:
        """Create a high-quality structured article if LLM backend is offline."""
        raw_title = video_title or ("Guide pratique et tutoriel complet" if lang_code == "fr" else "Complete Practical Guide & Tutorial")
        title = format_seo_title(raw_title, is_fr=(lang_code == "fr"))
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')[:50]
        
        sample_body = "\n\n".join(transcript_snippets[:12]) if transcript_snippets else ""
        
        if lang_code == "fr":
            clean_t = title
            if ":" not in clean_t:
                clean_t = f"{clean_t} : guide complet et installation"
            return f"""---SEO_METADATA---
TITLE: {clean_t}
SLUG: {slug}
META_DESCRIPTION: Découvrez notre tutoriel pas à pas sur {title}. Toutes les étapes, conseils d'experts et astuces pour maîtriser le sujet facilement.
FOCUS_KEYWORD: {title}
SECONDARY_KEYWORDS: tutoriel {title}, guide pas à pas, astuces, installation et configuration
---END_SEO_METADATA---

---ARTICLE_CONTENT---
# {clean_t}

Ce tutoriel détaille l'ensemble des étapes et des bonnes pratiques pour prendre en main ce sujet rapidement et éviter les écueils courants.

## Contexte et prérequis réels

Avant de débuter, assurez-vous de disposer des éléments nécessaires pour suivre la mise en place dans de bonnes conditions :

- Un environnement configuré selon les recommandations présentées.
- Les outils et bibliothèques requis installés.
- Les accès et autorisations adéquats.
- 4 Go de RAM recommandés au minimum pour démarrer le serveur sans carte graphique dédiée obligatoire.
- La configuration étape par étape avec les meilleures pratiques.
- Les erreurs courantes à éviter et les astuces pour aller plus vite.

## Étapes de mise en place pas à pas

1. **Préparation de l'environnement** : Assurez-vous d'avoir tous les accès et outils nécessaires.
2. **Exécution des premières commandes** : Suivez précisément l'ordre des étapes pour garantir la stabilité.
3. **Validation et vérification** : Testez le bon fonctionnement à l'aide des indicateurs présentés.

> 💡 **Conseil pratique :** Prenez le temps de documenter vos réglages dès le départ pour faciliter vos futures mises à jour.

## Ce qu'il faut retenir

En appliquant ces étapes méthodiquement, vous disposez d'une base solide et pérenne. N'hésitez pas à adapter ces principes à vos besoins spécifiques pour maximiser vos résultats.
---END_ARTICLE_CONTENT---

---IMAGE_PROMPTS---
IMAGE_1: Sleek high-tech banner illustration representing {title}, dark mode glassmorphism, 4k resolution
IMAGE_2: Minimalist step-by-step tutorial diagram illustration for {title}, clean vector aesthetics
IMAGE_3: Modern workspace screenshot mockup demonstrating productivity benefits of {title}
---END_IMAGE_PROMPTS---
"""
        else:
            clean_t = title
            if ":" not in clean_t and " - " not in clean_t:
                clean_t = f"{clean_t} - Complete Installation & Setup Guide"
            return f"""---SEO_METADATA---
TITLE: {clean_t}
SLUG: {slug}
META_DESCRIPTION: Master {title} with our complete step-by-step walkthrough. Discover key workflows, pro tips, and best practices to get started effortlessly.
FOCUS_KEYWORD: {title}
SECONDARY_KEYWORDS: {title} tutorial, step by step guide, workflow tips, best practices
---END_SEO_METADATA---

---ARTICLE_CONTENT---
# {clean_t}

Getting started with a new workflow or technical system can seem daunting. With the right structured approach, you can unlock its full potential quickly.

## Key Objectives & Overview

Whether you are just getting started or looking to optimize existing processes, this guide covers:
- Core prerequisites and environment setup (4 GB RAM minimum recommended).
- Step-by-step walkthrough of each essential milestone.
- Common pitfalls to avoid and performance tips.

## Step-by-Step Implementation

1. **Environment Setup**: Ensure all required dependencies are ready.
2. **Configuration & Execution**: Follow the recommended sequence closely.
3. **Verification & Testing**: Validate your setup with the checklist provided.

> 💡 **Pro Tip:** Keep your configuration parameters saved in a centralized note to streamline future updates.

## Key Takeaways & Next Steps

By following these structured guidelines, you now have a reliable foundation. Adapt these best practices to your specific workflows to achieve maximum efficiency.
---END_ARTICLE_CONTENT---

---IMAGE_PROMPTS---
IMAGE_1: Modern digital concept artwork representing {title}, futuristic neon accents, photorealistic
IMAGE_2: Clean infographic style workflow layout explaining {title} step by step
IMAGE_3: High performance dashboard mockup showing real-world results of {title}
---END_IMAGE_PROMPTS---
"""

    def markdown_to_gutenberg(self, markdown_text: str) -> str:
        """
        Convert Markdown content into standard WordPress Gutenberg Block HTML.
        Blocks generated:
          - <!-- wp:heading {"level":X} --><hX>...</hX><!-- /wp:heading -->
          - <!-- wp:paragraph --><p>...</p><!-- /wp:paragraph -->
          - <!-- wp:list --><ul><li>...</li></ul><!-- /wp:list -->
          - <!-- wp:quote --><blockquote class="wp-block-quote"><p>...</p></blockquote><!-- /wp:quote -->
          - <!-- wp:code --><pre class="wp-block-code"><code>...</code></pre><!-- /wp:code -->
        """
        if not markdown_text:
            return ""

        lines = markdown_text.strip().split("\n")
        blocks = []
        in_list = False
        list_items = []
        in_code = False
        code_lines = []

        def flush_list():
            nonlocal in_list, list_items
            if in_list and list_items:
                items_html = "".join([f"<li>{item}</li>" for item in list_items])
                blocks.append(f"<!-- wp:list -->\n<ul class=\"wp-block-list\">{items_html}</ul>\n<!-- /wp:list -->")
                list_items = []
                in_list = False

        def flush_code():
            nonlocal in_code, code_lines
            if in_code and code_lines:
                code_content = "\n".join(code_lines)
                # Escape html chars in code block
                code_content = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                blocks.append(f"<!-- wp:code -->\n<pre class=\"wp-block-code\"><code>{code_content}</code></pre>\n<!-- /wp:code -->")
                code_lines = []
                in_code = False

        for line in lines:
            line_str = line.strip()

            # Handle code fence
            if line_str.startswith("```"):
                if in_code:
                    flush_code()
                else:
                    flush_list()
                    in_code = True
                    code_lines = []
                continue

            if in_code:
                code_lines.append(line)
                continue

            # Handle Headings
            heading_match = re.match(r'^(#{1,6})\s+(.*)$', line_str)
            if heading_match:
                flush_list()
                level = len(heading_match.group(1))
                h_text = heading_match.group(2).strip()
                # Inline markdown formatting
                h_text = self._inline_markdown_to_html(h_text)
                blocks.append(f"<!-- wp:heading {{\"level\":{level}}} -->\n<h{level} class=\"wp-block-heading\">{h_text}</h{level}>\n<!-- /wp:heading -->")
                continue

            # Handle Bullet lists
            list_match = re.match(r'^[\*\-\+]\s+(.*)$', line_str) or re.match(r'^\d+\.\s+(.*)$', line_str)
            if list_match:
                in_list = True
                item_text = self._inline_markdown_to_html(list_match.group(1).strip())
                list_items.append(item_text)
                continue
            else:
                if in_list:
                    flush_list()

            # Handle Blockquotes / Callouts
            if line_str.startswith(">"):
                quote_text = line_str.lstrip(">").strip()
                quote_text = self._inline_markdown_to_html(quote_text)
                blocks.append(f"<!-- wp:quote -->\n<blockquote class=\"wp-block-quote\"><p>{quote_text}</p></blockquote>\n<!-- /wp:quote -->")
                continue

            # Handle normal paragraphs (if not empty)
            if line_str:
                p_text = self._inline_markdown_to_html(line_str)
                blocks.append(f"<!-- wp:paragraph -->\n<p>{p_text}</p>\n<!-- /wp:paragraph -->")

        # Flush any remaining items
        flush_list()
        flush_code()

        return "\n\n".join(blocks)

    def _inline_markdown_to_html(self, text: str) -> str:
        """Convert inline bold, italic, code, and links to clean HTML."""
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        # Inline code
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        # Links
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        return text

    def _get_video_duration(self, video_path: str) -> float:
        """Probe actual video duration using OpenCV or FFprobe."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if fps > 0 and frames > 0:
                    return float(frames / fps)
        except Exception:
            pass

        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass
        return 0.0

    def _get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """Probe actual video resolution (width, height) using OpenCV or FFprobe."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass

        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0", video_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and "x" in res.stdout:
                parts = res.stdout.strip().split("x")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 1920, 1080

    def _find_best_video_source(self, video_path: str, duration: float, topic_title: str = "") -> str:
        """
        Check if a higher-resolution version or valid video source exists in temp/ or adjacent folders.
        Handles:
        - Low-res video upgraded to 1080p source.
        - Audio-only inputs (.wav, .mp3, etc.) automatically mapped to the best matching video candidate in temp/.
        - Excludes intermediate render artifacts (test_*, preview_*, short_*, seg_*).
        """
        try:
            is_audio = os.path.splitext(video_path)[1].lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            w, h = (0, 0) if is_audio else self._get_video_resolution(video_path)
            
            # If already 1080p or higher (>= 1920x1080) and not an audio file, it's already top tier
            if not is_audio and w >= 1920 and h >= 1080:
                return video_path

            search_dirs = [os.path.dirname(video_path), "temp", "output"]
            best_path = video_path
            max_pixels = w * h

            ignore_prefixes = ('test_', 'preview_', 'short_', 'seg_')
            title_keywords = [k for k in re.split(r'[^a-zA-Z0-9]+', topic_title.lower()) if len(k) > 3]

            candidates = []
            for sdir in set(search_dirs):
                if not sdir or not os.path.exists(sdir):
                    continue
                for f in os.listdir(sdir):
                    f_lower = f.lower()
                    if not f_lower.endswith(('.mp4', '.mkv', '.mov', '.webm')):
                        continue
                    if any(f_lower.startswith(p) for p in ignore_prefixes):
                        continue
                    cand = os.path.join(sdir, f)
                    if os.path.abspath(cand) == os.path.abspath(video_path) or not os.path.isfile(cand):
                        continue
                    
                    cw, ch = self._get_video_resolution(cand)
                    cand_dur = self._get_video_duration(cand)
                    
                    score = 0
                    if duration > 0 and abs(cand_dur - duration) <= 4.0:
                        score += 100
                    elif duration > 0 and abs(cand_dur - duration) <= 30.0:
                        score += 50
                    
                    for kw in title_keywords:
                        if kw in f_lower:
                            score += 15
                    if 'zasttranslate' in f_lower:
                        score += 25
                    if 'doublage' in f_lower:
                        score += 15
                    if 'tuto' in f_lower:
                        score += 10
                    
                    pixels = cw * ch
                    size_mb = os.path.getsize(cand) / (1024 * 1024)
                    candidates.append((score, pixels, size_mb, cand, cw, ch, cand_dur))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                top_cand = candidates[0]
                cand_score, cand_pix, _, cand_path, cw, ch, cand_dur = top_cand

                if is_audio or cand_score >= 100 or cand_pix > max_pixels:
                    print(f"[BLOG] Automatically switched keyframe source from '{os.path.basename(video_path)}' to '{os.path.basename(cand_path)}' ({cw}x{ch}, {cand_dur:.1f}s)")
                    return cand_path

            return best_path
        except Exception as e:
            print(f"[BLOG] Video candidate search note: {e}")
            return video_path

    def extract_article_keyframes(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        output_dir: str,
        num_images: int = 4,
        topic_title: str = "article",
        target_resolution: str = "1080p (Full HD - 1920x1080) [Recommandé Articles & Google SEO]",
        enhance_text_clarity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract high-quality video keyframes with FFmpeg across key milestones.
        Supports 1080p/2K target resolution, Lanczos high-fidelity scaling, and text/code sharpening.
        Returns a list of image metadata dicts with image_path, timestamp, alt_text, caption, and dimensions.
        """
        if not video_path or not os.path.exists(video_path):
            print(f"[BLOG] Video path not found: {video_path}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        clean_slug = re.sub(r'[^a-zA-Z0-9]+', '-', topic_title.lower()).strip('-')[:35] or "guide"

        # Determine actual video duration
        real_duration = self._get_video_duration(video_path)
        seg_duration = segments[-1].get("end", 0.0) if segments else 0.0

        if real_duration > 0.2 and seg_duration > 0.2:
            total_duration = min(real_duration, seg_duration)
        elif real_duration > 0.2:
            total_duration = real_duration
        else:
            total_duration = seg_duration

        # Automatically use higher-resolution or matching video candidate if input is low-res or audio
        active_video_path = self._find_best_video_source(video_path, total_duration, topic_title=clean_slug)
        is_audio = os.path.splitext(active_video_path)[1].lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        if is_audio:
            print(f"[BLOG] [WARN] Keyframe extraction skipped: source '{active_video_path}' is an audio file and no video candidate was found in temp/.")
            return []

        real_vid_dur = self._get_video_duration(active_video_path)
        frame_duration = real_vid_dur if real_vid_dur > 0.2 else total_duration
        if frame_duration < 0.2:
            print(f"[BLOG] [WARN] Keyframe extraction skipped: duration {frame_duration:.2f}s is too short.")
            return []

        src_w, src_h = self._get_video_resolution(active_video_path)
        print(f"[BLOG] Extracting {num_images} milestone HD keyframes from '{os.path.basename(active_video_path)}' ({src_w}x{src_h}, duration: {frame_duration:.1f}s)...")

        # Parse target resolution
        res_lower = str(target_resolution).lower()
        if "2k" in res_lower or "1440" in res_lower:
            target_max = 2560
        elif "720" in res_lower:
            target_max = 1280
        elif "native" in res_lower or "source" in res_lower:
            if src_w < 1280 or src_h < 720:
                target_max = 1920
            else:
                target_max = None
        else:
            target_max = 1920

        # Build video filter chain (Lanczos scaling + Unsharp contrast mask for text/code readability)
        vf_filters = []
        if target_max:
            if src_w >= src_h:  # Landscape orientation
                vf_filters.append(f"scale={target_max}:-2:flags=lanczos+accurate_rnd")
            else:  # Portrait orientation
                vf_filters.append(f"scale=-2:{target_max}:flags=lanczos+accurate_rnd")

        if enhance_text_clarity:
            vf_filters.append("unsharp=lx=5:ly=5:la=1.0:cx=5:cy=5:ca=0.0")

        vf_arg = ",".join(vf_filters) if vf_filters else None

        # Select evenly spaced milestone timestamps across the video (from 10% to 90%)
        if num_images <= 1:
            pcts = [0.5]
        else:
            step = 0.80 / (num_images - 1)
            pcts = [0.10 + i * step for i in range(num_images)]
        extracted_images = []

        for idx, pct in enumerate(pcts):
            target_ts = frame_duration * pct
            target_seg_ts = (seg_duration * pct) if seg_duration > 0.2 else target_ts
            
            # Find nearest segment text to generate a rich, contextual ALT tag
            matching_seg = None
            min_dist = float('inf')
            if segments:
                for s in segments:
                    dist = abs(s.get("start", 0.0) - target_seg_ts)
                    if dist < min_dist:
                        min_dist = dist
                        matching_seg = s

            seg_text = matching_seg.get("text", "").strip() if matching_seg else f"Étape {idx+1}"
            clean_caption = re.sub(r'[\r\n]+', ' ', seg_text)[:110]

            img_filename = f"{clean_slug}-capture-{idx+1}.jpg"
            img_path = os.path.join(output_dir, img_filename)

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{target_ts:.2f}",
                "-i", active_video_path,
                "-vframes", "1",
            ]
            if vf_arg:
                cmd.extend(["-vf", vf_arg])
            cmd.extend([
                "-q:v", "1",
                img_path
            ])

            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                    try:
                        from PIL import Image
                        with Image.open(img_path) as im:
                            w_out, h_out = im.size
                    except Exception:
                        w_out, h_out = (target_max or src_w, int(target_max * 9 / 16) if target_max else src_h)

                    actual_res_str = f"{w_out}x{h_out}"
                    alt_text = f"Capture d'écran ({actual_res_str}) - {clean_slug.replace('-', ' ').title()} : {clean_caption}"
                    extracted_images.append({
                        "filename": img_filename,
                        "path": img_path,
                        "timestamp": target_ts,
                        "timestamp_display": f"{int(target_ts // 60):02d}:{int(target_ts % 60):02d}",
                        "alt_text": alt_text,
                        "caption": clean_caption,
                        "width": w_out,
                        "height": h_out,
                        "resolution_str": actual_res_str
                    })
                else:
                    err_hint = proc.stderr.strip().split("\n")[-1] if proc.stderr else "File not generated"
                    print(f"[BLOG] Warning: FFmpeg failed to extract frame {idx+1} at {target_ts:.2f}s: {err_hint}")
            except Exception as e:
                print(f"[BLOG] Failed to extract frame at {target_ts:.2f}s: {e}")

        print(f"[BLOG] [OK] Successfully extracted {len(extracted_images)}/{num_images} HD keyframes to '{output_dir}'")
        return extracted_images

    def package_wordpress_zip(
        self,
        article_data: Dict[str, Any],
        images_list: List[Dict[str, Any]],
        output_zip_path: str
    ) -> Optional[str]:
        """
        Package Markdown, Gutenberg HTML, SEO metadata, and all extracted images into a ZIP archive.
        """
        try:
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 1. Save Markdown article
                md_content = article_data.get("markdown_article", "")
                zipf.writestr("article_wordpress.md", md_content.encode("utf-8") if isinstance(md_content, str) else md_content)

                # 2. Save Gutenberg HTML
                html_content = article_data.get("gutenberg_html", "")
                zipf.writestr("article_gutenberg.html", html_content.encode("utf-8") if isinstance(html_content, str) else html_content)

                # 3. Save SEO Metadata JSON
                seo_meta = {
                    "title": article_data.get("title", ""),
                    "slug": article_data.get("slug", ""),
                    "meta_description": article_data.get("meta_description", ""),
                    "focus_keyword": article_data.get("focus_keyword", ""),
                    "secondary_keywords": article_data.get("secondary_keywords", []),
                    "word_count": article_data.get("word_count", 0),
                    "image_prompts": article_data.get("image_prompts", []),
                    "images": [
                        {
                            "filename": img["filename"],
                            "timestamp": img["timestamp_display"],
                            "alt_text": img["alt_text"],
                            "caption": img["caption"],
                            "resolution": img.get("resolution_str", "1920x1080")
                        } for img in images_list
                    ]
                }
                zipf.writestr("seo_metadata.json", json.dumps(seo_meta, ensure_ascii=False, indent=2).encode("utf-8"))

                # 4. Save Extracted Images
                for img in images_list:
                    if os.path.exists(img["path"]):
                        zipf.write(img["path"], arcname=os.path.join("images", img["filename"]))

            return output_zip_path
        except Exception as e:
            print(f"[SEOArticleGenerator] Error packaging ZIP: {e}")
            return None


# Global singleton instance
blog_generator = SEOArticleGenerator()
