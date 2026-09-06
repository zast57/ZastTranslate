import json
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

def fetch_youtube_suggestions(query: str, max_results: int = 10) -> List[str]:
    """
    Fetch live YouTube search autocomplete suggestions (0 API key required).
    Uses the public YouTube Suggest endpoint.
    """
    if not query or not query.strip():
        return []
    
    clean_q = query.strip()
    encoded = urllib.parse.quote(clean_q)
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={encoded}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
                return [s.lower().strip() for s in data[1][:max_results] if s.strip()]
    except Exception as e:
        print(f"[SEO Assistant] YouTube suggest note for '{query}': {e}")
        
    return []

def format_timestamp_short(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format for YouTube chapters."""
    secs = max(0, int(round(seconds)))
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    s = secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{s:02d}"
    return f"{minutes:02d}:{s:02d}"

def clean_youtube_text(text: str, is_fr: bool = True) -> str:
    """
    Strip markdown bold/italic asterisks, clean quotes, and apply Humanizer Anti-AI Cleanup
    (Charte WikiProject AI Cleanup 35 patterns & blader/humanizer).
    """
    if not text:
        return ""
    # Remove thinking tags or chatter
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = text.replace('<think>', '').replace('</think>', '').strip()
    
    # Remove **bold** and *italic* markdown artifacts
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    # Remove leading/trailing quotes
    cleaned = cleaned.strip().strip('"\'')
    
    # Apply Humanizer anti-cliché cleanup from Tab 7 blog_generator
    try:
        from modules.blog_generator import blog_generator
        if blog_generator and hasattr(blog_generator, "clean_ai_artifacts"):
            cleaned = blog_generator.clean_ai_artifacts(cleaned, is_fr=is_fr)
    except Exception:
        pass

    # YouTube video-specific AI chatter patterns
    if is_fr:
        yt_cliches_fr = [
            (r"\bDans cette vidéo,?\s*(?:nous allons|nous verrons|découvrez|plongeons dans|je vais vous présenter)\b[^.!?\n]*[.!?\n]?", ""),
            (r"\bBienvenue dans cette vidéo\b[^.!?\n]*[.!?\n]?", ""),
            (r"\bPlongeons (?:sans plus attendre )?dans (?:le vif du sujet|les détails)[^.!?\n]*[.!?\n]?", ""),
            (r"\bSans plus attendre,?\s*", ""),
            (r"\bVoyons sans plus tarder,?\s*", ""),
            (r"\bUn véritable couteau suisse\b", "Un outil polyvalent"),
            (r"\b(?:une|cette) solution révolutionnaire\b", "une solution performante"),
            (r"\brévolutionnaire\b", "performant"),
            (r"\bun outil incontournable\b", "un outil adapté"),
            (r"\bdes performances époustouflantes\b", "de bonnes performances"),
            (r"\bÀ l'ère (?:du numérique|de l'intelligence artificielle|de la transformation digitale)[^,\n]*,?", ""),
            (r"\bDans un monde (?:en constante évolution|de plus en plus connecté|numérique)[^,\n]*,?", ""),
            (r"\bDans le paysage (?:actuel|technologique|numérique)[^,\n]*,?", ""),
        ]
        for pat, rep in yt_cliches_fr:
            cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    else:
        yt_cliches_en = [
            (r"\bIn this video,?\s*(?:we will|we'll|let's|I am going to)\s*(?:explore|dive into|discover|show you)\b[^.!?\n]*[.!?\n]?", ""),
            (r"\bWelcome to this video\b[^.!?\n]*[.!?\n]?", ""),
            (r"\bLet's dive (?:right )?into[^.!?\n]*[.!?\n]?", ""),
            (r"\bWithout further ado,?\s*", ""),
            (r"\bA game changer\b", "An effective solution"),
            (r"\bgroundbreaking\b", "notable"),
            (r"\bbreathtaking\b", "impressive"),
            (r"\bIn (?:today's )?(?:fast-paced )?digital (?:world|age|landscape)[^,\n]*,?", ""),
        ]
        for pat, rep in yt_cliches_en:
            cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)

    # Double space / multi-newline cleanup
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def format_youtube_title(title: str, is_fr: bool = True) -> str:
    """Format and polish YouTube titles: capital first letter, proper punctuation and spacing."""
    if not title:
        return ""
    t = clean_youtube_text(title).strip()
    # Strip leading/trailing punctuation or quotes
    t = re.sub(r"^[\s\"'«»\-:_.]+", "", t)
    t = re.sub(r"[\s\"'«»\-:_.]+$", "", t)
    
    # Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]
        
    # Brand normalizations
    brand_map = {
        r"\bhermes\b": "Hermès",
        r"\bhermesagent\b": "Hermès Agent",
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
        r"\busb-c\b": "USB-C",
        r"\busb\b": "USB",
    }
    for pat, rep in brand_map.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        
    return t

class YouTubeSEOAssistant:
    """
    Elite YouTube Growth, Search Intent & Metadata Optimization Assistant.
    - Multi-seed YouTube Search autocomplete keyword mining (0 API key).
    - Front-loaded Search-Intent Titles (Optimized for Mobile CTR & Search Indexing).
    - No Markdown Asterisks (`**`) in descriptions for seamless copy-pasting to YouTube Studio.
    - Specific, named-entity YouTube Chapters in natural sentence case.
    - Comprehensive, long-form (300-500 words) rich YouTube descriptions.
    - Real-world Long-Tail YouTube Tags.
    """

    def __init__(self):
        pass

    def extract_entities_and_topics(self, text: str) -> Dict[str, Any]:
        """Extract brand names, URLs, hardware specs, and key concepts from text."""
        domains = set(re.findall(r"\b(?:https?://)?([a-zA-Z0-9.-]+\.(?:com|org|io|ai|net|fr|dev|app|co))\b", text, re.IGNORECASE))
        clean_domains = [d for d in domains if not d.endswith(".")]
        
        # Comprehensive conversational stopwords to keep only meaningful nouns / brands / tech terms
        stop = {
            "les", "des", "une", "pour", "avec", "dans", "sur", "par", "qui", "que", "test", "tuto",
            "the", "and", "with", "from", "this", "that", "how", "what", "video", "review", "bonjour",
            "amis", "aujourd", "hui", "faire", "tout", "tous", "bien", "voici", "voila", "donc", "alors",
            "vais", "allez", "faire", "plus", "comme", "aussi", "voir", "petit", "petite", "chose", "ici",
            "vous", "nous", "montrer", "dire", "mettre", "aller", "avoir", "etre", "peut", "peux", "veut",
            "vraiment", "assez", "beaucoup", "encore", "toujours", "deja", "apres", "avant", "entre",
            "show", "showing", "look", "know", "think", "want", "need", "like", "just", "very", "also", "come",
            "salut", "aviez", "avez", "travaille", "travailler", "sommes", "etes", "etaient", "etait", "sera",
            "seront", "serait", "fait", "faits", "faudra", "va", "vont", "ceci", "cela", "leur", "leurs", "notre",
            "votre", "leurs", "mien", "tiens", "celle", "celui", "ceux", "celles", "meme", "memes", "autre", "autres",
            "tres", "trop", "sans", "sous", "vers", "chez", "pendant", "depuis", "vers", "quand", "lorsque", "puis",
            "donc", "ainsi", "alors", "car", "parce", "puisque", "si", "soit", "ou", "mais", "non", "oui",
            "voulez", "pouvez", "savoir", "passer", "parler", "donner", "prendre", "regarder", "ecouter", "trouver",
            "laisser", "penser", "arriver", "croire", "aimer", "falloir", "tenir", "semble", "parle", "reste",
            "partir", "demander", "rendre", "venir", "comprendre", "sortir", "mettre", "suite", "premier", "deuxieme"
        }
        
        # Check compound tech / brand entities
        text_lower = text.lower()
        extracted_entities = []
        if "hermes" in text_lower or "hermès" in text_lower:
            extracted_entities.append("HermesAgent")
            extracted_entities.append("Hermes")
        if "dragon ball" in text_lower or "dbz" in text_lower:
            extracted_entities.append("DragonBall")
        if "nikolatoy" in text_lower:
            extracted_entities.append("Nikolatoy")
        if "kamehameha" in text_lower:
            extracted_entities.append("Kamehameha")
        if "whisper" in text_lower:
            extracted_entities.append("WhisperX")
        if "demucs" in text_lower:
            extracted_entities.append("Demucs")
        if "claude" in text_lower:
            extracted_entities.append("ClaudeAI")
        if "chatgpt" in text_lower or "gpt" in text_lower:
            extracted_entities.append("ChatGPT")
        if "open source" in text_lower or "opensource" in text_lower:
            extracted_entities.append("OpenSource")
            
        words = re.findall(r"\b[A-Za-z0-9À-ÖØ-öø-ÿ]{3,}\b", text)
        filtered_keywords = [w for w in words if w.lower() not in stop and len(w) >= 4]
        
        combined = extracted_entities + list(dict.fromkeys(filtered_keywords))
        unique_keywords = list(dict.fromkeys(combined))
        
        return {
            "domains": clean_domains,
            "keywords": unique_keywords[:15]
        }

    def mine_search_trends(self, title: str, keywords: List[str], lang: str = "fr") -> List[str]:
        """Mine live YouTube Autocomplete queries across multiple search intent angles with domain disambiguation."""
        seeds = []
        
        # Qualify short or ambiguous titles
        clean_title = re.sub(r'[\._\-]+', ' ', (title or "")).strip()
        if clean_title:
            if "hermes" in clean_title.lower() and "agent" not in clean_title.lower():
                clean_title = clean_title.lower().replace("hermes", "hermes agent").strip()
            seeds.append(clean_title)
        
        # Compound multi-word entity seeds
        if len(keywords) >= 2:
            k1, k2 = keywords[0], keywords[1]
            if "hermes" in k1.lower() and "agent" not in k2.lower():
                k1 = "hermes agent"
            seeds.append(f"{k1} {k2}")
            if lang.startswith("fr"):
                seeds.append(f"tuto {k1}")
                seeds.append(f"{k1} test")
                seeds.append(f"{k1} installation")
            else:
                seeds.append(f"{k1} tutorial")
                seeds.append(f"{k1} review")
                seeds.append(f"how to install {k1}")
        elif keywords:
            k1 = keywords[0]
            if "hermes" in k1.lower() and "agent" not in k1.lower():
                k1 = "hermes agent"
            seeds.append(k1)
            if lang.startswith("fr"):
                seeds.append(f"tuto {k1}")
                seeds.append(f"{k1} test")
            else:
                seeds.append(f"{k1} tutorial")
                seeds.append(f"{k1} review")
            
        all_trends = []
        for seed in seeds[:6]:
            if not seed or len(seed) < 3:
                continue
            suggestions = fetch_youtube_suggestions(seed, max_results=8)
            all_trends.extend(suggestions)
            
        # Semantic domain filtering: purge fashion/clothing homonyms for tech topics
        fashion_blacklist = {
            "foulard", "twilly", "sac", "lacet", "carre", "carré", "parfum", "ceinture",
            "birkin", "kelly", "robe", "maroquinerie", "bijoux", "couture", "mode", "chaussure"
        }
        clean_trends = []
        for s in all_trends:
            s_lower = s.lower()
            if not any(fw in s_lower for fw in fashion_blacklist):
                clean_trends.append(s)
            
        return list(dict.fromkeys(clean_trends))[:20]

    def generate_chapters(self, segments: List[Dict[str, Any]], llm_backend=None, source_lang: str = "fr", product_name: str = "") -> str:
        """
        Generate SEO-dense YouTube Chapters from transcript.
        Rules:
        - Must start with 00:00.
        - Must cover the entire video timeline from start to finish.
        - Must capture all key tools and features (Ollama, Qwen, Telegram, API, smartphone, automation).
        - Natural sentence case (no English Title Case with capital on every word).
        """
        is_fr = str(source_lang).lower().startswith("fr")
        if not segments:
            return "00:00 - Introduction & Présentation\n" if is_fr else "00:00 - Introduction & Overview\n"
            
        total_duration = segments[-1].get("end", 0.0) if segments else 0.0
        
        # Sample uniformly across the entire timeline so all sections (beginning, middle, end) are covered
        max_samples = 40
        if len(segments) <= max_samples:
            sampled_segments = segments
        else:
            step = len(segments) / max_samples
            sampled_segments = [segments[int(i * step)] for i in range(max_samples)]
            
        sampled = []
        for seg in sampled_segments:
            st = seg.get("start", 0.0)
            sampled.append(f"[{format_timestamp_short(st)}] {clean_youtube_text(seg.get('text', ''))[:110]}")
            
        sampled_text = "\n".join(sampled)
        
        # Scan full transcript for landmark keywords and their exact timestamps
        target_landmarks_fr = [
            ("Installation sur Windows", ["windows", "install"]),
            ("Configuration des clés API", ["clé api", "api key", "mes api"]),
            ("Configuration d'Ollama et LLM local (Qwen)", ["ollama", "qwen"]),
            ("Automatisation et planification de tâches (Jobs)", ["job", "automatisation", "planifi"]),
            ("Exécution locale et sécurité des données", ["données ne sortent pas", "rien ne sort", "travaille en local"]),
            ("Intégration Telegram et bot", ["telegram"]),
            ("Démonstration du contrôle à distance via smartphone", ["smartphone", "téléphone", "telephone", "mobile"])
        ]
        
        detected_moments_fr = []
        for label, kws in target_landmarks_fr:
            for s in segments:
                txt = s.get("text", "").lower()
                if any(kw in txt for kw in kws):
                    st = s.get("start", 0.0)
                    detected_moments_fr.append(f"• [{format_timestamp_short(st)}] {label}")
                    break
                    
        landmarks_hint = "\nÉTAPES CLÉS ET OUTILS DÉTECTÉS DANS LA VIDÉO (À INCLURE DANS LES CHAPITRES) :\n" + "\n".join(detected_moments_fr) if detected_moments_fr else ""
        landmarks_hint_en = "\nKEY LANDMARKS AND TOOLS DETECTED IN VIDEO (MUST BE REFLECTED IN CHAPTERS):\n" + "\n".join(detected_moments_fr) if detected_moments_fr else ""

        if llm_backend is not None:
            if is_fr:
                prompt = f"""Tu es un expert mondial en SEO YouTube et chapitrage vidéo. À partir des extraits horodatés de cette vidéo, génère entre 7 et 10 chapitres YouTube majeurs bien espacés couvrant l'INTÉGRALITÉ de la vidéo de 00:00 jusqu'à la fin.
{landmarks_hint}

RÈGLES STRICTES :
1. Le premier chapitre DOIT obligatoirement être : 00:00 - [Titre d'introduction nommant le sujet/outil en français]
2. ESPACEMENT OBLIGATOIRE : Laisse au moins 1 à 3 minutes entre chaque chapitre (ne génère JAMAIS de micro-chapitres toutes les 30 secondes !).
3. COUVERTURE TOTALE : Les chapitres doivent couvrir toutes les phases de la vidéo du début (installation) jusqu'à la fin (Ollama local, Telegram, smartphone, conclusion).
4. Format obligatoire par ligne : MM:SS - Titre précis du chapitre en FRANÇAIS
5. NOMME SYSTÉMATIQUEMENT LES OUTILS ET FONCTIONNALITÉS CLÉS (ex: "Installation d'Hermès Agent sur Windows", "Configuration des clés API", "Utilisation d'Ollama avec LLM local Qwen", "Intégration Telegram et contrôle à distance via smartphone").
6. RÈGLE DE CASSE : Casse standard française naturelle (majuscule UNIQUEMENT au premier mot et aux noms propres comme Hermès, Windows, Ollama, Qwen, Telegram, API). INTERDICTION d'écrire en Title Case anglais avec des majuscules à chaque mot !
7. Rédige TOUS les titres exclusivement en FRANÇAIS.
8. Retourne UNIQUEMENT la liste des chapitres formatés.

Sujet de la vidéo : {product_name or 'Tutoriel et guide complet'}
Extraits horodatés sur toute la durée :
{sampled_text}

Chapitres YouTube optimisés :"""
            else:
                prompt = f"""You are an elite YouTube SEO strategist. Based on the timestamped transcript excerpts, generate 7 to 10 major, well-spaced YouTube chapters covering the ENTIRE video from start to finish.
{landmarks_hint_en}

STRICT RULES:
1. First chapter MUST be: 00:00 - [Introduction title naming the exact product/topic]
2. SPACING: Leave at least 1 to 3 minutes between chapters (do NOT create micro-chapters every 30 seconds).
3. FULL COVERAGE: Must cover all phases from beginning to end (including local Ollama setup, Telegram bot, smartphone remote control, and conclusion).
4. Exact format per line: MM:SS - Chapter Title
5. Explicitly name the exact tools, features, and actions (e.g. "Installing Hermes Agent on Windows", "API keys setup", "Local LLM setup with Ollama and Qwen", "Remote control via Telegram and smartphone").
6. Use natural sentence case (capital only on first word and proper nouns).
7. Output ONLY the list of timestamped chapters.

Product topic: {product_name or 'Complete Guide and Tutorial'}
Transcript excerpts across full duration:
{sampled_text}

Optimized YouTube Chapters:"""

            messages = [
                {"role": "system", "content": "You are a professional YouTube SEO specialist. Output only the requested chapter list."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                res = llm_backend.generate(messages, max_new_tokens=650, temperature=0.2)
                lines = [clean_youtube_text(line) for line in res.strip().splitlines() if re.match(r"^\d{1,2}:\d{2}", line.strip())]
                if lines:
                    if not lines[0].startswith("00:00"):
                        first_title = f"00:00 - Présentation de {product_name or 'l outil'}" if is_fr else f"00:00 - Overview of {product_name or 'the tool'}"
                        lines.insert(0, first_title)
                    return "\n".join(lines)
            except Exception as e:
                print(f"[SEO Assistant] LLM chapter generation fallback: {e}")

        # Fallback rule-based chapters
        chapters = [f"00:00 - Découverte de {product_name or 'la figurine'}" if is_fr else f"00:00 - Overview of {product_name or 'the figure'}"]
        if total_duration > 40:
            step = max(30.0, total_duration / 5.0)
            curr = step
            while curr < total_duration - 20:
                best_seg = min(segments, key=lambda s: abs(s.get("start", 0.0) - curr))
                title_snip = clean_youtube_text(best_seg.get("text", ""))[:40].strip()
                chapters.append(f"{format_timestamp_short(best_seg.get('start', curr))} - {title_snip}")
                curr += step
        chapters.append(f"{format_timestamp_short(max(0.0, total_duration - 30.0))} - Mon avis et conclusion" if is_fr else f"{format_timestamp_short(max(0.0, total_duration - 30.0))} - Final verdict and review")
        return "\n".join(chapters)

    def get_hashtag_packs(self, main_topic: str, keywords: List[str], lang: str = "fr") -> Dict[str, str]:
        """Generate 4 targeted, search-intent hashtag packs."""
        clean_keys = [re.sub(r'[^a-zA-Z0-9À-ÖØ-öø-ÿ]', '', k) for k in (keywords or []) if k]
        clean_keys = [k for k in clean_keys if len(k) >= 3]
        
        main_tag = f"#{re.sub(r'[^a-zA-Z0-9]', '', main_topic).lower()}" if main_topic else ("#ia" if is_fr else "#ai")
        is_fr = str(lang).lower().startswith("fr")
        
        # Primary subject tags
        p1 = [main_tag] + [f"#{k.lower()}" for k in clean_keys[:4]]
        if len(p1) < 4:
            p1 += ["#ia", "#doublage", "#clonagevocal", "#pinokio"] if is_fr else ["#ai", "#dubbing", "#voicecloning", "#pinokio"]
            
        # Format / Intent tags
        p2_tags = [main_tag] + (["#tutoriel", "#guide", "#tuto", "#test"] if is_fr else ["#tutorial", "#guide", "#setup", "#review"])
        
        # Ecosystem & Technology tags
        p3_tags = [main_tag] + (["#ia", "#clonagevocal", "#doublage", "#pinokio"] if is_fr else ["#ai", "#voicecloning", "#dubbing", "#pinokio"])
        
        # Trends & Community tags
        p4_tags = [main_tag] + (["#opensource", "#whisper", "#innovation", "#dev"] if is_fr else ["#opensource", "#whisper", "#innovation", "#dev"])

        return {
            "Pack 1: Subject & Specific": " ".join(list(dict.fromkeys(p1))[:4]),
            "Pack 2: Review & Unboxing": " ".join(list(dict.fromkeys(p2_tags))[:4]),
            "Pack 3: Collector & Tech": " ".join(list(dict.fromkeys(p3_tags))[:4]),
            "Pack 4: Community & Trends": " ".join(list(dict.fromkeys(p4_tags))[:4])
        }

    def generate_full_seo_package(
        self,
        segments: List[Dict[str, Any]],
        current_title: str = "",
        source_lang: str = "fr",
        llm_backend=None
    ) -> Dict[str, Any]:
        """
        Orchestrates full SEO optimization based on #1 YouTube ranking guidelines:
        1. Deep keyword & entity extraction from full transcript.
        2. Live YouTube autocomplete query mining.
        3. Front-Loaded Search-Intent Title (Search Query + Topic First, No artificial Title Case).
        4. In-depth, high-ranking description WITHOUT markdown asterisks `**` (clean for copy-pasting).
        5. Specific, named-product timestamped chapters.
        6. Dynamic, high-conversion 4 Hashtag Packs.
        7. Real Long-Tail YouTube tags.
        """
        full_transcript = " ".join(s.get("text", "") for s in segments)
        entities = self.extract_entities_and_topics(f"{current_title} {full_transcript}")
        keywords = entities.get("keywords", [])
        domains = entities.get("domains", [])
        
        is_fr = str(source_lang).lower().startswith("fr")
        
        # Mine search autocomplete trends
        trends = self.mine_search_trends(current_title, keywords, lang=source_lang)
        
        # Identify main subject
        main_subject = ""
        if len(keywords) >= 2:
            main_subject = f"{keywords[0]} {keywords[1]}"
        elif keywords:
            main_subject = keywords[0]
        else:
            main_subject = current_title or ("Tutoriel & Guide" if is_fr else "Tutorial & Guide")
            
        # Generate chapters with exact product context
        chapters_text = self.generate_chapters(segments, llm_backend=llm_backend, source_lang=source_lang, product_name=main_subject)
        
        # Generate initial hashtag packs
        primary_tag = keywords[0] if keywords else main_subject
        hashtag_packs = self.get_hashtag_packs(primary_tag, keywords, lang=source_lang)
        default_hashtags = hashtag_packs["Pack 1: Subject & Specific"]
        
        # High-intent Title, Hook & In-depth Description Generation
        transcript_sample = full_transcript[:2200]
        
        # Fallbacks
        optimized_title = (
            f"Tutoriel et guide complet : {main_subject}"
            if is_fr else
            f"Complete Guide and Tutorial: {main_subject}"
        )
        hook_text = (
            f"Découvrez le guide complet et le test de {main_subject} ! Toutes les explications détaillées étape par étape pour maîtriser cet outil."
            if is_fr else
            f"Discover the complete guide and full hands-on test of {main_subject}! Step-by-step instructions to get the most out of it."
        )
        full_desc_body = ""
        
        if llm_backend is not None:
            if is_fr:
                prompt = f"""Tu es le meilleur consultant mondial en référencement naturel YouTube (YouTube SEO Specialist).
Analyse les sous-titres de cette vidéo pour créer un kit de publication YouTube de niveau professionnel.

CONTEXTE DE LA VIDÉO :
Titre actuel : {current_title}
Mots-clés extraits : {', '.join(keywords[:8])}
Requêtes de recherche réelles (YouTube Suggest) : {', '.join(trends[:8])}
Transcription intégrale :
{transcript_sample}

RÈGLES CAPITALES DE RÉDACTION NATURELLE (ANTI-DÉTECTION IA — CHARTE HUMANIZER & WIKIPROJECT AI CLEANUP) :
1. ZÉRO BOURSOUFLURE NI HYPE : N'écris JAMAIS 'témoigne de', 'joue un rôle charnière', 'marque un tournant', 'une avancée révolutionnaire', 'un véritable couteau suisse'. Bannis les superlatifs vides ('époustouflant', 'incontournable').
2. ZÉRO FORMULES D'INTRO/OUTRO ARTIFICIELLES : Bannis formellement 'Plongeons sans plus attendre dans...', 'Dans cette vidéo nous allons explorer...', 'À l'ère du numérique', 'Dans un monde connecté'. L'accroche doit entrer directement dans le vif du sujet sans formule de politesse artificielle.
3. PAS DE PARTICIPES PRÉSENTS ARTIFICIELS : Évite les incises pompeuses ('permettant ainsi de', 'mettant en lumière', 'illustrant parfaitement').
4. INTERDICTION FORMELLE D'ASTÉRISQUES MARKDOWN ** : Rédige en texte brut avec des émojis sobres et une casse normale de phrase (PAS DE TITRES EN MAJUSCULES QUI CRIENT).
5. RYTHME NATUREL HUMAIN (BURSTINESS) : Alterne phrases courtes percutantes et explications techniques directes. Voix active, humaine, ton direct de créateur d'expérience.

CONSIGNES POUR LE TITRE :
- INTENTION DE RECHERCHE FORTE : Place les mots-clés exacts recherchés par les spectateurs dès les 45 premiers caractères (ex: "Comment doubler une vidéo avec sa voix...", "Traduire et doubler ses vidéos YouTube...", "Comment cloner sa voix pour...").
- BANNIS LES PRÉFIXES PARASITES qui mangent l'espace sur smartphone (pas de "TUTO :", pas de "Guide :", pas de crochets [ ]).
- RÈGLE DE CASSE : Casse standard française naturelle (majuscule UNIQUEMENT au premier mot et aux noms propres : Windows, API, Python, etc.). INTERDICTION d'écrire en Title Case anglais avec des majuscules à chaque mot !
- Maximum 70 caractères, direct, percutant et cliquable.

CONSIGNES POUR LA DESCRIPTION (200 à 300 mots sans aucun astérisque) :
- Structure attendue :
  1. ACCROCHE (2 phrases) : Directe, contenant l'intention de recherche et les mots-clés principaux dès les 100 premiers caractères.
  2. AU PROGRAMME : 5 à 7 tirets simples (- ) sans markdown listant les étapes réelles montrées.
  3. Des intertitres sobres en minuscules (ex: "Au programme :", "Liens utiles :", "Configuration requise :"). Pas de titres en majuscules criardes.

CONSIGNES POUR LES HASHTAGS (4 PACKS DE 4 HASHTAGS COURTS ET POPULAIRES) :
- Choisis des hashtags YouTube courts, en minuscules et réellement recherchés (ex: #ia #doublage #clonagevocal #pinokio).
- N'inclus JAMAIS de verbes conjugués isolés ni de mots parasites (pas de #aviez, #salut, #travaille).

Format de réponse STRICT :
TITRE: [Titre ici]
ACCROCHE: [Accroche ici]
CORPS: [Texte détaillé sans astérisques ici]
PACK1: [#tag1 #tag2 #tag3 #tag4]
PACK2: [#tag1 #tag2 #tag3 #tag4]
PACK3: [#tag1 #tag2 #tag3 #tag4]
PACK4: [#tag1 #tag2 #tag3 #tag4]"""
            else:
                prompt = f"""You are an elite YouTube SEO Strategist and video metadata specialist.
Analyze the video transcript below and create a professional, high-ranking YouTube publishing kit.

VIDEO CONTEXT:
Current Title: {current_title}
Key entities: {', '.join(keywords[:8])}
Live YouTube Search Suggest queries: {', '.join(trends[:8])}
Transcript:
{transcript_sample}

CRITICAL HUMANIZER WRITING RULES (ANTI-AI DETECTION — 35 WIKIPEDIA AI CLEANUP PATTERNS):
1. NO INFLATED CLAIMS OR LEGACY HYPE: Avoid 'pivotal moment', 'testament to', 'crucial role', 'indelible mark', 'game changer', 'revolutionary'.
2. NO STOCK FORMULAS OR CHATTER: Ban 'In this video we will explore', 'Without further ado', 'Let's dive into', 'In today's fast-paced digital world'. Hook immediately into search intent.
3. NO SHALLOW -ING PHRASES: Cut pseudo-profound participial clauses ('highlighting the significance', 'showcasing how', 'fostering a sense of').
4. NO MARKDOWN ASTERISKS ** : Plain text headers with clean emojis.
5. BURSTINESS & RHYTHM: Alternate punchy short sentences with nuanced direct explanations. Active creator voice.

STRICT TITLE GUIDELINES:
- Front-load the PRODUCT NAME / TOOL or SEARCH INTENT for maximum mobile search CTR.
- Natural sentence casing (capital only on first word and proper names).
- Max 70 characters.

STRICT DESCRIPTION GUIDELINES (250-350 words):
- First 150 characters must front-load primary search keywords.
- Structure:
  1. HOOK: 2 compelling sentences with target keywords.
  2. OVERVIEW: 2 paragraphs explaining features and workflow.
  3. KEY HIGHLIGHTS: 3-4 simple bullet points (- ) without asterisks.

HASHTAG PACKS (4 PACKS OF 5 RELEVANT HASHTAGS):
- Generate 5 specific, high-intent hashtags per pack tailored to the video topic.

STRICT Output format:
TITRE: [SEO Title here]
ACCROCHE: [Hook here]
CORPS: [Detailed overview without asterisks here]
PACK1: [#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5]
PACK2: [#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5]
PACK3: [#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5]
PACK4: [#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5]"""

            messages = [
                {"role": "system", "content": "You are a professional YouTube SEO specialist following the Humanizer anti-AI writing charter. Output strictly the requested sections without any markdown asterisks **."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                res = llm_backend.generate(messages, max_new_tokens=750, temperature=0.3)
                corps_lines = []
                in_corps = False
                for line in res.strip().splitlines():
                    clean_line = clean_youtube_text(line)
                    if clean_line.startswith("TITRE:"):
                        optimized_title = clean_line.replace("TITRE:", "").strip()
                    elif clean_line.startswith("ACCROCHE:"):
                        hook_text = clean_line.replace("ACCROCHE:", "").strip()
                    elif clean_line.startswith("PACK1:"):
                        in_corps = False
                        p1_val = clean_line.replace("PACK1:", "").strip().strip("[]")
                        if p1_val:
                            hashtag_packs["Pack 1: Subject & Specific"] = p1_val
                    elif clean_line.startswith("PACK2:"):
                        in_corps = False
                        p2_val = clean_line.replace("PACK2:", "").strip().strip("[]")
                        if p2_val:
                            hashtag_packs["Pack 2: Review & Unboxing"] = p2_val
                    elif clean_line.startswith("PACK3:"):
                        in_corps = False
                        p3_val = clean_line.replace("PACK3:", "").strip().strip("[]")
                        if p3_val:
                            hashtag_packs["Pack 3: Collector & Tech"] = p3_val
                    elif clean_line.startswith("PACK4:"):
                        in_corps = False
                        p4_val = clean_line.replace("PACK4:", "").strip().strip("[]")
                        if p4_val:
                            hashtag_packs["Pack 4: Community & Trends"] = p4_val
                    elif clean_line.startswith("CORPS:"):
                        in_corps = True
                        rest = clean_line.replace("CORPS:", "").strip()
                        if rest:
                            corps_lines.append(rest)
                    elif in_corps:
                        corps_lines.append(clean_line)
                if corps_lines:
                    full_desc_body = "\n".join(corps_lines).strip()
            except Exception as e:
                print(f"[SEO Assistant] LLM description generation note: {e}")

        default_hashtags = hashtag_packs.get("Pack 1: Subject & Specific", default_hashtags)

        # Ensure description body, hook, and full description are 100% free of markdown asterisks and AI cliches
        full_desc_body = clean_youtube_text(full_desc_body, is_fr=is_fr)
        hook_text = clean_youtube_text(hook_text, is_fr=is_fr)
        optimized_title = format_youtube_title(optimized_title, is_fr=is_fr)

        # Assemble Full Structured Description
        links_block = ""
        if domains:
            header_links = "Liens et outils mentionnés :\n" if is_fr else "Resources and links:\n"
            links_block = header_links + "\n".join(f"• https://{d}" for d in domains) + "\n\n"
            
        ch_header = "Chapitres :" if is_fr else "Chapters:"
        cta_text = (
            "👍 Si la vidéo vous a plu, pensez à laisser un like, partager votre avis en commentaire et vous abonner pour ne manquer aucun tutoriel !"
            if is_fr else
            "👍 If you enjoyed this video, please leave a like, share your thoughts in the comments, and subscribe for more in-depth guides!"
        )
        
        body_section = f"\n\n{full_desc_body}\n\n" if full_desc_body else "\n\n"
        
        full_description = (
            f"{hook_text}"
            f"{body_section}"
            f"{ch_header}\n"
            f"{chapters_text}\n\n"
            f"{links_block}"
            f"{cta_text}\n\n"
            f"{default_hashtags}"
        )
        full_description = clean_youtube_text(full_description, is_fr=is_fr)
        
        # Build High-Value Long-Tail YouTube Tags
        tags_pool = []
        tags_pool.extend(trends)
        if len(keywords) >= 2:
            k1, k2 = keywords[0].lower(), keywords[1].lower()
            if is_fr:
                tags_pool.extend([
                    f"tuto {k1} {k2}",
                    f"guide {k1} {k2}",
                    f"installation {k1}",
                    f"avis {k1}",
                    f"test {k1}",
                    f"{k1} windows",
                    f"{k1} gratuit"
                ])
            else:
                tags_pool.extend([
                    f"{k1} {k2} tutorial",
                    f"{k1} {k2} guide",
                    f"how to install {k1}",
                    f"{k1} review",
                    f"{k1} setup windows",
                    f"{k1} test"
                ])
        for kw in keywords[:6]:
            tags_pool.append(kw.lower())
            
        unique_tags = []
        for t in tags_pool:
            clean_t = re.sub(r'[\'\"#*]', '', t).strip()
            if clean_t and clean_t not in unique_tags and len(clean_t) >= 3:
                unique_tags.append(clean_t)
                
        tags_str = ", ".join(unique_tags[:18])
        
        return {
            "title": optimized_title,
            "description": full_description,
            "chapters": chapters_text,
            "hook": hook_text,
            "body": full_desc_body,
            "links": domains,
            "hashtag_packs": hashtag_packs,
            "selected_hashtags": default_hashtags,
            "tags": tags_str,
            "trends": trends
        }

seo_assistant = YouTubeSEOAssistant()
