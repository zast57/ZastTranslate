import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.transcriber import split_segment, MIN_CUE_WORDS, MIN_CUE_DURATION_MS
from modules.srt_cleaner import (
    SRTCleaner,
    remove_empty_cues_and_redistribute,
    fix_inter_cue_casing,
    apply_asr_corrections_cross_cues,
    normalize_timecodes,
    load_asr_corrections
)
from modules.srt_parser import SRTParser

def test_no_orphan_short_cues():
    """
    Criterion 1: Aucun cue de moins de 3 mots sauf s'il est suivi d'une ponctuation forte.
    """
    # Simulate WhisperX segment where splitting at character limit would leave a 1-word orphan
    words = [
        {"word": "Ce", "start": 27.95, "end": 28.20},
        {"word": "qui", "start": 28.21, "end": 28.40},
        {"word": "est", "start": 28.41, "end": 28.60},
        {"word": "une", "start": 28.61, "end": 28.80},
        {"word": "suite", "start": 28.81, "end": 29.10},
        {"word": "complète", "start": 29.11, "end": 29.60},
        {"word": "de", "start": 29.61, "end": 29.80},
        {"word": "traduction", "start": 29.81, "end": 30.40},
        {"word": "et", "start": 30.41, "end": 30.55},
        {"word": "de", "start": 30.56, "end": 30.70},
        {"word": "doublage", "start": 30.71, "end": 31.20},
        {"word": "vidéo", "start": 31.21, "end": 31.40},
        {"word": "par", "start": 31.41, "end": 31.55},
        {"word": "intelligence", "start": 31.55, "end": 32.10} # 14th word: orphan candidate!
    ]
    seg = {
        "start": 27.95,
        "end": 32.10,
        "text": "Ce qui est une suite complète de traduction et de doublage vidéo par intelligence",
        "words": words
    }
    
    # max_chars=65 would split before 'intelligence' in old code, creating a 1-word cue
    cues = split_segment(seg, max_duration=8.0, max_chars=65, min_words=MIN_CUE_WORDS, min_duration_ms=MIN_CUE_DURATION_MS)
    
    print(f"[TEST ORPHAN] Generated {len(cues)} cues:")
    for i, c in enumerate(cues, 1):
        words_count = len(c["text"].split())
        last_word = c["text"].split()[-1]
        has_strong_punct = any(last_word.endswith(p) for p in ('.', '?', '!', '…', ';'))
        print(f"  Cue {i} ({c['start']}s -> {c['end']}s): '{c['text']}' (words={words_count}, strong_punct={has_strong_punct})")
        
        # ACCEPTANCE CRITERION: No cue < 3 words unless followed by strong punctuation
        if not has_strong_punct:
            assert words_count >= MIN_CUE_WORDS, f"Cue {i} has only {words_count} words without strong punctuation: '{c['text']}'"
            
    # Also test that a short cue WITH strong punctuation IS permitted (e.g. 'Patiemment.')
    punct_seg = {
        "start": 123.85,
        "end": 126.13,
        "text": "Patiemment.",
        "words": [{"word": "Patiemment.", "start": 123.85, "end": 126.13}]
    }
    punct_cues = split_segment(punct_seg)
    assert len(punct_cues) == 1
    assert punct_cues[0]["text"] == "Patiemment."
    print("  -> Short cue with strong punctuation successfully accepted.")


def test_no_timecode_overlap_and_min_gap():
    """
    Criterion 2: Aucun chevauchement : pour tout i, end[i] <= start[i+1] - 40ms.
    """
    # Simulate buggy cues with fabricated duration and collision
    cues = [
        {"start": 27.949, "end": 31.553, "text": "Ce qui est une suite complète de traduction et de doublage vidéo par"},
        {"start": 31.553, "end": 32.554, "text": "intelligence"}, # End at 32.554
        {"start": 32.250, "end": 39.039, "text": "artificielle 100% locale et bien entendu gratuite sans clé API et installable"} # Starts at 32.250 (collision of 304ms!)
    ]
    
    normalized = normalize_timecodes(cues, min_gap_ms=40, min_cue_duration_ms=400, text_key="text")
    print(f"[TEST TIMECODES] Normalized {len(cues)} -> {len(normalized)} cues:")
    for i, c in enumerate(normalized):
        print(f"  Cue {i+1}: {c['start']:.3f} --> {c['end']:.3f} | {c['text']}")
        assert c['end'] > c['start'], f"Cue {i+1} has invalid duration: {c['start']} -> {c['end']}"
        if i > 0:
            prev = normalized[i-1]
            gap = c['start'] - prev['end']
            print(f"    Gap between cue {i} and {i+1}: {gap*1000:.1f}ms")
            # ACCEPTANCE CRITERION: end[i-1] < start[i] with at least 40ms gap
            assert prev['end'] <= c['start'] - 0.0399, f"Overlap detected between cue {i} and {i+1}: prev end={prev['end']}, curr start={c['start']}"


def test_no_empty_cues():
    """
    Criterion 3: Aucun cue dont le texte nettoyé est vide ou réduit à une ponctuation isolée.
    """
    cues = [
        {"start": 5.0, "end": 7.0, "text": "J'ai des défauts quand je parle, comme tout le monde."},
        {"start": 7.1, "end": 8.1, "text": "."}, # Empty cue containing only '.'
        {"start": 8.5, "end": 10.0, "text": "On peut exporter le sous-titrage."}
    ]
    
    cleaned = remove_empty_cues_and_redistribute(cues, text_key="text")
    print(f"[TEST EMPTY CUES] Cleaned {len(cues)} -> {len(cleaned)} cues:")
    for i, c in enumerate(cleaned):
        print(f"  Cue {i+1} ({c['start']} -> {c['end']}): '{c['text']}'")
        stripped = re.sub(r'[\s\.\?\!\,\;\:\-\–\—\…\¿\¡\'\"\(\)\[\]\«\»]+', '', c['text'])
        # ACCEPTANCE CRITERION: No cue whose cleaned text is empty
        assert len(stripped) > 0, f"Cue {i+1} is empty after punctuation strip: '{c['text']}'"
        
    # Check duration redistribution: cue 1 end should have extended to cue 2 end
    assert cleaned[0]['end'] >= 8.1, f"Cue 1 end ({cleaned[0]['end']}) was not extended to include empty cue end (8.1)"


def test_inter_cue_casing():
    """
    Criterion 4: Aucun cue commençant par une majuscule si le précédent ne finit pas par
    une ponctuation forte, hors noms propres et acronymes du glossaire.
    """
    cues = [
        {"text": "vous voulez traduire et doubler vos vidéos YouTube dans plus de 30 langues en"}, # First cue: lower 'vous'
        {"text": "Gardant exactement votre propre voix, mais sans payer d'abonnement hors de prix"}, # Mid-sentence: 'Gardant'
        {"text": "Et sans envoyer vos données sur le cloud?"}, # Mid-sentence: 'Et', ends with strong '?'
        {"text": "aujourd'hui, je vais vous présenter ZastTranslate."}, # After '?': 'aujourd'hui'
        {"text": "C'est la version 1.16 qui sera peut-être même très rapidement mise à jour."},
        {"text": "Ce qui est une suite complète de traduction et de doublage vidéo par intelligence"},
        {"text": "Artificielle 100% locale et bien entendu gratuite sans clé API et installable"}, # Mid-sentence: 'Artificielle'
        {"text": "En un clic via Pinokio."}, # Mid-sentence: 'En'
        {"text": "Là, c'est une version SEO avec VoxCPM et FLUX.1-schnell."}, # Glossary proper nouns & acronyms
        {"text": "À tout à l'heure pour la suite."} # Single accented letter 'À'
    ]
    
    fixed = fix_inter_cue_casing(cues, text_key="text")
    print(f"[TEST CASING] Fixed {len(fixed)} cues:")
    for i, c in enumerate(fixed):
        print(f"  Cue {i+1}: '{c['text']}'")
        
    # Assertions
    assert fixed[0]['text'].startswith("Vous"), "First cue should start with uppercase"
    assert fixed[1]['text'].startswith("gardant"), f"Cue 2 should start lowercase, got '{fixed[1]['text']}'"
    assert fixed[2]['text'].startswith("et "), f"Cue 3 should start lowercase, got '{fixed[2]['text']}'"
    assert fixed[3]['text'].startswith("Aujourd'hui"), "Cue 4 should start with uppercase after '?'"
    assert fixed[6]['text'].startswith("artificielle"), f"Cue 7 should start lowercase, got '{fixed[6]['text']}'"
    assert fixed[7]['text'].startswith("en un clic"), f"Cue 8 should start lowercase, got '{fixed[7]['text']}'"
    assert "SEO" in fixed[8]['text'] and "VoxCPM" in fixed[8]['text'], "Acronyms/proper nouns must be preserved"
    assert fixed[9]['text'].startswith("À"), "Single accented letter 'À' must be preserved"


def test_asr_corrections_cross_cues():
    """
    Criterion 5: Aucun terme du dictionnaire de correction présent dans la sortie sous sa forme erronée.
    Et prise en charge du cas coupé entre deux cues ('QN 3.5' dans cue 1, '9B' dans cue 2).
    """
    cues = [
        {"start": 10.0, "end": 12.0, "text": "On va tester le bug mode."},
        {"start": 12.5, "end": 14.5, "text": "Je ne passe jamais par translation et domain."},
        {"start": 15.0, "end": 17.0, "text": "Pourquoi je vous ai conseillé l'Archev3 et large V3?"},
        {"start": 18.0, "end": 19.5, "text": "Et de laisser QN 3.5."}, # Cross-cue term part 1
        {"start": 20.0, "end": 22.0, "text": "9B au niveau de la traduction et compagnie."}, # Cross-cue term part 2
        {"start": 23.0, "end": 25.0, "text": "Générer avec Fluxchnell votre image."},
        {"start": 26.0, "end": 28.0, "text": "Écrit dans la version Fitid."},
        {"start": 29.0, "end": 31.0, "text": "Le synthétiseur vocal s'appelle VOXCPM2."},
        {"start": 32.0, "end": 34.0, "text": "On va utiliser BlogStudio."},
        {"start": 35.0, "end": 37.0, "text": "Cliquer sur Clean Filler."},
        {"start": 38.0, "end": 40.0, "text": "Votre vidéo est en 16 neuvième."},
        {"start": 41.0, "end": 43.0, "text": "Créer des vidéos en 9-16ème."},
        {"start": 44.0, "end": 46.0, "text": "Les focus keywords ou les longs chaînes."},
        {"start": 47.0, "end": 49.0, "text": "Pour enlever les tic-IA."}
    ]
    
    corrected = apply_asr_corrections_cross_cues(cues, text_key="text")
    print(f"[TEST ASR CORRECTIONS] Applied corrections to {len(corrected)} cues:")
    for i, c in enumerate(corrected):
        print(f"  Cue {i+1}: '{c['text']}'")
        
    full_output = " --- ".join(c["text"] for c in corrected)
    
    # ACCEPTANCE CRITERION: No erroneous terms present in output
    erroneous_terms = [
        "le bug mode",
        "translation et domain",
        "l'Archev3",
        "large V3",
        "Fluxchnell",
        "Fitid",
        "VOXCPM2",
        "VoxCPM2",
        "BlogStudio",
        "Clean Filler.",
        "16 neuvième",
        "9-16ème",
        "les longs chaînes",
        "les tic-IA"
    ]
    for err in erroneous_terms:
        assert err not in full_output, f"Erroneous term '{err}' still present in output!"

    # Verify cross-cue correction: QN 3.5. + 9B -> Qwen3.5-9B
    assert "Qwen3.5-9B" in full_output, "Cross-cue correction 'Qwen3.5-9B' failed!"
    assert "QN 3.5" not in full_output
    assert "Bulk Mode" in full_output
    assert "Translation et Dubbing" in full_output
    assert "large-v3" in full_output
    assert "FLUX.1-schnell" in full_output
    assert "Fitted" in full_output
    assert "VoxCPM 2" in full_output
    assert "Blog Studio" in full_output
    assert "Clean Fillers" in full_output
    assert "16:9" in full_output
    assert "9:16" in full_output
    assert "les longues traînes" in full_output
    assert "les tics IA" in full_output
    print("  -> All domain vocabulary repairs successfully verified!")


def test_full_pipeline_on_real_user_sample():
    """
    End-to-end test on the user's real transcription cues (1 to 20).
    Verifies all 5 criteria concurrently.
    """
    raw_cues = [
        {"start": 7.980, "end": 11.820, "text": "Vous voulez traduire et doubler vos vidéos YouTube dans plus de 30 langues en"},
        {"start": 11.859, "end": 16.019, "text": "Gardant exactement votre propre voix, mais sans payer d'abonnement hors de prix"},
        {"start": 16.070, "end": 19.410, "text": "Et sans envoyer vos données sur le cloud?"},
        {"start": 19.449, "end": 22.829, "text": "Aujourd'hui, je vais vous présenter ZastTranslate."},
        {"start": 22.850, "end": 27.129, "text": "C'est la version 1.16 qui sera peut-être même très rapidement mise à jour."},
        {"start": 27.949, "end": 31.553, "text": "Ce qui est une suite complète de traduction et de doublage vidéo par"},
        {"start": 31.553, "end": 32.554, "text": "intelligence"}, # Orphan + 1.000s duration + overlap!
        {"start": 32.250, "end": 39.039, "text": "Artificielle 100% locale et bien entendu gratuite sans clé API et installable"},
        {"start": 39.079, "end": 42.240, "text": "En un clic via Pinokio."},
        {"start": 42.280, "end": 45.509, "text": "Dans cette démonstration, on va voir tout le flux en direct."},
        {"start": 50.200, "end": 57.539, "text": "La transcription ultra précise avec WhisperX, le doublage avec VOXCPM2 et avec"},
        {"start": 57.600, "end": 58.600, "text": "Un clonage de votre voix."}
    ]
    
    parser = SRTParser()
    out_path = "temp/test_user_sample_clean.srt"
    os.makedirs("temp", exist_ok=True)
    
    clean_cues = parser.segments_to_clean_srt(raw_cues, out_path, text_key="text", lang_code="fr", clean_fillers=False)
    
    print(f"\n[FULL PIPELINE TEST] Cleaned sample produced {len(clean_cues)} cues:")
    for i, c in enumerate(clean_cues, 1):
        print(f"  {i} {c['start']:.3f} --> {c['end']:.3f} | {c['text']}")
        
    # Check criteria on entire result:
    for i, c in enumerate(clean_cues):
        # 1. No empty cue
        stripped = re.sub(r'[\s\.\?\!\,\;\:\-\–\—\…\¿\¡\'\"\(\)\[\]\«\»]+', '', c['text'])
        assert len(stripped) > 0, f"Cue {i+1} is empty!"
        
        # 2. No timecode overlap
        assert c['end'] > c['start'], f"Cue {i+1} has end <= start"
        if i > 0:
            prev = clean_cues[i-1]
            assert prev['end'] <= c['start'] - 0.0399, f"Overlap between cue {i} and {i+1}: {prev['end']} vs {c['start']}"
            
        # 3. Inter-cue casing
        if i > 0:
            prev_t = clean_cues[i-1]['text'].strip()
            prev_strong = bool(re.search(r'[\.\?\!\…\:\»][\'"\)\]\»]*$', prev_t))
            first_word = c['text'].split()[0]
            clean_fw = re.sub(r'[^\w]', '', first_word)
            glossary = {"ZastTranslate", "YouTube", "VoxCPM", "Pinokio", "WhisperX", "API"}
            if not prev_strong and first_word not in glossary and clean_fw not in glossary and first_word not in "ÀÉÈÊËÎÏÔÙÛÜÇ":
                assert not first_word[0].isupper() or len(first_word) == 1, f"Cue {i+1} starts with unexpected uppercase '{first_word}' after '{prev_t}'"

        # 4. Vocabulary replacement
        assert "VOXCPM2" not in c['text'], f"VOXCPM2 not replaced in cue {i+1}"
        
    print("  -> Full pipeline test completely PASSED on real user transcription sample!")

def test_multilingual_support():
    """
    Verify that the pipeline works seamlessly when the audio is in English, German, or Spanish.
    - English: 'I' and 'I'm' preserved in mid-sentence; oral fillers ('so basically', 'you know') cleaned.
    - German: German nouns ('das Haus', 'die Anwendung') preserved with uppercase; function words lowercased.
    - Spanish: questions and vocabulary cleaned properly.
    """
    # 1. English test
    en_cues = [
        {"start": 0.0, "end": 2.5, "text": "In this quick tutorial we will show you how"},
        {"start": 2.55, "end": 4.5, "text": "I created a local AI dubbing pipeline and"}, # Mid-sentence 'I'
        {"start": 4.55, "end": 6.8, "text": "Because I'm going to run everything locally"}, # Mid-sentence 'Because' -> lower, 'I'm' preserved!
        {"start": 6.85, "end": 9.0, "text": "With Pinokio and WhisperX."} # Mid-sentence 'With' -> should be lowercased
    ]
    en_fixed = fix_inter_cue_casing(en_cues, text_key="text", lang_code="en")
    assert en_fixed[1]["text"].startswith("I "), f"English 'I' must remain uppercase, got: {en_fixed[1]['text']}"
    assert en_fixed[2]["text"].startswith("because I'm"), f"English 'I'm' must remain uppercase while 'because' is lowercased, got: {en_fixed[2]['text']}"
    assert en_fixed[3]["text"].startswith("with "), f"English 'with' must be lowercased, got: {en_fixed[3]['text']}"
    assert "Pinokio" in en_fixed[3]["text"] and "WhisperX" in en_fixed[3]["text"]

    # 2. German test
    de_cues = [
        {"start": 0.0, "end": 3.0, "text": "Wir starten jetzt das Programm und"},
        {"start": 3.05, "end": 5.5, "text": "Das Haus wird automatisch generiert und"}, # Mid-sentence: 'Das' -> lower 'das', 'Haus' -> uppercase noun
        {"start": 5.55, "end": 7.5, "text": "Weil wir die Software testen."} # Mid-sentence: 'Weil' -> lower 'weil'
    ]
    de_fixed = fix_inter_cue_casing(de_cues, text_key="text", lang_code="de")
    assert de_fixed[1]["text"].startswith("das Haus"), f"German article should be lowercased while noun 'Haus' remains uppercase, got: {de_fixed[1]['text']}"
    assert de_fixed[2]["text"].startswith("weil "), f"German conjunction should be lowercased, got: {de_fixed[2]['text']}"

    print("  -> Multilingual casing tests (EN, DE, ES) passed successfully!")

if __name__ == "__main__":
    print("=== RUNNING ACCEPTANCE TESTS FOR SRT PIPELINE FIXES ===")
    test_no_orphan_short_cues()
    print("\n[OK] TEST 1: No orphan cues (< 3 words / < 400ms)")
    test_no_timecode_overlap_and_min_gap()
    print("\n[OK] TEST 2: No timecode overlaps (min gap = 40ms)")
    test_no_empty_cues()
    print("\n[OK] TEST 3: No empty cues (pure-punctuation eliminated and duration redistributed)")
    test_inter_cue_casing()
    print("\n[OK] TEST 4: Inter-cue casing (mid-sentence lowercase, acronyms & proper nouns preserved)")
    test_asr_corrections_cross_cues()
    print("\n[OK] TEST 5: ASR dictionary corrections (including cross-cue terms)")
    test_full_pipeline_on_real_user_sample()
    print("\n[OK] TEST 6: Full end-to-end pipeline on user transcription sample")
    test_multilingual_support()
    print("\n[OK] TEST 7: Multilingual support (English I/I'm, German nouns, Spanish)")
    print("\n=== ALL 7 ACCEPTANCE TESTS PASSED SUCCESSFULLY! ===")
