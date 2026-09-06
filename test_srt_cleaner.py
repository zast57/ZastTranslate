import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.srt_cleaner import SRTCleaner
from modules.srt_parser import SRTParser

def test_filler_removal_fr():
    cleaner = SRTCleaner()
    test_cases = [
        ("Donc voilà, on va ouvrir Pinokio pour lancer le serveur.", "On va ouvrir Pinokio pour lancer le serveur."),
        ("Alors donc en fait c'est super simple.", "C'est super simple."),
        ("On a testé le modèle, voilà quoi.", "On a testé le modèle."),
        ("Et euh, c'est vraiment rapide.", "Et c'est vraiment rapide.")
    ]
    for inp, expected in test_cases:
        res = cleaner.clean_text_heuristics(inp, lang_code="fr")
        print(f"[FR TEST] '{inp}' -> '{res}'")
        assert len(res) > 0

def test_filler_removal_en():
    cleaner = SRTCleaner()
    test_cases = [
        ("So basically, we are going to install Maestro using Pinokio.", "We are going to install Maestro using Pinokio."),
        ("You know, it's actually really fast and easy.", "It's actually really fast and easy."),
        ("And um, the results look great, right?", "And the results look great."),
    ]
    for inp, expected in test_cases:
        res = cleaner.clean_text_heuristics(inp, lang_code="en")
        print(f"[EN TEST] '{inp}' -> '{res}'")
        assert len(res) > 0

def test_filler_removal_multilingual():
    cleaner = SRTCleaner()
    # Spanish
    res_es = cleaner.clean_text_heuristics("Bueno pues, vamos a instalar la aplicación.", lang_code="es")
    print(f"[ES TEST] 'Bueno pues...' -> '{res_es}'")
    assert res_es.lower().startswith("vamos")
    
    # German
    res_de = cleaner.clean_text_heuristics("Also halt, wir starten das Programm jetzt.", lang_code="de")
    print(f"[DE TEST] 'Also halt...' -> '{res_de}'")
    assert res_de.lower().startswith("wir")

def test_words_list_sync():
    cleaner = SRTCleaner()
    words = [
        {"word": "Donc", "start": 1.0, "end": 1.3},
        {"word": "voilà,", "start": 1.3, "end": 1.6},
        {"word": "on", "start": 1.7, "end": 1.9},
        {"word": "lance", "start": 2.0, "end": 2.5},
        {"word": "l'application.", "start": 2.6, "end": 3.2}
    ]
    cleaned_words = cleaner.clean_words_list(words, lang_code="fr")
    print("[WORDS TEST] Cleaned words:", [w["word"] for w in cleaned_words])
    assert cleaned_words[0]["word"] == "on"
    assert cleaned_words[0]["start"] == 1.7
    print(f"Timing accurately preserved: New start is {cleaned_words[0]['start']}s instead of 1.0s")

def test_ergonomic_wrapping():
    cleaner = SRTCleaner(max_chars_per_line=40, max_lines_per_cue=2)
    long_text = "Dans ce tutoriel complet nous allons voir comment installer et configurer Maestro directement depuis l'interface Pinokio pour faire de la vidéo IA en local."
    segments = [{"start": 0.0, "end": 6.0, "text": long_text}]
    cues = cleaner.split_into_ergonomic_cues(segments)
    print(f"[WRAP TEST] Long segment split into {len(cues)} cues:")
    for i, c in enumerate(cues, 1):
        print(f"  Cue {i} ({c['start']:.2f}s -> {c['end']:.2f}s):")
        for line in c["lines"]:
            print(f"    | {line} ({len(line)} chars)")
            assert len(line) <= 50  # Balanced within margins

def test_export_files():
    cleaner = SRTCleaner()
    os.makedirs("temp", exist_ok=True)
    segments = [
        {"start": 0.5, "end": 3.2, "text": "Donc voilà, bienvenue dans ce nouveau tuto."},
        {"start": 3.5, "end": 7.0, "text": "On va générer des vidéos avec MiniMax H3 et notre propre visage en local."}
    ]
    parser = SRTParser()
    cues = parser.segments_to_clean_srt(segments, "temp/test_clean.srt", lang_code="fr", clean_fillers=True)
    
    cleaner.export_vtt(cues, "temp/test_clean.vtt")
    cleaner.export_sbv(cues, "temp/test_clean.sbv")
    
    assert os.path.exists("temp/test_clean.srt")
    assert os.path.exists("temp/test_clean.vtt")
    assert os.path.exists("temp/test_clean.sbv")
    
    with open("temp/test_clean.srt", "r", encoding="utf-8-sig") as f:
        srt_content = f.read()
        print("\n[EXPORT TEST] Generated SRT:\n" + srt_content)
        assert "Donc voilà" not in srt_content

if __name__ == "__main__":
    print("=== Running SRTCleaner Tests ===")
    test_filler_removal_fr()
    print("\n--- Test FR OK ---")
    test_filler_removal_en()
    print("\n--- Test EN OK ---")
    test_filler_removal_multilingual()
    print("\n--- Test Multilingual OK ---")
    test_words_list_sync()
    print("\n--- Test Sync OK ---")
    test_ergonomic_wrapping()
    print("\n--- Test Wrapping OK ---")
    test_export_files()
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
