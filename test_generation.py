import os
import sys

def run_test():
    print("=== TEST QWEN 3.5 LLM ===")
    try:
        from modules.llm_backends.factory import get_backend as get_llm
        llm = get_llm("Qwen3.5-9B")
        print("Loading LLM...")
        llm.load()
        messages = [{"role": "user", "content": "Hello, this is a test. Reply with 'OK'."}]
        print("Generating LLM text...")
        out = llm.generate(messages, max_new_tokens=10)
        print("LLM Output:", out)
        print("Unloading LLM...")
        llm.unload()
    except Exception as e:
        print("LLM TEST FAILED:", e)
        return False

    print("\n=== TEST QWEN TTS ===")
    try:
        from modules.tts_backends.factory import get_backend as get_tts
        tts = get_tts("Qwen3-TTS")
        print("Loading QwenTTS...")
        tts.load()
        print("Generating QwenTTS audio...")
        audio = tts.synthesize("Bonjour, test de la synthèse vocale.", "fr")
        print("QwenTTS output success:", audio is not None)
        print("Unloading QwenTTS...")
        tts.unload()
    except Exception as e:
        print("QwenTTS TEST FAILED:", e)
        return False

    print("\n=== TEST OMNIVOICE ===")
    try:
        from modules.tts_backends.factory import get_backend as get_tts
        tts_omni = get_tts("OmniVoice")
        if tts_omni.is_available():
            print("Loading OmniVoice...")
            tts_omni.load()
            print("Generating OmniVoice audio...")
            audio = tts_omni.synthesize("Bonjour, test de la synthèse vocale.", "fr")
            print("OmniVoice output success:", audio is not None)
            print("Unloading OmniVoice...")
            tts_omni.unload()
        else:
            print("OmniVoice not available (dependencies missing).")
    except Exception as e:
        print("OmniVoice TEST FAILED:", e)
        return False

    print("\nALL TESTS PASSED WITH TRANSFORMERS 5.X!")
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
