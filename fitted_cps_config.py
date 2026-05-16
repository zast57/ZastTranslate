"""
Per-language fitted characters-per-second (CPS) calibration for VoxCPM 2.

Each value represents the measured average speaking rate of the TTS engine
in the given language. Used to compute the maximum translated text length
that fits in a segment's time slot without truncation.

Adjust values after measuring real generation output if needed.
"""

FITTED_CPS_BY_LANG = {
    # Latin européen
    "fr": 9.0, "en": 8.5, "de": 8.5, "es": 9.0, "it": 9.0, "pt": 9.0,
    "nl": 8.5, "pl": 8.0, "da": 8.0, "sv": 8.0, "no": 8.0, "fi": 8.5,
    "tr": 8.5, "el": 8.0,
    # Latin autres
    "id": 8.5, "ms": 8.5, "vi": 9.0, "tl": 8.5, "sw": 8.5,
    # Cyrillique
    "ru": 8.0,
    # Abjads
    "ar": 7.0, "he": 7.0,
    # Abugidas
    "hi": 7.0, "th": 7.0, "lo": 7.0, "km": 7.0, "my": 7.0,
    # CJK
    "zh": 6.0, "ja": 7.0, "ko": 6.5,
    # Dialectes chinois
    "zh-yue": 6.0, "zh-sichuan": 6.0, "zh-wu": 6.0, "zh-ne": 6.0,
    "zh-henan": 6.0, "zh-shaanxi": 6.0, "zh-shandong": 6.0,
    "zh-tianjin": 6.0, "zh-minnan": 6.0,
    "_default": 7.5,
}


def get_fitted_cps(lang_code: str) -> float:
    """Return the calibrated CPS for a given ISO 639-1 language code.

    Falls back to prefix match (e.g. 'zh-hans' → 'zh'), then to _default.
    """
    lang_code = (lang_code or "").lower().replace("_", "-")
    if lang_code in FITTED_CPS_BY_LANG:
        return FITTED_CPS_BY_LANG[lang_code]
    prefix = lang_code.split("-")[0]
    if prefix in FITTED_CPS_BY_LANG:
        return FITTED_CPS_BY_LANG[prefix]
    return FITTED_CPS_BY_LANG["_default"]


# ---------------------------------------------------------------------------
# User overrides — persisted in user_cps_config.json (project root)
# ---------------------------------------------------------------------------
import json as _json
import os as _os

_USER_CPS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "user_cps_config.json")


def load_user_cps() -> dict:
    """Load user CPS overrides. Returns {} if file absent or corrupt."""
    try:
        if _os.path.exists(_USER_CPS_PATH):
            with open(_USER_CPS_PATH, "r", encoding="utf-8") as f:
                return {k: float(v) for k, v in _json.load(f).items()}
    except Exception:
        pass
    return {}


def save_user_cps(overrides: dict) -> None:
    """Persist user CPS overrides to JSON file."""
    with open(_USER_CPS_PATH, "w", encoding="utf-8") as f:
        _json.dump(overrides, f, indent=2, ensure_ascii=False)


def get_effective_cps(lang_code: str, user_overrides: dict = None) -> float:
    """Return CPS for lang_code: user override takes priority over the default table."""
    lc = (lang_code or "").lower().replace("_", "-")
    if user_overrides:
        if lc in user_overrides:
            return float(user_overrides[lc])
        prefix = lc.split("-")[0]
        if prefix in user_overrides:
            return float(user_overrides[prefix])
    return get_fitted_cps(lang_code)
