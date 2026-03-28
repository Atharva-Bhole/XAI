"""
Language detection utility.
Uses langdetect as primary engine with a custom Hindi/Marathi heuristic.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Devanagari unicode range: \u0900-\u097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

# Marathi-specific common words (rough heuristic)
MARATHI_WORDS = {
    "आहे", "आहेत", "आणि", "हे", "हा", "ती", "तो", "मला", "तुला", "आपण",
    "नाही", "काय", "कसे", "मराठी", "महाराष्ट्र", "छान", "चित्रपट",
}

# Hindi-specific common words (rough heuristic)
HINDI_WORDS = {
    "है", "हैं", "और", "यह", "वह", "मैं", "तुम", "आप", "नहीं", "क्या",
    "कैसे", "हिंदी", "भारत", "बहुत", "अच्छा", "बुरा",
}


def detect_language(text: str) -> str:
    """Return ISO-639-1 language code or descriptive name."""
    if not text or not text.strip():
        return "en"

    # Check for Devanagari script first
    dev_count = len(DEVANAGARI_RE.findall(text))
    if dev_count > 0:
        words = set(text.split())
        marathi_hits = len(words & MARATHI_WORDS)
        hindi_hits = len(words & HINDI_WORDS)
        if marathi_hits >= hindi_hits:
            return "mr"   # Marathi
        return "hi"       # Hindi

    # Fall back to langdetect
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        return lang
    except Exception:
        logger.warning("langdetect failed; defaulting to 'en'.")
        return "en"


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}


def language_display_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code.upper())
