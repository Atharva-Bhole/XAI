"""
Translation utility.
Translates detected non-English text to English using deep_translator (Google backend).
"""
import logging

logger = logging.getLogger(__name__)


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate *text* from *source_lang* to English. Returns original on failure."""
    if source_lang == "en" or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        return translated or text
    except Exception as exc:
        logger.warning("Translation failed (%s): %s — returning original text.", source_lang, exc)
        return text
