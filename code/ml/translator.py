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
        import re
        
        # Regex to match emojis (matching standard ranges including surrogate pairs on some systems)
        emoji_pattern = re.compile(
            r'['
            r'\U0001f600-\U0001f64f'  # emoticons
            r'\U0001f300-\U0001f5ff'  # symbols & pictographs
            r'\U0001f680-\U0001f6ff'  # transport & map symbols
            r'\U0001f1e0-\U0001f1ff'  # flags (iOS)
            r'\u2702-\u27b0'          # dingbats
            r'\u24c2-\U0001f251'
            r']+', flags=re.UNICODE
        )
        
        # 1. Extract and replace emojis
        extracted_emojis = []
        
        def replace_with_placeholder(match):
            extracted_emojis.append(match.group(0))
            return f" __EMOJI_{len(extracted_emojis) - 1}__ "
            
        sanitized_text = emoji_pattern.sub(replace_with_placeholder, text)
        
        # 2. Translate text (Google Translate handles placeholders well if spaced out)
        translated = GoogleTranslator(source=source_lang, target="en").translate(sanitized_text)
        if not translated:
            return text
            
        # 3. Restore emojis
        for i, emoji_chars in enumerate(extracted_emojis):
            translated = translated.replace(f"__EMOJI_{i}__", emoji_chars)
            # Sometimes translation strips underscores or changes capitalization
            translated = translated.replace(f"__emoji_{i}__", emoji_chars)
            translated = translated.replace(f"__Emoji_{i}__", emoji_chars)
            translated = translated.replace(f"__ EMOJI_{i} __", emoji_chars)
            translated = translated.replace(f"__EMOJI_ {i}__", emoji_chars)
            
        # Clean up any leftover double spaces around emojis
        translated = translated.replace("  ", " ").strip()
        
        return translated
    except Exception as exc:
        logger.warning("Translation failed (%s): %s — returning original text.", source_lang, exc)
        return text
