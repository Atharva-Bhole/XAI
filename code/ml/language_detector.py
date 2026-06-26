"""
Language detection utility.
Uses langdetect as primary engine with a custom Hindi/Marathi heuristic.

Design notes (read before editing the word lists):
------------------------------------------------------------------
Romanized Hindi and Romanized Marathi share a large pool of function
words and Sanskrit-derived vocabulary ("kay/kya", "kuthe/kahan", "tu",
"aaj", "kal", "nahi/nhi", "ho", etc). If the *same* spelling is put into
both MARATHI_WORDS and HINDI_WORDS, it cannot discriminate between the
two languages -- it only adds noise and biases whichever language wins
ties. So:

  * SHARED_WORDS holds spellings that are genuinely ambiguous between
    Hindi and Marathi (or also common standalone English words like
    "ho", "to", "kal"-as-in-English-name). These are EXCLUDED from both
    language counts entirely. They still help detect "this is one of
    these two languages" generally without taking a side.
  * MARATHI_WORDS / HINDI_WORDS hold only spellings that are
    *distinctive* to that language -- i.e. not also valid, common
    spellings in the other. This is what lets the heuristic actually
    discriminate instead of guessing.
  * Decision uses a normalized score (hits / set size is NOT used --
    instead we just compare raw hit counts, but a tie with hits>0 in
    both no longer defaults to Marathi; see decide_language()).
  * "hinglish" is returned when both Hindi/Marathi distinctive words
    AND English content words are detected, indicating code-mixed text.
------------------------------------------------------------------
"""
import re
import string
import logging

logger = logging.getLogger(__name__)

# Devanagari unicode range: \u0900-\u097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

# ---------------------------------------------------------------------------
# SHARED / AMBIGUOUS words -- present in both Hindi and Marathi (or too
# generic / overlapping with English) -- deliberately excluded from both
# language-specific sets so they don't bias the tie-break either way.
# ---------------------------------------------------------------------------
SHARED_WORDS = {
    # Devanagari
    "नाही", "नहीं", "है", "आहे",  # both can render close variants if user mixes scripts
    # Romanized -- negation / copula / generic discourse markers common to both
    "nahi", "nhi", "nahin", "nai", "nako", "naako",
    "ho", "hoy", "hoye",
    "aaj", "kal", "abhi", "ata", "aata",  # "kal" = yesterday/tomorrow in both
    "kay", "kya",  # both languages use kay/kya forms for "what" colloquially in border dialects
    "tu", "mi", "to", "the", "main", "mat", "pan", # extremely short, high false-positive risk with English
    "pn", "par", "pr",  # "but" -- spelled almost identically, both languages
    "ani", "aani", "aur",  # technically distinct (ani=mr, aur=hi) but frequently cross-used in code-mixed text near the border; keep distinctive spellings below instead
    "bhai", "yaar", "boss", "scene", "set",  # Bollywood-influenced slang used identically in both
    "thoda", "thodi", "thode",  # "a little" -- identical spelling/meaning in both
    "sahi", "karte", "jate", "hota", "hoti", "hote",  # shared verb-form spellings
    "tujhe",  # Hindi dative "to you" vs Marathi informal possessive -- same spelling
    # Additional shared sentiment/slang words
    "mast", "badhiya", "badiya", "accha", "achha", "acha",
    "kharab", "bakwas", "bakwaas", "bekar", "bekaar",
    "dost", "bro", "dude",
}

# ---------------------------------------------------------------------------
# MARATHI -- words/spellings NOT commonly used in Hindi.
# ---------------------------------------------------------------------------
MARATHI_WORDS = {
    # Devanagari -- distinctive Marathi function words / verb forms
    "आहेत", "आणि", "हे", "हा", "ती", "तो", "मला", "तुला", "आपण",
    "काय", "कसे", "मराठी", "महाराष्ट्र", "छान", "चित्रपट", "नको",
    "तुझा", "माझा", "त्याचा", "तिचा", "आमचा", "तुमचा", "करतोय", "करतेय",
    "झाला", "झाली", "गेला", "गेली", "आला", "आली", "होता", "होती",

    # Romanized -- pronouns / possessives distinctive to Marathi
    "mala", "mla", "tula", "tla", "tyala", "tila", "amhala", "tumhala",
    "majha", "majhi", "majhe", "tujha", "tujhi",
    "tyacha", "tyachi", "tyache", "ticha", "tichi", "tiche",
    "amcha", "amchi", "amche", "tumcha", "tumchi", "tumche",
    "apla", "apli", "aple",
    "amhi", "tumhi", "te", "tya",

    # Romanized -- existential / copula forms distinctive to Marathi
    "ahe", "aahe", "ahes", "aahes", "ahot", "aahot", "aahet", "ahet",
    "hoto",

    # Romanized -- distinctive verb endings (-toy/-tey/-tay/-naar family)
    "karto", "kartay", "kartoy", "kartey", "krtoy", "krtey", "krtes",
    "jato", "jatoy", "jatey",
    "yeto", "yete", "yetoy", "yetey",
    "denar", "ghenar", "jananar", "honar", "karnar",
    "kela", "keli", "kele", "zala", "zali", "zale",
    "gela", "geli", "gele", "gelo",
    "ala", "ali", "ale", "alo",
    "jevla", "jevli", "jevle", "jevlas",
    "zopla", "zopli", "zople",

    # Romanized -- distinctive vocabulary / adjectives / question words
    "kasa", "kashi", "kase", "kuthe", "kadhi", "kiti", "kashala", "konala",
    "khup", "khoop", "bhaari", "bhari",
    "changla", "changli", "changle", "vait", "waeet",
    "mulgi", "mulga", "porga", "porgi",
    "divas", "ratra", "udya", "parva",
    "kaay", "konta", "konti", "kontya",
    "lavkar", "halu", "jara",
    "barobar", "agdi", "ekdum",
    "kr", "krto", "krte",  # very short SMS-style Marathi verb stems (low weight, but distinctive enough)

    # Additional distinctive Marathi words
    "mazya", "maza", "mazhi", "tumchya", "tyanchya",
    "mhanje", "mhanun", "mhantat", "mhantla", "mhantli",
    "basla", "basli", "basle", "ubha", "ubhi",
    "sangto", "sangte", "sangtoy", "sangtey",
    "vicharla", "vicharli", "vicharle",
    "mhanala", "mhanali",
    "sakkhat", "zakas", "jhakas", "jhakkas", "zhakaas",
    "kadak", "kadhak",
    "aapla", "aapli", "aaple",
    "chitrapat", "natak",
    "samjla", "samjli", "samjle",
    "pahila", "pahili", "pahile",
    "aikla", "aikli", "aikle",
    "apratim", "uttam",
    "vaitagla", "vaitagli", "kantala", "kantaala",
}

# ---------------------------------------------------------------------------
# HINDI -- words/spellings NOT commonly used in Marathi.
# ---------------------------------------------------------------------------
HINDI_WORDS = {
    # Devanagari -- distinctive Hindi function words / verb forms
    "हैं", "और", "यह", "वह", "मैं", "तुम", "आप", "क्या",
    "कैसे", "हिंदी", "भारत", "बहुत", "अच्छा", "बुरा",
    "मेरा", "तेरा", "उसका", "इसका", "हमारा", "तुम्हारा", "अपना",
    "रहा", "रही", "रहे", "गया", "गयी", "गए", "आया", "आयी", "आए",

    # Romanized -- pronouns / possessives distinctive to Hindi
    "mujhe", "humein", "humko", "tumko", "usko", "isko",
    "mera", "meri", "mere", "tera", "teri", "tere",
    "uska", "uski", "uske", "iska", "iski", "iske",
    "hamara", "hamari", "hamare", "tumhara", "tumhari", "tumhare",
    "apna", "apni", "apne",
    "hum", "tum", "aap", "woh", "yeh", "ye", "inko", "unko",

    # Romanized -- existential / copula forms distinctive to Hindi
    "hai", "hain", "hoon", "hun", "thi", "tha",

    # Romanized -- distinctive verb endings (-na/-ta hai family, -raha)
    "karna", "karta", "karti", "kar", "kiya", "ki", "kiye",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "aaya", "aayi", "aaye", "jaana", "jata", "jati",
    "hoga", "hogi", "honge", "hona",
    "dena", "deta", "deti", "dete", "lena", "leta", "leti", "lete",

    # Romanized -- distinctive vocabulary / question words
    "kaise", "kaisa", "kaisi", "kahan", "kab", "kitna", "kitni", "kitne",
    "kaun", "kiska", "kisko", "kyun", "kyon", "kuch", "sab", "sabhi",
    "bahut", "bhot", "bohot", "acchi",
    "bura", "buri", "bure", "theek", "thik",
    "ladka", "ladki", "bachcha", "bachche",
    "din", "raat", "subah", "shaam",
    "jaldi", "dheere", "zara",
    "matlab", "shayad", "zaroor", "lekin", "magar", "isliye", "kyunki",
    "fir", "phir", "toh", "tabhi", "wahi", "yahi",

    # Additional distinctive Hindi words
    "samajh", "samjho", "samjha", "samajhna",
    "batao", "bataao", "batana", "btao",
    "chalo", "chalte", "chalna", "chal",
    "dekho", "dekhna", "dekha", "dekhi",
    "suno", "sunna", "suna", "suni",
    "bolo", "bolna", "bola", "boli",
    "padha", "padhi", "padhna", "padho",
    "likha", "likhi", "likhna", "likho",
    "khana", "khaya", "khayi", "khao",
    "peena", "peeya", "piyo",
    "wala", "wali", "wale",
    "waala", "waali", "waale",
    "itna", "itni", "itne",
    "utna", "utni", "utne",
    "jaise", "jitna", "jitni",
    "bilkul", "sachmuch", "sachme",
    "paagal", "pagal", "pagla", "pagli",
}

# ---------------------------------------------------------------------------
# ENGLISH CONTENT WORDS -- used to detect code-mixed Hinglish text.
# These are common English content words (not function words) that
# indicate English-mixed speech when found alongside Hindi/Marathi words.
# ---------------------------------------------------------------------------
ENGLISH_CONTENT_WORDS = {
    "movie", "movies", "film", "song", "songs", "music", "video", "photo",
    "time", "work", "place", "food", "game", "phone", "class", "college",
    "school", "office", "party", "friend", "friends", "family", "people",
    "money", "problem", "life", "story", "thing", "things", "world",
    "feeling", "actually", "really", "totally", "seriously", "obviously",
    "literally", "basically", "honestly", "definitely", "absolutely",
    "performance", "acting", "direction", "dialogue", "dialogues",
    "camera", "review", "rating", "experience", "quality", "service",
    "product", "delivery", "amazing", "awesome", "perfect", "worst",
    "boring", "interesting", "funny", "cool", "nice", "fine",
}

# Precompute case where a word accidentally ends up in more than one set --
# fail loudly during development rather than silently mis-detecting.
_overlap_check = (MARATHI_WORDS & HINDI_WORDS) | (MARATHI_WORDS & SHARED_WORDS) | (HINDI_WORDS & SHARED_WORDS)
if _overlap_check:
    logger.warning(
        "Language word lists have unexpected overlap (%d words): %s",
        len(_overlap_check), sorted(_overlap_check)
    )


def _tokenize(text: str) -> set:
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return set(cleaned_text.split())


def detect_language(text: str) -> str:
    """Return ISO-639-1 language code, 'hinglish', or descriptive name."""
    if not text or not text.strip():
        return "en"

    # Check for Devanagari script first
    dev_count = len(DEVANAGARI_RE.findall(text))

    words = _tokenize(text)

    marathi_hits = len(words & MARATHI_WORDS)
    hindi_hits = len(words & HINDI_WORDS)
    shared_hits = len(words & SHARED_WORDS)
    english_hits = len(words & ENGLISH_CONTENT_WORDS)

    # Total Indic signal (distinctive + shared)
    indic_signal = marathi_hits + hindi_hits + shared_hits

    if dev_count > 0:
        if marathi_hits > hindi_hits:
            return "mr"
        if hindi_hits > marathi_hits:
            return "hi"
        # True tie on distinctive words (including 0-0): Devanagari script
        # with no distinctive hits at all is far more often Hindi in
        # general text, so prefer Hindi only as a script-level fallback.
        # If there were at least some shared hits, we still can't be sure,
        # but Devanagari + ambiguity defaults to Hindi (larger corpus/base rate).
        return "hi"

    # Romanized (Hinglish/Marlish) detection.
    # Only trust distinctive hits; shared_hits alone never decides the
    # language, it just confirms "this is Hindi-or-Marathi-ish text".

    # Detect code-mixed "Hinglish" -- both Indic and English content words present
    if indic_signal >= 1 and english_hits >= 1:
        # This is code-mixed text. Determine the Indic base language.
        if marathi_hits > hindi_hits and marathi_hits >= 2:
            return "mr"  # Marathi-English code-mix, treat as Marathi
        if hindi_hits > marathi_hits and hindi_hits >= 2:
            return "hinglish"
        if marathi_hits >= 2 and hindi_hits >= 2:
            # Genuine tie -- default to hinglish
            return "hinglish"
        # Some Indic signal + English = Hinglish (default to Hindi-base)
        if indic_signal >= 2:
            return "hinglish"

    if marathi_hits > hindi_hits and marathi_hits >= 2:
        return "mr"
    if hindi_hits > marathi_hits and hindi_hits >= 2:
        return "hi"
    if marathi_hits == hindi_hits and marathi_hits >= 2:
        # Genuine tie on distinctive words from both languages in the same
        # string (common in code-switched border dialects) -- without more
        # signal we can't pick a side reliably, so fall through to
        # langdetect rather than defaulting to one language.
        pass

    # Check if shared words alone suggest Indic language
    if shared_hits >= 3 and english_hits >= 1:
        return "hinglish"

    # Fall back to langdetect
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)

        # If langdetect incorrectly identifies Romanized Hindi/Marathi as
        # something unrelated, trust our distinctive-word signal if we have any.
        if lang not in ('en', 'hi', 'mr'):
            if marathi_hits > hindi_hits:
                return "mr"
            if hindi_hits > marathi_hits:
                return "hi"
            if shared_hits >= 2:
                # Ambiguous Hindi/Marathi signal with no langdetect support;
                # default to Hindi as the more common general case.
                return "hi"

        # If langdetect says Hindi but we also detected English content words,
        # it's likely Hinglish
        if lang == 'hi' and english_hits >= 2:
            return "hinglish"

        return lang
    except Exception:
        logger.warning("langdetect failed; defaulting to 'en'.")
        return "en"


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "hinglish": "Hinglish",
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
