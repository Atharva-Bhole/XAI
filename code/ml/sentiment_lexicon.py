"""
Hindi/Marathi/Hinglish sentiment lexicon.

Design notes:
------------------------------------------------------------------
Unlike lang_detect.py's word lists (which must stay small and
*distinctive* per language to avoid collisions), this lexicon is
deliberately language-agnostic and as exhaustive as practical. A
sentiment word's polarity doesn't depend on whether it "belongs" to
Hindi or Marathi -- "mast" means "great" whether a Hindi speaker or
a Marathi speaker types it, so it gets one polarity entry, not two
language-tagged copies.

This sidesteps the core problem from the language-ID lists: there
*is* no clean way to assign many Hindi-belt slang/sentiment words to
one language exclusively, but for sentiment scoring we don't need to
-- we only need to know "is this word positive, negative, a negator,
or an intensifier", which is a strictly easier and safer question to
answer exhaustively.

All entries are single tokens. score_sentiment() tokenizes on
whitespace, so a multi-word string like "thank you" or "kya baat hai"
can never match as one token -- it would just be dead weight in the
set. Phrases are now handled via PHRASE_PATTERNS (regex-based) which
runs *before* token-level scoring.

Spelling variation handling:
  Romanization isn't standardized (one word can be spelled several
  ways: doubled vowels, dropped vowels, transliterated retroflex
  consonants, etc). Rather than hand-listing every variant forever,
  this module:
    1. Lists the most common variants directly.
    2. Applies a light normalization step (collapse 2+ repeated
       letters down to 1, strip trailing punctuation) before lookup,
       so unlisted variants like "masttt" or "achhhha" still often
       resolve correctly.

Negation & intensifiers:
  "nahi mast hai" (not great) must flip polarity, and "khup mast"
  (very great) should weight it up. NEGATION_WORDS and
  INTENSIFIER_WORDS support a windowed negation/intensity scan in
  score_sentiment(). Negation window is 3 tokens. Intensifiers
  multiply the score of the following sentiment word by 1.5×.

Contrast conjunctions:
  Words like "par", "lekin", "magar" signal a shift -- the clause
  *after* the contrast word often carries the speaker's true
  sentiment ("achha hai par bahut slow" → negative). CONTRAST_WORDS
  triggers post-contrast weighting (1.3×) in score_sentiment().

Emoji sentiment:
  EMOJI_SENTIMENT maps common emoji characters to float scores
  in [-1.0, +1.0], accumulated alongside word-level scores.

Phrase patterns:
  PHRASE_PATTERNS is a list of (compiled_regex, score) tuples for
  multi-word Hinglish/Hindi/Marathi idioms. Checked first, before
  word-level tokenization.
------------------------------------------------------------------
"""
import re
import string
from collections import namedtuple

# ---------------------------------------------------------------------------
# POSITIVE sentiment words (Hindi + Marathi + Hinglish slang, Devanagari +
# Romanized). Language-agnostic by design -- see module docstring.
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    # ---- Devanagari positives ----
    "अच्छा", "अच्छी", "अच्छे", "बढ़िया", "उम्दा", "शानदार", "उत्तम", "खुश",
    "खुशी", "प्रसन्न", "सुंदर", "बेहतरीन", "मस्त", "जबरदस्त", "लाजवाब",
    "छान", "उत्कृष्ट", "आनंद", "आनंदी", "सुखद", "प्रिय", "प्यारा", "प्यारी",
    "धन्यवाद", "शुक्रिया", "आभारी", "सही", "बरोबर", "मजा", "मज्जा", "मजेदार",
    "हुशार", "प्रेम", "प्यार", "वाह", "कमाल", "स्वागत",
    # Additional Devanagari positives
    "प्रशंसा", "सराहना", "तारीफ", "सफल", "जीत", "विजय", "गर्व", "इज्जत",
    "शान", "जोश", "उत्साह", "प्रेरणा", "हौसला", "हिम्मत", "भरोसा", "विश्वास",
    "समृद्ध",

    # ---- Romanized -- general "good/nice/great" register ----
    "accha", "achha", "acha", "achaa", "achi", "acchi", "achhi",
    "ache", "acche", "acchhe",
    "badhiya", "badiya", "umda", "shaandar", "shandar", "shandaar",
    "uttam", "behtareen", "behtarin",
    "lajawab", "lajavab", "lajavaab", "jabardast", "zabardast",
    "changla", "changli", "changle", "changlay", "changlaa", "changlaay",
    "chaan", "chan", "utkrisht",
    "mast", "zakas", "jhakas", "jhakkas", "zhakaas",
    "bhari", "bhaari", "bhaaree",
    "sahi", "sahii",
    "kadak", "kadhak",
    "lavkar", "barobar",
    "sundar", "sunder", "khoobsurat", "khubsurat",
    "pyaar", "pyar", "pyara", "pyari", "pyare", "preet",
    "khush", "khushi", "khushhal", "anand", "anandi",
    "maja", "mazza", "mazedaar", "majedaar",
    "dhanyavad", "dhanyawad", "shukriya", "abhari",
    "hushar", "hoshar", "smart",
    "awesome", "best", "love", "loved", "amazing", "great", "nice", "good",
    "fantastic", "wonderful", "excellent", "superb", "brilliant", "perfect",
    "happy", "thanks",
    "vah", "wah", "waah", "kamaal", "kamal",
    "lovely", "cute", "sweet", "top", "topclass",
    "khaas", "khas",
    "yogya",
    "swagat", "welcome",

    # ---- Hinglish social media slang positives ----
    "lit", "fire", "solid", "dope", "chill", "op", "savage", "slay",
    "vibe", "vibes", "sick", "killing", "killed", "nailed", "crushed",
    "rocked", "slaps", "banger", "goat", "goated", "based", "bussin",
    "valid", "peak", "insane", "mental", "unreal", "clutch", "iconic",
    "legendary", "epic",

    # ---- More Romanized Hindi positives ----
    "khubsurat", "dilchasp", "mazedaar", "rochak", "manoranjak",
    "safal", "kamyab", "jeet", "jeeta", "jeeti", "jiit",
    "shandar", "atiuttam", "sarvottam", "achcha", "bahut",
    "shandaar", "bemisaal", "anokha", "apaar",
    "zindadil", "josh", "josheela", "umang", "utsaah",
    "prerna", "prerit", "hausla", "himmat", "izzat", "shaan",

    # ---- More Romanized Marathi positives ----
    "apratim", "avishkar", "yashashvi", "yashasvi",
    "samadhan", "samadhankaarak", "harkhun", "harakh",
    "prem", "jiv", "jivaab", "jivaabhar", "shaabasski",
    "abhimaan", "bhavishya", "umeed", "ummeed",
    "vishwas", "bharosa", "sampann", "samruddh",

    # ---- Celebratory ----
    "jai", "zindabad", "shabaash", "shabash", "shaabaash",
    "badhai", "vadhdivas", "congratulations", "congrats", "bravo",

    # ---- Agreement / approval expressions ----
    "sach", "sachchi", "sacchi", "bilkul", "zaroor", "zarur",
    "pakka", "pakkaa", "haan", "ha", "ji", "ekdum", "agdi",
}

# ---------------------------------------------------------------------------
# NEGATIVE sentiment words
# ---------------------------------------------------------------------------
NEGATIVE_WORDS = {
    # ---- Devanagari negatives ----
    "बुरा", "बुरी", "बुरे", "खराब", "गंदा", "गंदी", "गंदे", "घटिया", "बेकार",
    "निराश", "दुखी", "दुख", "गुस्सा", "नाराज", "क्रोध", "वाईट", "त्रास",
    "नापसंद", "नफरत", "घृणा", "बकवास", "फालतू", "बोरिंग", "थकाऊ",
    "डर", "डरावना", "चिंता", "परेशान", "तकलीफ", "दर्द", "वैताग",
    # Additional Devanagari negatives
    "निराशा", "मायूस", "थका", "थकी", "तनाव", "झगड़ा", "लड़ाई", "हिंसा",
    "ज़ुल्म", "अन्याय", "अत्याचार", "शोषण", "अफ़सोस", "पछतावा", "मुसीबत",
    "परेशानी", "हार", "पराजय", "निष्फल",

    # ---- Romanized negatives ----
    "kharab", "kharaab",
    "ganda", "gandi", "gande", "gandagi",
    "ghatiya", "ghatia", "ghatiyaa",
    "bekar", "bekaar", "faltu", "faltoo", "fizul", "fizool",
    "bakwas", "bakwaas", "bakar",
    "boring", "thakau", "thakaau",
    "nirash", "niraash", "dukhi", "dukh", "dukhad",
    "gussa", "gussha", "naraaz", "naraz", "krodh",
    "vait", "waeet", "vaeet", "vaiit",
    "tras", "trass",
    "napasand", "napasandi",
    "nafrat", "nafrath", "ghrina", "ghrna",
    "dar", "daravna", "daravana", "bhayanak",
    "chinta", "chintit", "tension",
    "pareshan", "pareshaan", "takleef", "taklif",
    "dard", "dukhta", "dukhti",
    "bura", "buri", "bure", "burai",
    "ghin", "ghinerda", "ghinercha",
    "kantala", "kantaal",
    "vaitaag", "vaitag", "vaitagla", "vaitagli",
    "vaitlay", "vaitla", "vaitli",
    "hopeless", "useless", "worst", "bad", "sad", "angry", "annoying",
    "terrible", "horrible", "awful", "pathetic", "disgusting", "hate", "hated",
    "disappoint", "disappointed", "disappointing", "fail", "failed", "failure",
    "rubbish", "nonsense", "waste", "wasteful",

    # ---- Hinglish abusive / harsh ----
    "wahiyat", "wahiyaat", "ghatiya", "kamina", "kamini", "kamne",
    "harami", "haramkhor", "gadha", "gadhe", "ullu",
    "bevkoof", "bevakoof", "bewkoof", "buddhu",
    "gaddar", "makkar", "dhokha", "dhoka", "dhokhebaaz", "dhokhebaaj",

    # ---- Emotional distress ----
    "rona", "roye", "roi", "royee", "rula", "rulaya",
    "taklif", "takleef", "kasht", "kashta", "peeda", "peedaa",
    "vyatha", "tanav", "tanaav", "stress", "depression", "depressed",
    "anxiety", "anxious", "loneliness",
    "akela", "akeli", "akele", "tanha", "tanhaai",

    # ---- Conflict / violence ----
    "jhagda", "jhagra", "ladai", "ladaai",
    "maara", "maari", "maarpeet",
    "hinsa", "zulm", "anyaay", "anyay",
    "atyachaar", "atyachar", "shoshit", "shoshan",

    # ---- Disappointment / frustration ----
    "nirasha", "niraasha", "mayus", "maayus",
    "thaka", "thaki", "thak", "tang", "tangg",
    "jhanjhat", "jhanjhaat", "pareshani", "musibat", "musibaat",
    "afsos", "afasos", "pachtava", "pachtaava",

    # ---- More Romanized Marathi negatives ----
    "vaeet", "vait", "khalas", "khallaas",
    "dhokebaaz", "bhayaanak",
    # ---- Violence, Fight, Injury, Blood, Crime, Death ----
    "maramari", "marhan", "maarpeet", "marpeet", "mara", "mari", "marla", "marli", "marlo", "hanla", "badavla", "thopla",
    "rakta", "rakt", "bleeding", "blood", "jakham", "jakhmi", "zakhmi", "ghayaal", "ghayal",
    "khun", "khoon", "murder", "qatl", "katl", "hatya", "shatru", "dushman", "dushmani", "vair", "badla",
    "hospital", "icu", "serious", "gambhir", "death", "mele", "mela", "meli", "mayat", "shav", "laash", "lash", "dead",
    "suicide", "atmaghaat", "accident", "apghaat", "apghat", "fracture", "dard", "vedna", "khalas", "nighala",
    "chaku", "talwar", "goli", "golibar", "firing", "bomb", "sphot", "dhamaka", "threat", "dhamki",
    # ---- Abuses & Slang Curses ----
    "bhadva", "bhadve", "bhadvyachya", "chutiya", "chutya", "yedzava", "yedzavya", "yedjawa", "gaand", "gand",
    "lavda", "lawda", "bocha", "raand", "randi", "saala", "sala", "harami", "haramkhor", "nalayak",
    "kutta", "kutre", "kutrya", "suar", "dukkar", "gadha", "bhikari", "darudya", "nashedi", "bakchod",
    "madarchod", "behenchod", "benchod", "bchod", "bhosdike", "bsdk", "mc", "bc", "gandu",
    # ---- Cheating, Betrayal, Corruption ----
    "lutla", "lutli", "lootle", "fasaav", "fasavla", "fasavli", "fasvnuk", "chori", "robbery", "bhrashtachar",
    "scam", "ghotala", "jhol", "fraud", "nakli", "fake", "jhut", "jhutha", "khota", "khoti", "khote", "laach", "rishwat",
}

# ---------------------------------------------------------------------------
# NEGATION words -- flip polarity of nearby sentiment word.
# Multi-word negations are decomposed: the distinctive marker word is
# included on its own so token-level lookup still works.
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    # Core single-word negations
    "nahi", "nhi", "nahin", "nai", "na", "nako", "naako", "naa",
    "mat", "nakos", "nakoos",
    # Devanagari
    "नाही", "नहीं", "ना", "मत",
    # Additional negation markers
    "hargiz", "never", "nope",
    "kabhi",      # part of "kabhi nahi"
    "kadhi",      # part of "kadhi nahi"
    "kahi",       # part of "kahi nahi"
    "mushkil",    # "impossible" sense in context
    "impossible",
}

# ---------------------------------------------------------------------------
# INTENSIFIER words -- amplify the nearby sentiment word's weight.
# An intensifier immediately preceding a sentiment word multiplies
# that word's contribution by INTENSIFIER_MULTIPLIER (1.5×).
# ---------------------------------------------------------------------------
INTENSIFIER_MULTIPLIER = 1.5

INTENSIFIER_WORDS = {
    "khup", "khoop", "bahut", "bohot", "bhot", "atishay", "atyant",
    "ekdum", "agdi", "purn", "jastach", "bilkul", "itna", "itni", "itne",
    "kiti", "kititari",
    # Devanagari
    "बहुत", "खूप", "अत्यंत", "बिलकुल",
    # Expanded intensifiers
    "ati", "sakkhat", "sachchi", "sachmuch", "sacchi",
    "poora", "puri", "pure", "pura",
    "seedha", "full", "totally", "completely",
    "bharpoor", "bharpur", "jabardast", "zabardast",
    "zyada", "jyada",
    "sachme", "sachmein", "asli",
    "waakai", "wakai", "kafi",
    "kamaal", "kamal",
    # Marathi slang intensifiers (lay / khoop)
    "lay", "laay", "lai", "laai", "laii", "khoopch", "khupch",
}

# ---------------------------------------------------------------------------
# CONTRAST words -- conjunctions that signal a sentiment shift.
# The clause after the contrast word is weighted 1.3× since it often
# carries the speaker's true/dominant sentiment.
# ---------------------------------------------------------------------------
CONTRAST_WEIGHT = 1.3

CONTRAST_WORDS = {
    "par", "lekin", "magar", "parantu", "kintu",
    "haalanki", "halanki",
    "but", "however", "although", "yet", "though",
    "pn", "pan",
    # Multi-word contrast markers -- also include component words that
    # serve as contrast cues on their own:
    "phir", "toh", "fir",  # from "phir bhi", "toh bhi", "fir bhi"
    "bhi",
}

# ---------------------------------------------------------------------------
# EMOJI_SENTIMENT -- map common emoji characters to float scores [-1, +1].
# Accumulated alongside word-level scores in score_sentiment().
# ---------------------------------------------------------------------------
EMOJI_SENTIMENT = {
    # Positive emoji
    "😊": 0.8,  "😍": 0.9,  "❤️": 0.8,  "🥰": 0.9,
    "😂": 0.6,  "🤣": 0.6,  "👍": 0.7,  "🎉": 0.8,
    "🔥": 0.7,  "💯": 0.8,  "✨": 0.6,  "🙌": 0.7,
    "👏": 0.7,  "💪": 0.6,  "🤗": 0.7,  "😎": 0.6,
    "🥳": 0.8,  "💖": 0.8,  "💕": 0.7,  "😀": 0.7,
    "😃": 0.7,  "😄": 0.8,  "😁": 0.7,  "🤩": 0.8,
    "👌": 0.7,  "✅": 0.6,  "🏆": 0.7,  "🌟": 0.7,
    "💐": 0.6,  "🎊": 0.7,
    # Also handle bare ❤ without variation selector
    "❤": 0.8,

    # Negative emoji
    "😢": -0.8,  "😭": -0.9,  "😡": -0.9,  "🤬": -1.0,
    "😤": -0.7,  "💔": -0.8,  "😞": -0.7,  "😔": -0.7,
    "😠": -0.8,  "👎": -0.8,  "🤮": -0.9,  "😰": -0.6,
    "😨": -0.7,  "😱": -0.8,  "🙄": -0.4,  "😒": -0.5,
    "😩": -0.6,  "😫": -0.7,  "💀": -0.3,  "🤢": -0.7,
    "☠️": -0.5,  "😿": -0.6,
    # Also handle bare ☠ without variation selector
    "☠": -0.5,
}

# ---------------------------------------------------------------------------
# PHRASE_PATTERNS -- (compiled_regex, sentiment_score) tuples for multi-word
# Hindi/Marathi/Hinglish idioms. Checked on the full (lowered) text *before*
# word-level tokenization, so multi-word expressions work correctly.
# Patterns use \b word boundaries and are case-insensitive.
# ---------------------------------------------------------------------------
PHRASE_PATTERNS = [
    # ---- Positive phrases ----
    (re.compile(r'\bkya\s+baat\b', re.IGNORECASE),          +0.8),
    (re.compile(r'\bek\s+number\b', re.IGNORECASE),         +0.9),
    (re.compile(r'\bek\s+no\b', re.IGNORECASE),             +0.9),
    (re.compile(r'\btop\s+class\b', re.IGNORECASE),         +0.8),
    (re.compile(r'\bfirst\s+class\b', re.IGNORECASE),       +0.8),
    (re.compile(r'\bnext\s+level\b', re.IGNORECASE),        +0.8),
    (re.compile(r'\bout\s+of\s+this\s+world\b', re.IGNORECASE), +0.9),
    (re.compile(r'\bdil\s+khush\b', re.IGNORECASE),         +0.9),
    (re.compile(r'\bmann\s+khush\b', re.IGNORECASE),        +0.8),
    (re.compile(r'\bbohot\s+acha\b', re.IGNORECASE),        +0.9),
    (re.compile(r'\bbahut\s+achh?a\b', re.IGNORECASE),      +0.9),
    (re.compile(r'\bkhup\s+chan\b', re.IGNORECASE),          +0.9),
    (re.compile(r'\bkhup\s+changla\b', re.IGNORECASE),      +0.9),

    # ---- Negative phrases ----
    (re.compile(r'\bscene\s+kharab\b', re.IGNORECASE),      -0.7),
    (re.compile(r'\bhaal\s+kharab\b', re.IGNORECASE),       -0.8),
    (re.compile(r'\bband\s+baj\b', re.IGNORECASE),          -0.7),
    (re.compile(r'\bdimag\s+kharab\b', re.IGNORECASE),      -0.6),
    (re.compile(r'\bdimag\s+ka\s+dahi\b', re.IGNORECASE),   -0.5),
    (re.compile(r'\bpaisa\s+barbaad\b', re.IGNORECASE),     -0.8),
    (re.compile(r'\btime\s+waste\b', re.IGNORECASE),        -0.7),
    (re.compile(r'\btime\s+barbaad\b', re.IGNORECASE),      -0.7),
    (re.compile(r'\bkuch\s+nahi\b', re.IGNORECASE),         -0.3),
    (re.compile(r'\baur\s+kuch\s+nahi\b', re.IGNORECASE),   -0.5),
    (re.compile(r'\bachh?a\s+nahi\b', re.IGNORECASE),       -0.7),
    (re.compile(r'\bchangla\s+nahi\b', re.IGNORECASE),      -0.7),
    (re.compile(r'\btheek\s+nahi\b', re.IGNORECASE),        -0.5),
    (re.compile(r'\bdok[ae]\s+kharab\b', re.IGNORECASE),     -0.8),
    (re.compile(r'\bdokya\s+kharab\b', re.IGNORECASE),      -0.8),
    (re.compile(r'\bdok[ae]\s+satak\b', re.IGNORECASE),      -0.8),
    (re.compile(r'\bsatakl[ia]\b', re.IGNORECASE),          -0.8),
    (re.compile(r'\bm[uo]od\s+kharab\b', re.IGNORECASE),    -0.7),
    (re.compile(r'\bmara\s*mari\b', re.IGNORECASE),         -0.9),
    (re.compile(r'\brakt[a]?\s+nighal[a]?\b', re.IGNORECASE), -0.9),
    (re.compile(r'\bkhup\s+rakt[a]?\b', re.IGNORECASE),     -0.9),
    (re.compile(r'\bmaar\s*peet\b', re.IGNORECASE),         -0.8),
    (re.compile(r'\bmurder\s+zala\b', re.IGNORECASE),       -1.0),
    (re.compile(r'\bkhun\s+zala\b', re.IGNORECASE),         -1.0),
    (re.compile(r'\baccident\b', re.IGNORECASE),            -0.8),

    # ---- Neutral phrases ----
    (re.compile(r'\btheek\s+thaak\b', re.IGNORECASE),       0.0),
    (re.compile(r'\bthik\s+tha?k\b', re.IGNORECASE),        0.0),
    (re.compile(r'\btime\s*pass\b', re.IGNORECASE),         0.0),
]

# ---------------------------------------------------------------------------
# SentimentResult -- returned by score_sentiment().
# Fields:
#   label        : str   -- "positive", "negative", or "neutral"
#   pos_hits     : int   -- count of positive signals (words + phrases + emoji)
#   neg_hits     : int   -- count of negative signals
#   details      : list  -- per-signal detail tuples
#   confidence   : float -- 0.0-1.0, based on how many sentiment signals found
#   phrase_hits  : list  -- matched phrase patterns
# ---------------------------------------------------------------------------
SentimentResult = namedtuple(
    "SentimentResult",
    ["label", "pos_hits", "neg_hits", "details", "confidence", "phrase_hits"],
)


def _normalize_word(word: str) -> str:
    """Normalize a romanized word for fuzzy lexicon lookup.

    Steps:
      1. Collapse any letter repeated 2+ times down to 1.
         ("baadhiya" → "badhiya", "achhhha" → "acha", "mastttt" → "mast")
      2. Strip trailing ASCII punctuation noise.
      3. Return lowered result (caller already lowers, but be safe).

    Note: Devanagari tokens pass through unchanged since repeated
    Devanagari characters are extremely rare in real text.
    """
    # Strip trailing punctuation (periods, commas, exclamation, etc.)
    word = word.rstrip(string.punctuation)
    # Collapse 2+ repeated characters → 1
    word = re.sub(r'(.)\1+', r'\1', word)
    return word.lower()


# Self-normalize all lexicon sets so double-letter spelling variations match seamlessly
POSITIVE_WORDS = {_normalize_word(w) for w in POSITIVE_WORDS} | POSITIVE_WORDS
NEGATIVE_WORDS = {_normalize_word(w) for w in NEGATIVE_WORDS} | NEGATIVE_WORDS
INTENSIFIER_WORDS = {_normalize_word(w) for w in INTENSIFIER_WORDS} | INTENSIFIER_WORDS
NEGATION_WORDS = {_normalize_word(w) for w in NEGATION_WORDS} | NEGATION_WORDS
CONTRAST_WORDS = {_normalize_word(w) for w in CONTRAST_WORDS} | CONTRAST_WORDS


def _tokenize(text: str) -> list:
    """Tokenize text: lowercase, strip punctuation, split on whitespace."""
    cleaned = text.lower().translate(str.maketrans('', '', string.punctuation))
    return cleaned.split()


def _extract_emoji_scores(text: str) -> list:
    """Scan the raw text for emoji characters and return (emoji, score) pairs."""
    hits = []
    for char_or_seq, score in EMOJI_SENTIMENT.items():
        if char_or_seq in text:
            # Count occurrences (handles repeated emoji)
            count = text.count(char_or_seq)
            for _ in range(count):
                hits.append((char_or_seq, score))
    return hits


def _match_phrases(text: str) -> list:
    """Match PHRASE_PATTERNS against the full (lowered) text.
    Returns a list of (matched_string, score) tuples.
    """
    lowered = text.lower()
    hits = []
    for pattern, score in PHRASE_PATTERNS:
        match = pattern.search(lowered)
        if match:
            hits.append((match.group(), score))
    return hits


def score_sentiment(text: str) -> SentimentResult:
    """
    Score sentiment for Hindi/Marathi/Hinglish (or English) text.
    Returns SentimentResult(label, pos_hits, neg_hits, details,
                            confidence, phrase_hits).

    Processing order:
      1. PHRASE_PATTERNS   -- regex on full text, multi-word idioms
      2. EMOJI_SENTIMENT   -- scan raw text for emoji characters
      3. Word-level scoring with negation (window=3) and intensifier
         (1.5× multiplier) handling
      4. CONTRAST_WORDS    -- if present, post-contrast clause is
         weighted 1.3× (it usually carries the true sentiment)

    Label is one of 'positive', 'negative', 'neutral'.
    Confidence is 0.0-1.0, proportional to the number of sentiment
    signals found (capped at 1.0 when ≥5 signals are present).
    """
    if not text or not text.strip():
        return SentimentResult("neutral", 0, 0, [], 0.0, [])

    pos_score = 0.0
    neg_score = 0.0
    details = []
    phrase_hits_list = []

    # ------------------------------------------------------------------
    # Step 1: Phrase-level scoring (multi-word idioms)
    # ------------------------------------------------------------------
    phrase_hits = _match_phrases(text)
    text_for_words = text
    for phrase_text, pscore in phrase_hits:
        phrase_hits_list.append((phrase_text, pscore))
        # Strip matched phrase from text so individual words aren't double counted
        text_for_words = re.sub(re.escape(phrase_text), ' ', text_for_words, flags=re.IGNORECASE)
        if pscore > 0:
            pos_score += pscore
            details.append((phrase_text, "phrase-positive"))
        elif pscore < 0:
            neg_score += abs(pscore)
            details.append((phrase_text, "phrase-negative"))
        else:
            details.append((phrase_text, "phrase-neutral"))

    # ------------------------------------------------------------------
    # Step 2: Emoji scoring
    # ------------------------------------------------------------------
    emoji_hits = _extract_emoji_scores(text)
    for emoji_char, escore in emoji_hits:
        if escore > 0:
            pos_score += escore
            details.append((emoji_char, "emoji-positive"))
        else:
            neg_score += abs(escore)
            details.append((emoji_char, "emoji-negative"))

    # ------------------------------------------------------------------
    # Step 3: Word-level scoring with negation + intensifier handling
    # ------------------------------------------------------------------
    raw_tokens = _tokenize(text_for_words)
    tokens = [_normalize_word(t) for t in raw_tokens]

    # Locate contrast word positions for post-contrast weighting
    contrast_indices = set()
    for i, tok in enumerate(tokens):
        if tok in CONTRAST_WORDS:
            contrast_indices.add(i)

    # Determine if there is a contrast word; if so, all tokens after
    # the *last* contrast word get a weighting boost.
    last_contrast_idx = max(contrast_indices) if contrast_indices else -1

    for i, tok in enumerate(tokens):
        # Record intensifiers in details but don't score them alone
        if tok in INTENSIFIER_WORDS:
            details.append((raw_tokens[i], "intensifier"))
            continue

        # Record contrast words in details
        if tok in CONTRAST_WORDS:
            details.append((raw_tokens[i], "contrast"))
            continue

        is_pos = tok in POSITIVE_WORDS
        is_neg = tok in NEGATIVE_WORDS
        if not is_pos and not is_neg:
            continue

        # --- Negation check (window = 3 tokens on each side) ---
        window_start = max(0, i - 3)
        window_end = min(len(tokens), i + 4)
        negated = any(
            t in NEGATION_WORDS
            for t in tokens[window_start:i] + tokens[i + 1:window_end]
        )

        # --- Intensifier check (immediately preceding token) ---
        intensified = (i > 0 and tokens[i - 1] in INTENSIFIER_WORDS)
        multiplier = INTENSIFIER_MULTIPLIER if intensified else 1.0

        # --- Contrast weighting (post-contrast clause gets boost) ---
        if last_contrast_idx >= 0 and i > last_contrast_idx:
            multiplier *= CONTRAST_WEIGHT

        # --- Accumulate score ---
        if is_pos:
            if negated:
                neg_score += 1.0 * multiplier
                details.append(
                    (raw_tokens[i], "positive-word-negated->negative")
                )
            else:
                pos_score += 1.0 * multiplier
                details.append((raw_tokens[i], "positive"))
        else:  # is_neg
            if negated:
                pos_score += 1.0 * multiplier
                details.append(
                    (raw_tokens[i], "negative-word-negated->positive")
                )
            else:
                neg_score += 1.0 * multiplier
                details.append((raw_tokens[i], "negative"))

    # ------------------------------------------------------------------
    # Step 4: Determine label
    # ------------------------------------------------------------------
    if pos_score == 0.0 and neg_score == 0.0:
        label = "neutral"
    elif pos_score > neg_score:
        label = "positive"
    elif neg_score > pos_score:
        label = "negative"
    else:
        # Equal nonzero pos/neg -- mixed sentiment, no clean majority
        label = "neutral"

    # ------------------------------------------------------------------
    # Step 5: Confidence score (0.0-1.0)
    # Based on total number of sentiment signals found.
    # 0 signals → 0.0, 1 → 0.2, 2 → 0.4, ... 5+ → 1.0
    # ------------------------------------------------------------------
    total_signals = len([d for d in details if d[1] not in
                         ("intensifier", "contrast")])
    confidence = min(1.0, total_signals * 0.2)

    # Round scores to int for the hit counts (backward compat)
    return SentimentResult(
        label,
        int(round(pos_score)),
        int(round(neg_score)),
        details,
        round(confidence, 2),
        phrase_hits_list,
    )
