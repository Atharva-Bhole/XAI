"""
Emotion-first sentiment analysis.

Primary model: j-hartmann/emotion-english-distilroberta-base
Fallback model: cardiffnlp/twitter-roberta-base-sentiment-latest (mapped to emotions)

The models are lazy-loaded on first use and cached in memory.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

_emotion_pipeline = None
_polarity_pipeline = None

_EMOTION_LABEL_MAP = {
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "neutral": "Calm",
    "sadness": "Sad",
    "surprise": "Surprised",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
}

_POLARITY_LABEL_MAP = {
    "positive": "Happy",
    "negative": "Sad",
    "neutral": "Calm",
    "label_0": "Sad",
    "label_1": "Calm",
    "label_2": "Happy",
}


def _get_emotion_pipeline(local_files_only: bool = False):
    global _emotion_pipeline
    if _emotion_pipeline is None:
        logger.info("Loading emotion model …")
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification

            _model_name = "j-hartmann/emotion-english-distilroberta-base"
            tokenizer = AutoTokenizer.from_pretrained(_model_name, local_files_only=local_files_only)
            model = AutoModelForSequenceClassification.from_pretrained(_model_name, local_files_only=local_files_only)
            _emotion_pipeline = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Emotion model loaded.")
        except Exception as exc:
            logger.warning("Failed to load emotion model: %s", exc)
            _emotion_pipeline = None
    return _emotion_pipeline


def _get_polarity_pipeline(local_files_only: bool = False):
    global _polarity_pipeline
    if _polarity_pipeline is None:
        logger.info("Loading polarity fallback model …")
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification

            _model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            tokenizer = AutoTokenizer.from_pretrained(_model_name, local_files_only=local_files_only)
            model = AutoModelForSequenceClassification.from_pretrained(_model_name, local_files_only=local_files_only)
            _polarity_pipeline = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=512,
            )
        except Exception as exc:
            logger.warning("Failed to load polarity fallback model: %s", exc)
            _polarity_pipeline = None
    return _polarity_pipeline


def preload_sentiment_models(preload_fallback: bool = False, local_files_only: bool = False) -> bool:
    """Warm-load sentiment models into memory at app startup."""
    ok = _get_emotion_pipeline(local_files_only=local_files_only) is not None
    if preload_fallback:
        ok = (_get_polarity_pipeline(local_files_only=local_files_only) is not None) and ok
    return ok


def _normalize_emotion_label(raw: str) -> str:
    return _EMOTION_LABEL_MAP.get(raw.lower(), raw.capitalize())


def _normalize_polarity_to_emotion(raw: str) -> str:
    return _POLARITY_LABEL_MAP.get(raw.lower(), raw.capitalize())


def _emotion_to_polarity(emotion_scores: Dict[str, float]) -> Dict[str, float]:
    """Derive positive/negative/neutral distribution from emotion scores.

    Uses temperature-scaled softmax to sharpen the dominant category so that
    clear-cut inputs produce decisive scores (e.g. >85 % for the winner).
    """
    import math

    positive = (
        float(emotion_scores.get("happy", 0.0))
        + 0.7 * float(emotion_scores.get("surprised", 0.0))
    )
    negative = (
        float(emotion_scores.get("sad", 0.0))
        + float(emotion_scores.get("angry", 0.0))
        + float(emotion_scores.get("fear", 0.0))
        + float(emotion_scores.get("disgust", 0.0))
    )
    neutral = (
        float(emotion_scores.get("calm", 0.0))
        + 0.3 * float(emotion_scores.get("surprised", 0.0))
    )

    total = positive + negative + neutral
    if total <= 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    # Normalize to [0, 1]
    raw = {
        "positive": positive / total,
        "negative": negative / total,
        "neutral": neutral / total,
    }

    # Temperature-scaled softmax sharpening (lower temp → sharper peaks)
    _TEMP = 0.4
    exp_vals = {k: math.exp(v / _TEMP) for k, v in raw.items()}
    exp_total = sum(exp_vals.values())

    return {k: round(v / exp_total, 4) for k, v in exp_vals.items()}


def _canonicalize_emotion_scores(emotion_scores: Dict[str, float]) -> Dict[str, float]:
    ordered = ["happy", "sad", "angry", "calm", "fear", "surprised", "disgust"]
    out = {k: float(emotion_scores.get(k, 0.0)) for k in ordered}
    total = sum(out.values())
    if total > 0:
        out = {k: round(v / total, 4) for k, v in out.items()}
    return out


def _result_from_emotions(emotion_scores: Dict[str, float]) -> Dict:
    normalized = _canonicalize_emotion_scores(emotion_scores)
    dominant = max(normalized, key=normalized.get) if normalized else "calm"
    sentiment = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "calm": "Calm",
        "fear": "Fear",
        "surprised": "Surprised",
        "disgust": "Disgust",
    }.get(dominant, "Calm")
    return {
        "sentiment": sentiment,
        "emotion_scores": normalized,
        "scores": _emotion_to_polarity(normalized),
    }


def analyze_text_sentiment(text: str) -> Dict:
    """
    Returns an emotion-first dict:
    {
        "sentiment": "Happy" | "Sad" | "Angry" | "Calm" | ...,
        "emotion_scores": {"happy": float, "sad": float, ...},
        "scores": {"positive": float, "negative": float, "neutral": float}
    }
    `scores` are retained for backward compatibility with stored analytics/reporting.
    """
    if not text or not text.strip():
        return _result_from_emotions({"calm": 1.0})

    pipe = _get_emotion_pipeline()
    if pipe:
        try:
            raw_results = pipe(text[:512])[0]  # list of {label, score}
            emotion_scores: Dict[str, float] = {}
            for item in raw_results:
                norm = _normalize_emotion_label(item.get("label", "")).lower()
                emotion_scores[norm] = emotion_scores.get(norm, 0.0) + float(item.get("score", 0.0))
            if emotion_scores:
                return _result_from_emotions(emotion_scores)
        except Exception as exc:
            logger.warning("Emotion inference error: %s — trying polarity fallback.", exc)

    # Fallback to polarity model and map to coarse emotions.
    pol_pipe = _get_polarity_pipeline()
    if pol_pipe:
        try:
            raw_results = pol_pipe(text[:512])[0]
            emotion_scores: Dict[str, float] = {"happy": 0.0, "sad": 0.0, "calm": 0.0}
            for item in raw_results:
                mapped = _normalize_polarity_to_emotion(item.get("label", "")).lower()
                emotion_scores[mapped] = emotion_scores.get(mapped, 0.0) + float(item.get("score", 0.0))
            return _result_from_emotions(emotion_scores)
        except Exception:
            logger.warning("Polarity fallback inference failed; using lexical fallback.")

    # ---- Keyword fallback ----
    return _keyword_emotion(text)


# ---- Simple keyword-based fallback ------------------------------------------------

_POS_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "best", "happy", "joy", "awesome", "brilliant", "superb", "nice", "beautiful",
    "delightful", "perfect", "outstanding", "positive", "pleasant",
}
_NEG_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly", "poor",
    "disgusting", "dreadful", "pathetic", "negative", "nasty", "sad", "angry",
    "disappointed", "frustrating", "useless", "broken", "failure",
}


_ANGER_WORDS = {"angry", "furious", "rage", "annoyed", "irritated", "mad", "hate"}
_SAD_WORDS = {"sad", "depressed", "upset", "down", "heartbroken", "cry", "miserable"}
_FEAR_WORDS = {"afraid", "scared", "fear", "terrified", "worried", "anxious", "panic"}
_SURPRISE_WORDS = {"surprised", "shocked", "unexpected", "wow", "astonished", "amazed"}


def _keyword_emotion(text: str) -> Dict:
    """Keyword-based emotion fallback with boosted scores for decisive output.

    When sentiment keywords are found, their counts are amplified so the
    dominant category reaches 80%+ after temperature sharpening in
    ``_result_from_emotions``.  The 'calm' baseline is kept tiny to avoid
    diluting genuine signals.
    """
    words = set(text.lower().split())
    pos_count = len(words & _POS_WORDS)
    sad_count = len(words & _SAD_WORDS)
    angry_count = len(words & _ANGER_WORDS)
    fear_count = len(words & _FEAR_WORDS)
    surprise_count = len(words & _SURPRISE_WORDS)
    disgust_count = len(words & {"disgust", "disgusted", "revolting", "gross"})

    total_hits = pos_count + sad_count + angry_count + fear_count + surprise_count + disgust_count

    if total_hits == 0:
        # No sentiment keywords — default to calm
        return _result_from_emotions({"calm": 1.0})

    # Boost matched emotions; use a power curve so more matches → much higher score
    _BOOST = 3.0
    emotion_counts = {
        "happy": pos_count * _BOOST,
        "sad": sad_count * _BOOST,
        "angry": angry_count * _BOOST,
        "fear": fear_count * _BOOST,
        "surprised": surprise_count * _BOOST,
        "disgust": disgust_count * _BOOST,
        "calm": 0.1,  # Tiny residual so calm doesn't dilute real signals
    }

    return _result_from_emotions(emotion_counts)
