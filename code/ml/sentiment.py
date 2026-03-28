"""
Sentiment Analysis using a pre-trained transformer model.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Labels: Positive, Negative, Neutral

The model is lazy-loaded on first use and cached in memory.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

_pipeline = None
_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    # some model variants use numbered labels
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
}


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Loading sentiment model …")
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment model loaded.")
        except Exception as exc:
            logger.error("Failed to load transformer model: %s", exc)
            _pipeline = None
    return _pipeline


def _normalize_label(raw: str) -> str:
    return _LABEL_MAP.get(raw.lower(), raw.capitalize())


def analyze_text_sentiment(text: str) -> Dict:
    """
    Returns a dict:
    {
        "sentiment": "Positive" | "Negative" | "Neutral",
        "scores": {"positive": float, "negative": float, "neutral": float}
    }
    Falls back to keyword-based analysis if the model is unavailable.
    """
    if not text or not text.strip():
        return {"sentiment": "Neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}

    pipe = _get_pipeline()
    if pipe:
        try:
            raw_results = pipe(text[:512])[0]  # list of {label, score}
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for item in raw_results:
                norm = _normalize_label(item["label"]).lower()
                scores[norm] = round(float(item["score"]), 4)
            sentiment = max(scores, key=scores.get).capitalize()
            return {"sentiment": sentiment, "scores": scores}
        except Exception as exc:
            logger.warning("Transformer inference error: %s — falling back to keyword method.", exc)

    # ---- Keyword fallback ----
    return _keyword_sentiment(text)


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


def _keyword_sentiment(text: str) -> Dict:
    words = set(text.lower().split())
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg + 1e-9
    pos_score = round(pos / total, 4)
    neg_score = round(neg / total, 4)
    neu_score = round(1 - pos_score - neg_score, 4)

    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "scores": {"positive": pos_score, "negative": neg_score, "neutral": max(neu_score, 0)},
    }
