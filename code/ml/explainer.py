"""
Explainable AI (XAI) module.

Uses LIME TextExplainer to highlight which words most influence the sentiment prediction.
Falls back to a simple TF-IDF keyword extraction when LIME is unavailable.
"""
import logging
import json
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  LIME-based explainability                                                   #
# --------------------------------------------------------------------------- #

def _lime_explain(text: str, predict_fn) -> Dict:
    """Run LIME and return explanation dict mapped to a 1D sentiment scale."""
    try:
        from lime.lime_text import LimeTextExplainer
        import numpy as np

        explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
        exp = explainer.explain_instance(
            text,
            predict_fn,
            labels=(0, 2),
            num_features=15,
            num_samples=300,
        )
        
        neg_dict = {w: float(weight) for w, weight in exp.as_list(label=0)}
        pos_dict = {w: float(weight) for w, weight in exp.as_list(label=2)}
        
        STOPWORDS = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
            "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now"
        }
        
        all_words = set(neg_dict.keys()).union(set(pos_dict.keys()))
        
        combined = []
        for w in all_words:
            w_lower = w.lower()
            if w_lower in STOPWORDS or len(w) < 2:
                continue
            # Universal weight: + pushes toward Positive, - pushes toward Negative
            w_score = pos_dict.get(w, 0.0) - neg_dict.get(w, 0.0)
            combined.append({"word": w, "weight": round(w_score, 4)})
            
        # Sort by absolute impact and keep the top 10
        combined.sort(key=lambda x: abs(x["weight"]), reverse=True)
        combined = combined[:10]

        explanation = {
            "method": "LIME",
            "word_weights": combined,
        }
        return explanation
    except Exception as exc:
        logger.warning("LIME explanation failed: %s", exc)
        return {}


def _build_predict_fn():
    """Build a predict function compatible with LIME (returns 2-D probability array)."""
    from ml.sentiment import analyze_text_sentiment
    import numpy as np

    def predict_fn(texts: List[str]):
        results = []
        for t in texts:
            res = analyze_text_sentiment(t)
            s = res["scores"]
            results.append([s.get("negative", 0), s.get("neutral", 0), s.get("positive", 0)])
        return np.array(results)

    return predict_fn


# --------------------------------------------------------------------------- #
#  Keyword fallback                                                            #
# --------------------------------------------------------------------------- #

_SENTIMENT_LEXICON = {
    # Positive
    "good": 1, "great": 1, "excellent": 1, "amazing": 1, "wonderful": 1,
    "fantastic": 1, "love": 1, "best": 1, "happy": 1, "joy": 1, "awesome": 1,
    "brilliant": 1, "superb": 1, "nice": 1, "beautiful": 1, "delightful": 1,
    "perfect": 1, "outstanding": 1, "pleasant": 1, "enjoyed": 1, "recommend": 1,
    # Negative
    "bad": -1, "terrible": -1, "awful": -1, "horrible": -1, "hate": -1,
    "worst": -1, "ugly": -1, "poor": -1, "disgusting": -1, "dreadful": -1,
    "pathetic": -1, "nasty": -1, "sad": -1, "angry": -1, "broken": -1,
    "failure": -1, "useless": -1, "disappointed": -1, "frustrating": -1,
    "waste": -1,
}


def _keyword_explain(text: str) -> Dict:
    words = re.findall(r"\b\w+\b", text.lower())
    hits = []
    for word in words:
        if word in _SENTIMENT_LEXICON:
            hits.append({"word": word, "weight": float(_SENTIMENT_LEXICON[word])})
    # Deduplicate, keep highest abs weight per word
    seen: Dict[str, float] = {}
    for h in hits:
        if h["word"] not in seen or abs(h["weight"]) > abs(seen[h["word"]]):
            seen[h["word"]] = h["weight"]
    word_weights = [{"word": w, "weight": v} for w, v in seen.items()]
    return {"method": "keyword", "word_weights": word_weights}


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def generate_text_explanation(text: str, use_lime: bool = True) -> Dict:
    """
    Returns an explanation dict with word-level importance scores.
    {
        "method": "LIME" | "keyword",
        "word_weights": [{"word": str, "weight": float}, ...],
        "summary": str
    }
    """
    explanation: Dict = {}

    if use_lime:
        try:
            predict_fn = _build_predict_fn()
            explanation = _lime_explain(text, predict_fn)
        except Exception as exc:
            logger.warning("LIME setup failed: %s — using keyword fallback.", exc)

    if not explanation:
        explanation = _keyword_explain(text)

    # Build human-readable summary
    pos_words = [e["word"] for e in explanation.get("word_weights", []) if e["weight"] > 0]
    neg_words = [e["word"] for e in explanation.get("word_weights", []) if e["weight"] < 0]

    parts: List[str] = []
    if pos_words:
        parts.append(f"Positive indicators detected: {', '.join(pos_words[:5])}")
    if neg_words:
        parts.append(f"Negative indicators detected: {', '.join(neg_words[:5])}")
    if not parts:
        parts.append("No strong sentiment indicators detected in text.")

    explanation["summary"] = " | ".join(parts)
    return explanation


def extract_key_words(explanation: Dict) -> List[str]:
    """Return a sorted list of the most influential words."""
    weights = explanation.get("word_weights", [])
    sorted_w = sorted(weights, key=lambda x: abs(x["weight"]), reverse=True)
    return [w["word"] for w in sorted_w[:10]]
