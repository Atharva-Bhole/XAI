"""
Image sentiment analysis.

Strategy:
  1. Extract dominant colours & brightness → map to valence (warm=positive, dark=negative).
  2. If a face is detected via OpenCV Haar Cascade → approximate expression.
  3. Return combined sentiment + confidence scores.
"""
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_VISION_PIPELINE = None


def _polarity_to_emotions(scores: Dict) -> Dict:
    positive = float(scores.get("positive", 0.0))
    negative = float(scores.get("negative", 0.0))
    neutral = float(scores.get("neutral", 0.0))
    emotions = {
        "happy": round(positive * 0.9, 4),
        "sad": round(negative * 0.45, 4),
        "angry": round(negative * 0.35, 4),
        "fear": round(negative * 0.12, 4),
        "disgust": round(negative * 0.08, 4),
        "surprised": round(neutral * 0.20, 4),
        "calm": round(neutral * 0.80 + positive * 0.10, 4),
    }
    total = sum(emotions.values()) or 1.0
    return {k: round(v / total, 4) for k, v in emotions.items()}


def _top_emotion_label(emotion_scores: Dict) -> str:
    key = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "calm"
    return {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "calm": "Calm",
        "fear": "Fear",
        "surprised": "Surprised",
        "disgust": "Disgust",
    }.get(key, "Calm")


# ---- Colour-valence heuristic ------------------------------------------------

def _color_valence(img_path: str) -> Dict:
    """Returns rough sentiment based on dominant colour temperature & brightness."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert("RGB").resize((100, 100))
        arr = np.array(img, dtype=float)

        r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        brightness = (r_mean + g_mean + b_mean) / 3.0
        warmth = r_mean - b_mean   # positive = warm tones

        pos, neg, neu = 0.0, 0.0, 0.0
        if brightness > 160 and warmth > 20:
            pos = 0.65; neg = 0.10; neu = 0.25
        elif brightness < 80 or warmth < -20:
            neg = 0.60; pos = 0.10; neu = 0.30
        else:
            neu = 0.60; pos = 0.25; neg = 0.15

        sentiment = max({"positive": pos, "negative": neg, "neutral": neu},
                        key=lambda k: {"positive": pos, "negative": neg, "neutral": neu}[k])
        return {
            "method": "color_heuristic",
            "sentiment": sentiment.capitalize(),
            "scores": {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)},
            "explanation": (
                f"Image brightness={brightness:.1f}/255, warmth index={warmth:.1f}. "
                f"Warm/bright images tend toward positive sentiment; dark/cool images toward negative."
            ),
        }
    except Exception as exc:
        logger.warning("Color valence analysis failed: %s", exc)
        return {}


def _texture_valence(img_path: str) -> Dict:
    """Estimate sentiment from contrast, saturation, and edge density."""
    try:
        from PIL import Image
        import numpy as np
        import cv2

        img = Image.open(img_path).convert("RGB").resize((256, 256))
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        saturation = float(hsv[:, :, 1].mean()) / 255.0
        value = float(hsv[:, :, 2].mean()) / 255.0
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        contrast = float(gray.std()) / 128.0
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float((edges > 0).sum()) / float(edges.size)

        # Heuristic mapping: vivid + well-lit tends positive, very dark/flat tends negative.
        pos = min(0.85, max(0.05, 0.20 + 0.35 * saturation + 0.30 * value + 0.15 * edge_density))
        neg = min(0.85, max(0.05, 0.20 + 0.40 * max(0.0, 0.45 - value) + 0.20 * max(0.0, 0.20 - saturation)))
        neu = min(0.90, max(0.05, 1.0 - (pos + neg)))

        total = pos + neg + neu
        pos, neg, neu = pos / total, neg / total, neu / total

        sentiment_key = max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda k: {"positive": pos, "negative": neg, "neutral": neu}[k])
        return {
            "method": "texture_heuristic",
            "sentiment": sentiment_key.capitalize(),
            "scores": {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)},
            "explanation": (
                f"Texture cues: saturation={saturation:.2f}, value={value:.2f}, "
                f"contrast={contrast:.2f}, edge density={edge_density:.3f}."
            ),
        }
    except Exception as exc:
        logger.warning("Texture valence analysis failed: %s", exc)
        return {}


def _get_vision_pipeline(local_files_only: bool = False):
    """Lazily load a zero-shot image classifier for semantic visual sentiment cues."""
    global _VISION_PIPELINE
    if _VISION_PIPELINE is not None:
        return _VISION_PIPELINE

    try:
        from transformers import pipeline

        _VISION_PIPELINE = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            local_files_only=local_files_only,
        )
        return _VISION_PIPELINE
    except Exception as exc:
        logger.warning("Vision pipeline load failed: %s", exc)
        _VISION_PIPELINE = False
        return None


def preload_image_models(local_files_only: bool = False) -> bool:
    """Warm-load image models into memory at app startup."""
    return _get_vision_pipeline(local_files_only=local_files_only) is not None


def _vision_model_valence(img_path: str) -> Dict:
    """Use CLIP zero-shot labels as a semantic sentiment signal."""
    try:
        vision = _get_vision_pipeline()
        if not vision:
            return {}

        labels = [
            "people smiling and celebrating",
            "joyful and happy scene",
            "sad or upset people",
            "angry or tense scene",
            "calm neutral scene",
        ]
        results = vision(img_path, candidate_labels=labels)
        if not results:
            return {}

        # `results` can be dict(list) or list(dict) depending on transformers version.
        if isinstance(results, dict):
            pairs = list(zip(results.get("labels", []), results.get("scores", [])))
        else:
            pairs = [(item.get("label", ""), item.get("score", 0.0)) for item in results]

        score_map = {label: float(score) for label, score in pairs}
        pos = score_map.get("people smiling and celebrating", 0.0) + score_map.get("joyful and happy scene", 0.0)
        neg = score_map.get("sad or upset people", 0.0) + score_map.get("angry or tense scene", 0.0)
        neu = score_map.get("calm neutral scene", 0.0)

        total = pos + neg + neu
        if total <= 0:
            return {}

        pos, neg, neu = pos / total, neg / total, neu / total
        sentiment_key = max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda k: {"positive": pos, "negative": neg, "neutral": neu}[k])
        top_label = max(score_map, key=score_map.get) if score_map else ""

        return {
            "method": "vision_model",
            "sentiment": sentiment_key.capitalize(),
            "scores": {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)},
            "explanation": f"Vision model semantic match favored: '{top_label}'.",
        }
    except Exception as exc:
        logger.warning("Vision model valence analysis failed: %s", exc)
        return {}


# ---- Face / expression detection -------------------------------------------

def _face_expression(img_path: str) -> Dict:
    """Detect multiple faces and estimate smile-driven sentiment."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Haar cascade for face detection
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return {}

        smile_cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_smile.xml")
        smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

        smiling_faces = 0
        mouth_variances = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y: y + h, x: x + w]
            mouth_region = roi_gray[int(h * 0.60): h, :]
            if mouth_region.size:
                mouth_variances.append(float(mouth_region.std()))

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.8,
                minNeighbors=20,
                minSize=(max(15, w // 5), max(15, h // 5)),
            )
            if len(smiles) > 0:
                smiling_faces += 1

        face_count = len(faces)
        smile_ratio = smiling_faces / float(face_count)
        avg_mouth_std = sum(mouth_variances) / float(len(mouth_variances) or 1)

        # Stronger positive confidence when many faces are smiling.
        pos = min(0.92, 0.20 + 0.65 * smile_ratio + 0.10 * min(1.0, avg_mouth_std / 35.0))
        neg = max(0.03, 0.20 * (1.0 - smile_ratio))
        neu = max(0.03, 1.0 - (pos + neg))
        total = pos + neg + neu
        pos, neg, neu = pos / total, neg / total, neu / total

        sentiment_key = max({"positive": pos, "negative": neg, "neutral": neu}, key=lambda k: {"positive": pos, "negative": neg, "neutral": neu}[k])
        return {
            "method": "face_detection",
            "sentiment": sentiment_key.capitalize(),
            "scores": {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)},
            "explanation": (
                f"Detected {face_count} face(s); {smiling_faces} with smile cues "
                f"(smile ratio {smile_ratio:.2f})."
            ),
        }
    except Exception as exc:
        logger.warning("Face expression analysis failed: %s", exc)
        return {}


# ---- Public API -------------------------------------------------------------

def analyze_image_sentiment(img_path: str) -> Dict:
    """
    Analyse sentiment of an image.
    Returns:
    {
        "sentiment": str,
        "scores": {"positive": float, "negative": float, "neutral": float},
        "explanation": str,
        "key_words": []
    }
    """
    face_result = _face_expression(img_path)
    color_result = _color_valence(img_path)
    texture_result = _texture_valence(img_path)
    vision_result = _vision_model_valence(img_path)

    signals = []
    if color_result:
        signals.append(("color", color_result, 0.20))
    if texture_result:
        signals.append(("texture", texture_result, 0.20))
    if face_result:
        signals.append(("face", face_result, 0.30))
    if vision_result:
        signals.append(("vision_model", vision_result, 0.30))

    # Re-normalize weights if one of the signals is unavailable.
    weight_sum = sum(w for _n, _r, w in signals) or 1.0
    merged_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for _name, res, w in signals:
        scores = res.get("scores", {})
        for key in merged_scores:
            merged_scores[key] += (w / weight_sum) * float(scores.get(key, 0.0))

    if signals:
        sentiment_key = max(merged_scores, key=merged_scores.get)
        result = {
            "sentiment": sentiment_key.capitalize(),
            "scores": {
                "positive": round(merged_scores["positive"], 4),
                "negative": round(merged_scores["negative"], 4),
                "neutral": round(merged_scores["neutral"], 4),
            },
            "explanation": " ".join(
                item.get("explanation", "")
                for _n, item, _w in signals
                if item.get("explanation")
            )[:1200],
            "xai_data": {
                "method": "visual_fusion",
                "components": [
                    {
                        "name": name,
                        "weight": round(weight / weight_sum, 3),
                        "scores": res.get("scores", {}),
                    }
                    for name, res, weight in signals
                ],
            },
        }
    else:
        result = {}

    if not result:
        result = {
            "sentiment": "Calm",
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "explanation": "Could not extract meaningful visual features from the image.",
        }

    emotion_scores = _polarity_to_emotions(result.get("scores", {}))
    result["emotion_scores"] = emotion_scores
    result["sentiment"] = _top_emotion_label(emotion_scores)
    result["key_words"] = []
    return result
