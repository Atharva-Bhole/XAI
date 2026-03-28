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


# ---- Face / expression detection -------------------------------------------

def _face_expression(img_path: str) -> Dict:
    """Detect faces and approximate expression using OpenCV."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return {}

        # Use mouth region brightness as a crude smile detector
        (x, y, w, h) = faces[0]
        mouth_region = gray[y + int(h * 0.65): y + h, x: x + w]
        mouth_std = float(mouth_region.std())

        if mouth_std > 25:
            return {
                "method": "face_detection",
                "sentiment": "Positive",
                "scores": {"positive": 0.70, "negative": 0.10, "neutral": 0.20},
                "explanation": f"Face detected. High mouth-region variance ({mouth_std:.1f}) suggests a smile.",
            }
        else:
            return {
                "method": "face_detection",
                "sentiment": "Neutral",
                "scores": {"positive": 0.25, "negative": 0.25, "neutral": 0.50},
                "explanation": f"Face detected. Low mouth-region variance ({mouth_std:.1f}) suggests a neutral expression.",
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

    # Face detection takes priority if found
    result = face_result if face_result else color_result

    if not result:
        result = {
            "sentiment": "Neutral",
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "explanation": "Could not extract meaningful visual features from the image.",
        }

    result["key_words"] = []
    return result
