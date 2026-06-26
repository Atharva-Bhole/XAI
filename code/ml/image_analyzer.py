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
    """Map polarity scores to emotion distribution.

    Uses a sharper mapping so that decisive polarity (e.g. positive > 0.6)
    concentrates almost entirely on the corresponding primary emotion instead
    of being diffused across all seven categories.
    """
    import math

    positive = float(scores.get("positive", 0.0))
    negative = float(scores.get("negative", 0.0))
    neutral = float(scores.get("neutral", 0.0))

    # Primary emotion gets the lion's share; secondaries get residual.
    if positive >= max(negative, neutral) and positive > 0.45:
        # Clearly positive — concentrate on happy
        emotions = {
            "happy": positive ** 0.7,
            "sad": negative * 0.15,
            "angry": negative * 0.10,
            "fear": negative * 0.05,
            "disgust": negative * 0.02,
            "surprised": neutral * 0.10,
            "calm": neutral * 0.25 + positive * 0.05,
        }
    elif negative >= max(positive, neutral) and negative > 0.45:
        # Clearly negative — concentrate on sad/angry
        emotions = {
            "happy": positive * 0.10,
            "sad": negative * 0.50,
            "angry": negative * 0.30,
            "fear": negative * 0.12,
            "disgust": negative * 0.08,
            "surprised": neutral * 0.05,
            "calm": neutral * 0.15,
        }
    else:
        # Neutral / ambiguous — lean calm
        emotions = {
            "happy": positive * 0.40,
            "sad": negative * 0.25,
            "angry": negative * 0.15,
            "fear": negative * 0.05,
            "disgust": negative * 0.03,
            "surprised": neutral * 0.10,
            "calm": neutral * 0.70 + positive * 0.10,
        }

    total = sum(emotions.values()) or 1.0
    normalized = {k: v / total for k, v in emotions.items()}

    # Temperature sharpening to further amplify the dominant emotion
    _TEMP = 0.45
    exp_vals = {k: math.exp(v / _TEMP) for k, v in normalized.items()}
    exp_total = sum(exp_vals.values())
    return {k: round(v / exp_total, 4) for k, v in exp_vals.items()}


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
    """Returns rough sentiment based on dominant colour temperature & brightness.

    Uses a continuous gradient instead of hard-coded caps so that very bright/warm
    images can reach 0.85+ positive and very dark/cold images reach 0.85+ negative.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert("RGB").resize((100, 100))
        arr = np.array(img, dtype=float)

        r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        brightness = (r_mean + g_mean + b_mean) / 3.0
        warmth = r_mean - b_mean   # positive = warm tones

        # Continuous gradient: map brightness [0,255] and warmth [-128,128] to sentiment
        # Bright + warm → positive; dark + cold → negative
        bright_norm = brightness / 255.0  # [0, 1]
        warm_norm = max(-1.0, min(1.0, warmth / 80.0))  # [-1, 1]

        pos = max(0.02, 0.15 + 0.45 * bright_norm + 0.35 * max(0.0, warm_norm))
        neg = max(0.02, 0.15 + 0.45 * (1.0 - bright_norm) + 0.35 * max(0.0, -warm_norm))
        neu = max(0.02, 1.0 - pos - neg)

        total = pos + neg + neu
        pos, neg, neu = pos / total, neg / total, neu / total

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
    """Estimate sentiment from contrast, saturation, and edge density.

    Caps removed so vivid, well-lit images can score 0.90+ positive.
    """
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

        # Continuous mapping without hard caps
        pos = max(0.03, 0.10 + 0.40 * saturation + 0.35 * value + 0.15 * edge_density)
        neg = max(0.03, 0.10 + 0.50 * max(0.0, 0.50 - value) + 0.25 * max(0.0, 0.25 - saturation))
        neu = max(0.03, 1.0 - (pos + neg))

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
    """Use CLIP zero-shot labels as a semantic sentiment signal.

    Uses 15 diverse candidate labels (was 5) so CLIP's softmax produces far
    more discriminating scores — a smiling-child image now concentrates
    probability on the matching positive labels instead of splitting evenly.
    """
    try:
        import math

        vision = _get_vision_pipeline()
        if not vision:
            return {}

        # Expanded label set for much better discrimination
        _POS_LABELS = [
            "a child smiling and laughing happily",
            "people celebrating with joy and excitement",
            "a joyful happy scene with bright colors",
            "a group of friends having fun together",
            "a beautiful scenic landscape with sunshine",
            "a couple hugging lovingly",
        ]
        _NEG_LABELS = [
            "a sad or upset person crying",
            "an angry or tense confrontation",
            "destruction disaster or damage",
            "a lonely person in a dark room",
            "fear horror or danger",
        ]
        _NEU_LABELS = [
            "a plain neutral everyday scene",
            "an empty room or office space",
            "a calm ordinary street view",
            "a simple object on a table",
        ]

        all_labels = _POS_LABELS + _NEG_LABELS + _NEU_LABELS
        results = vision(img_path, candidate_labels=all_labels)
        if not results:
            return {}

        # `results` can be dict(list) or list(dict) depending on transformers version.
        if isinstance(results, dict):
            pairs = list(zip(results.get("labels", []), results.get("scores", [])))
        else:
            pairs = [(item.get("label", ""), item.get("score", 0.0)) for item in results]

        score_map = {label: float(score) for label, score in pairs}
        pos = sum(score_map.get(lbl, 0.0) for lbl in _POS_LABELS)
        neg = sum(score_map.get(lbl, 0.0) for lbl in _NEG_LABELS)
        neu = sum(score_map.get(lbl, 0.0) for lbl in _NEU_LABELS)

        total = pos + neg + neu
        if total <= 0:
            return {}

        pos, neg, neu = pos / total, neg / total, neu / total

        # Temperature sharpening on the aggregated polarity
        _TEMP = 0.35
        vals = {"positive": pos, "negative": neg, "neutral": neu}
        exp_vals = {k: math.exp(v / _TEMP) for k, v in vals.items()}
        exp_total = sum(exp_vals.values())
        sharpened = {k: v / exp_total for k, v in exp_vals.items()}

        sentiment_key = max(sharpened, key=sharpened.get)
        top_label = max(score_map, key=score_map.get) if score_map else ""

        return {
            "method": "vision_model",
            "sentiment": sentiment_key.capitalize(),
            "scores": {"positive": round(sharpened["positive"], 4), "negative": round(sharpened["negative"], 4), "neutral": round(sharpened["neutral"], 4)},
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
                minNeighbors=12,
                minSize=(max(12, w // 6), max(12, h // 6)),
            )
            if len(smiles) > 0:
                smiling_faces += 1

        face_count = len(faces)
        smile_ratio = smiling_faces / float(face_count)
        avg_mouth_std = sum(mouth_variances) / float(len(mouth_variances) or 1)

        # High positive confidence when faces are smiling
        pos = min(0.95, 0.15 + 0.75 * smile_ratio + 0.10 * min(1.0, avg_mouth_std / 30.0))
        neg = max(0.02, 0.15 * (1.0 - smile_ratio))
        neu = max(0.02, 1.0 - (pos + neg))
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


# ---- OCR text extraction for meme/screenshot images -----------------------

def _ocr_text_sentiment(img_path: str) -> Dict:
    """Extract text from image via EasyOCR and run sentiment on it.

    This catches Hindi/Marathi/English text in memes, screenshots, and
    text-overlaid images that visual-only analysis cannot understand.
    """
    try:
        import easyocr

        reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=False, verbose=False)
        results = reader.readtext(img_path, detail=0, paragraph=True)
        extracted_text = " ".join(results).strip()

        if not extracted_text or len(extracted_text) < 5:
            return {}

        from ml.language_detector import detect_language
        from ml.translator import translate_to_english
        from ml.sentiment import analyze_text_sentiment

        lang_code = detect_language(extracted_text)

        if lang_code in ("hi", "mr", "hinglish"):
            from ml.sentiment_lexicon import score_sentiment
            import math as _math

            lex_res = score_sentiment(extracted_text)
            source_lang = lang_code if lang_code in ("hi", "mr") else "hi"
            translated = translate_to_english(extracted_text, source_lang)
            trans_result = analyze_text_sentiment(translated)

            lex_has_signal = lex_res.pos_hits + lex_res.neg_hits > 0
            if lex_has_signal:
                lw, tw = 0.25, 0.75
                total_h = lex_res.pos_hits + lex_res.neg_hits
                lex_scores = {
                    "positive": lex_res.pos_hits / total_h,
                    "negative": lex_res.neg_hits / total_h,
                    "neutral": 0.05,
                }
            else:
                lw, tw = 0.0, 1.0
                lex_scores = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

            trans_scores = trans_result.get("scores", {"positive": 0.33, "negative": 0.33, "neutral": 0.34})
            fused = {
                "positive": lw * lex_scores["positive"] + tw * float(trans_scores.get("positive", 0.0)),
                "negative": lw * lex_scores["negative"] + tw * float(trans_scores.get("negative", 0.0)),
                "neutral": lw * lex_scores["neutral"] + tw * float(trans_scores.get("neutral", 0.0)),
            }
            sentiment_key = max(fused, key=fused.get)
            return {
                "method": "ocr_text",
                "sentiment": sentiment_key.capitalize(),
                "scores": {k: round(v, 4) for k, v in fused.items()},
                "explanation": f"OCR extracted text ({lang_code}): \"{extracted_text[:100]}...\"" if len(extracted_text) > 100 else f"OCR extracted text ({lang_code}): \"{extracted_text}\"",
                "extracted_text": extracted_text,
            }
        else:
            if lang_code != "en":
                analysis_text = translate_to_english(extracted_text, lang_code)
            else:
                analysis_text = extracted_text
            result = analyze_text_sentiment(analysis_text)
            scores = result.get("scores", {"positive": 0.33, "negative": 0.33, "neutral": 0.34})
            sentiment_key = max(scores, key=scores.get)
            return {
                "method": "ocr_text",
                "sentiment": sentiment_key.capitalize(),
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
                "explanation": f"OCR extracted text: \"{extracted_text[:100]}...\"" if len(extracted_text) > 100 else f"OCR extracted text: \"{extracted_text}\"",
                "extracted_text": extracted_text,
            }
    except ImportError:
        logger.debug("easyocr not available; skipping OCR text extraction.")
        return {}
    except Exception as exc:
        logger.warning("OCR text extraction failed: %s", exc)
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
    ocr_result = _ocr_text_sentiment(img_path)

    # Rebalanced weights: CLIP is the strongest signal, heuristics are supplementary.
    # OCR text gets significant weight when available since it directly captures
    # meaning in text-heavy images (memes, screenshots).
    signals = []
    if color_result:
        signals.append(("color", color_result, 0.08))
    if texture_result:
        signals.append(("texture", texture_result, 0.07))
    if face_result:
        signals.append(("face", face_result, 0.25))
    if vision_result:
        signals.append(("vision_model", vision_result, 0.40 if not ocr_result else 0.25))
    if ocr_result:
        signals.append(("ocr_text", ocr_result, 0.40))

    # Re-normalize weights if one of the signals is unavailable.
    weight_sum = sum(w for _n, _r, w in signals) or 1.0
    merged_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for _name, res, w in signals:
        scores = res.get("scores", {})
        for key in merged_scores:
            merged_scores[key] += (w / weight_sum) * float(scores.get(key, 0.0))

    if signals:
        # Temperature sharpening on final fused scores
        import math
        _TEMP = 0.35
        exp_vals = {k: math.exp(v / _TEMP) for k, v in merged_scores.items()}
        exp_total = sum(exp_vals.values())
        sharpened = {k: round(v / exp_total, 4) for k, v in exp_vals.items()}

        sentiment_key = max(sharpened, key=sharpened.get)
        result = {
            "sentiment": sentiment_key.capitalize(),
            "scores": sharpened,
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

    # If OCR extracted text, add it as key words
    if ocr_result and ocr_result.get("extracted_text"):
        result["ocr_text"] = ocr_result["extracted_text"]

    return result
