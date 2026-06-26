import json
import os
import re
import math
import string
import logging
from flask import Blueprint, request, jsonify, session, current_app
from werkzeug.utils import secure_filename

from database.db import db
from models.analysis import Analysis
from ml.sentiment import analyze_text_sentiment
from ml.language_detector import detect_language, language_display_name
from ml.translator import translate_to_english
from ml.explainer import generate_text_explanation, extract_key_words
from ml.image_analyzer import analyze_image_sentiment
from ml.audio_analyzer import analyze_audio_sentiment
from utils.social_media import fetch_text_from_url
from utils.report import generate_pdf_report

logger = logging.getLogger(__name__)
analysis_bp = Blueprint("analysis", __name__, url_prefix="/api/analysis")


def _login_required(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)

    return decorated


def _allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def _save_upload(file, allowed_extensions: set) -> str:
    """Validate and save an uploaded file; return its absolute path."""
    filename = secure_filename(file.filename)
    if not filename or not _allowed_file(filename, allowed_extensions):
        raise ValueError(f"File type not allowed: {filename}")
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, filename)
    file.save(path)
    return path


def _persist_analysis(user_id, input_type, raw_input, lang_code, translated,
                       result_dict, xai_dict, report_path=None, transcript: str = "") -> Analysis:
    """Save an Analysis record and return it."""
    scores = result_dict.get("scores", {})
    keywords = extract_key_words(xai_dict) if xai_dict else result_dict.get("key_words", [])
    rec = Analysis(
        user_id=user_id,
        input_type=input_type,
        raw_input=str(raw_input)[:2000],
        detected_language=language_display_name(lang_code) if lang_code else result_dict.get("detected_language", ""),
        translated_text=translated[:2000] if translated else "",
        transcript=transcript[:5000] if transcript else "",
        sentiment=result_dict["sentiment"],
        positive_score=scores.get("positive", 0),
        negative_score=scores.get("negative", 0),
        neutral_score=scores.get("neutral", 0),
        explanation=json.dumps(xai_dict) if xai_dict else result_dict.get("explanation", ""),
        key_words=",".join(keywords),
        report_path=report_path,
    )
    db.session.add(rec)
    db.session.commit()
    return rec


def _aggregate_post_analyses(post_analyses: list) -> dict:
    """Aggregate per-post emotion and polarity scores into one summary."""
    if not post_analyses:
        return {
            "sentiment": "Calm",
            "emotion_scores": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 1.0, "fear": 0.0, "surprised": 0.0, "disgust": 0.0},
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        }

    total = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    emo_total = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 0.0, "fear": 0.0, "surprised": 0.0, "disgust": 0.0}
    for item in post_analyses:
        scores = item.get("scores", {})
        total["positive"] += float(scores.get("positive", 0.0))
        total["negative"] += float(scores.get("negative", 0.0))
        total["neutral"] += float(scores.get("neutral", 0.0))
        emotion_scores = item.get("emotion_scores", {})
        for k in emo_total:
            emo_total[k] += float(emotion_scores.get(k, 0.0))

    n = float(len(post_analyses))
    avg = {
        "positive": total["positive"] / n,
        "negative": total["negative"] / n,
        "neutral": total["neutral"] / n,
    }
    emo_avg = {k: round(v / n, 4) for k, v in emo_total.items()}
    emotion_to_label = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "calm": "Calm",
        "fear": "Fear",
        "surprised": "Surprised",
        "disgust": "Disgust",
    }
    top_emo = max(emo_avg, key=emo_avg.get)
    sentiment = emotion_to_label.get(top_emo, "Calm")
    return {"sentiment": sentiment, "scores": avg, "emotion_scores": emo_avg}


def _analyze_native_text(text: str, lang_code: str = "hi") -> dict:
    """Dual-engine sentiment analysis for Hindi/Marathi/Hinglish text.

    Engine 1 (Lexicon): Fast word-match on original text — catches slang,
    code-mixed words, and romanized sentiment the translator may garble.

    Engine 2 (Transformer): Translates to English via Google Translate, then
    runs the full transformer emotion pipeline for nuanced 7-emotion scoring.

    Fusion: Weighted average — transformer 70%, lexicon 30%.  If the lexicon
    found zero sentiment hits, transformer gets 100%.
    """
    import math
    from ml.sentiment_lexicon import score_sentiment

    # ---- Engine 1: Lexicon on original text ----
    sentiment_res = score_sentiment(text)
    lexicon_has_signal = sentiment_res.pos_hits + sentiment_res.neg_hits > 0

    # Build lexicon emotion scores (granular, not binary)
    if lexicon_has_signal:
        total_hits = sentiment_res.pos_hits + sentiment_res.neg_hits
        pos_ratio = sentiment_res.pos_hits / total_hits
        neg_ratio = sentiment_res.neg_hits / total_hits
        lex_scores = {"positive": pos_ratio, "negative": neg_ratio, "neutral": 0.05}

        # Map to emotions based on lexicon polarity and specific keyword discrimination
        if sentiment_res.label == "positive":
            lex_emotions = {
                "happy": pos_ratio * 0.75, "sad": neg_ratio * 0.30,
                "angry": neg_ratio * 0.20, "calm": 0.10,
                "fear": neg_ratio * 0.05, "surprised": pos_ratio * 0.15,
                "disgust": neg_ratio * 0.05,
            }
        elif sentiment_res.label == "negative":
            text_l = text.lower()
            violence_cues = ["mara", "mari", "maramari", "marhan", "maarpeet", "rakta", "rakt", "bleeding", "blood", "jakham", "jakhmi", "khun", "khoon", "murder", "qatl", "katl", "hatya", "chaku", "talwar", "goli", "firing", "bomb", "accident", "apghat", "suicide", "dead", "mele", "mela", "lash", "dhishum", "petla"]
            anger_cues = ["doka", "doke", "dokya", "dimag", "dimaag", "satak", "firla", "firli", "kharab", "bakwas", "bakwaas", "ghatiya", "wahiyat", "bekar", "kamina", "harami", "chutiya", "gadha", "ullu", "jhagda", "ladai", "maar", "hinsa", "zulm", "atyachar", "dhokha", "vaitaag", "vatag", "traas", "tras", "chirchir", "rage", "annoy", "irritat", "hate", "kantala", "bore", "boring", "barbaad", "waste", "yedzava", "yeda", "pagal", "chidan", "rag", "gussa", "vaiteen", "chirden", "mood", "gaand", "lavda", "bocha", "madarchod", "behenchod", "benchod", "bhosdike", "mc", "bc"]
            sad_cues = ["rona", "roi", "roye", "rula", "dukhi", "dukh", "sad", "akela", "depress", "tanha", "vyatha", "peeda", "kasht", "afsos", "pachtava", "musibat", "pareshani", "haar", "apyash", "mayus", "radlo", "radla"]
            fear_cues = ["dar", "darr", "bhaya", "bhiti", "ghabar", "anxi", "stress", "tanav", "tanaav", "bhayanak", "ghabarlo", "dhamki", "threat"]

            is_violence = any(w in text_l for w in violence_cues)
            is_angry = any(w in text_l for w in anger_cues)
            is_sad = any(w in text_l for w in sad_cues)
            is_fear = any(w in text_l for w in fear_cues)

            if is_violence:
                lex_emotions = {
                    "happy": pos_ratio * 0.02, "sad": neg_ratio * 0.20, "angry": neg_ratio * 0.48,
                    "calm": 0.02, "fear": neg_ratio * 0.35, "surprised": 0.05,
                    "disgust": neg_ratio * 0.10,
                }
            elif is_angry and not is_sad:
                lex_emotions = {
                    "happy": pos_ratio * 0.05, "sad": neg_ratio * 0.15, "angry": neg_ratio * 0.60,
                    "calm": 0.03, "fear": neg_ratio * 0.05, "surprised": 0.02,
                    "disgust": neg_ratio * 0.15,
                }
            elif is_sad and not is_angry:
                lex_emotions = {
                    "happy": pos_ratio * 0.05, "sad": neg_ratio * 0.60, "angry": neg_ratio * 0.10,
                    "calm": 0.03, "fear": neg_ratio * 0.15, "surprised": 0.02,
                    "disgust": neg_ratio * 0.05,
                }
            elif is_fear:
                lex_emotions = {
                    "happy": pos_ratio * 0.05, "sad": neg_ratio * 0.20, "angry": neg_ratio * 0.10,
                    "calm": 0.03, "fear": neg_ratio * 0.60, "surprised": 0.02,
                    "disgust": neg_ratio * 0.05,
                }
            else:
                lex_emotions = {
                    "happy": pos_ratio * 0.10, "sad": neg_ratio * 0.35, "angry": neg_ratio * 0.35,
                    "calm": 0.05, "fear": neg_ratio * 0.10, "surprised": 0.02,
                    "disgust": neg_ratio * 0.13,
                }
        else:
            lex_emotions = {
                "happy": 0.10, "sad": 0.10, "angry": 0.05, "calm": 0.55,
                "fear": 0.03, "surprised": 0.05, "disgust": 0.02,
            }
        # Normalize
        lex_emo_total = sum(lex_emotions.values()) or 1.0
        lex_emotions = {k: v / lex_emo_total for k, v in lex_emotions.items()}
    else:
        lex_scores = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        lex_emotions = {
            "happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 1.0,
            "fear": 0.0, "surprised": 0.0, "disgust": 0.0,
        }

    # ---- Engine 2: Translate + Transformer ----
    # Always translate, even for hi/mr, to feed the English transformer
    source_for_translation = lang_code if lang_code in ("hi", "mr") else "hi"
    translated = translate_to_english(text, source_for_translation)
    transformer_result = analyze_text_sentiment(translated)
    transformer_xai = generate_text_explanation(translated)

    trans_scores = transformer_result.get("scores", {"positive": 0.33, "negative": 0.33, "neutral": 0.34})
    trans_emotions = transformer_result.get("emotion_scores", {
        "happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 1.0,
        "fear": 0.0, "surprised": 0.0, "disgust": 0.0,
    })

    # ---- Fusion ----
    # Check if Google Translate actually converted the Indic text or returned the original romanized string
    clean_orig = re.sub(r'\s+', ' ', text.lower()).strip()
    clean_trans = re.sub(r'\s+', ' ', (translated or "").lower()).strip()
    translation_unchanged = (not clean_trans or clean_trans == clean_orig)

    if lexicon_has_signal:
        if translation_unchanged:
            # Translation returned original romanized string -> English transformer is blind. Trust native lexicon 100%!
            lw, tw = 1.0, 0.0
        else:
            # Translation worked -> weight native lexicon 75%, transformer 25% (native slang is ground truth)
            lw, tw = 0.75, 0.25
    else:
        lw, tw = 0.0, 1.0

    # Fuse polarity scores
    fused_scores = {
        "positive": lw * lex_scores["positive"] + tw * float(trans_scores.get("positive", 0.0)),
        "negative": lw * lex_scores["negative"] + tw * float(trans_scores.get("negative", 0.0)),
        "neutral": lw * lex_scores["neutral"] + tw * float(trans_scores.get("neutral", 0.0)),
    }

    # Fuse emotion scores
    emotion_keys = ["happy", "sad", "angry", "calm", "fear", "surprised", "disgust"]
    fused_emotions = {}
    for k in emotion_keys:
        fused_emotions[k] = lw * lex_emotions.get(k, 0.0) + tw * float(trans_emotions.get(k, 0.0))

    # Power sharpening on fused probabilities to accentuate dominant emotion without e^0 floor noise
    _POW = 2.5
    pow_emo = {k: math.pow(max(0.0, v), _POW) for k, v in fused_emotions.items()}
    pow_emo_total = sum(pow_emo.values()) or 1.0
    fused_emotions = {k: round(v / pow_emo_total, 4) for k, v in pow_emo.items()}

    # Sharpen polarity scores too
    pow_pol = {k: math.pow(max(0.0, v), _POW) for k, v in fused_scores.items()}
    pow_pol_total = sum(pow_pol.values()) or 1.0
    fused_scores = {k: round(v / pow_pol_total, 4) for k, v in pow_pol.items()}

    # Determine final sentiment from dominant emotion
    dominant_emotion = max(fused_emotions, key=fused_emotions.get)
    sentiment_label = {
        "happy": "Happy", "sad": "Sad", "angry": "Angry", "calm": "Calm",
        "fear": "Fear", "surprised": "Surprised", "disgust": "Disgust",
    }.get(dominant_emotion, "Calm")

    result = {
        "sentiment": sentiment_label,
        "emotion_scores": fused_emotions,
        "scores": fused_scores,
    }

    # ---- XAI: Merge word weights from both engines ----
    lexicon_weights = []
    for word, reason in sentiment_res.details:
        if "positive" in reason:
            lexicon_weights.append({"word": word, "weight": 0.5, "source": "lexicon"})
        elif "negative" in reason:
            lexicon_weights.append({"word": word, "weight": -0.5, "source": "lexicon"})
        elif "intensifier" in reason:
            lexicon_weights.append({"word": word, "weight": 0.2, "source": "lexicon"})

    transformer_weights = transformer_xai.get("word_weights", [])
    if isinstance(transformer_weights, list):
        for item in transformer_weights:
            if isinstance(item, dict):
                item["source"] = "transformer"
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                transformer_weights[transformer_weights.index(item)] = {
                    "word": item[0], "weight": item[1], "source": "transformer"
                }

    # Combine and deduplicate (keep higher abs weight)
    all_weights = lexicon_weights + (transformer_weights if isinstance(transformer_weights, list) else [])
    seen = {}
    for w in all_weights:
        word = w.get("word", "") if isinstance(w, dict) else ""
        weight = w.get("weight", 0.0) if isinstance(w, dict) else 0.0
        if word and (word not in seen or abs(weight) > abs(seen[word].get("weight", 0.0))):
            seen[word] = w
    merged_weights = sorted(seen.values(), key=lambda x: abs(x.get("weight", 0.0)), reverse=True)[:15]

    # Build summary
    parts = []
    if lexicon_has_signal:
        pos_words = [w["word"] for w in lexicon_weights if w.get("weight", 0) > 0]
        neg_words = [w["word"] for w in lexicon_weights if w.get("weight", 0) < 0]
        if pos_words:
            parts.append(f"Lexicon positive: {', '.join(pos_words[:5])}")
        if neg_words:
            parts.append(f"Lexicon negative: {', '.join(neg_words[:5])}")

    trans_summary = transformer_xai.get("summary", "")
    if trans_summary:
        parts.append(f"Transformer: {trans_summary}")

    summary = " | ".join(parts) if parts else "Dual-engine analysis completed."

    xai = {
        "method": "dual_engine",
        "summary": summary,
        "word_weights": merged_weights,
        "emotion_scores": fused_emotions,
        "engine_details": {
            "lexicon_label": sentiment_res.label,
            "lexicon_has_signal": lexicon_has_signal,
            "lexicon_weight": lw,
            "transformer_weight": tw,
            "translated_text": translated,
        },
    }
    result["key_words"] = [w.get("word", "") for w in merged_weights if isinstance(w, dict)][:10]
    return {"translated": translated, "result": result, "xai": xai}


# ---- Text analysis ---------------------------------------------------------

@analysis_bp.route("/text", methods=["POST"])
@_login_required
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Text too long (max 10 000 characters)."}), 400

    lang_code = detect_language(text)
    
    if lang_code in ("hi", "mr", "hinglish"):
        native_res = _analyze_native_text(text, lang_code)
        translated = native_res["translated"]
        result = native_res["result"]
        xai = native_res["xai"]
    else:
        translated = translate_to_english(text, lang_code)
        result = analyze_text_sentiment(translated)
        xai = generate_text_explanation(translated)
        xai["emotion_scores"] = result.get("emotion_scores", {})
        result["key_words"] = extract_key_words(xai)

    rec = _persist_analysis(
        session["user_id"], "text", text, lang_code, translated if lang_code not in ("en",) else "", result, xai
    )
    response = rec.to_dict()
    response["emotion_scores"] = result.get("emotion_scores", {})
    response["xai"] = xai
    return jsonify(response), 200


# ---- Image analysis --------------------------------------------------------

@analysis_bp.route("/image", methods=["POST"])
@_login_required
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    try:
        img_path = _save_upload(file, current_app.config["ALLOWED_IMAGE_EXTENSIONS"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = analyze_image_sentiment(img_path)
    xai = {"method": "visual", "summary": result.get("explanation", ""), "word_weights": []}
    rec = _persist_analysis(
        session["user_id"], "image", img_path, "en", "", result, xai
    )
    response = rec.to_dict()
    response["emotion_scores"] = result.get("emotion_scores", {})
    response["xai"] = xai
    return jsonify(response), 200


# ---- Audio analysis --------------------------------------------------------

@analysis_bp.route("/audio", methods=["POST"])
@_login_required
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    file = request.files["audio"]
    allowed = current_app.config["ALLOWED_AUDIO_EXTENSIONS"] | current_app.config["ALLOWED_VIDEO_EXTENSIONS"]
    try:
        file_path = _save_upload(file, allowed)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    ext = file_path.rsplit(".", 1)[1].lower()
    is_video = ext in current_app.config["ALLOWED_VIDEO_EXTENSIONS"]
    result = analyze_audio_sentiment(file_path, is_video=is_video)

    lang_code = result.get("detected_language", "en")
    translated = result.get("translated_text", "")
    xai = result.get("xai_data", {"method": "audio_transcript", "summary": result.get("explanation", ""), "word_weights": []})
    xai["emotion_scores"] = result.get("emotion_scores", {})

    rec = _persist_analysis(
        session["user_id"], "video" if is_video else "audio",
        file_path, lang_code, translated, result, xai, transcript=result.get("transcript", "")
    )
    response = rec.to_dict()
    response["transcript"] = result.get("transcript", "")
    response["frame_analyses"] = result.get("frame_analyses", [])
    response["emotion_scores"] = result.get("emotion_scores", {})
    response["xai"] = xai
    return jsonify(response), 200


# ---- URL analysis ----------------------------------------------------------

@analysis_bp.route("/url", methods=["POST"])
@_login_required
def analyze_url():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    bearer = current_app.config.get("TWITTER_BEARER_TOKEN", "")
    insta_token = current_app.config.get("INSTAGRAM_ACCESS_TOKEN", "")
    fetched = fetch_text_from_url(url, bearer_token=bearer, instagram_token=insta_token)
    if fetched.get("error"):
        return jsonify({"error": fetched["error"]}), 400

    text = fetched.get("text", "")
    media_path = fetched.get("media_path")
    media_type = fetched.get("media_type")

    if not text and not media_path:
        return jsonify({"error": "Could not extract text or media from the provided URL."}), 422

    posts = fetched.get("posts", [])
    post_analyses = []

    for post in posts:
        post_text = str(post.get("text", "")).strip()
        if post_text:
            post_lang = detect_language(post_text)
            if post_lang in ("hi", "mr", "hinglish"):
                native_res = _analyze_native_text(post_text, post_lang)
                post_translated = native_res["translated"]
                post_result = native_res["result"]
            else:
                post_translated = translate_to_english(post_text, post_lang)
                post_result = analyze_text_sentiment(post_translated)
                
            post_analyses.append({
                "id": post.get("id", ""),
                "text": post_text,
                "detected_language": language_display_name(post_lang),
                "translated_text": post_translated if post_lang not in ("en",) else "",
                "sentiment": post_result.get("sentiment", "Calm"),
                "emotion_scores": post_result.get("emotion_scores", {}),
                "scores": post_result.get("scores", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}),
            })

    media_result = None
    media_xai = None
    if media_path:
        if media_type == "image":
            media_result = analyze_image_sentiment(media_path)
            media_xai = {"method": "visual", "summary": media_result.get("explanation", ""), "word_weights": []}
        elif media_type == "video":
            media_result = analyze_audio_sentiment(media_path, is_video=True)
            media_xai = media_result.get("xai_data", {"method": "audio_transcript", "summary": media_result.get("explanation", ""), "word_weights": []})
            
        if media_result:
            post_analyses.append({
                "id": "media",
                "text": media_result.get("transcript", "") or "[Media Content]",
                "detected_language": media_result.get("detected_language", "en"),
                "translated_text": media_result.get("translated_text", ""),
                "sentiment": media_result.get("sentiment", "Calm"),
                "emotion_scores": media_result.get("emotion_scores", {}),
                "scores": media_result.get("scores", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}),
            })
        
        try:
            os.remove(media_path)
        except OSError:
            pass

    if not post_analyses:
        return jsonify({"error": "Could not extract analyzable content from the provided URL."}), 422

    aggregate = _aggregate_post_analyses(post_analyses)

    combined_text = "\n\n".join(item.get("translated_text") or item.get("text", "") for item in post_analyses if item.get("id") != "media")
    combined_text = combined_text[:8000]

    lang_code = detect_language(text) if text else "en"
    if text and lang_code in ("hi", "mr", "hinglish"):
        native_res = _analyze_native_text(text, lang_code)
        translated = native_res["translated"]
        xai = native_res["xai"]
    else:
        translated = translate_to_english(text, lang_code) if text else ""
        xai = generate_text_explanation(combined_text) if combined_text else {}
    if not xai and media_xai:
        xai = media_xai

    result = {
        "sentiment": aggregate["sentiment"],
        "scores": aggregate["scores"],
        "emotion_scores": aggregate.get("emotion_scores", {}),
        "key_words": extract_key_words(xai) if xai else [],
        "explanation": xai.get("summary", "") if xai else "",
    }
    xai["emotion_scores"] = result.get("emotion_scores", {})

    rec = _persist_analysis(
        session["user_id"], "url", url, lang_code, translated if lang_code not in ("en",) else "", result, xai
    )
    response = rec.to_dict()
    response["source"] = fetched.get("source", "web_scrape")
    response["post_count"] = len(post_analyses)
    response["post_analyses"] = post_analyses
    response["emotion_scores"] = result.get("emotion_scores", {})
    response["fetched_text_preview"] = text[:500]
    response["xai"] = xai
    return jsonify(response), 200


# ---- Report download -------------------------------------------------------

@analysis_bp.route("/<int:analysis_id>/report", methods=["GET"])
@_login_required
def download_report(analysis_id: int):
    rec = Analysis.query.filter_by(id=analysis_id, user_id=session["user_id"]).first()
    if not rec:
        return jsonify({"error": "Analysis not found."}), 404

    if not rec.report_path or not os.path.exists(rec.report_path):
        try:
            path = generate_pdf_report(rec, current_app.config["REPORTS_FOLDER"])
            rec.report_path = path
            db.session.commit()
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            return jsonify({"error": "Could not generate report."}), 500

    from flask import send_file
    return send_file(rec.report_path, as_attachment=True, download_name=f"xsense_report_{analysis_id}.pdf")


# ---- History ---------------------------------------------------------------

@analysis_bp.route("/history", methods=["GET"])
@_login_required
def history():
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(50, request.args.get("per_page", 20, type=int))
    records = (
        Analysis.query.filter_by(user_id=session["user_id"])
        .order_by(Analysis.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify({
        "items": [r.to_dict() for r in records.items],
        "total": records.total,
        "pages": records.pages,
        "page": page,
    }), 200


# ---- Single record ---------------------------------------------------------

@analysis_bp.route("/<int:analysis_id>", methods=["GET"])
@_login_required
def get_analysis(analysis_id: int):
    rec = Analysis.query.filter_by(id=analysis_id, user_id=session["user_id"]).first()
    if not rec:
        return jsonify({"error": "Analysis not found."}), 404
    data = rec.to_dict()
    try:
        data["xai"] = json.loads(rec.explanation) if rec.explanation and rec.explanation.startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        data["xai"] = {}
    data["emotion_scores"] = data.get("xai", {}).get("emotion_scores", {})
    return jsonify(data), 200
