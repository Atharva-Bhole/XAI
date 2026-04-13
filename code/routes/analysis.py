import json
import os
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
                       result_dict, xai_dict, report_path=None) -> Analysis:
    """Save an Analysis record and return it."""
    scores = result_dict.get("scores", {})
    keywords = extract_key_words(xai_dict) if xai_dict else result_dict.get("key_words", [])
    rec = Analysis(
        user_id=user_id,
        input_type=input_type,
        raw_input=str(raw_input)[:2000],
        detected_language=language_display_name(lang_code) if lang_code else result_dict.get("detected_language", ""),
        translated_text=translated[:2000] if translated else "",
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
    translated = translate_to_english(text, lang_code)
    result = analyze_text_sentiment(translated)
    xai = generate_text_explanation(translated)
    xai["emotion_scores"] = result.get("emotion_scores", {})
    result["key_words"] = extract_key_words(xai)

    rec = _persist_analysis(
        session["user_id"], "text", text, lang_code, translated if lang_code != "en" else "", result, xai
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
        result.get("transcript", ""), lang_code, translated, result, xai
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
    if not text:
        return jsonify({"error": "Could not extract text from the provided URL."}), 422

    posts = fetched.get("posts", [])
    post_analyses = []

    for post in posts:
        post_text = str(post.get("text", "")).strip()
        if not post_text:
            continue
        post_lang = detect_language(post_text)
        post_translated = translate_to_english(post_text, post_lang)
        post_result = analyze_text_sentiment(post_translated)
        post_analyses.append({
            "id": post.get("id", ""),
            "text": post_text,
            "detected_language": language_display_name(post_lang),
            "translated_text": post_translated if post_lang != "en" else "",
            "sentiment": post_result.get("sentiment", "Calm"),
            "emotion_scores": post_result.get("emotion_scores", {}),
            "scores": post_result.get("scores", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}),
        })

    if not post_analyses:
        return jsonify({"error": "Could not extract analyzable post text from the provided URL."}), 422

    aggregate = _aggregate_post_analyses(post_analyses)

    # Build explanation from concatenated translated content for a single coherent XAI output.
    combined_text = "\n\n".join(item.get("translated_text") or item.get("text", "") for item in post_analyses)
    combined_text = combined_text[:8000]

    lang_code = detect_language(text)
    translated = translate_to_english(text, lang_code)
    xai = generate_text_explanation(combined_text)
    result = {
        "sentiment": aggregate["sentiment"],
        "scores": aggregate["scores"],
        "emotion_scores": aggregate.get("emotion_scores", {}),
        "key_words": extract_key_words(xai),
        "explanation": xai.get("summary", ""),
    }
    xai["emotion_scores"] = result.get("emotion_scores", {})

    rec = _persist_analysis(
        session["user_id"], "url", url, lang_code, translated if lang_code != "en" else "", result, xai
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
