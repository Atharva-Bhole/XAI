"""
Audio / Video sentiment analysis.

Pipeline:
  1. If video → extract audio track using ffmpeg (subprocess, no shell injection).
  2. Transcribe audio using OpenAI Whisper (or SpeechRecognition as fallback).
  3. Run text sentiment on the transcript.
"""
import os
import logging
import importlib
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple

from ml.sentiment import analyze_text_sentiment
from ml.explainer import generate_text_explanation, extract_key_words
from ml.language_detector import detect_language, language_display_name
from ml.translator import translate_to_english
from ml.image_analyzer import analyze_image_sentiment

logger = logging.getLogger(__name__)
_WHISPER_MODEL = None


# ---- Audio extraction from video -------------------------------------------

def _resolve_ffmpeg_executable() -> str:
    """Return a usable ffmpeg executable path, or an empty string if unavailable."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            # Ensure downstream libraries (like Whisper) can also find ffmpeg.
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return ffmpeg_path
    except Exception as exc:
        logger.debug("imageio-ffmpeg lookup failed: %s", exc)

    return ""


def _run_ffmpeg(input_path: str, output_path: str, audio_only: bool = False) -> Tuple[bool, str]:
    """Run ffmpeg to convert media; returns (ok, stderr_or_reason)."""
    ffmpeg_exe = _resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        logger.warning("FFmpeg is not available. Install ffmpeg or imageio-ffmpeg.")
        return False, "ffmpeg_not_available"

    cmd = [ffmpeg_exe, "-y", "-i", input_path]
    if audio_only:
        cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
    cmd.append(output_path)

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
        if result.returncode == 0:
            return True, ""
        stderr_text = result.stderr.decode(errors="replace")
        logger.warning("ffmpeg returned non-zero: %s", stderr_text)
        return False, stderr_text
    except Exception as exc:
        logger.warning("ffmpeg execution failed: %s", exc)
        return False, str(exc)

def _extract_audio(video_path: str) -> Tuple[str, str]:
    """Extract audio from video file to a temp .wav file. Returns (path, reason)."""
    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio.close()
    ok, error_detail = _run_ffmpeg(video_path, tmp_audio.name, audio_only=True)
    if ok:
        return tmp_audio.name, ""

    try:
        os.unlink(tmp_audio.name)
    except OSError:
        pass

    normalized_error = (error_detail or "").lower()
    if "does not contain any stream" in normalized_error or "stream #0" in normalized_error and "audio" in normalized_error:
        reason = "no_audio_stream"
    elif error_detail == "ffmpeg_not_available":
        reason = "ffmpeg_not_available"
    else:
        reason = "extraction_failed"

    logger.warning("Audio extraction failed for file: %s (reason=%s)", video_path, reason)
    return "", reason


def _convert_audio_to_wav(audio_path: str) -> str:
    """Convert an audio file to 16k mono wav for robust transcription fallback."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    ok, _error_detail = _run_ffmpeg(audio_path, tmp_wav.name, audio_only=False)
    if ok:
        return tmp_wav.name

    try:
        os.unlink(tmp_wav.name)
    except OSError:
        pass

    return ""


def _extract_video_frames(video_path: str, max_frames: int = 6) -> Tuple[List[str], str]:
    """Extract representative frames from a video; returns (frame_paths, reason)."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], "video_open_failed"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return [], "no_frames"

        step = max(1, frame_count // max_frames)
        frame_paths: List[str] = []
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp_img.close()
                cv2.imwrite(tmp_img.name, frame)
                frame_paths.append(tmp_img.name)
                if len(frame_paths) >= max_frames:
                    break
            idx += 1

        cap.release()
        return frame_paths, ""
    except Exception as exc:
        logger.warning("Frame extraction failed: %s", exc)
        return [], "frame_extraction_failed"


def _aggregate_visual_results(frame_results: List[Dict]) -> Dict:
    if not frame_results:
        return {}

    total = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    emo_total = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 0.0, "fear": 0.0, "surprised": 0.0, "disgust": 0.0}
    for item in frame_results:
        scores = item.get("scores", {})
        total["positive"] += float(scores.get("positive", 0.0))
        total["negative"] += float(scores.get("negative", 0.0))
        total["neutral"] += float(scores.get("neutral", 0.0))
        emotion_scores = item.get("emotion_scores", {})
        for key in emo_total:
            emo_total[key] += float(emotion_scores.get(key, 0.0))

    n = float(len(frame_results))
    avg = {
        "positive": total["positive"] / n,
        "negative": total["negative"] / n,
        "neutral": total["neutral"] / n,
    }
    sentiment = max(avg, key=avg.get).capitalize()
    emo_avg = {k: round(v / n, 4) for k, v in emo_total.items()}
    emo_label = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "calm": "Calm",
        "fear": "Fear",
        "surprised": "Surprised",
        "disgust": "Disgust",
    }.get(max(emo_avg, key=emo_avg.get), "Calm")
    return {
        "sentiment": emo_label or sentiment,
        "scores": {k: round(v, 4) for k, v in avg.items()},
        "emotion_scores": emo_avg,
        "explanation": f"Visual sentiment computed from {len(frame_results)} sampled video frames.",
    }


def _fuse_modal_scores(audio_scores: Dict, visual_scores: Dict, has_transcript: bool) -> Dict:
    """Fuse audio/text and visual scores into one final score distribution."""
    if audio_scores and visual_scores:
        audio_w = 0.65 if has_transcript else 0.0
        visual_w = 1.0 - audio_w
        if audio_w == 0.0:
            visual_w = 1.0
        fused = {
            "positive": audio_w * float(audio_scores.get("positive", 0.0)) + visual_w * float(visual_scores.get("positive", 0.0)),
            "negative": audio_w * float(audio_scores.get("negative", 0.0)) + visual_w * float(visual_scores.get("negative", 0.0)),
            "neutral": audio_w * float(audio_scores.get("neutral", 0.0)) + visual_w * float(visual_scores.get("neutral", 0.0)),
        }
        sentiment = max(fused, key=fused.get).capitalize()
        return {"sentiment": sentiment, "scores": {k: round(v, 4) for k, v in fused.items()}}
    if audio_scores:
        return {
            "sentiment": max(audio_scores, key=audio_scores.get).capitalize(),
            "scores": {k: round(float(v), 4) for k, v in audio_scores.items()},
        }
    if visual_scores:
        return {
            "sentiment": max(visual_scores, key=visual_scores.get).capitalize(),
            "scores": {k: round(float(v), 4) for k, v in visual_scores.items()},
        }
    return {
        "sentiment": "Neutral",
        "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
    }


def _fuse_emotion_scores(audio_emotions: Dict, visual_emotions: Dict, has_transcript: bool) -> Dict:
    keys = ["happy", "sad", "angry", "calm", "fear", "surprised", "disgust"]
    audio_emotions = audio_emotions or {}
    visual_emotions = visual_emotions or {}

    if audio_emotions and visual_emotions:
        aw = 0.65 if has_transcript else 0.0
        vw = 1.0 - aw if aw > 0 else 1.0
        fused = {k: aw * float(audio_emotions.get(k, 0.0)) + vw * float(visual_emotions.get(k, 0.0)) for k in keys}
    elif audio_emotions:
        fused = {k: float(audio_emotions.get(k, 0.0)) for k in keys}
    elif visual_emotions:
        fused = {k: float(visual_emotions.get(k, 0.0)) for k in keys}
    else:
        fused = {k: 0.0 for k in keys}
        fused["calm"] = 1.0

    total = sum(fused.values()) or 1.0
    return {k: round(v / total, 4) for k, v in fused.items()}


# ---- Transcription ----------------------------------------------------------

def _has_local_whisper_weights(download_root: str = "") -> bool:
    root = download_root or os.environ.get("WHISPER_MODEL_DIR", "")
    if not root:
        return False
    if not os.path.isdir(root):
        return False
    # OpenAI Whisper base model is typically stored as base.pt in download_root.
    expected = os.path.join(root, "base.pt")
    if os.path.exists(expected):
        return True
    # Fallback: any whisper checkpoint-like file starting with 'base'.
    try:
        return any(name.startswith("base") and name.endswith(".pt") for name in os.listdir(root))
    except OSError:
        return False


def _get_whisper_model(allow_download: bool = True):
    """Load Whisper model once and keep it in memory."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        import whisper

        download_root = os.environ.get("WHISPER_MODEL_DIR", "") or None
        if not allow_download and not _has_local_whisper_weights(download_root or ""):
            logger.info("Skipping Whisper preload; local weights not found.")
            return None
        _WHISPER_MODEL = whisper.load_model("base", download_root=download_root)
        return _WHISPER_MODEL
    except Exception as exc:
        logger.warning("Whisper model load failed: %s", exc)
        _WHISPER_MODEL = None
        return None


def preload_audio_models(allow_download: bool = True) -> bool:
    """Warm-load audio models into memory at app startup."""
    return _get_whisper_model(allow_download=allow_download) is not None

def _transcribe_whisper(audio_path: str) -> str:
    """Transcribe using openai-whisper."""
    try:
        model = _get_whisper_model(allow_download=True)
        if model is None:
            return ""
        result = model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return ""


def _transcribe_sr(audio_path: str) -> str:
    """Transcribe using SpeechRecognition (Google Web Speech API)."""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as exc:
        logger.warning("SpeechRecognition failed: %s", exc)
        return ""


def transcribe(audio_path: str) -> str:
    """Try Whisper first, fall back to SpeechRecognition."""
    text = _transcribe_whisper(audio_path)
    if not text:
        text = _transcribe_sr(audio_path)
    return text


# ---- Public API -------------------------------------------------------------

def analyze_audio_sentiment(file_path: str, is_video: bool = False) -> Dict:
    """
    Transcribe audio/video and run text sentiment analysis on the transcript.
    Returns full analysis dict including transcript.
    """
    audio_path = file_path
    cleanup_paths = []
    transcript = ""
    translated = ""
    lang_code = "en"
    audio_result = {}
    visual_result = {}
    frame_analyses = []
    audio_issue = ""

    if is_video:
        frame_paths, frame_reason = _extract_video_frames(file_path)
        for frame_path in frame_paths:
            frame_result = analyze_image_sentiment(frame_path)
            frame_analyses.append({
                "sentiment": frame_result.get("sentiment", "Calm"),
                "scores": frame_result.get("scores", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}),
                "emotion_scores": frame_result.get("emotion_scores", {}),
                "explanation": frame_result.get("explanation", ""),
            })
            cleanup_paths.append(frame_path)
        visual_result = _aggregate_visual_results(frame_analyses)

        audio_path, reason = _extract_audio(file_path)
        if not audio_path:
            if reason == "ffmpeg_not_available":
                audio_issue = "Audio track analysis unavailable because FFmpeg is not available."
            elif reason == "no_audio_stream":
                audio_issue = "Video does not contain an audio track."
            else:
                audio_issue = "Audio track extraction failed."
        else:
            cleanup_paths.append(audio_path)
    else:
        # SpeechRecognition works best with PCM WAV; normalize when possible.
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".wav":
            normalized = _convert_audio_to_wav(file_path)
            if normalized:
                audio_path = normalized
                cleanup_paths.append(audio_path)

    if audio_path and os.path.exists(audio_path):
        transcript = transcribe(audio_path)

    # Clean up temp audio files we created.
    for path in cleanup_paths:
        try:
            os.unlink(path)
        except OSError:
            pass

    if transcript:
        lang_code = detect_language(transcript)
        translated = translate_to_english(transcript, lang_code)
        analysis = analyze_text_sentiment(translated)
        audio_result = analysis
        xai = generate_text_explanation(translated)
        keywords = extract_key_words(xai)
    else:
        xai = {"method": "audio_transcript", "summary": "No transcript available.", "word_weights": []}
        keywords = []

    fused = _fuse_modal_scores(
        audio_scores=audio_result.get("scores", {}) if audio_result else {},
        visual_scores=visual_result.get("scores", {}) if visual_result else {},
        has_transcript=bool(transcript),
    )
    fused_emotions = _fuse_emotion_scores(
        audio_emotions=audio_result.get("emotion_scores", {}) if audio_result else {},
        visual_emotions=visual_result.get("emotion_scores", {}) if visual_result else {},
        has_transcript=bool(transcript),
    )
    emotion_label = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "calm": "Calm",
        "fear": "Fear",
        "surprised": "Surprised",
        "disgust": "Disgust",
    }.get(max(fused_emotions, key=fused_emotions.get), "Calm")

    explanations = []
    if transcript and xai.get("summary"):
        explanations.append(f"Audio/text: {xai.get('summary', '')}")
    elif not transcript and not is_video:
        explanations.append("Could not transcribe the audio content.")

    if visual_result:
        explanations.append(visual_result.get("explanation", ""))
    if audio_issue:
        explanations.append(audio_issue)
    if is_video and not frame_analyses and frame_reason:
        explanations.append("Could not extract representative video frames.")

    explanation = " ".join(part for part in explanations if part).strip() or "Could not analyze media content."

    return {
        "sentiment": emotion_label,
        "scores": fused["scores"],
        "emotion_scores": fused_emotions,
        "transcript": transcript,
        "detected_language": language_display_name(lang_code),
        "translated_text": translated if lang_code != "en" else "",
        "explanation": explanation,
        "key_words": keywords,
        "frame_analyses": frame_analyses,
        "xai_data": {
            **xai,
            "method": "multimodal_video" if is_video else xai.get("method", "audio_transcript"),
            "modal_breakdown": {
                "audio": audio_result.get("scores", {}),
                "visual": visual_result.get("scores", {}),
            },
            "frame_count": len(frame_analyses),
        },
    }
