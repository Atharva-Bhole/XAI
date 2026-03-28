"""
Audio / Video sentiment analysis.

Pipeline:
  1. If video → extract audio track using ffmpeg (subprocess, no shell injection).
  2. Transcribe audio using OpenAI Whisper (or SpeechRecognition as fallback).
  3. Run text sentiment on the transcript.
"""
import os
import logging
import subprocess
import tempfile
from typing import Dict

from ml.sentiment import analyze_text_sentiment
from ml.explainer import generate_text_explanation, extract_key_words
from ml.language_detector import detect_language, language_display_name
from ml.translator import translate_to_english

logger = logging.getLogger(__name__)


# ---- Audio extraction from video -------------------------------------------

def _extract_audio(video_path: str) -> str:
    """Extract audio from video file to a temp .wav file. Returns path or ''."""
    try:
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_audio.close()
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_audio.name,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            return tmp_audio.name
        logger.warning("ffmpeg returned non-zero: %s", result.stderr.decode(errors="replace"))
        return ""
    except Exception as exc:
        logger.warning("Audio extraction failed: %s", exc)
        return ""


# ---- Transcription ----------------------------------------------------------

def _transcribe_whisper(audio_path: str) -> str:
    """Transcribe using openai-whisper."""
    try:
        import whisper
        model = whisper.load_model("base")
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

    if is_video:
        audio_path = _extract_audio(file_path)
        if not audio_path:
            return {
                "sentiment": "Neutral",
                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "transcript": "",
                "detected_language": "en",
                "translated_text": "",
                "explanation": "Could not extract audio from the video file.",
                "key_words": [],
            }

    transcript = transcribe(audio_path)

    # Clean up temp audio if we created it
    if is_video and audio_path != file_path:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    if not transcript:
        return {
            "sentiment": "Neutral",
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "transcript": "",
            "detected_language": "en",
            "translated_text": "",
            "explanation": "Could not transcribe the audio content.",
            "key_words": [],
        }

    lang_code = detect_language(transcript)
    translated = translate_to_english(transcript, lang_code)
    analysis = analyze_text_sentiment(translated)
    xai = generate_text_explanation(translated)
    keywords = extract_key_words(xai)

    return {
        "sentiment": analysis["sentiment"],
        "scores": analysis["scores"],
        "transcript": transcript,
        "detected_language": language_display_name(lang_code),
        "translated_text": translated if lang_code != "en" else "",
        "explanation": xai.get("summary", ""),
        "key_words": keywords,
        "xai_data": xai,
    }
