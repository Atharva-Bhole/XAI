import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "xsense-super-secret-key-change-in-production")
    DEBUG = False
    TESTING = False

    # JWT
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "xsense-jwt-secret-key-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)

    # Database  (SQLite default; override via env var for MySQL/Mongo)
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'xsense.db')}"
    )

    # File uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    # Audio/video inputs can be substantially larger than images, so keep a higher cap.
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
    ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "ogg", "flac"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

    # Reports
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

    # ML models cache
    MODELS_CACHE = os.path.join(BASE_DIR, "ml", "cache")
    PRELOAD_MODELS = _env_bool("PRELOAD_MODELS", True)
    PRELOAD_POLARITY_MODEL = _env_bool("PRELOAD_POLARITY_MODEL", False)
    PRELOAD_VISION_MODEL = _env_bool("PRELOAD_VISION_MODEL", False)
    HF_LOCAL_FILES_ONLY = _env_bool("HF_LOCAL_FILES_ONLY", False)
    PRELOAD_ONLY_CACHED_MODELS = _env_bool("PRELOAD_ONLY_CACHED_MODELS", True)

    # Social Media API keys (fill via .env)
    TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
    INSTAGRAM_ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
