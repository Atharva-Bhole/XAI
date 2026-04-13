import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from code/.env for local development.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from config import config
from database.db import db
from routes.auth import auth_bp
from routes.analysis import analysis_bp
from routes.dashboard import dashboard_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _configure_model_cache_env(models_cache_root: str) -> None:
    """Point model libraries to persistent cache directories inside the project."""
    hf_home = os.path.join(models_cache_root, "hf")
    whisper_home = os.path.join(models_cache_root, "whisper")
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(whisper_home, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["WHISPER_MODEL_DIR"] = whisper_home


def _preload_models_once(app: Flask) -> None:
    """Warm-load heavy ML models once per serving process."""
    if not app.config.get("PRELOAD_MODELS", True):
        app.logger.info("Model preloading is disabled (PRELOAD_MODELS=0).")
        return

    # In debug mode, Werkzeug spawns a parent and child process. Preload only in child.
    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    try:
        from ml.sentiment import preload_sentiment_models
        from ml.image_analyzer import preload_image_models
        from ml.audio_analyzer import preload_audio_models

        cached_only = bool(app.config.get("PRELOAD_ONLY_CACHED_MODELS", True))
        local_files_only = bool(app.config.get("HF_LOCAL_FILES_ONLY", False)) or cached_only

        app.logger.info("Preloading ML models into memory...")
        text_ok = preload_sentiment_models(
            preload_fallback=bool(app.config.get("PRELOAD_POLARITY_MODEL", False)),
            local_files_only=local_files_only,
        )
        app.logger.info("Sentiment preload: %s", "ready" if text_ok else "skipped")

        if app.config.get("PRELOAD_VISION_MODEL", False):
            vision_ok = preload_image_models(local_files_only=local_files_only)
            app.logger.info("Vision preload: %s", "ready" if vision_ok else "skipped")

        audio_ok = preload_audio_models(allow_download=not cached_only)
        app.logger.info("Whisper preload: %s", "ready" if audio_ok else "skipped")
        app.logger.info("ML model preload complete.")
    except Exception:
        app.logger.warning("ML model preload did not fully complete.")


def create_app(env: str = "default") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[env])
    _configure_model_cache_env(app.config["MODELS_CACHE"])

    # Allow React dev server (port 3000) and same-origin requests
    CORS(
        app,
        supports_credentials=True,
        origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    )

    # Ensure required directories exist
    for folder_key in ("UPLOAD_FOLDER", "REPORTS_FOLDER", "MODELS_CACHE"):
        os.makedirs(app.config[folder_key], exist_ok=True)

    # Database setup
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config["DATABASE_URL"]
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        from models import user, analysis  # noqa: F401 – ensure models are registered
        db.create_all()

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(dashboard_bp)
    _preload_models_once(app)

    # Health check
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "X-Sense API"}), 200

    # Generic error handlers
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "File too large. Max upload size is 50 MB."}), 413

    @app.errorhandler(500)
    def server_error(_e):
        return jsonify({"error": "Internal server error."}), 500

    return app


if __name__ == "__main__":
    flask_app = create_app(os.environ.get("FLASK_ENV", "development"))
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
