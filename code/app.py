import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from config import config
from database.db import db
from routes.auth import auth_bp
from routes.analysis import analysis_bp
from routes.dashboard import dashboard_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def create_app(env: str = "default") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[env])

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
