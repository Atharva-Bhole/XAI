import os
import re
import secrets
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request, jsonify, session
from database.db import db
from models.user import User
from utils.email import send_password_reset_email

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    confirm = str(data.get("confirm_password", ""))

    # --- Validation ---
    if not name or len(name) < 2:
        return jsonify({"error": "Name must be at least 2 characters."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email is already registered."}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Account created successfully.", "user": user.to_dict()}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user.id
    session["user_name"] = user.name
    session["user_email"] = user.email

    return jsonify({"message": "Login successful.", "user": user.to_dict()}), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."}), 200


@auth_bp.route("/me", methods=["GET"])
def me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated."}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"user": user.to_dict()}), 200


@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()

    if not email or not EMAIL_RE.match(email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    user = User.query.filter_by(email=email).first()
    if user:
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(minutes=30)
        db.session.commit()

        frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173").strip().rstrip("/")
        reset_link = f"{frontend_url}/reset-password?token={token}"
        send_password_reset_email(user.email, user.name, reset_link)

    return jsonify({"message": "If an account exists with this email, a password reset link has been sent to your inbox."}), 200


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    token = str(data.get("token", "")).strip()
    new_password = str(data.get("new_password", ""))

    if not token or not new_password:
        return jsonify({"error": "Reset token and new password are required."}), 400

    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    user = User.query.filter_by(reset_token=token).first()
    now_utc = datetime.now(timezone.utc)
    if not user or not user.reset_token_expiry:
        return jsonify({"error": "Invalid or expired password reset link."}), 400

    expiry = user.reset_token_expiry
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    if now_utc > expiry:
        return jsonify({"error": "This password reset link has expired. Please request a new one."}), 400

    user.set_password(new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.session.commit()

    return jsonify({"message": "Password reset successfully. You can now log in."}), 200
