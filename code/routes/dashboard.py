from flask import Blueprint, jsonify, session
from models.analysis import Analysis
from database.db import db
from sqlalchemy import func

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")


def _login_required(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)

    return decorated


@dashboard_bp.route("/stats", methods=["GET"])
@_login_required
def stats():
    uid = session["user_id"]

    total = Analysis.query.filter_by(user_id=uid).count()
    positive = Analysis.query.filter_by(user_id=uid, sentiment="Positive").count()
    negative = Analysis.query.filter_by(user_id=uid, sentiment="Negative").count()
    neutral = Analysis.query.filter_by(user_id=uid, sentiment="Neutral").count()

    # Breakdown by input type
    type_counts = (
        db.session.query(Analysis.input_type, func.count(Analysis.id))
        .filter(Analysis.user_id == uid)
        .group_by(Analysis.input_type)
        .all()
    )

    # Recent 10 analyses
    recent = (
        Analysis.query.filter_by(user_id=uid)
        .order_by(Analysis.created_at.desc())
        .limit(10)
        .all()
    )

    return jsonify({
        "total_analyses": total,
        "sentiment_distribution": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
        "input_type_distribution": {t: c for t, c in type_counts},
        "positive_pct": round(positive / total * 100, 1) if total else 0,
        "negative_pct": round(negative / total * 100, 1) if total else 0,
        "neutral_pct": round(neutral / total * 100, 1) if total else 0,
        "recent_analyses": [r.to_dict() for r in recent],
    }), 200
