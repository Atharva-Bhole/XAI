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

    # Mean polarity from persisted scores (backward-compatible regardless of emotion label).
    score_avgs = (
        db.session.query(
            func.avg(Analysis.positive_score),
            func.avg(Analysis.negative_score),
            func.avg(Analysis.neutral_score),
        )
        .filter(Analysis.user_id == uid)
        .one()
    )
    pos_avg = float(score_avgs[0] or 0.0)
    neg_avg = float(score_avgs[1] or 0.0)
    neu_avg = float(score_avgs[2] or 0.0)

    # Emotion/sentiment label distribution from latest structure.
    sentiment_rows = (
        db.session.query(Analysis.sentiment, func.count(Analysis.id))
        .filter(Analysis.user_id == uid)
        .group_by(Analysis.sentiment)
        .all()
    )
    sentiment_distribution = {str(label or "Unknown"): int(count) for label, count in sentiment_rows}

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
        "sentiment_distribution": sentiment_distribution,
        "polarity_distribution": {
            "positive": round(pos_avg, 4),
            "negative": round(neg_avg, 4),
            "neutral": round(neu_avg, 4),
        },
        "input_type_distribution": {t: c for t, c in type_counts},
        "positive_pct": round(pos_avg * 100, 1),
        "negative_pct": round(neg_avg * 100, 1),
        "neutral_pct": round(neu_avg * 100, 1),
        "recent_analyses": [r.to_dict() for r in recent],
    }), 200
