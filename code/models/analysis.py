from datetime import datetime, timezone
from database.db import db


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # Input metadata
    input_type = db.Column(db.String(20), nullable=False)   # text | image | audio | video | url
    raw_input = db.Column(db.Text)                          # original text / file path / URL
    detected_language = db.Column(db.String(50))
    translated_text = db.Column(db.Text)
    transcript = db.Column(db.Text)                         # audio/video transcript when available

    # Sentiment output
    sentiment = db.Column(db.String(20))                    # Positive | Negative | Neutral
    positive_score = db.Column(db.Float, default=0.0)
    negative_score = db.Column(db.Float, default=0.0)
    neutral_score = db.Column(db.Float, default=0.0)

    # XAI
    explanation = db.Column(db.Text)                        # JSON-serialized LIME / Grad-CAM output
    key_words = db.Column(db.Text)                          # comma-separated important tokens

    # Report file
    report_path = db.Column(db.String(512))

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "input_type": self.input_type,
            "raw_input": self.raw_input,
            "detected_language": self.detected_language,
            "translated_text": self.translated_text,
            "transcript": self.transcript,
            "sentiment": self.sentiment,
            "scores": {
                "positive": round(self.positive_score, 4),
                "negative": round(self.negative_score, 4),
                "neutral": round(self.neutral_score, 4),
            },
            "explanation": self.explanation,
            "key_words": self.key_words.split(",") if self.key_words else [],
            "report_path": self.report_path,
            "created_at": self.created_at.isoformat(),
        }
