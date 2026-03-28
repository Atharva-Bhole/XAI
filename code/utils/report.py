"""
PDF Report Generator using fpdf2.
"""
import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _latin1_safe(text: str) -> str:
    """Ensure text can be rendered by fpdf core fonts (latin-1)."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(analysis, reports_folder: str) -> str:
    """
    Generate a PDF report for an Analysis record.
    Returns the absolute path to the saved PDF.
    """
    from fpdf import FPDF, XPos, YPos

    os.makedirs(reports_folder, exist_ok=True)
    report_path = os.path.join(reports_folder, f"xsense_report_{analysis.id}.pdf")

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # ---- Header ----
    pdf.set_fill_color(30, 58, 138)   # deep blue
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(15, 8)
    pdf.cell(180, 12, "X-Sense Sentiment Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 22)
    pdf.cell(180, 8, "Explainable Multimodal Sentiment Intelligence System", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(30, 30, 30)
    pdf.ln(8)

    # ---- Meta info ----
    def section_title(title: str):
        pdf.set_fill_color(236, 240, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, _latin1_safe(f"  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(2)

    def kv(key: str, value: str):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 7, _latin1_safe(f"{key}:"), new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 7, _latin1_safe(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    section_title("Report Information")
    kv("Report ID", f"#{analysis.id}")
    kv("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    kv("Input Type", (analysis.input_type or "N/A").upper())
    kv("Detected Language", analysis.detected_language or "N/A")
    pdf.ln(4)

    # ---- Input data ----
    section_title("Input Data")
    input_preview = (analysis.raw_input or "")[:600]
    if len(analysis.raw_input or "") > 600:
        input_preview += " ..."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _latin1_safe(input_preview or "N/A"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if analysis.translated_text:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 6, _latin1_safe(f"[Translated] {analysis.translated_text[:400]}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(30, 30, 30)
    pdf.ln(4)

    # ---- Sentiment result ----
    section_title("Sentiment Result")
    sentiment = analysis.sentiment or "N/A"
    color_map = {"Positive": (22, 163, 74), "Negative": (220, 38, 38), "Neutral": (107, 114, 128)}
    r, g, b = color_map.get(sentiment, (30, 30, 30))
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 12, _latin1_safe(f"  {sentiment}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    kv("Positive Score", f"{analysis.positive_score * 100:.1f}%")
    kv("Negative Score", f"{analysis.negative_score * 100:.1f}%")
    kv("Neutral Score", f"{analysis.neutral_score * 100:.1f}%")
    pdf.ln(4)

    # ---- XAI Explanation ----
    section_title("Explainable AI - Why This Prediction?")
    pdf.set_font("Helvetica", "", 10)

    xai_text = ""
    try:
        xai_data = json.loads(analysis.explanation) if analysis.explanation and analysis.explanation.startswith("{") else {}
        xai_text = xai_data.get("summary", analysis.explanation or "")
    except (json.JSONDecodeError, TypeError):
        xai_text = analysis.explanation or ""

    pdf.multi_cell(0, 6, _latin1_safe(xai_text or "No explanation available."), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if analysis.key_words:
        pdf.ln(2)
        kv("Key Influencing Words", analysis.key_words.replace(",", ", "))
    pdf.ln(4)

    # ---- Score bar chart (ASCII-style) ----
    section_title("Sentiment Score Distribution")
    pdf.set_font("Helvetica", "", 9)
    bar_width = 120  # mm
    for label, score, (cr, cg, cb) in [
        ("Positive", analysis.positive_score, (22, 163, 74)),
        ("Negative", analysis.negative_score, (220, 38, 38)),
        ("Neutral", analysis.neutral_score, (107, 114, 128)),
    ]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(25, 7, label, new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_fill_color(cr, cg, cb)
        filled = max(2, int(score * bar_width))
        pdf.rect(pdf.get_x(), pdf.get_y() + 1, filled, 5, "F")
        pdf.set_x(pdf.get_x() + bar_width + 3)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(20, 7, f"{score * 100:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # ---- Footer ----
    pdf.set_y(-18)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 8, "Generated by X-Sense - Explainable Multimodal Sentiment Intelligence System", align="C")

    pdf.output(report_path)
    logger.info("PDF report saved: %s", report_path)
    return report_path
