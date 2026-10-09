"""Minimal text-to-PDF helper (reportlab only, fully offline).

Used for SBAR handoff, queue export and history export so that every
download in the app is a PDF. Text is wrapped inside the page margins.
"""
import io
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape


def text_to_pdf(title: str, body: str, subtitle: Optional[str] = None) -> bytes:
    """Render plain text as a wrapped, paginated PDF and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=title,
    )
    styles = getSampleStyleSheet()
    mono = ParagraphStyle(
        "mono", parent=styles["Normal"], fontName="Courier", fontSize=8.5, leading=11,
    )
    story = [Paragraph(escape(title), styles["Title"])]
    stamp = subtitle or f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story += [Paragraph(escape(stamp), styles["Italic"]), Spacer(1, 6 * mm)]
    # Wrap long lines manually so nothing runs outside the page.
    import textwrap
    wrapped = []
    for line in (body or "No data available.").splitlines():
        wrapped.extend(textwrap.wrap(line, 95, replace_whitespace=False) or [""])
    story.append(Preformatted("\n".join(wrapped), mono))
    doc.build(story)
    return buf.getvalue()