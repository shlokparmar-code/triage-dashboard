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

_ALLOWED_EXTRA = set("\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u2026")


def clean_pdf_text(text) -> str:
    """Drop characters the built-in PDF fonts cannot draw (emoji, symbols), keeping normal text.

    Keeps Latin-1 characters and common punctuation (en/em dash, curly quotes, bullet, ellipsis).
    """
    out = "".join(ch for ch in str(text) if ord(ch) < 256 or ch in _ALLOWED_EXTRA)
    return " ".join(out.split(" ")).replace("  ", " ") if out else out


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
    title, body = clean_pdf_text(title), clean_pdf_text(body or "")
    story = [Paragraph(escape(title), styles["Title"])]
    stamp = clean_pdf_text(subtitle) if subtitle else f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story += [Paragraph(escape(stamp), styles["Italic"]), Spacer(1, 6 * mm)]
    # Wrap long lines manually so nothing runs outside the page.
    import textwrap
    wrapped = []
    for line in (body.strip() and body or "No data available.").splitlines():
        wrapped.extend(textwrap.wrap(line, 95, replace_whitespace=False) or [""])
    story.append(Preformatted("\n".join(wrapped), mono))
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Patient history report (readable, clinician-friendly)
# ---------------------------------------------------------------------------
_URGENCY_COLORS = {"HIGH": "#fecaca", "MEDIUM": "#fde68a", "LOW": "#bbf7d0"}
_URGENCY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _get(obj, key, default=""):
    """Read key from a dict or sqlite Row; return default if missing or None."""
    try:
        val = obj[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if val is None else val


def _fmt_time(ts) -> str:
    text = str(ts or "").replace("T", " ")
    return text[:16] if text else "n/a"


def _fmt_score(score) -> str:
    try:
        return f"{float(score):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _findings_text(visit) -> str:
    """One readable block: complaint, vitals, BP, glucose and notes for a visit."""
    inputs = _get(visit, "inputs", {}) or {}
    if not isinstance(inputs, dict):
        inputs = {}
    parts = []
    symptoms = str(inputs.get("symptoms") or "").strip()
    parts.append(f"<b>Complaint:</b> {escape(symptoms) if symptoms else 'not recorded'}")
    vitals = inputs.get("vitals") if isinstance(inputs.get("vitals"), dict) else {}
    bits = []
    if vitals.get("hr") is not None:
        bits.append(f"HR {escape(str(vitals['hr']))} bpm")
    if vitals.get("spo2") is not None:
        bits.append(f"SpO2 {escape(str(vitals['spo2']))}%")
    if inputs.get("resting_bp") is not None:
        bits.append(f"BP (sys) {escape(str(inputs['resting_bp']))} mmHg")
    if inputs.get("glucose") is not None:
        bits.append(f"Glucose {escape(str(inputs['glucose']))} mg/dL")
    if bits:
        parts.append("<b>Findings:</b> " + ", ".join(bits))
    notes = str(_get(visit, "notes", "")).strip()
    if notes:
        parts.append(f"<b>Notes:</b> {escape(notes[:300])}")
    return "<br/>".join(parts)


def history_report_pdf(patient, visits, deterioration_msg: Optional[str] = None) -> bytes:
    """Readable multi-hospital patient history PDF.

    patient: dict with name, age, sex, mobile_masked, consent_given, patient_id (all optional).
    visits: list of visit dicts, newest first (hospital_name, timestamp, final_urgency,
            final_score, referral_level, inputs, notes).
    The mobile number is shown masked only.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    visits = list(visits or [])
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("hcell", parent=styles["BodyText"], fontSize=8, leading=10)
    cell_b = ParagraphStyle("hcell_b", parent=cell, fontName="Helvetica-Bold")
    small = ParagraphStyle("hsmall", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#475569"))

    def P(text, style=cell):
        return Paragraph(clean_pdf_text(text), style)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
        title="Patient History Summary",
    )
    story = [Paragraph("Patient History Summary", styles["Title"]),
             Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Confidential medical record", small),
             Spacer(1, 4 * mm)]

    # --- patient block
    name = escape(str(_get(patient, "name", "Unknown")))
    consent = "Active" if _get(patient, "consent_given", False) else "Revoked / not recorded"
    info_rows = [
        [P("<b>Patient</b>"), P(name), P("<b>Patient ID</b>"), P(escape(str(_get(patient, "patient_id", "n/a"))))],
        [P("<b>Age / Sex</b>"), P(f"{escape(str(_get(patient, 'age', 'n/a')))} yrs / {escape(str(_get(patient, 'sex', 'n/a')))}"),
         P("<b>Mobile (masked)</b>"), P(escape(str(_get(patient, "mobile_masked", "n/a"))))],
        [P("<b>Data-sharing consent</b>"), P(consent), P("<b>Visits recorded</b>"), P(str(len(visits)))],
    ]
    info = Table(info_rows, colWidths=[34 * mm, 54 * mm, 34 * mm, 52 * mm])
    info.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#94a3b8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [info, Spacer(1, 4 * mm)]

    # --- summary
    if visits:
        chrono = list(reversed(visits))  # oldest first
        levels = [str(_get(v, "final_urgency", "")).upper() for v in chrono]
        highest = max((u for u in levels if u in _URGENCY_RANK), key=lambda u: _URGENCY_RANK[u], default="n/a")
        hospitals = sorted({str(_get(v, "hospital_name", "")) for v in visits if _get(v, "hospital_name", "")})
        trend = " &rarr; ".join(escape(u or "?") for u in levels)
        story.append(Paragraph("Summary", styles["Heading3"]))
        story.append(P(f"<b>First visit:</b> {escape(_fmt_time(_get(chrono[0], 'timestamp')))} &nbsp;|&nbsp; "
                       f"<b>Latest visit:</b> {escape(_fmt_time(_get(chrono[-1], 'timestamp')))} &nbsp;|&nbsp; "
                       f"<b>Highest urgency:</b> {escape(highest)}"))
        story.append(P(f"<b>Urgency trend (oldest to newest):</b> {trend}"))
        story.append(P(f"<b>Hospitals ({len(hospitals)}):</b> {escape(', '.join(hospitals)) or 'n/a'}"))
        if deterioration_msg:
            warn = Table([[P(f"<b>Clinical deterioration warning:</b> {escape(str(deterioration_msg))}")]], colWidths=[174 * mm])
            warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fee2e2")),
                                      ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#dc2626")),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story += [Spacer(1, 2 * mm), warn]
        story.append(Spacer(1, 4 * mm))

        # --- visit table (newest first, same order as the app timeline)
        story.append(Paragraph("Visit timeline (newest first)", styles["Heading3"]))
        header = [P(h, cell_b) for h in ("Visit", "Date", "Hospital", "Urgency", "Score", "Care level", "Complaint and findings")]
        rows = [header]
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#94a3b8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        for c in range(7):
            header[c] = Paragraph(f'<font color="white"><b>{("Visit", "Date", "Hospital", "Urgency", "Score", "Care level", "Complaint and findings")[c]}</b></font>', cell)
        total = len(visits)
        for idx, v in enumerate(visits, start=1):
            urgency = str(_get(v, "final_urgency", "")).upper() or "n/a"
            rows.append([
                P(f"#{total - idx + 1}"), P(escape(_fmt_time(_get(v, "timestamp")))),
                P(escape(str(_get(v, "hospital_name", "n/a")))), P(f"<b>{escape(urgency)}</b>"),
                P(_fmt_score(_get(v, "final_score", None))), P(escape(str(_get(v, "referral_level", "n/a")))),
                P(_findings_text(v)),
            ])
            if urgency in _URGENCY_COLORS:
                style_cmds.append(("BACKGROUND", (3, idx), (3, idx), colors.HexColor(_URGENCY_COLORS[urgency])))
        table = Table(rows, colWidths=[11 * mm, 22 * mm, 30 * mm, 18 * mm, 13 * mm, 22 * mm, 58 * mm], repeatRows=1)
        table.setStyle(TableStyle(style_cmds))
        story.append(table)
    else:
        story.append(P("No visits are recorded for this patient."))

    story += [Spacer(1, 6 * mm),
              P("AI decision-support output. Not a medical diagnosis. Requires review by a qualified clinician. "
                "Handle according to your data-protection policy.", small)]

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 10 * mm, "Confidential patient record | AI triage decision support")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()