"""PDF Clinical Triage Summary Report Generator using ReportLab."""

import html
import io
from datetime import datetime
from typing import Any, Dict, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def safe_escape(text: Any) -> str:
    """Escape XML/HTML special characters (&, <, >) safely for ReportLab Paragraphs."""
    if text is None:
        return ""
    return html.escape(str(text))


def generate_triage_pdf(
    fusion_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    vitals_result: Dict[str, Any],
    symptom_result: Dict[str, Any],
    patient_info: Optional[Dict[str, Any]] = None
) -> bytes:
    """Compile patient triage findings into a downloadable clinical summary PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4b5563"),
        alignment=1
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK"
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6b7280"),
        alignment=1,
        wordWrap="CJK"
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("MULTI-MODAL AI MEDICAL TRIAGE SUMMARY", title_style))
    elements.append(Paragraph("Decision Support Prototype for Rural Healthcare Access | UN SDG 3", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # Patient Metadata & Timestamp Table (Total Width: 540 pt)
    patient_info = patient_info or {}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [
            Paragraph(f"<b>Assessment Date:</b> {now_str}", body_style),
            Paragraph(f"<b>Patient Age/Sex:</b> {safe_escape(patient_info.get('age', 45))} yrs / {safe_escape(patient_info.get('sex', 'N/A'))}", body_style)
        ],
        [
            Paragraph(f"<b>Resting BP:</b> {safe_escape(patient_info.get('resting_bp', 'N/A'))} mmHg", body_style),
            Paragraph(f"<b>Blood Glucose:</b> {safe_escape(patient_info.get('glucose', 'N/A'))} mg/dL", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # Urgency Callout Banner (Total Width: 540 pt)
    urgency = fusion_result.get("final_urgency", "LOW")
    score = fusion_result.get("final_score", 0.0)
    banner_bg = colors.HexColor("#fee2e2") if urgency == "HIGH" else (colors.HexColor("#fef3c7") if urgency == "MEDIUM" else colors.HexColor("#d1fae5"))
    banner_border = colors.HexColor("#ef4444") if urgency == "HIGH" else (colors.HexColor("#f59e0b") if urgency == "MEDIUM" else colors.HexColor("#10b981"))
    text_color = colors.HexColor("#991b1b") if urgency == "HIGH" else (colors.HexColor("#92400e") if urgency == "MEDIUM" else colors.HexColor("#065f46"))

    callout_data = [
        [
            Paragraph(f"<font size=14 color='{text_color}'><b>OVERALL URGENCY: {safe_escape(urgency)}</b></font>", body_style),
            Paragraph(f"<font size=12 color='{text_color}'><b>Calibrated Score: {score:.0%}</b></font>", body_style)
        ],
        [
            Paragraph(f"<b>Clinical Synthesis:</b> {safe_escape(fusion_result.get('clinical_synthesis', ''))}", body_style),
            ""
        ]
    ]
    callout_table = Table(callout_data, colWidths=[380, 160])
    callout_table.setStyle(TableStyle([
        ('SPAN', (0, 1), (1, 1)),
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(callout_table)
    elements.append(Spacer(1, 14))

    # Multi-Modal Breakdown Table (Total Width: 540 pt)
    elements.append(Paragraph("Multi-Modal Component Breakdown", h2_style))

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        wordWrap="CJK"
    )
    th_center_style = ParagraphStyle(
        "TableHeaderCenter",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.white,
        wordWrap="CJK"
    )
    td_modality_style = ParagraphStyle(
        "TableModality",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK"
    )
    td_tier_style = ParagraphStyle(
        "TableTier",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK"
    )
    td_score_style = ParagraphStyle(
        "TableScore",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK"
    )
    td_findings_style = ParagraphStyle(
        "TableFindings",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK"
    )

    breakdown_data = [
        [
            Paragraph("Modality", th_style),
            Paragraph("Model Tier", th_center_style),
            Paragraph("Score", th_center_style),
            Paragraph("Clinical Findings", th_style)
        ],
        [
            Paragraph("Risk Screening", td_modality_style),
            Paragraph(safe_escape(risk_result.get("urgency", "LOW")), td_tier_style),
            Paragraph(f"{risk_result.get('score', 0):.0%}", td_score_style),
            Paragraph(safe_escape(risk_result.get("explanation", "")), td_findings_style)
        ],
        [
            Paragraph("Vitals Anomaly", td_modality_style),
            Paragraph(safe_escape(vitals_result.get("urgency", "LOW")), td_tier_style),
            Paragraph(f"{vitals_result.get('score', 0):.0%}", td_score_style),
            Paragraph(safe_escape(vitals_result.get("explanation", "")), td_findings_style)
        ],
        [
            Paragraph("Symptom Triage", td_modality_style),
            Paragraph(safe_escape(symptom_result.get("urgency", "LOW")), td_tier_style),
            Paragraph(f"{symptom_result.get('score', 0):.0%}", td_score_style),
            Paragraph(safe_escape(symptom_result.get("explanation", "")), td_findings_style)
        ]
    ]

    # colWidths: Modality 20% (108pt), Tier 12% (65pt), Score 10% (54pt), Findings 58% (313pt) -> 540pt total
    breakdown_table = Table(breakdown_data, colWidths=[108, 65, 54, 313], repeatRows=1)
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(breakdown_table)
    elements.append(Spacer(1, 14))

    # Recommendations Section
    recs = fusion_result.get("recommendations", {})
    elements.append(Paragraph("Actionable Care Recommendations", h2_style))
    if recs.get("is_emergency", False):
        elements.append(Paragraph(f"<b>CRITICAL ACTION:</b> {safe_escape(recs.get('action'))}", body_style))
        elements.append(Paragraph(f"<b>Emergency Dispatcher:</b> {safe_escape(recs.get('emergency_number'))}", body_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph("<b>While waiting for medical responders:</b>", body_style))
        for step in recs.get("while_waiting_steps", []):
            elements.append(Paragraph(f"• {safe_escape(step)}", body_style))
    else:
        elements.append(Paragraph(f"<b>Action:</b> {safe_escape(recs.get('action'))}", body_style))
        if recs.get("yoga_asanas"):
            elements.append(Spacer(1, 4))
            elements.append(Paragraph("<b>Targeted Yoga Asana / Pranayama:</b>", body_style))
            for a in recs.get("yoga_asanas", []):
                elements.append(Paragraph(f"• <b>{safe_escape(a.get('name'))}</b>: {safe_escape(a.get('benefit'))}", body_style))
        if recs.get("dietary_guidelines"):
            elements.append(Spacer(1, 4))
            elements.append(Paragraph("<b>Dietary Adjustments:</b>", body_style))
            for d in recs.get("dietary_guidelines", []):
                for tip in d.get("tips", []):
                    elements.append(Paragraph(f"• {safe_escape(tip)}", body_style))
        if recs.get("exercise_plan"):
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"<b>Exercise Guidance:</b> {safe_escape(recs.get('exercise_plan'))}", body_style))

    # Disclaimer Footer
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9ca3af"), spaceAfter=8))
    elements.append(Paragraph(safe_escape(fusion_result.get("disclaimer", "")), disclaimer_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
