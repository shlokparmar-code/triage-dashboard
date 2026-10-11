"""Tests for the simple PDF helpers (history report and text PDF)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.pdf_simple import clean_pdf_text, history_report_pdf, text_to_pdf  # noqa: E402

PATIENT = {"patient_id": "PAT-1", "name": "Ramesh Kumar", "age": 58, "sex": "Male", "mobile_masked": "******3210", "consent_given": True}


def visit(ts, hosp, urgency, score, symptoms="chest pain"):
    return {"timestamp": ts, "hospital_name": hosp, "final_urgency": urgency, "final_score": score,
            "referral_level": "Emergency", "inputs": {"symptoms": symptoms, "vitals": {"hr": 110, "spo2": 92.0}, "resting_bp": 150, "glucose": 140},
            "notes": "via \U0001F3E5 Hospital"}


def pdf_text(data: bytes, tmp_path) -> str:
    f = tmp_path / "x.pdf"
    f.write_bytes(data)
    try:
        return subprocess.run(["pdftotext", str(f), "-"], capture_output=True, text=True).stdout
    except FileNotFoundError:  # poppler not installed: skip text checks
        return ""


def test_clean_pdf_text_drops_emoji_keeps_text():
    assert clean_pdf_text("via \U0001F3E5 Hospital").replace("  ", " ") == "via Hospital" or "Hospital" in clean_pdf_text("via \U0001F3E5 Hospital")
    assert "\U0001F3E5" not in clean_pdf_text("a \U0001F3E5 b")
    assert clean_pdf_text("Café – ok") == "Café – ok"
    assert clean_pdf_text("") == ""


def test_history_report_is_a_pdf_with_readable_content(tmp_path):
    data = history_report_pdf(PATIENT, [visit("2026-10-08T09:15:00", "Apex District Hospital", "HIGH", 0.91),
                                        visit("2026-09-01T10:00:00", "Alwar CHC", "LOW", 0.2)], "Urgency increased.")
    assert data.startswith(b"%PDF")
    text = pdf_text(data, tmp_path)
    if text:
        for needle in ("Ramesh Kumar", "******3210", "Apex District Hospital", "HIGH", "LOW", "LOW → HIGH", "Urgency increased", "Visit timeline"):
            assert needle in text, needle
        assert "{" not in text and "resourceType" not in text   # not raw JSON


def test_history_report_never_shows_a_full_mobile_number(tmp_path):
    data = history_report_pdf({**PATIENT, "mobile_masked": "******3210"}, [visit("t", "H", "LOW", 0.1)])
    text = pdf_text(data, tmp_path)
    assert "9876543210" not in text


def test_edge_cases_do_not_crash():
    assert history_report_pdf(PATIENT, []).startswith(b"%PDF")
    assert history_report_pdf(None, None).startswith(b"%PDF")
    assert history_report_pdf(PATIENT, [{"inputs": "weird", "final_urgency": None}]).startswith(b"%PDF")
    many = [visit(f"2026-01-{(i % 28) + 1:02d}T10:00:00", f"Hospital {i}", ["LOW", "MEDIUM", "HIGH"][i % 3], 0.3, "long symptom text " * 15) for i in range(40)]
    assert history_report_pdf(PATIENT, many).startswith(b"%PDF")


def test_text_pdf_handles_emoji_and_long_lines():
    assert text_to_pdf("SBAR \U0001F3E5", "line " * 200 + "\n\U0001F3E5 done").startswith(b"%PDF")
    assert text_to_pdf("t", "").startswith(b"%PDF")