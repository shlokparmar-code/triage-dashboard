"""Share a short triage summary with a secondary contact (friend, guardian or relative).

Privacy rules built in:
- the patient must consent to sharing,
- the message holds only the tier, plain advice and up to 3 key concerns
  (no full mobile number, no full medical record),
- the SMS sender is pluggable. The default is a MOCK that sends nothing.
"""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

RELATIONS = ["Family member / relative", "Guardian / caregiver", "Friend", "Other"]

TIER_TEXT = {
    "HIGH": ("needs urgent medical attention now",
             "Please contact them right away and help them reach the nearest hospital, or call the emergency number."),
    "MEDIUM": ("should see a doctor within 24 to 48 hours",
               "Please check on them and help them visit a health centre or doctor soon."),
    "LOW": ("has mild symptoms that can usually be managed at home",
            "Please check on them. If they get worse, they should see a doctor."),
}


def digits_only(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def mask_number(full: str) -> str:
    """Show only the last 4 digits, e.g. ******3210."""
    d = digits_only(full)
    return "******" + d[-4:] if len(d) >= 4 else "******"


def validate_secondary(country_code: str, number: str, primary_full: str = "") -> Tuple[bool, str, str]:
    """Validate the contact's number. Returns (ok, full_number, error)."""
    cfg = config.COUNTRY_PHONE_CONFIG.get(country_code)
    num = digits_only(number)
    if not num:
        return False, "", "Enter the contact's mobile number."
    if cfg is None:
        return False, "", "Unsupported country code."
    if not re.fullmatch(cfg["regex"], num):
        return False, "", f"Enter a valid {cfg['name']} mobile number (example: {cfg['example']})."
    full = f"{country_code}{num}"
    if primary_full and digits_only(full) == digits_only(primary_full):
        return False, "", "The contact's number must be different from the patient's own number."
    return True, full, ""


def _clean(text: Any, limit: int = 100) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


def build_contact_message(
    patient_name: str,
    urgency: str,
    concerns: Optional[List[Any]] = None,
    relation: str = "",
    emergency_number: Optional[str] = None,
) -> str:
    """Compose a short message for the contact. Contains no phone numbers except the emergency number."""
    urgency = (urgency or "LOW").upper()
    meaning, action = TIER_TEXT.get(urgency, TIER_TEXT["LOW"])
    emergency_number = emergency_number or getattr(config, "DEFAULT_EMERGENCY_NUMBER", "112 / 108")
    name = _clean(patient_name, 60) or "Your contact"
    lines = [f"Health alert: {name} used an AI triage tool and was assessed as {urgency} urgency. They {meaning}."]
    items = [_clean(c, 90) for c in (concerns or []) if _clean(c)]
    if items and urgency != "LOW":
        lines.append("Main concerns: " + "; ".join(items[:3]) + ".")
    lines.append(action)
    lines.append(f"Emergency: {emergency_number}.")
    lines.append("This is AI decision support, not a medical diagnosis.")
    return " ".join(lines)


def mock_sms_sender(to_number: str, message: str) -> Dict[str, Any]:
    """DEMO sender: sends nothing. Replace with a real SMS gateway function in production."""
    return {"status": "MOCK_NOT_SENT", "to_masked": mask_number(to_number),
            "note": "DEMO: SMS gateway not connected. No message was actually sent."}


_SMS_SENDER: Callable[[str, str], Dict[str, Any]] = mock_sms_sender


def set_sms_sender(fn: Callable[[str, str], Dict[str, Any]]) -> None:
    """Plug in a real SMS gateway: fn(to_number, message) -> dict with a 'status' key."""
    global _SMS_SENDER
    _SMS_SENDER = fn


def notify_contact(
    contact_full: str,
    consent: bool,
    patient_name: str,
    urgency: str,
    concerns: Optional[List[Any]] = None,
    relation: str = "",
    emergency_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Send (or mock-send) the summary. Refuses without consent or a valid number."""
    if not consent:
        return {"ok": False, "error": "The patient has not agreed to share this result with the contact."}
    if not digits_only(contact_full):
        return {"ok": False, "error": "No valid contact number."}
    message = build_contact_message(patient_name, urgency, concerns, relation, emergency_number)
    try:
        res = _SMS_SENDER(contact_full, message)
    except Exception as exc:
        return {"ok": False, "error": f"Sending failed: {type(exc).__name__}"}
    return {"ok": True, "message": message, "to_masked": mask_number(contact_full),
            "status": res.get("status", "UNKNOWN"), "note": res.get("note", "")}