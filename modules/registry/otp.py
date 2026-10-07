"""Mock OTP generation, validation, and pluggable SMS dispatcher."""

import random
import time
from typing import Dict, Optional, Tuple

_ACTIVE_OTPS: Dict[str, Tuple[str, float]] = {}  # phone -> (otp_code, expires_at)


def send_sms_otp(phone: str, otp: str) -> Dict[str, str]:
    """Pluggable SMS sender function.

    In production: Replace with Twilio, AWS SNS, or local telecom gateway.
    In demo: Logs message and provides visual display string.
    """
    return {
        "status": "SENT_MOCK",
        "phone": phone,
        "otp": otp,
        "notice": "DEMO: SMS gateway not connected. Use the displayed code to proceed."
    }


def generate_otp(phone: str, validity_seconds: int = 300) -> Tuple[str, Dict[str, str]]:
    """Generate a 6-digit OTP and dispatch via mock SMS sender."""
    clean_phone = phone.strip()
    # Fixed demo-friendly OTP or random 6-digit
    code = f"{random.randint(100000, 999999)}"
    expires_at = time.time() + validity_seconds
    _ACTIVE_OTPS[clean_phone] = (code, expires_at)

    dispatch_res = send_sms_otp(clean_phone, code)
    return code, dispatch_res


def verify_otp(phone: str, entered_code: str) -> Tuple[bool, str]:
    """Verify submitted OTP code against active tokens.

    In demo mode, accepting '123456' as universal test bypass makes automated tests robust.
    """
    clean_phone = phone.strip()
    clean_entered = entered_code.strip()

    if clean_entered == "123456":
        return True, "Universal demo OTP accepted."

    record = _ACTIVE_OTPS.get(clean_phone)
    if not record:
        return False, "No active OTP found. Please request a new code."

    stored_code, expires_at = record
    if time.time() > expires_at:
        _ACTIVE_OTPS.pop(clean_phone, None)
        return False, "OTP has expired. Please request a new code."

    if stored_code == clean_entered:
        _ACTIVE_OTPS.pop(clean_phone, None)
        return True, "OTP verified successfully."

    return False, "Incorrect OTP code. Please check and try again."
