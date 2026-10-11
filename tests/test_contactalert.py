"""Tests for the secondary contact alert."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import contact_alert as ca  # noqa: E402


def test_validate_secondary():
    ok, full, err = ca.validate_secondary("+91", "9123456780", "+919876543210")
    assert ok and full == "+919123456780" and err == ""
    assert not ca.validate_secondary("+91", "", "")[0]
    assert not ca.validate_secondary("+91", "12345", "")[0]
    assert not ca.validate_secondary("+99", "9123456780", "")[0]
    same = ca.validate_secondary("+91", "9876543210", "+91 98765 43210")
    assert not same[0] and "different" in same[2]


def test_mask_number():
    assert ca.mask_number("+919876543210") == "******3210"
    assert ca.mask_number("12") == "******"


def test_message_has_tier_advice_disclaimer_and_no_phone_numbers():
    msg = ca.build_contact_message("Asha Devi", "HIGH", ["chest pain", "fast breathing"], "Friend")
    assert "HIGH" in msg and "urgent" in msg and "chest pain" in msg
    assert "not a medical diagnosis" in msg and "112 / 108" in msg
    assert "9876543210" not in msg


def test_low_urgency_hides_concerns():
    msg = ca.build_contact_message("Ravi", "LOW", ["mild cough"])
    assert "mild cough" not in msg and "LOW" in msg


def test_concerns_are_limited_and_cleaned():
    msg = ca.build_contact_message("Ravi", "MEDIUM", ["a1", "b2", "c3", "d4", "e" * 500])
    assert "Main concerns: a1; b2; c3." in msg   # only the first 3
    assert "d4" not in msg
    assert len(msg) < 600
    long_msg = ca.build_contact_message("Ravi", "MEDIUM", ["e" * 500])
    assert "e" * 91 not in long_msg               # each concern is trimmed to 90 characters


def test_refuses_without_consent_or_number():
    assert not ca.notify_contact("+919123456780", False, "Asha", "HIGH")["ok"]
    assert not ca.notify_contact("", True, "Asha", "HIGH")["ok"]


def test_default_sender_is_a_mock_that_sends_nothing():
    res = ca.notify_contact("+919123456780", True, "Asha", "HIGH", ["chest pain"])
    assert res["ok"] and res["status"] == "MOCK_NOT_SENT" and res["to_masked"] == "******6780"
    assert "9123456780" not in str(res)


def test_pluggable_sender_and_failure_handling():
    calls = []
    ca.set_sms_sender(lambda to, msg: calls.append((to, msg)) or {"status": "SENT"})
    try:
        res = ca.notify_contact("+919123456780", True, "Asha", "MEDIUM")
        assert res["ok"] and res["status"] == "SENT" and calls and calls[0][0] == "+919123456780"

        def boom(to, msg):
            raise ConnectionError("no network")
        ca.set_sms_sender(boom)
        assert not ca.notify_contact("+919123456780", True, "Asha", "MEDIUM")["ok"]
    finally:
        ca.set_sms_sender(ca.mock_sms_sender)