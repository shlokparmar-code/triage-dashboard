"""Unit tests for Patient Registry, Encryption, Mobile Validation, OTP, and Multi-Hospital History."""

import pytest
import sqlite3
from pathlib import Path
import tempfile

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from modules.registry.crypto import (
    encrypt_mobile, decrypt_mobile, hash_mobile, mask_mobile, validate_mobile
)
from modules.registry.otp import generate_otp, verify_otp
from modules.registry.auth import authenticate_doctor, hash_password
from modules.registry.provider import LocalRegistryProvider, check_clinical_deterioration
from modules.registry.db import init_db


@pytest.fixture
def temp_provider(tmp_path):
    db_file = tmp_path / "test_registry.db"
    init_db(db_file)
    return LocalRegistryProvider(db_path=db_file)


def test_mobile_validation():
    """Verify regex validation across different country dial codes."""
    # India +91: 10 digits starting with 6-9
    valid_in, norm_in, err_in = validate_mobile("+91", "9876543210")
    assert valid_in
    assert norm_in == "+91 9876543210"
    assert err_in == ""

    invalid_in, _, err = validate_mobile("+91", "12345")
    assert not invalid_in
    assert "Invalid India number" in err

    # US +1: 10 digits starting 2-9
    valid_us, norm_us, _ = validate_mobile("+1", "4155552671")
    assert valid_us
    assert norm_us == "+1 4155552671"

    # UK +44: 10-11 digits
    valid_uk, norm_uk, _ = validate_mobile("+44", "7911123456")
    assert valid_uk


def test_encryption_and_masking():
    """Verify raw mobile number is strongly encrypted and masked."""
    raw = "+91 9876543210"
    encrypted = encrypt_mobile(raw)
    assert encrypted != raw
    assert raw not in encrypted  # Raw number must never appear in encrypted token

    decrypted = decrypt_mobile(encrypted)
    assert decrypted == raw

    masked = mask_mobile(raw)
    assert "3210" in masked
    assert "9876" not in masked


def test_otp_generation_and_verification():
    """Verify mock OTP generation, universal test bypass, and verification."""
    phone = "+91 9876543210"
    otp, res = generate_otp(phone)
    assert len(otp) == 6
    assert res["status"] == "SENT_MOCK"

    # Universal test code
    ok_bypass, _ = verify_otp(phone, "123456")
    assert ok_bypass

    # Correct code
    otp2, _ = generate_otp(phone)
    ok_real, msg = verify_otp(phone, otp2)
    assert ok_real

    # Wrong code
    bad, _ = verify_otp(phone, "000000")
    assert not bad


def test_patient_registration_and_consent(temp_provider):
    """Verify patient registration enforces consent and prevents plain text leakage."""
    phone = "+91 9123456789"
    # Fails without consent
    with pytest.raises(ValueError):
        temp_provider.register_patient("Asha Devi", 32, "Female", phone, consent_given=False)

    # Succeeds with consent
    p = temp_provider.register_patient("Asha Devi", 32, "Female", phone, consent_given=True)
    assert p["patient_id"].startswith("PAT-")
    assert p["name"] == "Asha Devi"
    assert "6789" in p["mobile_masked"]

    # Verify database file content: raw phone string must NOT be stored in DB
    conn = sqlite3.connect(str(temp_provider.db_path))
    cur = conn.cursor()
    cur.execute("SELECT mobile_hash, mobile_encrypted FROM patients WHERE patient_id = ?", (p["patient_id"],))
    row = cur.fetchone()
    conn.close()

    assert phone not in row[0]
    assert phone not in row[1]


def test_multi_hospital_history_and_deterioration(temp_provider):
    """Verify visits across multiple hospitals aggregate in a single patient timeline."""
    p = temp_provider.register_patient("Sunil Verma", 45, "Male", "+91 9988776655", consent_given=True)
    pid = p["patient_id"]

    # Visit 1 at Hospital A (LOW)
    v1 = temp_provider.add_visit(
        pid, "HOSP-001", {"bp": 120}, {"symptom": {"urgency": "LOW"}}, "LOW", 0.18, "Self-care"
    )
    # Visit 2 at Hospital B (HIGH)
    v2 = temp_provider.add_visit(
        pid, "HOSP-002", {"bp": 175}, {"symptom": {"urgency": "HIGH"}}, "HIGH", 0.88, "Emergency Hospital"
    )

    history = temp_provider.get_history(pid)
    assert len(history) == 2
    # Order is newest first
    assert history[0]["visit_id"] == v2
    assert history[0]["final_urgency"] == "HIGH"
    assert history[1]["visit_id"] == v1
    assert history[1]["final_urgency"] == "LOW"

    # Deterioration warning
    is_worse, msg = check_clinical_deterioration("HIGH", 0.90, [history[1]])
    assert is_worse
    assert "deteriorated" in msg.lower()


def test_audit_logging_and_erasure(temp_provider):
    """Verify audit log tracks access and patient deletion removes records."""
    p = temp_provider.register_patient("Meera Sen", 29, "Female", "+91 9000011111", consent_given=True)
    pid = p["patient_id"]

    logs = temp_provider.get_audit_logs(pid)
    assert len(logs) >= 1
    assert any(l["action"] == "PATIENT_CREATED" for l in logs)

    # Delete data
    ok = temp_provider.delete_patient_data(pid)
    assert ok
    assert temp_provider.get_patient(pid) is None
    assert len(temp_provider.get_history(pid)) == 0


def test_fhir_export(temp_provider):
    """Verify FHIR R4 JSON bundle export."""
    p = temp_provider.register_patient("Anil Roy", 60, "Male", "+91 9444455555", consent_given=True)
    pid = p["patient_id"]
    temp_provider.add_visit(pid, "HOSP-001", {}, {}, "MEDIUM", 0.55, "Clinic")

    fhir_bundle = temp_provider.export_fhir_json(pid)
    assert fhir_bundle["resourceType"] == "Bundle"
    assert fhir_bundle["type"] == "collection"
    assert len(fhir_bundle["entry"]) >= 2
    res_types = [e["resource"]["resourceType"] for e in fhir_bundle["entry"]]
    assert "Patient" in res_types
    assert "Encounter" in res_types
    assert "Observation" in res_types


def test_doctor_auth():
    """Verify doctor credentials verification."""
    doc = authenticate_doctor("dr.sharma", "doctor123")
    assert doc is not None
    assert doc["doctor_id"] == "DOC-001"
    assert "Sharma" in doc["name"]

    bad = authenticate_doctor("dr.sharma", "wrongpassword")
    assert bad is None
