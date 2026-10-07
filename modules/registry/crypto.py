"""Cryptographic utilities for patient identifier encryption, hashing, and masking."""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from cryptography.fernet import Fernet

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

SALT = "ANTIGRAVITY_TRIAGE_SALT_2026_SECURE"


def get_fernet_key() -> bytes:
    """Retrieve or generate Fernet key for mobile number encryption."""
    env_key = os.environ.get("FERNET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    key_file = config.FERNET_KEY_PATH
    if key_file.exists():
        try:
            return key_file.read_bytes().strip()
        except Exception:
            pass

    # Generate and persist new key
    key = Fernet.generate_key()
    try:
        key_file.write_bytes(key)
    except Exception:
        pass
    return key


_FERNET_INSTANCE = None


def get_fernet() -> Fernet:
    """Singleton Fernet instance."""
    global _FERNET_INSTANCE
    if _FERNET_INSTANCE is None:
        _FERNET_INSTANCE = Fernet(get_fernet_key())
    return _FERNET_INSTANCE


def encrypt_mobile(raw_mobile: str) -> str:
    """Encrypt raw mobile number using AES-128-CBC via Fernet."""
    if not raw_mobile:
        return ""
    f = get_fernet()
    return f.encrypt(raw_mobile.strip().encode("utf-8")).decode("utf-8")


def decrypt_mobile(encrypted_token: str) -> str:
    """Decrypt mobile number."""
    if not encrypted_token:
        return ""
    try:
        f = get_fernet()
        return f.decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
    except Exception:
        return "[Decryption Failed]"


def hash_mobile(raw_mobile: str) -> str:
    """Deterministic salted SHA-256 hash for fast database lookup."""
    clean = re.sub(r"[^\d+]", "", str(raw_mobile).strip())
    salted_input = f"{SALT}:{clean}".encode("utf-8")
    return hashlib.sha256(salted_input).hexdigest()


def mask_mobile(raw_mobile: str) -> str:
    """Mask mobile number for UI display (e.g., +91 ******3210)."""
    if not raw_mobile or len(raw_mobile) < 4:
        return "******"
    clean = raw_mobile.strip()
    if clean.startswith("+"):
        parts = clean.split()
        if len(parts) > 1:
            code = parts[0]
            rest = "".join(parts[1:])
            masked_rest = "*" * max(0, len(rest) - 4) + rest[-4:]
            return f"{code} {masked_rest}"
    # Standard masking keeping last 4 digits
    digits = re.sub(r"[^\d]", "", clean)
    if len(digits) >= 4:
        return "*" * (len(digits) - 4) + digits[-4:]
    return "******"


def validate_mobile(country_code: str, number_str: str) -> Tuple[bool, str, str]:
    """Validate phone number against country specific rules.

    Returns:
        (is_valid, normalized_full_number, error_message)
    """
    clean_code = country_code.strip()
    digits = re.sub(r"\D", "", number_str.strip())

    cfg = config.COUNTRY_PHONE_CONFIG.get(clean_code)
    if not cfg:
        # Generic international validation (7-15 digits)
        if 7 <= len(digits) <= 15:
            return True, f"{clean_code} {digits}", ""
        return False, "", f"Invalid phone format for {clean_code}."

    pattern = cfg["regex"]
    if not re.match(pattern, digits):
        expected_len = cfg["digits"]
        example = cfg["example"]
        return False, "", f"Invalid {cfg['name']} number. Must be {expected_len} digits (e.g., {example})."

    return True, f"{clean_code} {digits}", ""
