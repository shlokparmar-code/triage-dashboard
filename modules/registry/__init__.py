"""Patient Registry, Multi-Hospital History, and Data Privacy package."""

from .crypto import encrypt_mobile, decrypt_mobile, hash_mobile, mask_mobile, validate_mobile
from .db import init_db, get_db_connection
from .otp import generate_otp, verify_otp, send_sms_otp
from .auth import authenticate_doctor, list_doctors
from .provider import HistoryProvider, LocalRegistryProvider, check_clinical_deterioration

__all__ = [
    "encrypt_mobile",
    "decrypt_mobile",
    "hash_mobile",
    "mask_mobile",
    "validate_mobile",
    "init_db",
    "get_db_connection",
    "generate_otp",
    "verify_otp",
    "send_sms_otp",
    "authenticate_doctor",
    "list_doctors",
    "HistoryProvider",
    "LocalRegistryProvider",
    "check_clinical_deterioration"
]
