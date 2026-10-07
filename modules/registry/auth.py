"""Doctor authentication and credential verification for Hospital Mode."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.registry.db import get_db_connection, init_db


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()


def authenticate_doctor(
    username: str, password: str, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Authenticate clinician credentials against doctors table."""
    if not username or not password:
        return None

    path = db_path or config.REGISTRY_DB_PATH
    init_db(path)
    pw_hash = hash_password(password)
    conn = get_db_connection(path)
    cur = conn.cursor()
    cur.execute("""
    SELECT d.*, h.name as hospital_name, h.city as hospital_city
    FROM doctors d
    LEFT JOIN hospitals h ON d.hospital_id = h.hospital_id
    WHERE d.username = ? AND d.password_hash = ?
    """, (username.strip().lower(), pw_hash))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "doctor_id": row["doctor_id"],
        "username": row["username"],
        "name": row["name"],
        "hospital_id": row["hospital_id"],
        "hospital_name": row["hospital_name"] or row["hospital_id"],
        "hospital_city": row["hospital_city"] or "",
        "role": row["role"]
    }


def list_doctors(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List seeded demo doctors for convenience."""
    path = db_path or config.REGISTRY_DB_PATH
    init_db(path)
    conn = get_db_connection(path)
    cur = conn.cursor()
    cur.execute("""
    SELECT d.username, d.name, d.role, h.name as hospital_name
    FROM doctors d
    LEFT JOIN hospitals h ON d.hospital_id = h.hospital_id
    ORDER BY d.name ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
