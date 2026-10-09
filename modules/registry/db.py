"""SQLite Database schema, connection management, and demo seeding for Patient Registry."""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.registry.crypto import encrypt_mobile, hash_mobile


def get_age_from_dob(dob_str: str) -> dict:
    """Return {'years': int, 'months': int, 'display': str} from ISO date string (YYYY-MM-DD)."""
    if not dob_str:
        return {"years": 0, "months": 0, "display": "Unknown"}
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        total_months = (today.year - dob.year) * 12 + today.month - dob.month
        if today.day < dob.day:
            total_months -= 1
        months = total_months % 12
        if years < 5:
            display = f"{years} yr {months} mo" if years > 0 else f"{total_months} months"
        else:
            display = f"{years} years"
        return {"years": max(0, years), "months": max(0, months), "display": display}
    except (ValueError, TypeError):
        return {"years": 0, "months": 0, "display": "Unknown"}


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an active SQLite database connection with Row factory."""
    path = db_path or config.REGISTRY_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create all registry tables and seed demo hospitals, doctors, and multi-hospital patients."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Hospitals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS hospitals (
        hospital_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT NOT NULL,
        country TEXT NOT NULL
    );
    """)

    # 2. Patients
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        sex TEXT NOT NULL,
        mobile_hash TEXT UNIQUE NOT NULL,
        mobile_encrypted TEXT NOT NULL,
        consent_given INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );
    """)

    # 3. Visits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        visit_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        hospital_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        inputs TEXT NOT NULL,
        module_results TEXT NOT NULL,
        final_urgency TEXT NOT NULL,
        final_score REAL NOT NULL,
        referral_level TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
    );
    """)

    # 4. Queue
    cur.execute("""
    CREATE TABLE IF NOT EXISTS queue (
        queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        hospital_id TEXT NOT NULL,
        visit_id TEXT,
        name TEXT NOT NULL,
        urgency TEXT NOT NULL,
        score REAL NOT NULL,
        complaint TEXT NOT NULL,
        arrival_time TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'WAITING',
        is_priority INTEGER NOT NULL DEFAULT 0
    );
    """)

    # 5. Audit Log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        who TEXT NOT NULL,
        role TEXT NOT NULL,
        hospital_id TEXT,
        patient_id TEXT,
        action TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)

    # 6. Doctors
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        hospital_id TEXT NOT NULL,
        role TEXT NOT NULL
    );
    """)

    conn.commit()

    # --- Migration: add date_of_birth column if not present ---
    cur.execute("PRAGMA table_info(patients)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "date_of_birth" not in existing_cols:
        cur.execute("ALTER TABLE patients ADD COLUMN date_of_birth TEXT")
        # Back-fill estimated DOB from age (mid-year)
        cur.execute("SELECT patient_id, age FROM patients WHERE date_of_birth IS NULL OR date_of_birth = ''")
        rows = cur.fetchall()
        current_year = datetime.now().year
        for row in rows:
            est_dob = f"{current_year - row[1]}-07-01"
            cur.execute("UPDATE patients SET date_of_birth = ? WHERE patient_id = ?", (est_dob, row[0]))
        conn.commit()

    # Seed Initial Data if empty
    cur.execute("SELECT COUNT(*) FROM hospitals")
    if cur.fetchone()[0] == 0:
        seed_demo_data(conn)

    conn.close()


def seed_demo_data(conn: sqlite3.Connection) -> None:
    """Seed initial hospitals, doctors, and multi-hospital visit history."""
    cur = conn.cursor()
    now = datetime.now()

    # Seed Hospitals
    hospitals = [
        ("HOSP-001", "Apex District Hospital, Jaipur", "Jaipur", "India"),
        ("HOSP-002", "Alwar Community Health Centre", "Alwar", "India"),
        ("HOSP-003", "St. Jude Rural Clinic, Bangalore", "Bangalore", "India"),
        ("HOSP-004", "Metro General Emergency Hospital", "London", "UK")
    ]
    cur.executemany("INSERT INTO hospitals VALUES (?, ?, ?, ?)", hospitals)

    # Seed Doctors
    def hash_pw(pw: str) -> str:
        return hashlib.sha256(pw.encode()).hexdigest()

    doctors = [
        ("DOC-001", "dr.sharma", hash_pw("doctor123"), "Dr. Priya Sharma (Triage Lead)", "HOSP-001", "Medical Officer"),
        ("DOC-002", "dr.patel", hash_pw("doctor123"), "Dr. Rajesh Patel (General Physician)", "HOSP-002", "Senior Resident"),
        ("DOC-003", "dr.adams", hash_pw("doctor123"), "Dr. Sarah Adams (Consultant)", "HOSP-004", "Emergency Consultant")
    ]
    cur.executemany("INSERT INTO doctors VALUES (?, ?, ?, ?, ?, ?)", doctors)

    # Seed Demo Patient with multi-hospital history (Ramesh Kumar, +91 9876543210)
    demo_phone = "+91 9876543210"
    p_id = "PAT-DEMO-001"
    cur.execute("""
    INSERT INTO patients (patient_id, name, age, sex, mobile_hash, mobile_encrypted, consent_given, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        p_id,
        "Ramesh Kumar",
        58,
        "Male",
        hash_mobile(demo_phone),
        encrypt_mobile(demo_phone),
        1,
        (now - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    ))

    # Visit 1: 60 days ago at Alwar CHC (HOSP-002) - LOW urgency
    v1_inputs = {"resting_bp": 128, "glucose": 110, "symptoms": "Mild occasional headache and fatigue"}
    v1_results = {"risk_screening": {"urgency": "LOW", "score": 0.22}, "symptom_triage": {"urgency": "LOW", "score": 0.15}}
    cur.execute("""
    INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "VISIT-001",
        p_id,
        "HOSP-002",
        (now - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(v1_inputs),
        json.dumps(v1_results),
        "LOW",
        0.20,
        "Self-Care & Routine Follow-up",
        "Advised lifestyle changes and dietary sodium reduction."
    ))

    # Visit 2: 15 days ago at St. Jude (HOSP-003) - MEDIUM urgency
    v2_inputs = {"resting_bp": 145, "glucose": 165, "symptoms": "Persistent dizziness and elevated morning fasting blood sugar"}
    v2_results = {"risk_screening": {"urgency": "MEDIUM", "score": 0.52}, "symptom_triage": {"urgency": "MEDIUM", "score": 0.48}}
    cur.execute("""
    INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "VISIT-002",
        p_id,
        "HOSP-003",
        (now - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(v2_inputs),
        json.dumps(v2_results),
        "MEDIUM",
        0.51,
        "Primary Health Centre Consultation",
        "Suspected prediabetes / stage-1 hypertension. Ordered lab panel."
    ))

    # Visit 3: 2 days ago at Apex District Hospital (HOSP-001) - HIGH urgency
    v3_inputs = {"resting_bp": 172, "glucose": 210, "symptoms": "Substernal chest tightness and shortness of breath upon exertion"}
    v3_results = {"risk_screening": {"urgency": "HIGH", "score": 0.78}, "vitals_anomaly": {"urgency": "HIGH", "score": 0.85}, "symptom_triage": {"urgency": "HIGH", "score": 0.90}}
    cur.execute("""
    INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "VISIT-003",
        p_id,
        "HOSP-001",
        (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps(v3_inputs),
        json.dumps(v3_results),
        "HIGH",
        0.88,
        "Emergency-Capable District Hospital",
        "ECG performed; initiated emergency cardiology consult."
    ))

    # Seed Demo Audit Log
    cur.execute("""
    INSERT INTO audit_log (who, role, hospital_id, patient_id, action, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "dr.sharma",
        "Medical Officer",
        "HOSP-001",
        p_id,
        "RECORD_VIEW",
        (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
