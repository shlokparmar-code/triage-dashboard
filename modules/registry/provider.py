"""Patient Registry and Multi-Hospital History Provider.

Implements the abstract HistoryProvider interface with SQLite local storage,
FHIR-style export, clinical trend detection, audit logging, and data privacy safeguards.
Designed for drop-in replacement by ABDM (India Ayushman Bharat) or international FHIR gateways.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.registry.crypto import encrypt_mobile, decrypt_mobile, hash_mobile, mask_mobile
from modules.registry.db import get_db_connection, init_db, get_age_from_dob


class HistoryProvider(ABC):
    """Abstract interface for patient registry and multi-hospital historical health records."""

    @abstractmethod
    def find_patient_by_mobile(self, raw_mobile: str) -> Optional[Dict[str, Any]]:
        """Look up patient by mobile number using secure salted hash."""
        pass

    @abstractmethod
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient details by ID."""
        pass

    @abstractmethod
    def register_patient(
        self, name: str, age: int, sex: str, raw_mobile: str, consent_given: bool = True
    ) -> Dict[str, Any]:
        """Create a new patient record with encrypted mobile storage."""
        pass

    @abstractmethod
    def add_visit(
        self,
        patient_id: str,
        hospital_id: str,
        inputs: Dict[str, Any],
        module_results: Dict[str, Any],
        final_urgency: str,
        final_score: float,
        referral_level: str,
        notes: Optional[str] = None
    ) -> str:
        """Record an evaluated clinical triage visit."""
        pass

    @abstractmethod
    def get_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch all historical visits across hospitals for a patient."""
        pass

    @abstractmethod
    def delete_patient_data(self, patient_id: str) -> bool:
        """Right to erasure / Delete my data request."""
        pass

    @abstractmethod
    def log_audit(
        self, who: str, role: str, hospital_id: Optional[str], patient_id: Optional[str], action: str
    ) -> None:
        """Log data access or clinical modification."""
        pass

    @abstractmethod
    def export_fhir_json(self, patient_id: str) -> Dict[str, Any]:
        """Export patient history as standard FHIR R4 JSON bundle."""
        pass


class LocalRegistryProvider(HistoryProvider):
    """Local SQLite implementation of HistoryProvider with multi-hospital consolidation."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.REGISTRY_DB_PATH
        init_db(self.db_path)

    def find_patient_by_mobile(self, raw_mobile: str) -> Optional[Dict[str, Any]]:
        """Lookup patient using deterministic salted SHA-256 hash."""
        m_hash = hash_mobile(raw_mobile)
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM patients WHERE mobile_hash = ?", (m_hash,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return self._format_patient_dict(row)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return self._format_patient_dict(row)

    def register_patient(
        self, name: str, age: int, sex: str, raw_mobile: str, consent_given: bool = True,
        date_of_birth: Optional[str] = None
    ) -> Dict[str, Any]:
        if not consent_given:
            raise ValueError("Patient consent is required to create a health record.")

        # Check existing
        existing = self.find_patient_by_mobile(raw_mobile)
        if existing:
            return existing

        patient_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
        m_hash = hash_mobile(raw_mobile)
        m_enc = encrypt_mobile(raw_mobile)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Estimate DOB from age if not provided
        dob = date_of_birth or f"{datetime.now().year - int(age)}-07-01"

        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO patients (patient_id, name, age, sex, mobile_hash, mobile_encrypted, consent_given, created_at, date_of_birth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, name.strip(), int(age), sex.strip(), m_hash, m_enc, 1 if consent_given else 0, created_at, dob))
        conn.commit()
        conn.close()

        self.log_audit("system", "Registration", None, patient_id, "PATIENT_CREATED")
        return {
            "patient_id": patient_id,
            "name": name.strip(),
            "age": int(age),
            "date_of_birth": dob,
            "sex": sex.strip(),
            "mobile_masked": mask_mobile(raw_mobile),
            "consent_given": consent_given,
            "created_at": created_at
        }

    def add_visit(
        self,
        patient_id: str,
        hospital_id: str,
        inputs: Dict[str, Any],
        module_results: Dict[str, Any],
        final_urgency: str,
        final_score: float,
        referral_level: str,
        notes: Optional[str] = None
    ) -> str:
        visit_id = f"VISIT-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO visits (visit_id, patient_id, hospital_id, timestamp, inputs, module_results, final_urgency, final_score, referral_level, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            visit_id,
            patient_id,
            hospital_id,
            timestamp,
            json.dumps(inputs),
            json.dumps(module_results),
            final_urgency,
            float(final_score),
            referral_level,
            notes or ""
        ))
        conn.commit()
        conn.close()

        self.log_audit("clinician", "Triage", hospital_id, patient_id, f"VISIT_RECORDED_{final_urgency}")
        return visit_id

    def get_history(self, patient_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        SELECT v.rowid, v.*, h.name as hospital_name, h.city as hospital_city, h.country as hospital_country
        FROM visits v
        LEFT JOIN hospitals h ON v.hospital_id = h.hospital_id
        WHERE v.patient_id = ?
        ORDER BY v.timestamp DESC, v.rowid DESC
        """, (patient_id,))
        rows = cur.fetchall()
        conn.close()

        history = []
        for r in rows:
            history.append({
                "visit_id": r["visit_id"],
                "hospital_id": r["hospital_id"],
                "hospital_name": r["hospital_name"] or r["hospital_id"],
                "hospital_city": r["hospital_city"] or "",
                "hospital_country": r["hospital_country"] or "",
                "timestamp": r["timestamp"],
                "inputs": json.loads(r["inputs"]) if r["inputs"] else {},
                "module_results": json.loads(r["module_results"]) if r["module_results"] else {},
                "final_urgency": r["final_urgency"],
                "final_score": float(r["final_score"]),
                "referral_level": r["referral_level"],
                "notes": r["notes"] or ""
            })
        return history

    def delete_patient_data(self, patient_id: str) -> bool:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        # Delete queue entries
        cur.execute("DELETE FROM queue WHERE patient_id = ?", (patient_id,))
        # Delete visits
        cur.execute("DELETE FROM visits WHERE patient_id = ?", (patient_id,))
        # Delete patient
        cur.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        conn.commit()
        conn.close()

        self.log_audit("patient", "Self", None, patient_id, "PATIENT_DATA_PURGED")
        return True

    def log_audit(
        self, who: str, role: str, hospital_id: Optional[str], patient_id: Optional[str], action: str
    ) -> None:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO audit_log (who, role, hospital_id, patient_id, action, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (who, role, hospital_id, patient_id, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    def get_audit_logs(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        if patient_id:
            cur.execute("""
            SELECT a.*, h.name as hospital_name
            FROM audit_log a
            LEFT JOIN hospitals h ON a.hospital_id = h.hospital_id
            WHERE a.patient_id = ?
            ORDER BY a.timestamp DESC
            """, (patient_id,))
        else:
            cur.execute("""
            SELECT a.*, h.name as hospital_name
            FROM audit_log a
            LEFT JOIN hospitals h ON a.hospital_id = h.hospital_id
            ORDER BY a.timestamp DESC LIMIT 100
            """)
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_hospitals(self) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM hospitals ORDER BY name ASC")
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def export_fhir_json(self, patient_id: str) -> Dict[str, Any]:
        """Export consolidated patient history as FHIR R4 standard JSON bundle."""
        patient = self.get_patient(patient_id)
        if not patient:
            return {}

        history = self.get_history(patient_id)

        entries = [
            {
                "fullUrl": f"urn:uuid:{patient_id}",
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "identifier": [{"system": "https://triage-ai.org/patients", "value": patient_id}],
                    "name": [{"text": patient["name"]}],
                    "gender": "male" if patient["sex"].lower() == "male" else "female",
                    "extension": [
                        {"url": "https://triage-ai.org/fhir/consent", "valueBoolean": bool(patient.get("consent_given", True))}
                    ]
                }
            }
        ]

        for v in history:
            enc_id = v["visit_id"]
            entries.append({
                "fullUrl": f"urn:uuid:{enc_id}",
                "resource": {
                    "resourceType": "Encounter",
                    "id": enc_id,
                    "status": "finished",
                    "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "EMER", "display": "emergency"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "period": {"start": v["timestamp"]},
                    "serviceProvider": {"display": v["hospital_name"]},
                    "priority": {
                        "coding": [{"system": "https://triage-ai.org/urgency", "code": v["final_urgency"], "display": f"Urgency {v['final_urgency']}"}]
                    }
                }
            })

            # Observations for vitals / score
            obs_id = f"OBS-{enc_id}"
            entries.append({
                "fullUrl": f"urn:uuid:{obs_id}",
                "resource": {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]}],
                    "code": {"text": "Triage Urgency Score"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": f"Encounter/{enc_id}"},
                    "effectiveDateTime": v["timestamp"],
                    "valueQuantity": {"value": v["final_score"], "unit": "probability", "system": "http://unitsofmeasure.org", "code": "1"}
                }
            })

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "total": len(entries),
            "entry": entries
        }

    def _format_patient_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "patient_id": row["patient_id"],
            "name": row["name"],
            "age": int(row["age"]),
            "sex": row["sex"],
            "mobile_masked": mask_mobile(decrypt_mobile(row["mobile_encrypted"])),
            "mobile_raw": decrypt_mobile(row["mobile_encrypted"]),
            "consent_given": bool(row["consent_given"]),
            "created_at": row["created_at"]
        }


def check_clinical_deterioration(current_urgency: str, current_score: float, history: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Examine prior visits across hospitals and alert if patient condition is worsening."""
    if not history:
        return False, "Initial assessment: No previous visits recorded in multi-hospital network."

    tier_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    curr_rank = tier_order.get(current_urgency, 1)

    last_visit = history[0]
    prev_urgency = last_visit["final_urgency"]
    prev_rank = tier_order.get(prev_urgency, 1)
    prev_score = last_visit["final_score"]

    if curr_rank > prev_rank:
        return True, (
            f"⚠️ CLINICAL ALERT: Patient condition has deteriorated from {prev_urgency} "
            f"({prev_score:.0%}, recorded at {last_visit['hospital_name']}) to {current_urgency} ({current_score:.0%})."
        )
    elif curr_rank == prev_rank and current_score >= prev_score + 0.15:
        return True, (
            f"⚠️ NOTICE: Urgency score has escalated within the {current_urgency} tier "
            f"({prev_score:.0%} -> {current_score:.0%}) since last visit."
        )

    return False, f"Condition stable compared to prior visit ({prev_urgency} at {last_visit['hospital_name']})."
