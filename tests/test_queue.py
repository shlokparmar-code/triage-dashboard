"""Unit tests for Multi-Patient Triage Queue with Priority Lane and FCFS order."""

import pytest
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from dashboard.queue import (
    add_to_queue, get_queue, call_next, mark_as_seen, remove_from_queue, clear_queue, export_queue_csv
)
from modules.registry.db import init_db


@pytest.fixture
def queue_db(tmp_path):
    db_file = tmp_path / "test_queue.db"
    init_db(db_file)
    return db_file


def test_priority_lane_isolation(queue_db):
    """Verify HIGH urgency patients go strictly to Priority Lane ahead of regular queue."""
    # Add LOW patient
    add_to_queue("P1", "HOSP-001", "Alice", "LOW", 0.15, "Mild cough", db_path=queue_db)
    # Add MEDIUM patient
    add_to_queue("P2", "HOSP-001", "Bob", "MEDIUM", 0.50, "Fever 101F", db_path=queue_db)
    # Add HIGH patient
    add_to_queue("P3", "HOSP-001", "Charlie", "HIGH", 0.95, "Chest pain and dyspnea", db_path=queue_db)

    priority_lane, regular_queue = get_queue("HOSP-001", db_path=queue_db)

    assert len(priority_lane) == 1
    assert priority_lane[0]["name"] == "Charlie"
    assert priority_lane[0]["urgency"] == "HIGH"
    assert priority_lane[0]["is_priority"] == 1

    assert len(regular_queue) == 2
    assert regular_queue[0]["name"] == "Alice"
    assert regular_queue[1]["name"] == "Bob"


def test_call_next_priority_precedence(queue_db):
    """Verify call_next always takes Priority Lane before regular FCFS queue."""
    add_to_queue("P1", "HOSP-001", "Early Low Patient", "LOW", 0.10, "Sore throat", db_path=queue_db)
    add_to_queue("P2", "HOSP-001", "Late Emergency Patient", "HIGH", 0.90, "Severe hypoxia", db_path=queue_db)

    # First call_next must return the HIGH patient even though they arrived later
    first_called = call_next("HOSP-001", db_path=queue_db)
    assert first_called is not None
    assert first_called["patient_id"] == "P2"
    assert first_called["urgency"] == "HIGH"

    # Next call_next returns the LOW patient
    second_called = call_next("HOSP-001", db_path=queue_db)
    assert second_called is not None
    assert second_called["patient_id"] == "P1"

    # Queue should now be empty of waiting patients
    p_lane, reg_q = get_queue("HOSP-001", db_path=queue_db)
    assert len(p_lane) == 0
    assert len(reg_q) == 0


def test_reassessment_escalation_to_priority_lane(queue_db):
    """Verify re-assessment from LOW/MEDIUM to HIGH moves patient to Priority Lane."""
    # Patient arrives as MEDIUM
    res1 = add_to_queue("P1", "HOSP-001", "Ravi", "MEDIUM", 0.45, "Abdominal pain", db_path=queue_db)
    p_lane, reg_q = get_queue("HOSP-001", db_path=queue_db)
    assert len(p_lane) == 0
    assert len(reg_q) == 1

    # Patient condition deteriorates to HIGH
    res2 = add_to_queue("P1", "HOSP-001", "Ravi", "HIGH", 0.88, "Rigid abdomen & acute shock", db_path=queue_db)
    assert res2["action"] == "UPDATED"

    p_lane_after, reg_q_after = get_queue("HOSP-001", db_path=queue_db)
    assert len(p_lane_after) == 1
    assert p_lane_after[0]["name"] == "Ravi"
    assert p_lane_after[0]["urgency"] == "HIGH"
    assert len(reg_q_after) == 0


def test_no_duplicate_queue_entries(queue_db):
    """Verify adding same active patient updates instead of creating duplicate."""
    add_to_queue("P1", "HOSP-001", "Asha", "LOW", 0.20, "Headache", db_path=queue_db)
    add_to_queue("P1", "HOSP-001", "Asha", "LOW", 0.25, "Headache slightly worse", db_path=queue_db)

    _, reg_q = get_queue("HOSP-001", db_path=queue_db)
    assert len(reg_q) == 1
    assert reg_q[0]["complaint"] == "Headache slightly worse"


def test_csv_export(queue_db):
    """Verify CSV export format."""
    add_to_queue("P1", "HOSP-001", "John", "HIGH", 0.90, "Shortness of breath", db_path=queue_db)
    csv_str = export_queue_csv("HOSP-001", db_path=queue_db)
    assert "Queue_Position,Lane,Queue_ID" in csv_str
    assert "John" in csv_str
    assert "Priority Lane" in csv_str
