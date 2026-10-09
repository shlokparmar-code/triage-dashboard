"""Multi-Patient Triage Queue with Priority Lane and FCFS Waiting Rules.

Persisted in SQLite database (survives app refreshes and multi-session restarts).
HIGH urgency patients immediately enter the red-highlighted Priority Lane ahead of all others.
LOW and MEDIUM patients share an arrival-ordered FCFS queue (configurable via QUEUE_MEDIUM_BEFORE_LOW).
"""

import io
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config
from modules.registry.db import get_db_connection, init_db


# ------------------------------------------------------------
# WAITING-TIME ESCALATION
# Patients can get worse while waiting. Anyone waiting longer than the limit
# for their tier is flagged OVERDUE so staff re-check them.
# Limits are in minutes. Override in config.py with WAIT_ESCALATION_MINUTES.
# ------------------------------------------------------------
DEFAULT_WAIT_LIMITS_MIN = {"HIGH": 10, "MEDIUM": 30, "LOW": 60}


def get_wait_limits() -> Dict[str, int]:
    """Return waiting limits per tier (minutes), merging config overrides."""
    limits = dict(DEFAULT_WAIT_LIMITS_MIN)
    cfg = getattr(config, "WAIT_ESCALATION_MINUTES", None)
    if isinstance(cfg, dict):
        limits.update({k: int(v) for k, v in cfg.items() if k in limits})
    return limits


def minutes_waiting(arrival_time: str, now: Optional[datetime] = None) -> int:
    """Whole minutes since arrival_time ('YYYY-MM-DD HH:MM:SS'). Bad input gives 0."""
    now = now or datetime.now()
    try:
        arrived = datetime.fromisoformat(str(arrival_time).strip())
    except (ValueError, TypeError):
        return 0
    return max(0, int((now - arrived).total_seconds() // 60))


def format_wait(minutes: int) -> str:
    """Readable waiting time, e.g. '45 min' or '1 h 05 min'."""
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def is_overdue(
    patient: Dict[str, Any],
    now: Optional[datetime] = None,
    limits: Optional[Dict[str, int]] = None,
) -> bool:
    """True if the patient has waited longer than the limit for their tier."""
    limits = limits or get_wait_limits()
    limit = limits.get(str(patient.get("urgency", "")).upper())
    if limit is None:
        return False
    return minutes_waiting(patient.get("arrival_time", ""), now) > limit


def get_overdue_patients(
    patients: List[Dict[str, Any]], now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Return the overdue patients, longest wait first."""
    limits = get_wait_limits()
    late = [p for p in patients if is_overdue(p, now, limits)]
    return sorted(late, key=lambda p: minutes_waiting(p["arrival_time"], now), reverse=True)


def add_to_queue(
    patient_id: str,
    hospital_id: str,
    name: str,
    urgency: str,
    score: float,
    complaint: str,
    visit_id: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Add or re-triage a patient in the hospital queue. Prevents duplicate active waiting entries."""
    path = db_path or config.REGISTRY_DB_PATH
    init_db(path)
    conn = get_db_connection(path)
    cur = conn.cursor()

    is_priority = 1 if urgency == config.TIER_HIGH else 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if patient is already actively waiting in queue
    cur.execute("""
    SELECT queue_id, urgency, is_priority FROM queue
    WHERE patient_id = ? AND hospital_id = ? AND status = 'WAITING'
    """, (patient_id, hospital_id))
    existing = cur.fetchone()

    if existing:
        q_id = existing["queue_id"]
        # Update existing record (if re-assessed to HIGH, automatically promote to Priority Lane)
        cur.execute("""
        UPDATE queue
        SET urgency = ?, score = ?, complaint = ?, is_priority = ?, visit_id = ?
        WHERE queue_id = ?
        """, (urgency, float(score), complaint.strip(), is_priority, visit_id, q_id))
        conn.commit()
        conn.close()
        return {"queue_id": q_id, "action": "UPDATED", "is_priority": bool(is_priority)}

    # Insert new patient
    cur.execute("""
    INSERT INTO queue (patient_id, hospital_id, visit_id, name, urgency, score, complaint, arrival_time, status, is_priority)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'WAITING', ?)
    """, (patient_id, hospital_id, visit_id, name.strip(), urgency, float(score), complaint.strip(), now_str, is_priority))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {"queue_id": new_id, "action": "INSERTED", "is_priority": bool(is_priority)}


def get_queue(
    hospital_id: Optional[str] = None, db_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve active waiting patients split into (Priority Lane, Regular Queue)."""
    path = db_path or config.REGISTRY_DB_PATH
    init_db(path)
    conn = get_db_connection(path)
    cur = conn.cursor()

    # 1. Priority Lane (Strictly HIGH urgency or is_priority=1, sorted by arrival time)
    if hospital_id:
        cur.execute("""
        SELECT * FROM queue
        WHERE status = 'WAITING' AND (is_priority = 1 OR urgency = 'HIGH') AND hospital_id = ?
        ORDER BY arrival_time ASC, queue_id ASC
        """, (hospital_id,))
    else:
        cur.execute("""
        SELECT * FROM queue
        WHERE status = 'WAITING' AND (is_priority = 1 OR urgency = 'HIGH')
        ORDER BY arrival_time ASC, queue_id ASC
        """)
    priority_rows = [dict(r) for r in cur.fetchall()]

    # 2. Regular Queue (LOW and MEDIUM patients)
    order_clause = "arrival_time ASC, queue_id ASC"
    if getattr(config, "QUEUE_MEDIUM_BEFORE_LOW", False):
        order_clause = "CASE WHEN urgency = 'MEDIUM' THEN 1 ELSE 2 END, arrival_time ASC, queue_id ASC"

    if hospital_id:
        cur.execute(f"""
        SELECT * FROM queue
        WHERE status = 'WAITING' AND is_priority = 0 AND urgency != 'HIGH' AND hospital_id = ?
        ORDER BY {order_clause}
        """, (hospital_id,))
    else:
        cur.execute(f"""
        SELECT * FROM queue
        WHERE status = 'WAITING' AND is_priority = 0 AND urgency != 'HIGH'
        ORDER BY {order_clause}
        """)
    regular_rows = [dict(r) for r in cur.fetchall()]

    conn.close()
    return priority_rows, regular_rows


def call_next(
    hospital_id: Optional[str] = None, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Call the next patient in line (Priority Lane first, then FCFS regular queue)."""
    priority_lane, regular_queue = get_queue(hospital_id, db_path)

    next_patient = None
    if priority_lane:
        next_patient = priority_lane[0]
    elif regular_queue:
        next_patient = regular_queue[0]

    if not next_patient:
        return None

    path = db_path or config.REGISTRY_DB_PATH
    conn = get_db_connection(path)
    cur = conn.cursor()
    cur.execute("""
    UPDATE queue SET status = 'IN_CONSULTATION' WHERE queue_id = ?
    """, (next_patient["queue_id"],))
    conn.commit()
    conn.close()

    return next_patient


def mark_as_seen(queue_id: int, db_path: Optional[Path] = None) -> bool:
    """Mark a patient consultation as completed."""
    path = db_path or config.REGISTRY_DB_PATH
    conn = get_db_connection(path)
    cur = conn.cursor()
    cur.execute("UPDATE queue SET status = 'COMPLETED' WHERE queue_id = ?", (queue_id,))
    conn.commit()
    conn.close()
    return True


def remove_from_queue(queue_id: int, db_path: Optional[Path] = None) -> bool:
    """Remove or cancel a patient from the waiting queue."""
    path = db_path or config.REGISTRY_DB_PATH
    conn = get_db_connection(path)
    cur = conn.cursor()
    cur.execute("UPDATE queue SET status = 'REMOVED' WHERE queue_id = ?", (queue_id,))
    conn.commit()
    conn.close()
    return True


def clear_queue(hospital_id: Optional[str] = None, db_path: Optional[Path] = None) -> bool:
    """Clear waiting patients in queue."""
    path = db_path or config.REGISTRY_DB_PATH
    conn = get_db_connection(path)
    cur = conn.cursor()
    if hospital_id:
        cur.execute("DELETE FROM queue WHERE hospital_id = ? AND status = 'WAITING'", (hospital_id,))
    else:
        cur.execute("DELETE FROM queue WHERE status = 'WAITING'")
    conn.commit()
    conn.close()
    return True


def export_queue_csv(hospital_id: Optional[str] = None, db_path: Optional[Path] = None) -> str:
    """Export the active triage queue to a CSV string."""
    priority_lane, regular_queue = get_queue(hospital_id, db_path)
    all_patients = priority_lane + regular_queue

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Queue_Position", "Lane", "Queue_ID", "Patient_ID", "Name", "Urgency", "Score", "Complaint", "Arrival_Time"])

    for i, p in enumerate(priority_lane, start=1):
        writer.writerow([f"Priority-{i}", "Priority Lane", p["queue_id"], p["patient_id"], p["name"], p["urgency"], f"{p['score']:.0%}", p["complaint"], p["arrival_time"]])

    for i, p in enumerate(regular_queue, start=1):
        writer.writerow([f"Regular-{i}", "Standard Queue", p["queue_id"], p["patient_id"], p["name"], p["urgency"], f"{p['score']:.0%}", p["complaint"], p["arrival_time"]])

    return output.getvalue()


def export_queue_pdf(hospital_id: Optional[str] = None, db_path: Optional[Path] = None) -> bytes:
    """Export the active triage queue (Priority Lane first) as a PDF."""
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    priority_lane, regular_queue = get_queue(hospital_id, db_path)
    styles = getSampleStyleSheet()
    cell = styles["BodyText"]
    cell.fontSize = 8
    cell.leading = 10

    def P(text: Any) -> Paragraph:
        return Paragraph(escape(str(text)), cell)

    rows = [[P(h) for h in ["Position", "Lane", "Patient ID", "Name", "Urgency", "Score", "Complaint", "Arrival"]]]
    for i, p in enumerate(priority_lane, start=1):
        rows.append([P(f"Priority-{i}"), P("Priority Lane"), P(p["patient_id"]), P(p["name"]),
                     P(p["urgency"]), P(f"{p['score']:.0%}"), P(p["complaint"]), P(p["arrival_time"])])
    for i, p in enumerate(regular_queue, start=1):
        rows.append([P(f"Regular-{i}"), P("Standard Queue"), P(p["patient_id"]), P(p["name"]),
                     P(p["urgency"]), P(f"{p['score']:.0%}"), P(p["complaint"]), P(p["arrival_time"])])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Triage Queue")
    table = Table(rows, colWidths=[24*mm, 28*mm, 28*mm, 40*mm, 22*mm, 16*mm, 75*mm, 38*mm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for r in range(1, len(priority_lane) + 1):
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fee2e2")))
    table.setStyle(TableStyle(style))
    head = Paragraph(f"Triage Queue - {escape(str(hospital_id or 'All'))} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Title"])
    doc.build([head, Spacer(1, 4 * mm), table])
    return buf.getvalue()


def load_demo_queue_patients(hospital_id: str = "HOSP-001", db_path: Optional[Path] = None) -> None:
    """Seed a realistic set of demo waiting patients to demonstrate queue dynamics."""
    now = datetime.now()
    demo_queue_data = [
        # Priority Lane (HIGH)
        {
            "patient_id": "PAT-DEMO-001",
            "name": "Ramesh Kumar",
            "urgency": "HIGH",
            "score": 0.88,
            "complaint": "Acute crushing chest pain, cold sweating, SpO2 88%",
            "arrival_time": (now - timedelta(minutes=14)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_priority": 1
        },
        # Regular Queue (MEDIUM - arrived 35 mins ago)
        {
            "patient_id": "PAT-DEMO-002",
            "name": "Fatima Begum",
            "urgency": "MEDIUM",
            "score": 0.55,
            "complaint": "Persistent fever 102F for 3 days with lower abdominal tenderness",
            "arrival_time": (now - timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_priority": 0
        },
        # Regular Queue (LOW - arrived 45 mins ago)
        {
            "patient_id": "PAT-DEMO-003",
            "name": "Aakash Singh",
            "urgency": "LOW",
            "score": 0.18,
            "complaint": "Mild scratchy throat, runny nose for 2 days, no fever",
            "arrival_time": (now - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_priority": 0
        },
        # Regular Queue (MEDIUM - arrived 18 mins ago)
        {
            "patient_id": "PAT-DEMO-004",
            "name": "David Ochieng",
            "urgency": "MEDIUM",
            "score": 0.60,
            "complaint": "Severe throbbing migraine with nausea and light sensitivity",
            "arrival_time": (now - timedelta(minutes=18)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_priority": 0
        },
        # Regular Queue (LOW - arrived 10 mins ago)
        {
            "patient_id": "PAT-DEMO-005",
            "name": "Kavita Rao",
            "urgency": "LOW",
            "score": 0.12,
            "complaint": "Minor superficial scrape on arm, routine checkup",
            "arrival_time": (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "is_priority": 0
        }
    ]

    path = db_path or config.REGISTRY_DB_PATH
    init_db(path)
    conn = get_db_connection(path)
    cur = conn.cursor()

    for item in demo_queue_data:
        # Check existing
        cur.execute("SELECT queue_id FROM queue WHERE patient_id = ? AND status = 'WAITING'", (item["patient_id"],))
        if not cur.fetchone():
            cur.execute("""
            INSERT INTO queue (patient_id, hospital_id, visit_id, name, urgency, score, complaint, arrival_time, status, is_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'WAITING', ?)
            """, (
                item["patient_id"],
                hospital_id,
                None,
                item["name"],
                item["urgency"],
                item["score"],
                item["complaint"],
                item["arrival_time"],
                item["is_priority"]
            ))

    conn.commit()
    conn.close()


def render_queue_ui(hospital_id: str = "HOSP-001") -> None:
    """Render the interactive Hospital Mode Triage Queue workstation."""
    st.subheader("📋 Hospital Multi-Patient Triage Queue")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.5, 1.5, 1.5, 1.5])
    with col_btn1:
        if st.button("📢 Call Next Patient", type="primary", use_container_width=True):
            called = call_next(hospital_id)
            if called:
                st.session_state["_last_called_patient"] = called
                st.success(f"🔔 Calling **{called['name']}** ({called['urgency']} Urgency) to Consultation Room!")
                st.rerun()
            else:
                st.info("No patients currently waiting in queue.")

    with col_btn2:
        if st.button("👥 Load Demo Patients", use_container_width=True):
            load_demo_queue_patients(hospital_id)
            st.success("Loaded demo triage queue patients.")
            st.rerun()

    with col_btn3:
        st.download_button(
            "📥 Export Queue (PDF)",
            data=export_queue_pdf(hospital_id),
            file_name=f"triage_queue_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_btn4:
        if st.button("🗑️ Clear Queue", use_container_width=True):
            clear_queue(hospital_id)
            st.warning("Cleared waiting patients from queue.")
            st.rerun()

    # Fetch Queue data
    priority_lane, regular_queue = get_queue(hospital_id)
    n_priority = len(priority_lane)
    n_med = sum(1 for p in regular_queue if p["urgency"] == "MEDIUM")
    n_low = sum(1 for p in regular_queue if p["urgency"] == "LOW")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("🚨 Priority Lane", f"{n_priority} Patients", delta="Immediate" if n_priority > 0 else "Clear")
    with m_col2:
        st.metric("🟡 Medium Urgency", f"{n_med} Waiting")
    with m_col3:
        st.metric("🟢 Low Urgency", f"{n_low} Waiting")
    with m_col4:
        st.metric("👥 Total Waiting", f"{n_priority + n_med + n_low}")

    # Waiting-time escalation banner
    overdue = get_overdue_patients(priority_lane + regular_queue)
    if overdue:
        names = ", ".join(
            f"{p['name']} ({p['urgency']}, {format_wait(minutes_waiting(p['arrival_time']))})"
            for p in overdue
        )
        st.warning(
            f"⏰ **{len(overdue)} patient(s) have waited too long and need a re-check:** {names}. "
            "Re-assess them in the Clinical Triage tab, since their condition may have changed."
        )

    st.markdown("---")

    # ==========================================
    # 1. PRIORITY LANE (EMERGENCY HIGH PATIENTS)
    # ==========================================
    if priority_lane:
        st.error("### 🚨 PRIORITY LANE — IMMEDIATE CLINICAL ATTENTION")
        st.caption("Patients with life-threatening symptoms or critical physiological anomalies bypass the waiting queue.")

        for idx, p in enumerate(priority_lane, start=1):
            with st.container():
                st.markdown(f"""
                <div style="background-color: #fee2e2; border-left: 6px solid #dc2626; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #991b1b;">🚨 Priority #{idx}: {p['name']} ({p['patient_id']})</h4>
                        <span style="background: #dc2626; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.82rem;">HIGH ({p['score']:.0%})</span>
                    </div>
                    <p style="margin: 6px 0 2px 0; font-size: 0.94rem; color: #7f1d1d;"><b>Chief Complaint:</b> {p['complaint']}</p>
                    <small style="color: #991b1b;">Arrival: {p['arrival_time']} &nbsp;|&nbsp; Waiting: {format_wait(minutes_waiting(p['arrival_time']))}{' &nbsp;|&nbsp; <b>⏰ OVERDUE</b>' if is_overdue(p) else ''}</small>
                </div>
                """, unsafe_allow_html=True)

                c_action1, c_action2, _ = st.columns([1.2, 1.2, 4])
                with c_action1:
                    if st.button("Mark Seen", key=f"seen_p_{p['queue_id']}"):
                        mark_as_seen(p["queue_id"])
                        st.rerun()
                with c_action2:
                    if st.button("Remove", key=f"rem_p_{p['queue_id']}"):
                        remove_from_queue(p["queue_id"])
                        st.rerun()

    # ==========================================
    # 2. STANDARD QUEUE (FCFS LOW + MEDIUM)
    # ==========================================
    st.markdown("### ⏳ Standard Triage Queue (First-Come, First-Served)")
    if getattr(config, "QUEUE_MEDIUM_BEFORE_LOW", False):
        st.caption("Ordering rule: Medium urgency prioritized before Low, ordered by arrival time.")
    else:
        st.caption("Ordering rule: Strict First-Come-First-Served (FCFS) based on arrival timestamp.")

    if not regular_queue:
        st.info("No standard queue patients currently waiting.")
    else:
        for idx, p in enumerate(regular_queue, start=1):
            tier_color = "#fef3c7" if p["urgency"] == "MEDIUM" else "#d1fae5"
            border_color = "#f59e0b" if p["urgency"] == "MEDIUM" else "#10b981"
            if is_overdue(p):
                border_color = "#dc2626"
            badge_bg = "#d97706" if p["urgency"] == "MEDIUM" else "#059669"

            with st.container():
                st.markdown(f"""
                <div style="background-color: {tier_color}; border-left: 6px solid {border_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1f2937;">Position #{idx}: {p['name']} ({p['patient_id']})</h4>
                        <span style="background: {badge_bg}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.82rem;">{p['urgency']} ({p['score']:.0%})</span>
                    </div>
                    <p style="margin: 5px 0 2px 0; font-size: 0.92rem; color: #374151;"><b>Chief Complaint:</b> {p['complaint']}</p>
                    <small style="color: #6b7280;">Arrival: {p['arrival_time']} &nbsp;|&nbsp; Waiting: {format_wait(minutes_waiting(p['arrival_time']))}{' &nbsp;|&nbsp; <b style="color:#b91c1c;">⏰ OVERDUE - re-check patient</b>' if is_overdue(p) else ''}</small>
                </div>
                """, unsafe_allow_html=True)

                c_q1, c_q2, _ = st.columns([1.2, 1.2, 4])
                with c_q1:
                    if st.button("Mark Seen", key=f"seen_r_{p['queue_id']}"):
                        mark_as_seen(p["queue_id"])
                        st.rerun()
                with c_q2:
                    if st.button("Remove", key=f"rem_r_{p['queue_id']}"):
                        remove_from_queue(p["queue_id"])
                        st.rerun()