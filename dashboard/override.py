"""Clinician override of the AI urgency tier (Hospital mode only).

The AI result is never edited. A clinician can record a different final tier
together with a written reason. Both values are kept, and the override is
written to the audit log. Lowering the urgency needs an extra confirmation.
"""
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import streamlit as st

TIERS = ["LOW", "MEDIUM", "HIGH"]
MIN_REASON_CHARS = 10
_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_STATE_KEY = "_clinician_override"


def validate_override(
    ai_urgency: str,
    new_urgency: str,
    reason: str,
    confirmed_downgrade: bool = False,
) -> Tuple[bool, str]:
    """Return (ok, message). Pure function, no Streamlit needed."""
    if new_urgency not in TIERS or ai_urgency not in TIERS:
        return False, "Choose a valid urgency tier."
    if new_urgency == ai_urgency:
        return False, "The chosen tier is the same as the AI tier. Nothing to override."
    if len((reason or "").strip()) < MIN_REASON_CHARS:
        return False, f"Please write a reason (at least {MIN_REASON_CHARS} characters)."
    if _RANK[new_urgency] < _RANK[ai_urgency] and not confirmed_downgrade:
        return False, "Tick the box to confirm you are LOWERING the urgency."
    return True, ""


def build_override_record(
    ai_urgency: str, new_urgency: str, reason: str, doctor: str, patient_key: str
) -> Dict[str, Any]:
    """Create the override record stored in session_state."""
    return {
        "ai_urgency": ai_urgency,
        "clinician_urgency": new_urgency,
        "reason": reason.strip(),
        "doctor": doctor,
        "patient_key": str(patient_key),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def override_summary(record: Dict[str, Any]) -> str:
    """One-paragraph text used in the SBAR note, visit notes and audit log."""
    return (
        f"CLINICIAN OVERRIDE: AI tier {record['ai_urgency']} changed to "
        f"{record['clinician_urgency']} by {record['doctor']} at {record['timestamp']}. "
        f"Reason: {record['reason']}"
    )


def get_active_override(ai_urgency: str, patient_key: str) -> Optional[Dict[str, Any]]:
    """Return the stored override only if it still belongs to this assessment."""
    rec = st.session_state.get(_STATE_KEY)
    if not rec:
        return None
    if rec["patient_key"] != str(patient_key) or rec["ai_urgency"] != ai_urgency:
        # Inputs or patient changed, so the old override no longer applies.
        st.session_state.pop(_STATE_KEY, None)
        return None
    return rec


def _safe_audit(log_audit: Optional[Callable[..., Any]], **kwargs: Any) -> bool:
    if log_audit is None:
        return False
    try:
        log_audit(**kwargs)
        return True
    except Exception:
        return False


def render_clinician_override(
    ai_urgency: str,
    patient_key: str,
    patient_id: Optional[str],
    doctor_name: str,
    hospital_id: str,
    log_audit: Optional[Callable[..., Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Render the override panel. Returns (effective_urgency, active_override)."""
    active = get_active_override(ai_urgency, patient_key)

    st.markdown("### 🩺 Clinician Review of AI Urgency")
    with st.expander(
        "Override the AI urgency tier (doctor decision, recorded in audit log)",
        expanded=bool(active),
    ):
        st.caption(
            "The AI result is decision support only. The treating clinician has the final say. "
            "The original AI tier is always kept next to your decision."
        )
        audit_id = patient_id or "PAT-UNREG"

        if active:
            st.warning(
                f"**Override active:** AI said **{active['ai_urgency']}**, "
                f"clinician set **{active['clinician_urgency']}**  \n"
                f"By {active['doctor']} at {active['timestamp']}  \n"
                f"Reason: {active['reason']}"
            )
            if st.button("Remove override and return to AI tier", key="_ovr_remove"):
                _safe_audit(
                    log_audit,
                    who=doctor_name,
                    role="Medical Officer",
                    hospital_id=hospital_id,
                    patient_id=audit_id,
                    action="OVERRIDE_REMOVED: back to AI tier " + ai_urgency,
                )
                st.session_state.pop(_STATE_KEY, None)
                st.rerun()
            return active["clinician_urgency"], active

        options = [t for t in TIERS if t != ai_urgency]
        new_tier = st.selectbox(
            f"AI tier is {ai_urgency}. Set clinician tier to:", options, key="_ovr_tier"
        )
        reason = st.text_input(
            "Reason (required):",
            key="_ovr_reason",
            placeholder="e.g. Known COPD patient, SpO2 baseline is 90%",
        )
        is_downgrade = _RANK[new_tier] < _RANK[ai_urgency]
        confirmed = True
        if is_downgrade:
            confirmed = st.checkbox(
                "I confirm I am LOWERING the urgency and take responsibility for this decision.",
                key="_ovr_confirm",
            )
        if st.button("Apply override", type="primary", key="_ovr_apply"):
            ok, msg = validate_override(ai_urgency, new_tier, reason, confirmed)
            if not ok:
                st.error(msg)
            else:
                rec = build_override_record(ai_urgency, new_tier, reason, doctor_name, patient_key)
                st.session_state[_STATE_KEY] = rec
                logged = _safe_audit(
                    log_audit,
                    who=doctor_name,
                    role="Medical Officer",
                    hospital_id=hospital_id,
                    patient_id=audit_id,
                    action=override_summary(rec),
                )
                if not logged:
                    st.session_state["_override_audit_warning"] = True
                st.rerun()
        if st.session_state.pop("_override_audit_warning", False):
            st.warning("Override applied, but it could not be written to the audit log.")

    return ai_urgency, None