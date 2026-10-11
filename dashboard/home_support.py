"""Home-mode support panels for the Streamlit app.

- nearby hospitals (names + phone numbers) for every urgency tier
- for HIGH urgency: if the nearest emergency hospital's queue is long, suggest
  other nearby hospitals instead
- alert a friend / guardian / relative with a short summary (consent required)
- a detailed, personalised wellness plan from data/wellness_kb.json

Everything runs offline. The hospital directory is DEMO data until you replace
data/nearby_hospitals.json with verified hospitals and real phone numbers.
"""
import re
from typing import Any, Dict, List, Optional

import streamlit as st

from modules import contact_alert, facilities, wellness


def _tel(phone: str) -> str:
    digits = re.sub(r"[^\d+]", "", phone or "")
    return f"[{phone}](tel:{digits})" if digits else (phone or "n/a")


def _distance_text(km: float) -> str:
    return "in your city" if km < 1 else f"about {km:.0f} km away"


def _hospital_markdown(h: Dict[str, Any]) -> str:
    level = {"advanced": "24x7 emergency care", "basic": "basic emergency care", "none": "no emergency service"}.get(
        h.get("emergency_level", "none"), "")
    lines = [
        f"**{h['name']}** ({h.get('type', 'Hospital')}, {h.get('city', '')})",
        f"📞 {_tel(h.get('phone', ''))}  |  🚑 Ambulance: {_tel(h.get('ambulance', '108'))}",
        f"📍 {_distance_text(h.get('distance_km', 0))}  |  🏥 {level}",
    ]
    if h.get("note"):
        lines.append(f"ℹ️ {h['note']}")
    load = h.get("load")
    if load:
        lines.append(f"👥 Queue now: {load['priority']} emergency, {load['waiting']} waiting")
    return "  \n".join(lines)


def render_nearby_hospitals(urgency: str, city: Optional[str]) -> None:
    """Show hospital names and phone numbers; redirect HIGH cases away from busy hospitals."""
    st.markdown("### 🏥 Hospitals Near You")
    warn = facilities.directory_warning()
    if warn:
        st.caption(f"⚠️ {warn}")
    if not city:
        st.info("Choose your city in the sidebar to see nearby hospitals.")
        return

    if urgency == "HIGH":
        res = facilities.suggest_for_high(city)
        primary = res["primary"]
        if primary is None:
            st.error("No emergency hospital is listed for this city. Call **108 / 112** now.")
            return
        st.markdown("**Nearest emergency hospital**")
        with st.container(border=True):
            st.markdown(_hospital_markdown(primary))
        if res["busy"]:
            load = primary["load"]
            st.warning(
                f"⚠️ **{primary['name']} has a long queue right now** "
                f"({load['priority']} emergency and {load['waiting']} waiting patients). "
                "Please consider going to another hospital nearby instead:"
            )
            if res["alternatives"]:
                for alt in res["alternatives"]:
                    with st.container(border=True):
                        st.markdown(_hospital_markdown(alt))
            else:
                st.info("No other emergency hospital is listed nearby. Call **108 / 112** and tell them which hospital is full.")
        elif primary.get("load") is None:
            st.caption("Live queue information is not available for this hospital. Call ahead if you can, but do not delay.")
        else:
            st.caption("This hospital's queue is manageable right now.")
        st.caption("Queue information comes from hospitals using this system. Call 108 / 112 if you are unsure where to go.")
    else:
        hospitals = facilities.find_nearby(city, limit=3)
        if not hospitals:
            st.info("No hospitals are listed for this city yet.")
            return
        intro = ("Visit one of these within 24 to 48 hours:" if urgency == "MEDIUM"
                 else "If your symptoms get worse, these are the nearest places to go:")
        st.markdown(intro)
        for h in hospitals:
            with st.container(border=True):
                st.markdown(_hospital_markdown(h))
    st.caption("Distances are approximate straight-line distances between city centres, not road distances.")
    helplines = " | ".join(f"{x['name']}: {x['number']}" for x in facilities.national_helplines())
    if helplines:
        st.caption(f"📞 {helplines}")


def collect_concerns(special_pop_res: Optional[Dict[str, Any]], symptom_res: Optional[Dict[str, Any]],
                     symptom_text: str = "") -> List[str]:
    """Short list of main concerns for the contact message."""
    def label(x: Any) -> str:
        if isinstance(x, dict):
            return str(x.get("label") or x.get("name") or x.get("id") or "")
        return str(x)

    items: List[str] = []
    for res, keys in ((special_pop_res, ("danger_signs_detected", "red_flags_detected")),
                      (symptom_res, ("red_flags",))):
        details = (res or {}).get("details", {}) or {}
        for k in keys:
            for x in details.get(k, []) or []:
                s = label(x).replace("_", " ").strip()
                if s and s not in items:
                    items.append(s)
    if not items and symptom_text.strip():
        items.append(symptom_text.strip()[:90])
    return items[:3]


def render_contact_alert(patient_name: str, urgency: str, concerns: List[str], contact_full: str,
                         contact_valid: bool, share_consent: bool, relation: str) -> None:
    """Let the patient send a short summary to their contact (mock SMS until a gateway is plugged in)."""
    st.markdown("### 👨‍👩‍👧 Tell Your Contact")
    if not contact_full or not contact_valid:
        st.info("Add a friend, guardian or relative's mobile number in the sidebar (under **Emergency Contact**) "
                "so they can be told about this result.")
        return
    masked = contact_alert.mask_number(contact_full)
    if not share_consent:
        st.info(f"A contact is saved ({masked}). Tick the sharing consent in the sidebar to let them receive your result.")
        return
    message = contact_alert.build_contact_message(patient_name, urgency, concerns, relation)
    st.markdown(f"This short message will be shared with your **{relation or 'contact'}** ({masked}):")
    st.code(message, language=None)
    label = "🚨 Alert my contact now" if urgency == "HIGH" else "📨 Send summary to my contact"
    if st.button(label, type="primary" if urgency == "HIGH" else "secondary", key="_send_contact_alert"):
        res = contact_alert.notify_contact(contact_full, share_consent, patient_name, urgency, concerns, relation)
        if not res["ok"]:
            st.error(res["error"])
        elif res["status"] == "MOCK_NOT_SENT":
            st.warning("DEMO: no SMS gateway is connected, so **nothing was actually sent**. "
                       "Copy the message above and send it to your contact yourself.")
        else:
            st.success(f"Message sent to {res['to_masked']} (status: {res['status']}).")
    st.caption("Only the urgency level, up to 3 main concerns and basic advice are shared. Your full record and phone number are never included.")


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def _numbered(items: List[str]) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, start=1))


def render_wellness_plan(plan: Dict[str, Any]) -> None:
    """Detailed plan with tabs for yoga, exercise, diet, lifestyle and home care."""
    if plan.get("urgency") == "HIGH":
        return
    with st.expander("📚 Detailed Personalised Wellness Plan (yoga, exercise, diet, lifestyle)"):
        if plan.get("note"):
            st.warning(plan["note"])
        for r in plan.get("reasons", []):
            st.caption(f"• {r}")
        tabs = st.tabs(["🧘 Yoga & Breathing", "🏃 Exercise", "🥗 Diet", "🌿 Lifestyle", "🏠 Home Care"])

        with tabs[0]:
            if not plan["yoga"]:
                st.info("No yoga items match your profile right now.")
            for it in plan["yoga"]:
                with st.container(border=True):
                    st.markdown(f"**{it['name']}**  ·  _{it['type']}, {it['level']}_")
                    st.markdown(_numbered(it["steps"]))
                    st.markdown(f"⏱ {it['duration']}  |  🔁 {it['frequency']}")
                    st.markdown(f"**Benefit:** {it['benefits']}")
                    if it.get("cautions"):
                        st.caption("⚠️ " + " ".join(it["cautions"]))
            if plan["practices_to_avoid"]:
                st.markdown("**Practices to avoid for you:**")
                for pr in plan["practices_to_avoid"]:
                    st.markdown(f"- ❌ **{pr['name']}**: {pr['why']}")

        with tabs[1]:
            if not plan["exercise"]:
                st.info("No exercise items match your profile right now.")
            for it in plan["exercise"]:
                with st.container(border=True):
                    st.markdown(f"**{it['name']}**  ·  _{it['intensity']}_")
                    st.markdown(_numbered(it["steps"]))
                    st.markdown(f"⏱ {it['duration']}  |  🔁 {it['frequency']}")
                    st.markdown(f"**Benefit:** {it['benefits']}")
                    if it.get("cautions"):
                        st.caption("⚠️ " + " ".join(it["cautions"]))
                    if it.get("stop_if"):
                        st.caption("🛑 Stop and get help if: " + "; ".join(it["stop_if"]))

        with tabs[2]:
            if not plan["diet"]:
                st.info("No diet guidance matches your profile right now.")
            for it in plan["diet"]:
                with st.container(border=True):
                    st.markdown(f"**{it['focus']}**")
                    st.markdown("**Do:**\n" + _bullets(it["do"]))
                    st.markdown("**Limit or avoid:**\n" + _bullets(it["limit"]))
                    st.markdown("**Food examples:** " + ", ".join(it["indian_examples"]))
                    sd = it["sample_day"]
                    st.markdown(f"**Sample day:** Breakfast: {sd['breakfast']}. Lunch: {sd['lunch']}. "
                                f"Snack: {sd['snack']}. Dinner: {sd['dinner']}.")
                    st.caption("ℹ️ " + it["notes"])

        with tabs[3]:
            if not plan["lifestyle"]:
                st.info("No lifestyle tips match your profile right now.")
            for it in plan["lifestyle"]:
                with st.container(border=True):
                    st.markdown(f"**{it['title']}**")
                    st.markdown(_bullets(it["tips"]))
                    st.caption("Why: " + it["why"])

        with tabs[4]:
            if not plan["home_care"]:
                st.info("No home-care tips for your current symptoms. Home-care tips appear for mild symptoms only.")
            for it in plan["home_care"]:
                with st.container(border=True):
                    st.markdown(f"**{it['ailment']}**")
                    st.markdown("**What helps:**\n" + _bullets(it["home_care"]))
                    st.markdown("**Avoid:**\n" + _bullets(it["avoid"]))
                    st.markdown("**See a doctor if:**\n" + _bullets(it["see_doctor_if"]))

        st.markdown("---")
        st.markdown("**Get urgent help (call 108 / 112) for:**\n" + _bullets(plan.get("when_to_seek_care", [])))
        st.caption(plan.get("disclaimer", ""))