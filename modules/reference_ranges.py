"""Standard (normal) adult reference values for the sidebar inputs.

When the dashboard first opens, every vital-sign and chronic-baseline box is
pre-filled with the *standard* value below instead of an arbitrary number.
The numeric ranges come from public health authorities (see SOURCES).

IMPORTANT
- The RANGE is taken from the cited source.  The single DEFAULT number is a
  typical value chosen inside that range (sources give ranges, not one
  "official" number).
- These are adult resting reference ranges for screening/education.  They are
  not a diagnosis, and children / pregnancy have their own ranges (handled by
  the Special Population rules, not here).
- Pure Python, no network calls: works fully offline.
"""

# Session-state key used by app.py  ->  key in STANDARD_VALUES
SESSION_KEYS = {
    "manual_hr": "heart_rate",
    "manual_spo2": "spo2",
    "manual_temp": "temperature",
    "manual_rr": "resp_rate",
    "patient_bp": "bp_systolic",
    "patient_bp_dia": "bp_diastolic",
    "patient_glucose": "glucose",
    "patient_bmi": "bmi",
    "patient_chol": "cholesterol",
}

# low        : lowest normal value (inclusive); None = no lower limit
# high       : highest normal value; inclusive unless high_inclusive is False
# display    : the range exactly as shown to the user
STANDARD_VALUES = {
    "heart_rate": {
        "label": "Pulse / HR", "unit": "bpm",
        "default": 72, "low": 60, "high": 100, "high_inclusive": True,
        "display": "60-100 bpm",
        "source": "MedlinePlus - Vital signs (adult, at rest)",
    },
    "spo2": {
        "label": "SpO2", "unit": "%",
        "default": 98.0, "low": 95.0, "high": 100.0, "high_inclusive": True,
        "display": "95-100 %",
        "source": "StatPearls (NCBI) - Pulse oximetry, healthy adults at sea level",
    },
    "temperature": {
        "label": "Temperature", "unit": "°C",
        "default": 37.0, "low": 36.5, "high": 37.3, "high_inclusive": True,
        "display": "36.5-37.3 °C (average 37.0 °C / 98.6 °F)",
        "source": "MedlinePlus - Vital signs",
    },
    "resp_rate": {
        "label": "Breaths / min", "unit": "breaths/min",
        "default": 16, "low": 12, "high": 18, "high_inclusive": True,
        "display": "12-18 breaths/min",
        "source": "MedlinePlus - Vital signs (adult, at rest)",
    },
    "bp_systolic": {
        "label": "BP Systolic", "unit": "mmHg",
        "default": 115, "low": 90, "high": 120, "high_inclusive": False,
        "display": "90 to below 120 mmHg",
        "source": "American Heart Association (normal: below 120/80); "
                  "MedlinePlus (90/60 to 120/80)",
    },
    "bp_diastolic": {
        "label": "BP Diastolic", "unit": "mmHg",
        "default": 75, "low": 60, "high": 80, "high_inclusive": False,
        "display": "60 to below 80 mmHg",
        "source": "American Heart Association (normal: below 120/80); "
                  "MedlinePlus (90/60 to 120/80)",
    },
    "glucose": {
        "label": "Blood Glucose (fasting)", "unit": "mg/dL",
        "default": 90, "low": 70, "high": 99, "high_inclusive": True,
        "display": "70-99 mg/dL fasting",
        "source": "NIDDK / American Diabetes Association (normal fasting 99 or "
                  "lower; 100-125 prediabetes; 126+ diabetes); 70 = lower "
                  "alert limit for low blood sugar",
    },
    "bmi": {
        "label": "BMI", "unit": "kg/m²",
        "default": 22.0, "low": 18.5, "high": 25.0, "high_inclusive": False,
        "display": "18.5-24.9 (Asian public-health action point: 23.0)",
        "source": "CDC adult BMI categories; WHO Expert Consultation, "
                  "Lancet 2004 (Asian populations)",
    },
    "cholesterol": {
        "label": "Total Cholesterol", "unit": "mg/dL",
        "default": 180, "low": None, "high": 200, "high_inclusive": False,
        "display": "below 200 mg/dL",
        "source": "MedlinePlus - Cholesterol levels (adults 20+: total < 200)",
    },
}

SOURCES = [
    ("MedlinePlus - Vital signs", "https://medlineplus.gov/ency/article/002341.htm"),
    ("American Heart Association - Blood pressure readings",
     "https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings"),
    ("CDC - Adult BMI categories", "https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html"),
    ("NIDDK - Diabetes tests & diagnosis",
     "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis"),
    ("MedlinePlus - Cholesterol levels", "https://medlineplus.gov/lab-tests/cholesterol-levels/"),
    ("StatPearls (NCBI) - Pulse oximetry", "https://www.ncbi.nlm.nih.gov/books/NBK525974/"),
    ("WHO Expert Consultation, Lancet 2004 - BMI in Asian populations",
     "https://pubmed.ncbi.nlm.nih.gov/14726171/"),
]


def default(key):
    """Standard default value for a STANDARD_VALUES key."""
    return STANDARD_VALUES[key]["default"]


def session_defaults():
    """{session_state_key: standard default} for every sidebar input."""
    return {sk: STANDARD_VALUES[k]["default"] for sk, k in SESSION_KEYS.items()}


def reset_to_standard(state):
    """Write every standard default into a dict-like (st.session_state)."""
    for sk, value in session_defaults().items():
        state[sk] = value


def range_text(key):
    """Short tooltip text, e.g. 'Normal adult range: 60-100 bpm'."""
    return f"Normal adult range: {STANDARD_VALUES[key]['display']}"


def status(key, value):
    """Return 'low', 'normal' or 'high' for a value against the standard range."""
    spec = STANDARD_VALUES[key]
    try:
        v = round(float(value), 2)  # avoids 37.300000000000004 style noise
    except (TypeError, ValueError):
        return "normal"
    low, high = spec["low"], spec["high"]
    if low is not None and v < low:
        return "low"
    if high is not None:
        if v > high or (v == high and not spec["high_inclusive"]):
            return "high"
    return "normal"


def outside_range(values):
    """Check {STANDARD_VALUES key: value}; return only the abnormal ones.

    Each result is (label, value, unit, 'low'|'high', display_range).
    """
    out = []
    for key, value in values.items():
        if key not in STANDARD_VALUES:
            continue
        st_ = status(key, value)
        if st_ != "normal":
            spec = STANDARD_VALUES[key]
            out.append((spec["label"], value, spec["unit"], st_, spec["display"]))
    return out


def format_outside(items):
    """Markdown bullet list for the sidebar, or '' when everything is normal."""
    lines = []
    for label, value, unit, st_, display in items:
        lines.append(f"- **{label}: {value} {unit}** ({st_}) - normal {display}")
    return "\n".join(lines)