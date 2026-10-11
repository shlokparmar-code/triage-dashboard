"""Tests for the standard (normal) reference values used as sidebar defaults."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import reference_ranges as rr  # noqa: E402

# min / max of the matching st.number_input widgets in app.py
WIDGET_LIMITS = {
    "heart_rate": (30, 220),
    "spo2": (60.0, 100.0),
    "temperature": (33.0, 43.0),
    "resp_rate": (8, 80),
    "bp_systolic": (70, 240),
    "bp_diastolic": (40, 140),
    "glucose": (50, 400),
    "bmi": (12.0, 55.0),
    "cholesterol": (100, 480),
}


def test_every_session_key_maps_to_a_standard_value():
    assert set(rr.SESSION_KEYS.values()) == set(rr.STANDARD_VALUES)
    assert set(rr.session_defaults()) == set(rr.SESSION_KEYS)


def test_defaults_are_normal_by_their_own_range():
    for key in rr.STANDARD_VALUES:
        assert rr.status(key, rr.default(key)) == "normal", key


def test_defaults_fit_inside_the_widget_limits():
    for key, (lo, hi) in WIDGET_LIMITS.items():
        assert lo <= rr.default(key) <= hi, key


def test_published_ranges():
    sv = rr.STANDARD_VALUES
    assert (sv["heart_rate"]["low"], sv["heart_rate"]["high"]) == (60, 100)
    assert (sv["spo2"]["low"], sv["spo2"]["high"]) == (95.0, 100.0)
    assert (sv["temperature"]["low"], sv["temperature"]["high"]) == (36.5, 37.3)
    assert (sv["resp_rate"]["low"], sv["resp_rate"]["high"]) == (12, 18)
    assert sv["bp_systolic"]["high"] == 120 and not sv["bp_systolic"]["high_inclusive"]
    assert sv["bp_diastolic"]["high"] == 80 and not sv["bp_diastolic"]["high_inclusive"]
    assert (sv["glucose"]["low"], sv["glucose"]["high"]) == (70, 99)
    assert (sv["bmi"]["low"], sv["bmi"]["high"]) == (18.5, 25.0)
    assert sv["cholesterol"]["high"] == 200 and sv["cholesterol"]["low"] is None


def test_status_boundaries():
    assert rr.status("heart_rate", 59) == "low"
    assert rr.status("heart_rate", 60) == "normal"
    assert rr.status("heart_rate", 100) == "normal"
    assert rr.status("heart_rate", 101) == "high"
    # "below 120 / below 80" are exclusive limits
    assert rr.status("bp_systolic", 119) == "normal"
    assert rr.status("bp_systolic", 120) == "high"
    assert rr.status("bp_diastolic", 80) == "high"
    assert rr.status("cholesterol", 199) == "normal"
    assert rr.status("cholesterol", 200) == "high"
    assert rr.status("bmi", 24.9) == "normal"
    assert rr.status("bmi", 25.0) == "high"
    assert rr.status("glucose", 69) == "low"
    assert rr.status("glucose", 100) == "high"


def test_float_noise_does_not_flag_a_normal_temperature():
    # 37.2 + 0.1 style arithmetic can give 37.300000000000004
    assert rr.status("temperature", 37.2 + 0.1) == "normal"


def test_bad_input_is_not_flagged():
    assert rr.status("heart_rate", None) == "normal"
    assert rr.status("heart_rate", "abc") == "normal"


def test_outside_range_and_format():
    assert rr.outside_range({k: rr.default(k) for k in rr.STANDARD_VALUES}) == []
    items = rr.outside_range({"heart_rate": 130, "spo2": 91.0, "unknown": 5})
    assert [i[0] for i in items] == ["Pulse / HR", "SpO2"]
    assert items[0][3] == "high" and items[1][3] == "low"
    text = rr.format_outside(items)
    assert "130" in text and "60-100" in text and "91.0" in text
    assert rr.format_outside([]) == ""


def test_reset_to_standard_overwrites_everything():
    state = {k: 999 for k in rr.SESSION_KEYS}
    rr.reset_to_standard(state)
    assert state == rr.session_defaults()


def test_range_text_and_sources():
    assert rr.range_text("heart_rate") == "Normal adult range: 60-100 bpm"
    assert len(rr.SOURCES) >= 6
    assert all(url.startswith("https://") for _n, url in rr.SOURCES)
    assert all(spec["source"] for spec in rr.STANDARD_VALUES.values())