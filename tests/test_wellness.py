"""Tests for the wellness dataset and planner."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import wellness  # noqa: E402

KB = wellness.load_kb()


def ids(items):
    return [i["id"] for i in items]


def test_dataset_loads_and_has_content():
    for cat, minimum in (("yoga", 15), ("exercise", 8), ("diet", 10), ("lifestyle", 10), ("home_care", 8)):
        assert len(KB[cat]) >= minimum, cat


def test_ids_unique_and_tags_valid():
    seen = set()
    valid_patient = set(KB["meta"]["patient_tags"])
    valid_avoid = set(KB["meta"]["avoid_tags"])
    for cat in ("yoga", "exercise", "diet", "lifestyle", "home_care"):
        for it in KB[cat]:
            assert it["id"] not in seen
            seen.add(it["id"])
            assert set(it.get("conditions", [])) <= valid_patient, it["id"]
            assert set(it.get("avoid_if", [])) <= valid_avoid, it["id"]


def test_every_yoga_and_exercise_has_steps_and_safety_text():
    for it in KB["yoga"] + KB["exercise"]:
        assert it["steps"], it["id"]
        assert it["benefits"], it["id"]


def test_dataset_marked_for_clinical_review():
    assert KB["meta"]["needs_clinical_review"] is True
    assert "not a medical diagnosis" in KB["meta"]["disclaimer"].lower()


def test_high_urgency_returns_no_plan():
    plan = wellness.build_plan(age=40, urgency="HIGH")
    assert plan["yoga"] == [] and plan["diet"] == [] and plan["note"]


def test_hypertension_excludes_unsafe_yoga():
    plan = wellness.build_plan(age=50, urgency="LOW", systolic_bp=160, diastolic_bp=100)
    got = ids(plan["yoga"])
    assert "y_viparita_karani" not in got and "y_surya_namaskar" not in got
    assert any(i["id"] == "d_hypertension" for i in plan["diet"])
    assert plan["practices_to_avoid"]  # forceful breathing, inversions etc. flagged


def test_pregnancy_excludes_unsafe_items():
    plan = wellness.build_plan(age=28, urgency="LOW", sex="Female", is_pregnant=True)
    for it in plan["yoga"] + plan["exercise"]:
        assert it["pregnancy"] != "avoid"
    assert "y_bhujangasana" not in ids(plan["yoga"])
    assert any(i["id"] == "d_pregnancy" for i in plan["diet"])


def test_child_gets_age_appropriate_items():
    plan = wellness.build_plan(age=4, urgency="LOW")
    for cat in ("yoga", "exercise", "diet", "lifestyle"):
        for it in plan[cat]:
            assert it["min_age"] <= 4, it["id"]
    assert not any(i["id"] == "d_weight" for i in plan["diet"])


def test_elderly_and_medium_are_gentle_only():
    plan = wellness.build_plan(age=70, urgency="LOW")
    assert all(i["level"] == "gentle" for i in plan["yoga"] + plan["exercise"])
    med = wellness.build_plan(age=35, urgency="MEDIUM")
    assert all(i["level"] == "gentle" for i in med["yoga"] + med["exercise"])
    assert "24 to 48 hours" in med["note"]
    assert med["home_care"] == []  # home-care tips only for LOW


def test_user_limits_hide_items():
    plan = wellness.build_plan(age=30, urgency="LOW", limits=["knee_pain", "back_injury"])
    for it in plan["yoga"] + plan["exercise"]:
        assert not ({"knee_pain", "back_injury"} & set(it["avoid_if"]))


def test_symptom_negation_is_respected():
    assert "cold_cough" not in wellness.derive_tags(30, symptom_text="no cough and no cold, just tired")["tags"]
    assert "cold_cough" in wellness.derive_tags(30, symptom_text="mild cough since two days")["tags"]
    assert "back_pain" in wellness.derive_tags(30, symptom_text="back pain for a week")["tags"]


def test_chronic_conditions_outrank_minor_symptoms():
    plan = wellness.build_plan(age=48, urgency="LOW", systolic_bp=150, diastolic_bp=95, glucose=130,
                               symptom_text="mild cough")
    first_two = ids(plan["diet"])[:3]
    assert "d_hypertension" in first_two or "d_high_glucose" in first_two


def test_bmi_and_glucose_rules():
    t = wellness.derive_tags(40, bmi=27, glucose=130)["tags"]
    assert "overweight" in t and "high_glucose" in t
    assert "underweight" in wellness.derive_tags(40, bmi=17)["tags"]
    # children are not given adult BP/BMI tags
    kid = wellness.derive_tags(8, systolic_bp=150, bmi=30)["tags"]
    assert "hypertension" not in kid and "overweight" not in kid


def test_missing_dataset_does_not_crash(tmp_path):
    wellness.load_kb.cache_clear()
    empty = wellness.load_kb(str(tmp_path / "missing.json"))
    plan = wellness.build_plan(age=30, urgency="LOW", kb=empty)
    assert plan["yoga"] == []
    wellness.load_kb.cache_clear()


def test_json_and_csv_are_valid_files():
    root = Path(__file__).resolve().parents[1] / "data"
    json.loads((root / "wellness_kb.json").read_text(encoding="utf-8"))
    assert (root / "wellness_kb.csv").read_text(encoding="utf-8-sig").startswith("category,id,name")