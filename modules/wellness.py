"""Rule-based wellness planner using data/wellness_kb.json (no ML, fully offline).

Picks yoga, exercise, diet, lifestyle and home-care items that match the
patient's situation and drops anything marked unsafe for them (for example
inversions and breath-holding with high blood pressure, or deep backbends in
pregnancy). General wellness guidance only, never a diagnosis.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "wellness_kb.json"

# Limits the user can tick (these map to "avoid_if" tags in the dataset).
LIMIT_OPTIONS = {
    "knee_pain": "Knee pain",
    "back_injury": "Back pain or back injury",
    "neck_injury": "Neck problem",
    "vertigo": "Dizziness or vertigo",
    "hernia": "Hernia",
    "recent_surgery": "Recent surgery",
    "ear_infection": "Ear infection",
    "glaucoma": "Glaucoma",
}

SYMPTOM_TAGS = {
    "cold_cough": ["cold", "cough", "sneez", "runny nose", "blocked nose", "stuffy nose"],
    "sore_throat": ["sore throat", "throat pain", "scratchy throat"],
    "headache_tension": ["headache", "head ache"],
    "acidity": ["acidity", "heartburn", "acid reflux"],
    "digestion": ["gas", "bloating", "indigestion"],
    "constipation": ["constipation", "hard stool"],
    "muscle_ache": ["muscle ache", "body ache", "muscle pain", "sore muscles"],
    "back_pain": ["back pain"],
    "mild_diarrhea": ["loose motion", "diarrhoea", "diarrhea", "loose stool"],
    "skin_minor": ["scratch", "minor cut", "small cut", "insect bite", "mosquito bite", "itchy bump"],
    "sleep": ["poor sleep", "cant sleep", "can't sleep", "trouble sleeping", "insomnia", "tired", "fatigue"],
    "stress": ["stress", "anxious", "anxiety", "worried", "tension"],
    "desk_work": ["desk", "long hours sitting", "screen"],
}

LIMITS_PER_CATEGORY = {"yoga": 5, "exercise": 3, "diet": 4, "lifestyle": 4, "home_care": 3}


@lru_cache(maxsize=1)
def load_kb(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the wellness dataset. Returns an empty skeleton if the file is missing."""
    p = Path(path) if path else KB_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"meta": {}, "yoga": [], "exercise": [], "diet": [], "lifestyle": [], "home_care": [],
                "practices_to_avoid": [], "when_to_seek_care": []}


def _mentions(text: str, keyword: str) -> bool:
    """True if keyword appears and is not negated just before it ('no cough', 'without fever')."""
    for m in re.finditer(re.escape(keyword), text):
        before = text[max(0, m.start() - 18): m.start()]
        if re.search(r"\b(no|not|without|never|denies)\b(?:\s+\w+){0,2}\s*$", before):
            continue
        return True
    return False


def derive_tags(
    age: int,
    sex: str = "",
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    glucose: Optional[float] = None,
    bmi: Optional[float] = None,
    is_pregnant: bool = False,
    is_postpartum: bool = False,
    heart_risk: Optional[float] = None,
    diabetes_risk: Optional[float] = None,
    symptom_text: str = "",
    limits: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Work out the patient's tags and plain-language reasons. Returns {'tags', 'avoid', 'reasons'}."""
    tags: Set[str] = {"general", "stress"}
    avoid: Set[str] = set(limits or [])
    reasons: List[str] = []
    adult = age >= 18

    if age < 18:
        tags.add("child"); avoid.add("child")
    if age >= 60:
        tags.add("elderly"); avoid.add("elderly")
    if adult and systolic_bp is not None:
        if systolic_bp >= 130 or (diastolic_bp is not None and diastolic_bp >= 85):
            tags.add("hypertension"); avoid.add("hypertension")
            reasons.append("Blood pressure is above the normal range, so low-salt eating and calming practices are suggested.")
        elif systolic_bp < 90:
            tags.add("low_bp"); avoid.add("low_bp")
            reasons.append("Blood pressure is low, so standing and head-down poses are avoided.")
    if adult and glucose is not None and glucose >= 100:
        tags.add("high_glucose")
        reasons.append("Blood glucose is above the usual range (if this was a fasting reading), so blood-sugar friendly food and walking are suggested.")
    if adult and diabetes_risk is not None and diabetes_risk >= 0.4:
        tags.add("high_glucose")
        reasons.append("The risk screen shows a raised diabetes risk.")
    if adult and heart_risk is not None and heart_risk >= 0.4:
        tags.add("heart_risk"); avoid.add("heart_risk")
        reasons.append("The risk screen shows a raised heart risk, so gentle activity and heart-healthy eating are suggested.")
    if adult and bmi is not None:
        if bmi >= 23:
            tags.add("overweight")
            reasons.append("BMI is 23 or above (the cut-off for Asian adults), so gradual weight management is suggested.")
        elif bmi < 18.5:
            tags.add("underweight")
            reasons.append("BMI is below 18.5, so nourishing food is suggested rather than weight loss.")
    if is_pregnant:
        tags.add("pregnancy"); avoid.add("pregnancy")
        reasons.append("Pregnancy: only pregnancy-appropriate items are shown.")
    if is_postpartum:
        tags.add("postpartum"); avoid.add("postpartum")
        reasons.append("Recent childbirth: only gentle items are shown.")

    text = (symptom_text or "").lower()
    for tag, words in SYMPTOM_TAGS.items():
        if any(_mentions(text, w) for w in words):
            tags.add(tag)
    if "cold_cough" in tags:
        avoid.add("cold_cough")
    return {"tags": tags, "avoid": avoid, "reasons": reasons}


def _eligible(item: Dict[str, Any], age: int, avoid: Set[str], pregnant: bool, gentle_only: bool) -> bool:
    if age < item.get("min_age", 0):
        return False
    if set(item.get("avoid_if", [])) & avoid:
        return False
    if pregnant and item.get("pregnancy") == "avoid":
        return False
    if gentle_only and item.get("level", "gentle") != "gentle":
        return False
    return True


# Chronic and special-population tags matter more than minor symptoms when ranking items.
TAG_WEIGHTS = {
    "pregnancy": 4, "postpartum": 4, "child": 3, "hypertension": 3, "high_glucose": 3, "heart_risk": 3,
    "elderly": 2.5, "overweight": 2, "underweight": 2, "low_bp": 2,
}


def _score(item: Dict[str, Any], tags: Set[str]) -> float:
    matches = set(item.get("conditions", [])) & tags
    specific = matches - {"general", "stress"}
    return sum(TAG_WEIGHTS.get(t, 1) for t in specific) + (0.5 if matches else 0)


def build_plan(
    age: int,
    urgency: str = "LOW",
    sex: str = "",
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    glucose: Optional[float] = None,
    bmi: Optional[float] = None,
    is_pregnant: bool = False,
    is_postpartum: bool = False,
    heart_risk: Optional[float] = None,
    diabetes_risk: Optional[float] = None,
    symptom_text: str = "",
    limits: Optional[Iterable[str]] = None,
    kb: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a personalised wellness plan. HIGH urgency returns no plan (seek care first)."""
    kb = kb or load_kb()
    meta = kb.get("meta", {})
    base = {"urgency": urgency, "reasons": [], "yoga": [], "exercise": [], "diet": [], "lifestyle": [],
            "home_care": [], "practices_to_avoid": [], "note": "", "disclaimer": meta.get("disclaimer", ""),
            "when_to_seek_care": kb.get("when_to_seek_care", [])}
    if urgency == "HIGH":
        base["note"] = "A wellness plan is not shown for a high-urgency result. Get medical help first."
        return base

    info = derive_tags(age, sex, systolic_bp, diastolic_bp, glucose, bmi, is_pregnant, is_postpartum,
                       heart_risk, diabetes_risk, symptom_text, limits)
    tags, avoid = info["tags"], info["avoid"]
    gentle_only = urgency == "MEDIUM" or is_postpartum or age >= 60
    base["reasons"] = info["reasons"]
    if urgency == "MEDIUM":
        base["note"] = ("See a doctor within 24 to 48 hours. Until then keep activity light; only gentle items are shown.")

    for cat in ("yoga", "exercise", "diet", "lifestyle"):
        pool = [it for it in kb.get(cat, []) if _eligible(it, age, avoid, is_pregnant, gentle_only if cat in ("yoga", "exercise") else False)]
        scored = [(s, it) for it in pool if (s := _score(it, tags)) > 0]
        scored.sort(key=lambda p: (-p[0], p[1]["id"]))
        chosen = [it for _, it in scored[: LIMITS_PER_CATEGORY[cat]]]
        if cat == "yoga" and chosen and not any(i["type"] in ("pranayama", "relaxation", "meditation") for i in chosen):
            extra = next((it for _, it in scored if it["type"] in ("pranayama", "relaxation", "meditation")), None)
            if extra:
                chosen = chosen[:-1] + [extra]
        base[cat] = chosen

    # Home care for minor ailments only on a LOW result.
    if urgency == "LOW":
        care = [it for it in kb.get("home_care", []) if set(it.get("conditions", [])) & tags - {"general", "stress"}]
        care.sort(key=lambda it: it["id"])
        base["home_care"] = care[: LIMITS_PER_CATEGORY["home_care"]]

    avoid_list = []
    for pr in kb.get("practices_to_avoid", []):
        if set(pr.get("avoid_if", [])) & avoid:
            avoid_list.append(pr)
    base["practices_to_avoid"] = avoid_list
    return base