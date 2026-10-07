"""Special populations module package."""

from modules.special_populations.rules import (
    load_pediatric_data,
    detect_patient_category,
    get_pediatric_bracket,
    evaluate_pediatric_vitals,
    evaluate_special_population,
)

__all__ = [
    "load_pediatric_data",
    "detect_patient_category",
    "get_pediatric_bracket",
    "evaluate_pediatric_vitals",
    "evaluate_special_population",
]
