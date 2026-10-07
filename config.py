"""Configuration parameters, paths, and thresholds for AI Medical Triage Dashboard."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database & Data Paths
REGISTRY_DB_PATH = DATA_DIR / "registry.db"
REFERRAL_LEVELS_PATH = DATA_DIR / "referral_levels.json"
PEDIATRIC_RANGES_PATH = DATA_DIR / "pediatric_ranges.json"
FERNET_KEY_PATH = DATA_DIR / ".fernet_key"

# Urgency Tiers
TIER_LOW = "LOW"
TIER_MEDIUM = "MEDIUM"
TIER_HIGH = "HIGH"
TIERS = [TIER_LOW, TIER_MEDIUM, TIER_HIGH]

# Module A: Risk Screening Thresholds
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.65

# Module B: Vitals Thresholds
VITALS_HR_LOW_CRITICAL = 40
VITALS_HR_HIGH_CRITICAL = 140
VITALS_SPO2_CRITICAL = 90.0
VITALS_SPO2_WARNING = 93.0
VITALS_RR_LOW_CRITICAL = 8
VITALS_RR_HIGH_CRITICAL = 32
VITALS_TEMP_HIGH_CRITICAL = 39.5
VITALS_ANOMALY_PERCENTILE_THRESHOLD = 0.65

# Module C: Symptom Triage Thresholds
NLP_LOW_CONFIDENCE_THRESHOLD = 0.45  # escalate to MEDIUM if below
NLP_HIGH_RECALL_WEIGHT = 2.5        # penalty for missing HIGH tier

# Fusion Engine
FUSION_WEIGHTS = {
    "symptom_triage": 0.45,
    "vitals_anomaly": 0.35,
    "risk_screening": 0.20,
}

# Queue Configuration
QUEUE_MEDIUM_BEFORE_LOW = False  # False = strict arrival FCFS; True = prioritize MEDIUM before LOW

# Mode Configuration (Feature 1)
MODE_HOSPITAL = "🏥 Hospital"
MODE_HOME = "🏠 Home Use"
DEFAULT_MODE = MODE_HOME

MODE_CONFIG = {
    MODE_HOSPITAL: {
        "key": "hospital",
        "label": "Hospital Mode",
        "description": "Clinical workstation for triage nurses and doctors",
        "clinical_language": True,
        "show_scores": True,
        "show_module_breakdown": True,
        "show_technical_plots": True,
        "technical_plots_in_expander": False,
        "show_queue": True,
        "show_handoff": True,
        "show_registry_search": True,
        "allow_manual_vitals": True,
        "allow_stream_vitals": True,
        "pdf_type": "clinical",
        "enable_anonymous_check": False,
    },
    MODE_HOME: {
        "key": "home",
        "label": "Home Use Mode",
        "description": "Patient self-assessment and home care guidance",
        "clinical_language": False,
        "show_scores": False,
        "show_module_breakdown": False,
        "show_technical_plots": False,
        "technical_plots_in_expander": True,
        "show_queue": False,
        "show_handoff": False,
        "show_registry_search": False,
        "allow_manual_vitals": True,
        "allow_stream_vitals": False,
        "pdf_type": "patient_friendly",
        "enable_anonymous_check": True,
    }
}

# Mobile Validation Patterns by Country Code
COUNTRY_PHONE_CONFIG = {
    "+91": {
        "name": "India",
        "digits": 10,
        "regex": r"^[6-9]\d{9}$",
        "example": "9876543210",
        "emergency": "112 / 108"
    },
    "+1": {
        "name": "USA / Canada",
        "digits": 10,
        "regex": r"^[2-9]\d{9}$",
        "example": "4155552671",
        "emergency": "911"
    },
    "+44": {
        "name": "United Kingdom",
        "digits": 10,
        "regex": r"^[1-9]\d{9,10}$",
        "example": "7911123456",
        "emergency": "999"
    },
    "+254": {
        "name": "Kenya",
        "digits": 9,
        "regex": r"^[17]\d{8}$",
        "example": "712345678",
        "emergency": "999 / 112"
    },
    "+977": {
        "name": "Nepal",
        "digits": 10,
        "regex": r"^9[78]\d{8}$",
        "example": "9841234567",
        "emergency": "102 / 100"
    }
}

# Emergency and Deployment Info
DEFAULT_EMERGENCY_NUMBER = "112 / 108"
PROJECT_TITLE = "Multi-Modal AI Medical Triage Dashboard"
SDG_ALIGNMENT = "UN SDG 3: Good Health and Well-Being (Target 3.8: Universal Health Coverage)"
RANDOM_SEED = 42
