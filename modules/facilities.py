"""Nearby hospital suggestions and busy-hospital redirection (offline, rule-based).

Uses data/nearby_hospitals.json (a DEMO directory: replace with verified local
hospitals and real phone numbers). "Nearby" is judged from the city the user
picks, using straight-line distance between city-level coordinates, so no GPS
and no internet are needed.
"""
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

DIRECTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "nearby_hospitals.json"
_LEVEL_RANK = {"advanced": 0, "basic": 1, "none": 2}


@lru_cache(maxsize=1)
def load_directory() -> Dict[str, Any]:
    """Load the hospital directory; returns an empty directory if the file is missing."""
    try:
        return json.loads(DIRECTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"meta": {}, "national_helplines": [], "cities": [], "hospitals": []}


def list_cities() -> List[str]:
    return [c["name"] for c in load_directory().get("cities", [])]


def national_helplines() -> List[Dict[str, str]]:
    return load_directory().get("national_helplines", [])


def directory_warning() -> str:
    meta = load_directory().get("meta", {})
    return meta.get("warning", "") if meta.get("demo_data") else ""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance between two points in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _city(name: str) -> Optional[Dict[str, Any]]:
    return next((c for c in load_directory().get("cities", []) if c["name"].lower() == (name or "").lower()), None)


def find_nearby(city: str, limit: int = 5, emergency_only: bool = False) -> List[Dict[str, Any]]:
    """Hospitals sorted by approximate distance from the chosen city.

    emergency_only keeps only hospitals with basic or advanced emergency care.
    Each result has an added 'distance_km' (approximate, city level).
    """
    origin = _city(city)
    if origin is None:
        return []
    out = []
    for h in load_directory().get("hospitals", []):
        if emergency_only and h.get("emergency_level", "none") == "none":
            continue
        item = dict(h)
        item["distance_km"] = round(haversine_km(origin["lat"], origin["lon"], h["lat"], h["lon"]), 1)
        out.append(item)
    out.sort(key=lambda h: (h["distance_km"], _LEVEL_RANK.get(h.get("emergency_level", "none"), 3), h["name"]))
    return out[:limit]


# ---------------------------------------------------------------- queue load
def _read_queue(registry_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from dashboard.queue import get_queue  # imported late so this module works without Streamlit
    return get_queue(registry_id)


def get_hospital_load(registry_id: Optional[str]) -> Optional[Dict[str, int]]:
    """Current queue load for a hospital known to the registry, or None if unknown."""
    if not registry_id:
        return None
    try:
        priority, regular = _read_queue(registry_id)
    except Exception:
        return None
    return {"priority": len(priority), "waiting": len(regular), "total": len(priority) + len(regular)}


def is_busy(load: Optional[Dict[str, int]]) -> bool:
    """True when the queue is long. Thresholds can be set in config.py."""
    if not load:
        return False
    return (load["priority"] >= getattr(config, "QUEUE_BUSY_PRIORITY", 2)
            or load["total"] >= getattr(config, "QUEUE_BUSY_TOTAL", 4))


def suggest_for_high(city: str, max_alternatives: int = 3) -> Dict[str, Any]:
    """For a HIGH-urgency home user: check the nearest emergency hospital's queue.

    If it is busy, return other emergency-capable hospitals with a shorter or
    unknown queue. Always returns the nearest hospital as 'primary'.
    """
    candidates = find_nearby(city, limit=10, emergency_only=True)
    for c in candidates:
        c["load"] = get_hospital_load(c.get("registry_id"))
        c["busy"] = is_busy(c["load"])
    result = {"primary": None, "busy": False, "alternatives": [], "candidates": candidates}
    if not candidates:
        return result
    primary = candidates[0]
    result["primary"] = primary
    result["busy"] = primary["busy"]
    if primary["busy"]:
        others = [c for c in candidates[1:] if not c["busy"]]
        # Known-short queues first, then unknown, each nearest first.
        others.sort(key=lambda c: (0 if c["load"] is not None else 1,
                                   c["load"]["total"] if c["load"] else 0, c["distance_km"]))
        result["alternatives"] = others[:max_alternatives]
    return result