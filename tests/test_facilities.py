"""Tests for nearby-hospital suggestions and busy-hospital redirection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import facilities  # noqa: E402


def load(priority, waiting):
    return {"priority": priority, "waiting": waiting, "total": priority + waiting}


def test_directory_loads_and_is_flagged_demo():
    assert facilities.list_cities()
    assert "DEMO" in facilities.directory_warning()
    for h in facilities.load_directory()["hospitals"]:
        assert h["name"] and h["phone"]


def test_find_nearby_sorted_by_distance():
    res = facilities.find_nearby("Jaipur", limit=6)
    assert res and res[0]["distance_km"] == 0
    dists = [h["distance_km"] for h in res]
    assert dists == sorted(dists)


def test_unknown_city_gives_empty_list():
    assert facilities.find_nearby("Atlantis") == []
    assert facilities.suggest_for_high("Atlantis")["primary"] is None


def test_emergency_only_filters_clinics():
    for h in facilities.find_nearby("Bangalore", limit=10, emergency_only=True):
        assert h["emergency_level"] != "none"


def test_haversine():
    assert facilities.haversine_km(26.9, 75.8, 26.9, 75.8) == 0
    assert 100 < facilities.haversine_km(26.9124, 75.7873, 27.5530, 76.6346) < 130


def test_is_busy_thresholds():
    assert not facilities.is_busy(None)
    assert not facilities.is_busy(load(1, 2))
    assert facilities.is_busy(load(2, 0))      # 2 emergency patients
    assert facilities.is_busy(load(0, 4))      # 4 in total


def test_busy_hospital_gets_alternatives(monkeypatch):
    monkeypatch.setattr(facilities, "_read_queue", lambda rid: ([{}] * 3, [{}] * 3) if rid == "HOSP-001" else ([], []))
    res = facilities.suggest_for_high("Jaipur")
    assert res["primary"]["registry_id"] == "HOSP-001" and res["busy"]
    assert res["alternatives"]
    assert all(not a["busy"] for a in res["alternatives"])
    assert all(a["name"] != res["primary"]["name"] for a in res["alternatives"])
    assert all(a["emergency_level"] != "none" for a in res["alternatives"])


def test_quiet_hospital_has_no_alternatives(monkeypatch):
    monkeypatch.setattr(facilities, "_read_queue", lambda rid: ([], [{}]))
    res = facilities.suggest_for_high("Jaipur")
    assert not res["busy"] and res["alternatives"] == []


def test_unknown_queue_is_not_treated_as_busy():
    res = facilities.suggest_for_high("Mumbai")  # no registry ids in Mumbai
    assert not res["busy"] and res["primary"]["load"] is None


def test_queue_read_failure_is_handled(monkeypatch):
    def boom(rid):
        raise RuntimeError("db locked")
    monkeypatch.setattr(facilities, "_read_queue", boom)
    assert facilities.get_hospital_load("HOSP-001") is None
    assert facilities.get_hospital_load(None) is None