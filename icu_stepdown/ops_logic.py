from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .ops_store import list_procedure_los, list_theatre_schedule


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def latest_clinical_snapshot(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    latest = df.tail(1).to_dict(orient="records")[0]
    return latest


def compute_destination_recommendation(latest_row: Dict[str, Any], rules: Dict[str, Any], procedure_group: str | None) -> str:
    pacing_active = _to_float(latest_row.get("pacing_active")) == 1.0
    chest_drain = _to_float(latest_row.get("chest_drain_ml_30min"))
    fio2 = _to_float(latest_row.get("FiO2"))
    oxygen_device = str(latest_row.get("oxygen_device") or "").upper()
    pressor_any = any((_to_float(latest_row.get(k)) or 0) > 0 for k in [
        "noradrenaline_mcgkgmin",
        "adrenaline_mcgkgmin",
        "dobutamine_mcgkgmin",
        "milrinone_mcgkgmin",
    ])

    if rules.get("endocarditis_to_hdu") and procedure_group == "Endocarditis surgery":
        return "HDU"
    if rules.get("pacing_wires_require_hdu") and pacing_active:
        return "HDU"
    if rules.get("chest_drains_require_hdu") and chest_drain and chest_drain > 0:
        return "HDU"
    if rules.get("ward_exclusion_vasoactive") and pressor_any:
        return "HDU"
    if fio2 is not None and rules.get("ward_oxygen_fio2_threshold") is not None:
        try:
            if fio2 > float(rules.get("ward_oxygen_fio2_threshold")):
                return "HDU"
        except Exception:
            pass
    if oxygen_device in {"ETT", "NIV", "HFNC"}:
        return "HDU"
    return "Ward"


def compute_operational_blockers(
    latest_row: Dict[str, Any],
    destination: str,
    capacity: Dict[str, Any],
    bed_inventory: Dict[str, Any],
    staffing: Dict[str, Any],
    capability: Dict[str, Any],
    rules: Dict[str, Any],
) -> List[str]:
    blockers: List[str] = []

    def free_beds(total: Any, occupied: Any) -> Optional[int]:
        if total is None or occupied is None:
            return None
        try:
            return max(int(total) - int(occupied), 0)
        except Exception:
            return None

    icu_free = free_beds(capacity.get("icu_beds"), bed_inventory.get("icu_occupied"))
    hdu_free = free_beds(capacity.get("hdu_beds"), bed_inventory.get("hdu_occupied"))
    ward_free = free_beds(capacity.get("ward_beds"), bed_inventory.get("ward_occupied"))
    telemetry_free = free_beds(capacity.get("telemetry_beds"), bed_inventory.get("telemetry_occupied"))

    if destination == "HDU":
        if hdu_free is not None and hdu_free <= 0:
            blockers.append("no HDU bed")
    if destination == "Ward":
        if ward_free is not None and ward_free <= 0:
            blockers.append("no ward bed")
        if rules.get("min_telemetry_required"):
            if telemetry_free is not None and telemetry_free <= 0:
                blockers.append("no telemetry bed")

    if staffing.get("icu_nurse_available") is not None and staffing.get("icu_nurse_available") <= 0:
        blockers.append("ICU staffing unsafe")
    if destination == "HDU" and staffing.get("hdu_nurse_available") is not None and staffing.get("hdu_nurse_available") <= 0:
        blockers.append("HDU staffing unsafe")
    if destination == "Ward" and staffing.get("ward_nurse_available") is not None and staffing.get("ward_nurse_available") <= 0:
        blockers.append("Ward staffing unsafe")

    if staffing.get("telemetry_available") is not None and staffing.get("telemetry_available") <= 0 and destination == "Ward":
        blockers.append("telemetry staffing unavailable")

    pacing_active = _to_float(latest_row.get("pacing_active")) == 1.0
    chest_drain = _to_float(latest_row.get("chest_drain_ml_30min")) or 0
    insulin_infusion = _to_float(latest_row.get("insulin_infusion")) == 1.0
    low_oxygen = False
    fio2 = _to_float(latest_row.get("FiO2"))
    if fio2 is not None and rules.get("ward_oxygen_fio2_threshold") is not None:
        try:
            low_oxygen = fio2 > float(rules.get("ward_oxygen_fio2_threshold"))
        except Exception:
            low_oxygen = False

    if destination == "Ward":
        if pacing_active and not capability.get("can_manage_pacing_wires"):
            blockers.append("pacing wire capability unavailable")
        if chest_drain > 0 and not capability.get("can_manage_chest_drains"):
            blockers.append("chest drain capability unavailable")
        if low_oxygen and not capability.get("can_manage_low_oxygen"):
            blockers.append("oxygen requirement too high for ward")
        if insulin_infusion and not capability.get("can_manage_insulin_infusion"):
            blockers.append("insulin infusion capability unavailable")

    return blockers


def compute_transfer_feasibility(readiness_status: str, blockers: List[str]) -> str:
    if readiness_status == "GREEN":
        return "Yes" if not blockers else "No"
    if readiness_status == "AMBER":
        return "Uncertain"
    if readiness_status == "WAIT":
        return "Uncertain"
    return "No"


def _incoming_pressure_score(ops_db_path: str, window_hours: int = 24) -> float:
    now = datetime.utcnow()
    rows = list_theatre_schedule(ops_db_path)
    score = 0.0
    for row in rows:
        date_str = row.get("case_date")
        time_str = row.get("expected_arrival_time") or "00:00"
        try:
            when = datetime.fromisoformat(f"{date_str}T{time_str}")
        except Exception:
            continue
        if when < now and date_str == now.date().isoformat():
            rollover = when + timedelta(days=1)
            if rollover <= now + timedelta(hours=window_hours):
                when = rollover
        if when < now or when > now + timedelta(hours=window_hours):
            continue
        need = str(row.get("icu_need") or "").lower()
        if "definite" in need:
            score += 1.0
        elif "likely" in need:
            score += 0.5
    return score


def _procedure_los_hours(ops_db_path: str, procedure_group: str | None) -> Optional[float]:
    if not procedure_group:
        return None
    rows = list_procedure_los(ops_db_path)
    for row in rows:
        if row.get("procedure_group") == procedure_group:
            return _to_float(row.get("avg_icu_los_hours"))
    return None


def bed_priority_score(
    readiness_status: str,
    readiness_score: Optional[float],
    hours_since_ready: Optional[float],
    transfer_feasibility: str,
    procedure_group: str | None,
    ops_db_path: str,
) -> float:
    score = 0.0
    if readiness_status == "GREEN":
        score += 60
    elif readiness_status == "AMBER":
        score += 30
    if readiness_score is not None:
        score += min(10.0, max(0.0, (readiness_score - 50) / 5))
    if hours_since_ready is not None:
        score += min(20.0, hours_since_ready * 2)
    los = _procedure_los_hours(ops_db_path, procedure_group)
    if los and hours_since_ready and hours_since_ready > los:
        score += 10.0
    pressure = _incoming_pressure_score(ops_db_path, 24)
    score += min(10.0, pressure * 2)
    if transfer_feasibility == "Yes":
        score += 5
    elif transfer_feasibility == "No":
        score -= 5
    return max(0.0, min(100.0, score))


def forecast_bed_pressure(ops_db_path: str, capacity: Dict[str, Any], bed_inventory: Dict[str, Any]) -> Dict[str, Any]:
    free_icu = None
    if capacity.get("icu_beds") is not None and bed_inventory.get("icu_occupied") is not None:
        free_icu = max(int(capacity.get("icu_beds")) - int(bed_inventory.get("icu_occupied")), 0)
    pressure_24 = _incoming_pressure_score(ops_db_path, 24)
    pressure_48 = _incoming_pressure_score(ops_db_path, 48)
    return {
        "incoming_pressure_24h": pressure_24,
        "incoming_pressure_48h": pressure_48,
        "icu_free_beds": free_icu,
        "icu_pressure_flag": free_icu is not None and pressure_24 > free_icu,
    }
