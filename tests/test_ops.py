from datetime import datetime, timedelta

import pytest

from icu_stepdown.auth import resolve_expected_credentials, validate_credentials
from icu_stepdown.ops_logic import (
    compute_operational_blockers,
    compute_transfer_feasibility,
    forecast_bed_pressure,
)
from icu_stepdown.ops_store import (
    adjust_bed_inventory,
    list_patient_operational_status,
    init_ops_db,
    save_capacity,
    save_bed_inventory,
    save_theatre_schedule,
    get_latest_capacity,
    get_latest_bed_inventory,
    sync_patient_operational_status_from_datastores,
)
from icu_stepdown.patient_store import append_row


def test_auth_default_fallback():
    user, password, using_demo = resolve_expected_credentials(env={})
    assert user == "admin"
    assert password == "Test#12"
    assert using_demo is True


def test_auth_env_override():
    env = {"ICU_APP_USER": "alice", "ICU_APP_PASS": "secret"}
    user, password, using_demo = resolve_expected_credentials(env=env)
    assert user == "alice"
    assert password == "secret"
    assert using_demo is False
    valid, _ = validate_credentials("alice", "secret", env=env)
    assert valid is True


def test_save_and_load_capacity(tmp_path):
    db = tmp_path / "ops.sqlite"
    init_ops_db(str(db))
    save_capacity(
        str(db),
        {
            "icu_beds": 10,
            "hdu_beds": 6,
            "ward_beds": 20,
            "telemetry_beds": 4,
            "isolation_beds": 2,
            "surge_beds": 1,
            "shift_label": "Day",
        },
        user="tester",
    )
    latest = get_latest_capacity(str(db))
    assert latest["icu_beds"] == 10
    assert latest["telemetry_beds"] == 4


def test_blocked_due_to_no_telemetry_bed():
    latest_row = {"FiO2": 0.3, "pacing_active": 0}
    destination = "Ward"
    capacity = {"ward_beds": 10, "telemetry_beds": 2}
    inventory = {"ward_occupied": 5, "telemetry_occupied": 2}
    staffing = {"ward_nurse_available": 5}
    capability = {"can_manage_pacing_wires": 1, "can_manage_chest_drains": 1, "can_manage_low_oxygen": 1, "can_manage_insulin_infusion": 1}
    rules = {"min_telemetry_required": 1, "ward_oxygen_fio2_threshold": 0.4}
    blockers = compute_operational_blockers(latest_row, destination, capacity, inventory, staffing, capability, rules)
    assert "no telemetry bed" in blockers


def test_blocked_due_to_staffing():
    latest_row = {"FiO2": 0.3}
    destination = "Ward"
    capacity = {"ward_beds": 10}
    inventory = {"ward_occupied": 5}
    staffing = {"ward_nurse_available": 0}
    capability = {"can_manage_low_oxygen": 1}
    rules = {"ward_oxygen_fio2_threshold": 0.4}
    blockers = compute_operational_blockers(latest_row, destination, capacity, inventory, staffing, capability, rules)
    assert "Ward staffing unsafe" in blockers


def test_clinically_ready_but_no_downstream_bed():
    latest_row = {"FiO2": 0.3}
    destination = "Ward"
    capacity = {"ward_beds": 10}
    inventory = {"ward_occupied": 10}
    staffing = {"ward_nurse_available": 5}
    capability = {"can_manage_low_oxygen": 1}
    rules = {"ward_oxygen_fio2_threshold": 0.4}
    blockers = compute_operational_blockers(latest_row, destination, capacity, inventory, staffing, capability, rules)
    feasibility = compute_transfer_feasibility("GREEN", blockers)
    assert feasibility == "No"


def test_forecast_bed_pressure(tmp_path):
    db = tmp_path / "ops.sqlite"
    init_ops_db(str(db))
    save_capacity(
        str(db),
        {"icu_beds": 5, "hdu_beds": 0, "ward_beds": 0, "telemetry_beds": 0, "isolation_beds": 0, "surge_beds": 0, "shift_label": "Day"},
        user="tester",
    )
    save_bed_inventory(
        str(db),
        {"icu_occupied": 4, "hdu_occupied": 0, "ward_occupied": 0, "telemetry_occupied": 0},
        user="tester",
    )
    now = datetime.utcnow()
    save_theatre_schedule(
        str(db),
        [
            {
                "case_date": now.date().isoformat(),
                "procedure_group": "CABG",
                "expected_arrival_time": (now + timedelta(hours=4)).strftime("%H:%M"),
                "icu_need": "Definite",
                "is_emergency": False,
                "notes": "",
            }
        ],
        user="tester",
    )
    capacity = get_latest_capacity(str(db))
    inventory = get_latest_bed_inventory(str(db))
    forecast = forecast_bed_pressure(str(db), capacity, inventory)
    assert forecast["incoming_pressure_24h"] >= 1


def test_discharge_bed_updates_inventory(tmp_path):
    db = tmp_path / "ops.sqlite"
    init_ops_db(str(db))
    save_bed_inventory(
        str(db),
        {"icu_occupied": 3, "hdu_occupied": 2, "ward_occupied": 1, "telemetry_occupied": 1},
        user="tester",
    )
    adjust_bed_inventory(str(db), "ICU", delta=-1, note="Discharge ICU-1", user="tester")
    latest = get_latest_bed_inventory(str(db))
    assert latest["icu_occupied"] == 2


def test_transfer_readiness_separation():
    readiness_status = "RED"
    blockers = ["no ward bed"]
    feasibility = compute_transfer_feasibility(readiness_status, blockers)
    assert feasibility == "No"


def test_sync_patient_operational_status_from_datastores(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    clinical_db = data_root / "patient_a" / "icu_stepdown.sqlite"
    clinical_db.parent.mkdir()
    append_row(
        str(clinical_db),
        "1234567890",
        {
            "timestamp": "2025-01-01T08:00:00",
            "MAP": 75.0,
        },
    )

    ops_db = data_root / "icu_ops.sqlite"
    init_ops_db(str(ops_db))
    inserted = sync_patient_operational_status_from_datastores(str(ops_db), str(data_root))

    assert inserted == 1
    rows = list_patient_operational_status(str(ops_db))
    assert len(rows) == 1
    assert rows[0]["encounter_id"]
    assert rows[0]["patient_id"]
