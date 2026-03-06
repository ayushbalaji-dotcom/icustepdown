import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_PROCEDURE_GROUPS = [
    "CABG",
    "AVR",
    "MVR",
    "Double valve",
    "Valve + CABG",
    "Aortic surgery",
    "Endocarditis surgery",
    "Other complex cardiac procedure",
]


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_ops_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS capacity_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                icu_beds INTEGER,
                hdu_beds INTEGER,
                ward_beds INTEGER,
                telemetry_beds INTEGER,
                isolation_beds INTEGER,
                surge_beds INTEGER,
                shift_label TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS staffing_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                icu_nurse_available INTEGER,
                hdu_nurse_available INTEGER,
                ward_nurse_available INTEGER,
                ratio_icu REAL,
                ratio_hdu REAL,
                ratio_ward REAL,
                cardiac_trained_nurses INTEGER,
                telemetry_available INTEGER,
                outreach_available INTEGER,
                registrar_cover_available INTEGER,
                physio_available INTEGER,
                respiratory_support_available INTEGER,
                dialysis_support_available INTEGER,
                notes TEXT,
                shift_label TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ward_capability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                can_manage_pacing_wires INTEGER,
                can_manage_chest_drains INTEGER,
                can_manage_low_oxygen INTEGER,
                can_manage_insulin_infusion INTEGER,
                telemetry_monitoring_available INTEGER,
                notes TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS procedure_los_reference (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_group TEXT UNIQUE NOT NULL,
                avg_icu_los_hours REAL,
                avg_hdu_los_hours REAL,
                comments TEXT,
                last_reviewed TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS theatre_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_date TEXT NOT NULL,
                procedure_group TEXT NOT NULL,
                expected_arrival_time TEXT,
                icu_need TEXT,
                is_emergency INTEGER,
                notes TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bed_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                icu_occupied INTEGER,
                hdu_occupied INTEGER,
                ward_occupied INTEGER,
                telemetry_occupied INTEGER,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transfer_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                min_telemetry_required INTEGER,
                pacing_wires_require_hdu INTEGER,
                chest_drains_require_hdu INTEGER,
                ward_oxygen_fio2_threshold REAL,
                ward_exclusion_vasoactive INTEGER,
                ward_exclusion_invasive_lines INTEGER,
                endocarditis_to_hdu INTEGER,
                notes TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_operational_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                procedure_group TEXT,
                readiness_status TEXT,
                readiness_score REAL,
                destination_recommendation TEXT,
                transfer_feasibility TEXT,
                operational_blockers TEXT,
                bed_priority_score REAL,
                first_ready_at TEXT,
                last_ready_at TEXT,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operational_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                description TEXT,
                user TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(cur, "patient_operational_status", "procedure_group", "TEXT")
        conn.commit()
    finally:
        conn.close()


def log_audit(db_path: str, event_type: str, entity_type: str, entity_id: str | None, description: str, user: str | None) -> None:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO operational_audit_log (event_type, entity_type, entity_id, description, user, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event_type, entity_type, entity_id, description, user, _utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def seed_ops_data(db_path: str, user: str | None = None) -> None:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM admin_users")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                "INSERT INTO admin_users (username, role, created_at, updated_at) VALUES (?, ?, ?, ?)",
                ("admin", "admin", now, now),
            )
        cur.execute("SELECT COUNT(*) FROM procedure_los_reference")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            for group in DEFAULT_PROCEDURE_GROUPS:
                cur.execute(
                    """
                    INSERT INTO procedure_los_reference
                    (procedure_group, avg_icu_los_hours, avg_hdu_los_hours, comments, last_reviewed, updated_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (group, 24.0, 24.0, "Seeded demo value", now[:10], now, now),
                )
        cur.execute("SELECT COUNT(*) FROM capacity_config")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                """
                INSERT INTO capacity_config
                (icu_beds, hdu_beds, ward_beds, telemetry_beds, isolation_beds, surge_beds, shift_label, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (12, 8, 20, 6, 2, 2, "Day", now, now),
            )
        cur.execute("SELECT COUNT(*) FROM staffing_status")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                """
                INSERT INTO staffing_status
                (icu_nurse_available, hdu_nurse_available, ward_nurse_available, ratio_icu, ratio_hdu, ratio_ward,
                 cardiac_trained_nurses, telemetry_available, outreach_available, registrar_cover_available,
                 physio_available, respiratory_support_available, dialysis_support_available, notes, shift_label, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (6, 4, 10, 1.5, 2.0, 5.0, 3, 1, 1, 1, 1, 1, 1, "Seeded demo staffing", "Day", now, now),
            )
        cur.execute("SELECT COUNT(*) FROM ward_capability")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                """
                INSERT INTO ward_capability
                (can_manage_pacing_wires, can_manage_chest_drains, can_manage_low_oxygen,
                 can_manage_insulin_infusion, telemetry_monitoring_available, notes, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (0, 0, 1, 0, 1, "Seeded demo capability", now, now),
            )
        cur.execute("SELECT COUNT(*) FROM bed_inventory")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                """
                INSERT INTO bed_inventory
                (icu_occupied, hdu_occupied, ward_occupied, telemetry_occupied, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (8, 5, 14, 4, now, now),
            )
        cur.execute("SELECT COUNT(*) FROM transfer_rules")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            cur.execute(
                """
                INSERT INTO transfer_rules
                (min_telemetry_required, pacing_wires_require_hdu, chest_drains_require_hdu,
                 ward_oxygen_fio2_threshold, ward_exclusion_vasoactive, ward_exclusion_invasive_lines,
                 endocarditis_to_hdu, notes, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (1, 1, 1, 0.4, 1, 1, 1, "Seeded demo rules", now, now),
            )
        cur.execute("SELECT COUNT(*) FROM theatre_schedule")
        if int(cur.fetchone()[0]) == 0:
            now = _utc_now()
            today = now[:10]
            cur.execute(
                """
                INSERT INTO theatre_schedule
                (case_date, procedure_group, expected_arrival_time, icu_need, is_emergency, notes, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (today, "CABG", "14:00", "Definite", 0, "Seeded elective case", now, now),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "seed", "system", None, "Seeded demo operational data", user)


def _fetch_latest(conn: sqlite3.Connection, table: str) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} ORDER BY updated_at DESC, created_at DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def get_latest_capacity(db_path: str) -> Optional[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _fetch_latest(conn, "capacity_config")
    finally:
        conn.close()


def get_latest_staffing(db_path: str) -> Optional[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _fetch_latest(conn, "staffing_status")
    finally:
        conn.close()


def get_latest_capability(db_path: str) -> Optional[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _fetch_latest(conn, "ward_capability")
    finally:
        conn.close()


def get_latest_rules(db_path: str) -> Optional[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _fetch_latest(conn, "transfer_rules")
    finally:
        conn.close()


def get_latest_bed_inventory(db_path: str) -> Optional[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _fetch_latest(conn, "bed_inventory")
    finally:
        conn.close()


def list_procedure_los(db_path: str) -> List[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM procedure_los_reference ORDER BY procedure_group")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def list_theatre_schedule(db_path: str) -> List[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM theatre_schedule ORDER BY case_date, expected_arrival_time")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def save_capacity(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO capacity_config
            (icu_beds, hdu_beds, ward_beds, telemetry_beds, isolation_beds, surge_beds, shift_label, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("icu_beds"),
                payload.get("hdu_beds"),
                payload.get("ward_beds"),
                payload.get("telemetry_beds"),
                payload.get("isolation_beds"),
                payload.get("surge_beds"),
                payload.get("shift_label"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "capacity_config", None, "Updated capacity settings", user)


def save_staffing(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO staffing_status
            (icu_nurse_available, hdu_nurse_available, ward_nurse_available, ratio_icu, ratio_hdu, ratio_ward,
             cardiac_trained_nurses, telemetry_available, outreach_available, registrar_cover_available,
             physio_available, respiratory_support_available, dialysis_support_available, notes, shift_label, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("icu_nurse_available"),
                payload.get("hdu_nurse_available"),
                payload.get("ward_nurse_available"),
                payload.get("ratio_icu"),
                payload.get("ratio_hdu"),
                payload.get("ratio_ward"),
                payload.get("cardiac_trained_nurses"),
                payload.get("telemetry_available"),
                payload.get("outreach_available"),
                payload.get("registrar_cover_available"),
                payload.get("physio_available"),
                payload.get("respiratory_support_available"),
                payload.get("dialysis_support_available"),
                payload.get("notes"),
                payload.get("shift_label"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "staffing_status", None, "Updated staffing and capability status", user)


def save_capability(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ward_capability
            (can_manage_pacing_wires, can_manage_chest_drains, can_manage_low_oxygen,
             can_manage_insulin_infusion, telemetry_monitoring_available, notes, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("can_manage_pacing_wires"),
                payload.get("can_manage_chest_drains"),
                payload.get("can_manage_low_oxygen"),
                payload.get("can_manage_insulin_infusion"),
                payload.get("telemetry_monitoring_available"),
                payload.get("notes"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "ward_capability", None, "Updated ward capability", user)


def save_bed_inventory(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bed_inventory
            (icu_occupied, hdu_occupied, ward_occupied, telemetry_occupied, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("icu_occupied"),
                payload.get("hdu_occupied"),
                payload.get("ward_occupied"),
                payload.get("telemetry_occupied"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "bed_inventory", None, "Updated bed occupancy", user)


def adjust_bed_inventory(db_path: str, area: str, delta: int, note: str | None, user: str | None) -> Dict[str, int]:
    latest = get_latest_bed_inventory(db_path) or {}
    icu = int(latest.get("icu_occupied") or 0)
    hdu = int(latest.get("hdu_occupied") or 0)
    ward = int(latest.get("ward_occupied") or 0)
    telemetry = int(latest.get("telemetry_occupied") or 0)

    area_key = area.strip().lower()
    if area_key == "icu":
        icu = max(icu + delta, 0)
    elif area_key == "hdu":
        hdu = max(hdu + delta, 0)
    elif area_key == "ward":
        ward = max(ward + delta, 0)
    elif area_key == "telemetry":
        telemetry = max(telemetry + delta, 0)
    else:
        raise ValueError("Unknown bed area")

    save_bed_inventory(
        db_path,
        {
            "icu_occupied": icu,
            "hdu_occupied": hdu,
            "ward_occupied": ward,
            "telemetry_occupied": telemetry,
        },
        user,
    )
    if note:
        log_audit(db_path, "update", "bed_inventory", area, note, user)
    return {
        "icu_occupied": icu,
        "hdu_occupied": hdu,
        "ward_occupied": ward,
        "telemetry_occupied": telemetry,
    }


def save_transfer_rules(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO transfer_rules
            (min_telemetry_required, pacing_wires_require_hdu, chest_drains_require_hdu, ward_oxygen_fio2_threshold,
             ward_exclusion_vasoactive, ward_exclusion_invasive_lines, endocarditis_to_hdu, notes, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("min_telemetry_required"),
                payload.get("pacing_wires_require_hdu"),
                payload.get("chest_drains_require_hdu"),
                payload.get("ward_oxygen_fio2_threshold"),
                payload.get("ward_exclusion_vasoactive"),
                payload.get("ward_exclusion_invasive_lines"),
                payload.get("endocarditis_to_hdu"),
                payload.get("notes"),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "transfer_rules", None, "Updated transfer rules", user)


def save_procedure_los(db_path: str, rows: Iterable[Dict[str, Any]], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for row in rows:
            group = row.get("procedure_group")
            if not group:
                continue
            cur.execute(
                """
                INSERT INTO procedure_los_reference
                (procedure_group, avg_icu_los_hours, avg_hdu_los_hours, comments, last_reviewed, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(procedure_group) DO UPDATE SET
                    avg_icu_los_hours=excluded.avg_icu_los_hours,
                    avg_hdu_los_hours=excluded.avg_hdu_los_hours,
                    comments=excluded.comments,
                    last_reviewed=excluded.last_reviewed,
                    updated_at=excluded.updated_at
                """,
                (
                    group,
                    row.get("avg_icu_los_hours"),
                    row.get("avg_hdu_los_hours"),
                    row.get("comments"),
                    row.get("last_reviewed"),
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "procedure_los_reference", None, "Updated procedure LOS reference", user)


def save_theatre_schedule(db_path: str, rows: Iterable[Dict[str, Any]], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM theatre_schedule")
        for row in rows:
            if not row.get("case_date") or not row.get("procedure_group"):
                continue
            cur.execute(
                """
                INSERT INTO theatre_schedule
                (case_date, procedure_group, expected_arrival_time, icu_need, is_emergency, notes, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("case_date"),
                    row.get("procedure_group"),
                    row.get("expected_arrival_time"),
                    row.get("icu_need"),
                    1 if row.get("is_emergency") else 0,
                    row.get("notes"),
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(db_path, "update", "theatre_schedule", None, "Updated theatre schedule", user)


def upsert_patient_operational_status(db_path: str, payload: Dict[str, Any], user: str | None) -> None:
    init_ops_db(db_path)
    now = _utc_now()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, first_ready_at, readiness_status, transfer_feasibility, operational_blockers FROM patient_operational_status WHERE encounter_id = ?",
            (payload.get("encounter_id"),),
        )
        existing = cur.fetchone()
        first_ready_at = existing[1] if existing else None
        prev_status = existing[2] if existing else None
        prev_feas = existing[3] if existing else None
        prev_blockers = existing[4] if existing else None
        readiness_status = payload.get("readiness_status")
        if readiness_status == "GREEN" and not first_ready_at:
            first_ready_at = now
        if readiness_status != "GREEN":
            first_ready_at = None
        blockers_json = json.dumps(payload.get("operational_blockers") or [])
        cur.execute(
            """
            INSERT INTO patient_operational_status
            (encounter_id, patient_id, procedure_group, readiness_status, readiness_score, destination_recommendation,
             transfer_feasibility, operational_blockers, bed_priority_score, first_ready_at, last_ready_at, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(encounter_id) DO UPDATE SET
                patient_id=excluded.patient_id,
                procedure_group=excluded.procedure_group,
                readiness_status=excluded.readiness_status,
                readiness_score=excluded.readiness_score,
                destination_recommendation=excluded.destination_recommendation,
                transfer_feasibility=excluded.transfer_feasibility,
                operational_blockers=excluded.operational_blockers,
                bed_priority_score=excluded.bed_priority_score,
                first_ready_at=excluded.first_ready_at,
                last_ready_at=excluded.last_ready_at,
                updated_at=excluded.updated_at
            """,
            (
                payload.get("encounter_id"),
                payload.get("patient_id"),
                payload.get("procedure_group"),
                readiness_status,
                payload.get("readiness_score"),
                payload.get("destination_recommendation"),
                payload.get("transfer_feasibility"),
                blockers_json,
                payload.get("bed_priority_score"),
                first_ready_at,
                now if readiness_status == "GREEN" else None,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    if payload.get("readiness_status") and (
        payload.get("readiness_status") != prev_status
        or payload.get("transfer_feasibility") != prev_feas
        or blockers_json != prev_blockers
    ):
        log_audit(
            db_path,
            "patient_status",
            "patient_operational_status",
            payload.get("encounter_id"),
            f"Updated patient status: {payload.get('readiness_status')} / {payload.get('transfer_feasibility')}",
            user,
        )


def list_patient_operational_status(db_path: str) -> List[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM patient_operational_status ORDER BY updated_at DESC")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        results = []
        for row in rows:
            item = dict(zip(cols, row))
            try:
                item["operational_blockers"] = json.loads(item.get("operational_blockers") or "[]")
            except Exception:
                item["operational_blockers"] = []
            results.append(item)
        return results
    finally:
        conn.close()


def list_audit_log(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    init_ops_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM operational_audit_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()
