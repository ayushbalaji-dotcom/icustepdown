import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional


def _safe_nhs(nhs_number: str) -> str:
    digits = "".join(ch for ch in str(nhs_number) if ch.isdigit())
    if len(digits) < 6:
        raise ValueError("Invalid NHS number")
    return digits


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nhs_number TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS encounters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                encounter_id TEXT UNIQUE NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_icu_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                MAP REAL,
                HR REAL,
                RR REAL,
                SpO2 REAL,
                FiO2 REAL,
                noradrenaline_mcgkgmin REAL,
                adrenaline_mcgkgmin REAL,
                dobutamine_mcgkgmin REAL,
                milrinone_mcgkgmin REAL,
                urine_output_ml_30min REAL,
                chest_drain_ml_30min REAL,
                lactate REAL,
                haemoglobin_gL REAL,
                creatinine_umolL REAL,
                WCC_10e9L REAL,
                temperature_C REAL,
                RASS REAL,
                oxygen_device TEXT,
                arterial_line_present REAL,
                central_line_present REAL,
                insulin_infusion REAL,
                pacing_active REAL,
                imaging_summary TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _get_patient_id(conn: sqlite3.Connection, nhs_number: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM patients WHERE nhs_number = ?", (nhs_number,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        "INSERT INTO patients (nhs_number, created_at) VALUES (?, ?)",
        (nhs_number, datetime.utcnow().isoformat()),
    )
    return int(cur.lastrowid)


def start_encounter(db_path: str, nhs_number: str, force_new: bool = False) -> str:
    nhs = _safe_nhs(nhs_number)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        patient_id = _get_patient_id(conn, nhs)
        if not force_new:
            cur.execute(
                "SELECT encounter_id FROM encounters WHERE patient_id = ? ORDER BY started_at DESC LIMIT 1",
                (patient_id,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
        encounter_id = f"{nhs}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        cur.execute(
            "INSERT INTO encounters (patient_id, encounter_id, started_at) VALUES (?, ?, ?)",
            (patient_id, encounter_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return encounter_id
    finally:
        conn.close()


def get_latest_encounter(db_path: str, nhs_number: str) -> Optional[str]:
    nhs = _safe_nhs(nhs_number)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT encounter_id FROM encounters
            JOIN patients ON encounters.patient_id = patients.id
            WHERE patients.nhs_number = ?
            ORDER BY encounters.started_at DESC
            LIMIT 1
            """,
            (nhs,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def append_row(db_path: str, nhs_number: str, row: Dict[str, Any]) -> str:
    nhs = _safe_nhs(nhs_number)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        encounter_id = get_latest_encounter(db_path, nhs) or start_encounter(db_path, nhs)
        row = dict(row)
        row["patient_id"] = nhs
        row["encounter_id"] = encounter_id
        columns = [
            "encounter_id",
            "patient_id",
            "timestamp",
            "MAP",
            "HR",
            "RR",
            "SpO2",
            "FiO2",
            "noradrenaline_mcgkgmin",
            "adrenaline_mcgkgmin",
            "dobutamine_mcgkgmin",
            "milrinone_mcgkgmin",
            "urine_output_ml_30min",
            "chest_drain_ml_30min",
            "lactate",
            "haemoglobin_gL",
            "creatinine_umolL",
            "WCC_10e9L",
            "temperature_C",
            "RASS",
            "oxygen_device",
            "arterial_line_present",
            "central_line_present",
            "insulin_infusion",
            "pacing_active",
            "imaging_summary",
        ]
        values = [row.get(c) for c in columns]
        placeholders = ",".join(["?"] * len(columns))
        cur.execute(
            f"INSERT INTO raw_icu_data ({','.join(columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return encounter_id
    finally:
        conn.close()


def load_rows(db_path: str, nhs_number: str) -> list[Dict[str, Any]]:
    nhs = _safe_nhs(nhs_number)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        encounter_id = get_latest_encounter(db_path, nhs)
        if not encounter_id:
            return []
        cur.execute(
            "SELECT * FROM raw_icu_data WHERE encounter_id = ? ORDER BY timestamp",
            (encounter_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

