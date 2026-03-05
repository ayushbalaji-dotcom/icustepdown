import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional


KEY_FILE_NAME = ".icu_pseudo_key"


def _safe_nhs(nhs_number: str) -> str:
    digits = "".join(ch for ch in str(nhs_number) if ch.isdigit())
    if len(digits) < 6:
        raise ValueError("Invalid NHS number")
    return digits


def _default_key_dir(db_path: str | None) -> str:
    if db_path:
        abs_path = os.path.abspath(db_path)
        parts = abs_path.split(os.sep)
        if "database" in parts:
            idx = len(parts) - 1 - parts[::-1].index("database")
            return os.sep.join(parts[: idx + 1])
    cwd_db = os.path.join(os.getcwd(), "database")
    if os.path.isdir(cwd_db):
        return cwd_db
    if db_path:
        return os.path.dirname(os.path.abspath(db_path)) or "."
    return os.getcwd()


def _read_key_file(path: str) -> bytes:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().encode("utf-8")


def _secure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o700)
    except Exception:
        pass


def _secure_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.chmod(path, 0o600)
    except Exception:
        pass


def _load_pseudo_key(db_path: str | None) -> bytes:
    env_key = os.environ.get("ICU_PSEUDO_KEY")
    if env_key:
        return env_key.encode("utf-8")
    env_path = os.environ.get("ICU_PSEUDO_KEY_PATH")
    if env_path:
        if not os.path.exists(env_path):
            raise ValueError("ICU_PSEUDO_KEY_PATH does not exist")
        return _read_key_file(env_path)

    key_dir = _default_key_dir(db_path)
    key_path = os.path.join(key_dir, KEY_FILE_NAME)
    if os.path.exists(key_path):
        return _read_key_file(key_path)

    _secure_dir(key_dir)
    key = secrets.token_hex(32)
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key)
    _secure_file(key_path)
    return key.encode("utf-8")


def pseudonymize_nhs(nhs_number: str, db_path: str | None = None) -> str:
    nhs = _safe_nhs(nhs_number)
    key = _load_pseudo_key(db_path)
    return hmac.new(key, nhs.encode("utf-8"), hashlib.sha256).hexdigest()


def _encounter_id_for(patient_key: str, started_at: str | None = None) -> str:
    suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if started_at:
        try:
            suffix = datetime.fromisoformat(started_at).strftime("%Y%m%d%H%M%S")
        except Exception:
            pass
    return f"{patient_key}-{suffix}"


def _migrate_legacy_ids(conn: sqlite3.Connection, key: bytes) -> None:
    cur = conn.cursor()
    cur.execute("SELECT id, nhs_number FROM patients")
    rows = cur.fetchall()
    if not rows:
        return
    for patient_id, nhs_value in rows:
        if not nhs_value or not str(nhs_value).isdigit():
            continue
        old_nhs = str(nhs_value)
        new_hash = hmac.new(key, old_nhs.encode("utf-8"), hashlib.sha256).hexdigest()
        if new_hash == old_nhs:
            continue
        cur.execute("UPDATE patients SET nhs_number = ? WHERE id = ?", (new_hash, patient_id))
        cur.execute("UPDATE raw_icu_data SET patient_id = ? WHERE patient_id = ?", (new_hash, old_nhs))
        cur.execute("UPDATE preop_data SET patient_id = ? WHERE patient_id = ?", (new_hash, old_nhs))
        cur.execute(
            "SELECT id, encounter_id, started_at FROM encounters WHERE patient_id = ?",
            (patient_id,),
        )
        for enc_id, enc_value, started_at in cur.fetchall():
            if enc_value and str(enc_value).startswith(f"{old_nhs}-"):
                new_enc = f"{new_hash}-{str(enc_value)[len(old_nhs) + 1:]}"
            else:
                new_enc = _encounter_id_for(new_hash, started_at)
            if new_enc != enc_value:
                cur.execute("UPDATE encounters SET encounter_id = ? WHERE id = ?", (new_enc, enc_id))
                cur.execute(
                    "UPDATE raw_icu_data SET encounter_id = ? WHERE encounter_id = ?",
                    (new_enc, enc_value),
                )
                cur.execute(
                    "UPDATE preop_data SET encounter_id = ? WHERE encounter_id = ?",
                    (new_enc, enc_value),
                )
    conn.commit()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(db_path: str) -> None:
    key = _load_pseudo_key(db_path)
    _secure_dir(os.path.dirname(os.path.abspath(db_path)) or ".")
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
                rhythm TEXT,
                imaging_summary TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS preop_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encounter_id TEXT UNIQUE NOT NULL,
                patient_id TEXT NOT NULL,
                age_years REAL,
                bmi REAL,
                frailty_score REAL,
                renal_function REAL,
                lv_function REAL,
                diabetes REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(cur, "raw_icu_data", "rhythm", "TEXT")
        conn.commit()
        _migrate_legacy_ids(conn, key)
    finally:
        conn.close()
    _secure_file(db_path)


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
    patient_key = pseudonymize_nhs(nhs, db_path)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        patient_id = _get_patient_id(conn, patient_key)
        if not force_new:
            cur.execute(
                "SELECT encounter_id FROM encounters WHERE patient_id = ? ORDER BY started_at DESC LIMIT 1",
                (patient_id,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
        encounter_id = _encounter_id_for(patient_key)
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
    patient_key = pseudonymize_nhs(nhs, db_path)
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
            (patient_key,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def append_row(db_path: str, nhs_number: str, row: Dict[str, Any]) -> str:
    nhs = _safe_nhs(nhs_number)
    patient_key = pseudonymize_nhs(nhs, db_path)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        encounter_id = get_latest_encounter(db_path, nhs) or start_encounter(db_path, nhs)
        row = dict(row)
        row["patient_id"] = patient_key
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
            "rhythm",
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


def save_preop(db_path: str, nhs_number: str, preop: Dict[str, Any]) -> str:
    nhs = _safe_nhs(nhs_number)
    patient_key = pseudonymize_nhs(nhs, db_path)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        encounter_id = get_latest_encounter(db_path, nhs) or start_encounter(db_path, nhs)
        now = datetime.utcnow().isoformat()
        row = dict(preop)
        row["patient_id"] = patient_key
        row["encounter_id"] = encounter_id
        row["created_at"] = now
        row["updated_at"] = now
        columns = [
            "encounter_id",
            "patient_id",
            "age_years",
            "bmi",
            "frailty_score",
            "renal_function",
            "lv_function",
            "diabetes",
            "created_at",
            "updated_at",
        ]
        values = [row.get(c) for c in columns]
        placeholders = ",".join(["?"] * len(columns))
        cur.execute(
            f"""
            INSERT INTO preop_data ({','.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(encounter_id) DO UPDATE SET
                age_years=excluded.age_years,
                bmi=excluded.bmi,
                frailty_score=excluded.frailty_score,
                renal_function=excluded.renal_function,
                lv_function=excluded.lv_function,
                diabetes=excluded.diabetes,
                updated_at=excluded.updated_at
            """,
            values,
        )
        conn.commit()
        return encounter_id
    finally:
        conn.close()


def load_preop(db_path: str, nhs_number: str) -> Optional[Dict[str, Any]]:
    nhs = _safe_nhs(nhs_number)
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        encounter_id = get_latest_encounter(db_path, nhs)
        if not encounter_id:
            return None
        cur.execute(
            """
            SELECT age_years, bmi, frailty_score, renal_function, lv_function, diabetes
            FROM preop_data
            WHERE encounter_id = ?
            """,
            (encounter_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        keys = ["age_years", "bmi", "frailty_score", "renal_function", "lv_function", "diabetes"]
        return dict(zip(keys, row))
    finally:
        conn.close()
