import os
from typing import Any, List
from datetime import datetime

import pandas as pd

try:
    pd.options.mode.string_storage = "python"
except Exception:
    pass
import streamlit as st

from icu_stepdown.auth import resolve_expected_credentials, validate_credentials
from icu_stepdown.config import load_config
from icu_stepdown.features import compute_features, compute_features_latest
from icu_stepdown.ops_logic import (
    bed_priority_score,
    compute_destination_recommendation,
    compute_operational_blockers,
    compute_transfer_feasibility,
    forecast_bed_pressure,
    latest_clinical_snapshot,
)
from icu_stepdown.ops_store import (
    DEFAULT_PROCEDURE_GROUPS,
    adjust_bed_inventory,
    get_latest_bed_inventory,
    get_latest_capability,
    get_latest_capacity,
    get_latest_rules,
    get_latest_staffing,
    list_audit_log,
    list_patient_operational_status,
    list_procedure_los,
    list_theatre_schedule,
    save_bed_inventory,
    save_capacity,
    save_capability,
    save_procedure_los,
    save_staffing,
    save_theatre_schedule,
    save_transfer_rules,
    seed_ops_data,
    upsert_patient_operational_status,
)
from icu_stepdown.patient_store import append_row, load_preop, load_rows, save_preop, start_encounter, pseudonymize_nhs
from icu_stepdown.preprocess import preprocess
from icu_stepdown.quality import QualityLogger
from icu_stepdown.schema import validate_raw
from icu_stepdown.score import score_features, _fail_closed_dashboard, score_hard_stops_only


def _parse_timestamp(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        raise ValueError("invalid_timestamp_format")
    return ts.isoformat()


def _parse_optional_float(label: str, value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid number for {label}.") from exc


def _parse_yes_no(value: str | None) -> float | None:
    if value == "Yes":
        return 1.0
    if value == "No":
        return 0.0
    return None


def _safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    safe = df.copy()
    # Streamlit 1.19 + pyarrow can choke on LargeUtf8. Force Python strings for display.
    try:
        return safe.astype(str)
    except Exception:
        for col in safe.columns:
            try:
                safe[col] = safe[col].astype(str)
            except Exception:
                safe[col] = safe[col].map(lambda value: str(value))
        return safe


def _safe_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]) or pd.api.types.is_string_dtype(safe[col]):
            try:
                safe[col] = safe[col].astype(str)
            except Exception:
                safe[col] = safe[col].map(lambda value: str(value))
    return safe


def _pill(label: str, kind: str) -> None:
    st.markdown(f"<span class='pill {kind}'>{label}</span>", unsafe_allow_html=True)


def _free_beds(total: Any, occupied: Any) -> str:
    if total is None or occupied is None:
        return "--"
    try:
        return str(max(int(total) - int(occupied), 0))
    except Exception:
        return "--"


def _render_bed_board(read_only: bool = False) -> None:
    latest_capacity = get_latest_capacity(OPS_DB_PATH) or {}
    bed_inventory = get_latest_bed_inventory(OPS_DB_PATH) or {}
    patients = list_patient_operational_status(OPS_DB_PATH)
    forecast = forecast_bed_pressure(OPS_DB_PATH, latest_capacity, bed_inventory)

    st.markdown("**Capacity snapshot**")
    cap_cols = st.columns(4)
    cap_cols[0].metric("ICU beds", latest_capacity.get("icu_beds", "--"), f"Free {_free_beds(latest_capacity.get('icu_beds'), bed_inventory.get('icu_occupied'))}")
    cap_cols[1].metric("HDU beds", latest_capacity.get("hdu_beds", "--"), f"Free {_free_beds(latest_capacity.get('hdu_beds'), bed_inventory.get('hdu_occupied'))}")
    cap_cols[2].metric("Ward beds", latest_capacity.get("ward_beds", "--"), f"Free {_free_beds(latest_capacity.get('ward_beds'), bed_inventory.get('ward_occupied'))}")
    cap_cols[3].metric("Telemetry beds", latest_capacity.get("telemetry_beds", "--"), f"Free {_free_beds(latest_capacity.get('telemetry_beds'), bed_inventory.get('telemetry_occupied'))}")

    if not read_only:
        with st.form("bed_inventory_form"):
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                icu_occ = st.number_input("ICU occupied", value=int(bed_inventory.get("icu_occupied") or 0), min_value=0, step=1)
            with col_b2:
                hdu_occ = st.number_input("HDU occupied", value=int(bed_inventory.get("hdu_occupied") or 0), min_value=0, step=1)
            with col_b3:
                ward_occ = st.number_input("Ward occupied", value=int(bed_inventory.get("ward_occupied") or 0), min_value=0, step=1)
            with col_b4:
                telemetry_occ = st.number_input("Telemetry occupied", value=int(bed_inventory.get("telemetry_occupied") or 0), min_value=0, step=1)
            submitted = st.form_submit_button("Update bed occupancy")
            if submitted:
                save_bed_inventory(
                    OPS_DB_PATH,
                    {
                        "icu_occupied": icu_occ,
                        "hdu_occupied": hdu_occ,
                        "ward_occupied": ward_occ,
                        "telemetry_occupied": telemetry_occ,
                    },
                    st.session_state.get("current_user"),
                )
                st.success("Bed occupancy updated.")

        st.markdown("**Discharge bed**")
        with st.form("discharge_bed_form"):
            col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
            with col_d1:
                discharge_area = st.selectbox("Area", ["ICU", "HDU", "Ward", "Telemetry"])
            with col_d2:
                bed_label = st.text_input("Bed label (e.g. ICU-3)", value="")
            with col_d3:
                submit_discharge = st.form_submit_button("Discharge bed")
            if submit_discharge:
                try:
                    note = f"Discharged bed {bed_label or 'unknown'} in {discharge_area}"
                    adjust_bed_inventory(
                        OPS_DB_PATH,
                        discharge_area,
                        delta=-1,
                        note=note,
                        user=st.session_state.get("current_user"),
                    )
                    st.success(note)
                except Exception as e:
                    st.error(str(e))

    st.markdown("**Operational watchlists**")
    green_blocked = [p for p in patients if p.get("readiness_status") == "GREEN" and p.get("transfer_feasibility") == "No"]
    likely_stepdown = [p for p in patients if p.get("readiness_status") in {"GREEN", "AMBER"}]
    beds_needed = forecast.get("incoming_pressure_24h", 0)
    st.write(f"Beds needed for tomorrow's list (24h): {beds_needed}")

    with st.expander("Clinically green but operationally blocked"):
        if not green_blocked:
            st.write("None.")
        else:
            for item in green_blocked:
                st.write(f"- {item.get('patient_id')} (blockers: {', '.join(item.get('operational_blockers') or [])})")

    with st.expander("Likely step-down in next 24h"):
        if not likely_stepdown:
            st.write("None.")
        else:
            for item in likely_stepdown:
                st.write(f"- {item.get('patient_id')} ({item.get('readiness_status')})")

    st.markdown("**Potential bottlenecks**")
    if forecast.get("icu_pressure_flag"):
        _pill("ICU pressure expected", "blocked")
    else:
        _pill("No ICU pressure flag", "ready")


def _render_ward_view() -> None:
    st.header("Ward and ICU View")
    st.caption("Operational visibility only. Readiness remains clinically driven.")
    _render_bed_board(read_only=True)

    patients = list_patient_operational_status(OPS_DB_PATH)
    icu_candidates = [p for p in patients if p.get("readiness_status") in {"GREEN", "AMBER"}]
    ward_candidates = [p for p in patients if p.get("destination_recommendation") == "Ward"]
    blocked = [p for p in patients if p.get("transfer_feasibility") == "No"]

    col_i, col_w = st.columns(2)
    with col_i:
        st.subheader("ICU view")
        if not icu_candidates:
            st.write("No step-down candidates yet.")
        else:
            for item in icu_candidates:
                st.write(f"- {item.get('patient_id')} ({item.get('readiness_status')}, {item.get('transfer_feasibility')})")
    with col_w:
        st.subheader("Ward view")
        if not ward_candidates:
            st.write("No ward-destination candidates yet.")
        else:
            for item in ward_candidates:
                st.write(f"- {item.get('patient_id')} ({item.get('transfer_feasibility')})")

    with st.expander("Operational bottlenecks"):
        if not blocked:
            st.write("No active blockers.")
        else:
            for item in blocked:
                st.write(f"- {item.get('patient_id')}: {', '.join(item.get('operational_blockers') or [])}")


def _legacy_db_dir_candidates(nhs_number: str) -> list[str]:
    raw = str(nhs_number)
    digits = "".join(ch for ch in raw if ch.isdigit())
    candidates = [os.path.join("database", raw)]
    if digits and digits != raw:
        candidates.append(os.path.join("database", digits))
    return candidates


def _secret_get(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _data_editor(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    if hasattr(st, "data_editor"):
        return st.data_editor(df, **kwargs)
    if hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor(df, **kwargs)
    st.warning("Editable tables are not supported in this Streamlit version.")
    return df


def _require_login() -> bool:
    st.sidebar.header("Login")
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    submit = st.sidebar.button("Sign in")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.using_demo_creds = False
    if submit:
        valid, using_demo = validate_credentials(user, password, secret_get=_secret_get)
        if valid:
            st.session_state.authenticated = True
            st.session_state.current_user = user
            st.session_state.using_demo_creds = using_demo
        else:
            st.sidebar.error("Invalid credentials.")
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.using_demo_creds = False
    if st.session_state.get("authenticated"):
        _, _, using_demo = resolve_expected_credentials(secret_get=_secret_get)
        if using_demo:
            st.sidebar.warning("Demo credentials in use. Set ICU_APP_USER/ICU_APP_PASS for production.")
    return st.session_state.authenticated


def _score_from_db(db_path: str, cfg, model_path: str | None, hard_stops_only: bool):
    ql = QualityLogger()
    rows = load_rows(db_path, st.session_state.nhs_number)
    if not rows:
        return {"status": "no_data"}
    raw = pd.DataFrame(rows)
    raw, _ = validate_raw(raw, cfg, ql)
    raw = preprocess(raw, cfg, ql)
    t0 = pd.to_datetime(raw["timestamp"]).min()
    t1 = pd.to_datetime(raw["timestamp"]).max()
    hours_span = 0 if pd.isna(t0) or pd.isna(t1) else (t1 - t0).total_seconds() / 3600.0
    if not hard_stops_only:
        if pd.isna(t0) or pd.isna(t1) or hours_span < 4:
            return {
                "status": "insufficient_data",
                "hours": hours_span,
                "min_timestamp": None if pd.isna(t0) else str(t0),
                "max_timestamp": None if pd.isna(t1) else str(t1),
                "row_count": int(len(raw)),
            }
        features = compute_features(raw, cfg, ql)
    else:
        features = compute_features_latest(raw, cfg, ql)
    if features.empty:
        return {"status": "insufficient_data"}
    model_bundle = None
    if hard_stops_only:
        scores, _, dashboard, _ = score_hard_stops_only(features, cfg, ql)
        warning = "hard_stops_only"
    else:
        if model_path and os.path.exists(model_path):
            try:
                from icu_stepdown.train import load_model_bundle

                model_bundle = load_model_bundle(model_path)
            except Exception:
                model_bundle = None
        if model_bundle is None:
            scores, _, dashboard, _ = _fail_closed_dashboard(features, ql)
            warning = "no_model_loaded_fail_closed"
        else:
            scores, _, dashboard, _ = score_features(features, model_bundle, cfg, ql, force_schema=True)
            warning = None
    latest = dashboard.iloc[0].to_dict() if not dashboard.empty else {}
    return {"status": "ok", "dashboard": latest, "warning": warning}


st.set_page_config(page_title="ICU to HDU Step-Down", layout="wide")

if not _require_login():
    st.stop()

OPS_DB_PATH = os.path.join("database", "icu_ops.sqlite")
seed_ops_data(OPS_DB_PATH, st.session_state.get("current_user"))

st.markdown(
    """
    <style>
    .pill {display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;}
    .pill.ready {background:#e6f4ea;color:#1e7a46;}
    .pill.blocked {background:#fde8e8;color:#b42318;}
    .pill.pending {background:#fff3cd;color:#8a6d1d;}
    .pill.info {background:#e8f1ff;color:#2457a6;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ICU → HDU Step-Down Readiness")
st.caption("Decision support only. Conservative, fail-closed.")

cfg = load_config("configs/default.yaml")
model_path = st.sidebar.text_input("Model path", value="")
hard_stops_only = st.sidebar.checkbox("Hard-stops only (no ML)", value=not bool(model_path))
nav = st.sidebar.radio(
    "Navigation",
    ["Operational management", "Ward view", "Clinical readiness assessment"],
    index=2,
)

if nav == "Ward view":
    _render_ward_view()
    st.stop()

if nav == "Clinical readiness assessment":
    st.subheader("Patient")
    nhs_number = st.text_input("NHS number", value=st.session_state.get("nhs_number", ""))
    col1, col2 = st.columns(2)
    if col1.button("Start patient"):
        if not nhs_number:
            st.error("Enter NHS number")
        else:
            st.session_state.nhs_number = nhs_number
            pseudo_id = pseudonymize_nhs(nhs_number, os.path.join("database", "icu_stepdown.sqlite"))
            st.session_state.pseudo_id = pseudo_id
            db_dir = os.path.join("database", pseudo_id)
            for legacy_dir in _legacy_db_dir_candidates(nhs_number):
                if legacy_dir != db_dir and os.path.isdir(legacy_dir) and not os.path.isdir(db_dir):
                    try:
                        os.rename(legacy_dir, db_dir)
                    except Exception:
                        pass
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "icu_stepdown.sqlite")
            st.session_state.db_path = db_path
            enc = start_encounter(db_path, nhs_number, force_new=False)
            st.success(f"Using encounter {enc}")
            st.session_state.last_score = _score_from_db(db_path, cfg, model_path or None, hard_stops_only)
    if col2.button("Start new encounter"):
        if not nhs_number:
            st.error("Enter NHS number")
        else:
            st.session_state.nhs_number = nhs_number
            pseudo_id = pseudonymize_nhs(nhs_number, os.path.join("database", "icu_stepdown.sqlite"))
            st.session_state.pseudo_id = pseudo_id
            db_dir = os.path.join("database", pseudo_id)
            for legacy_dir in _legacy_db_dir_candidates(nhs_number):
                if legacy_dir != db_dir and os.path.isdir(legacy_dir) and not os.path.isdir(db_dir):
                    try:
                        os.rename(legacy_dir, db_dir)
                    except Exception:
                        pass
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "icu_stepdown.sqlite")
            st.session_state.db_path = db_path
            enc = start_encounter(db_path, nhs_number, force_new=True)
            st.success(f"New encounter {enc} started")
            st.session_state.last_score = _score_from_db(db_path, cfg, model_path or None, hard_stops_only)

    if "db_path" not in st.session_state:
        st.info("Start a patient to begin data entry.")
        st.stop()

    if st.session_state.get("pseudo_id"):
        st.caption(f"Pseudo ID: {st.session_state.pseudo_id}")

    st.subheader("Pre-op characteristics")
    preop_existing = load_preop(st.session_state.db_path, st.session_state.nhs_number)
    preop_age_default = "" if not preop_existing else preop_existing.get("age_years") or ""
    preop_bmi_default = "" if not preop_existing else preop_existing.get("bmi") or ""
    preop_frailty_default = "" if not preop_existing else preop_existing.get("frailty_score") or ""
    preop_renal_default = "" if not preop_existing else preop_existing.get("renal_function") or ""
    preop_lv_default = "" if not preop_existing else preop_existing.get("lv_function") or ""
    preop_diabetes_default = "" if not preop_existing else ("Yes" if preop_existing.get("diabetes") == 1 else "No" if preop_existing.get("diabetes") == 0 else "")

    with st.form("preop_entry"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            preop_age = st.text_input("Age (years)", value=preop_age_default)
            preop_bmi = st.text_input("BMI (kg/m2)", value=preop_bmi_default)
        with col_p2:
            preop_frailty = st.text_input("Frailty score (1-9)", value=preop_frailty_default)
            preop_renal = st.text_input("Renal function (eGFR ml/min)", value=preop_renal_default)
        with col_p3:
            preop_lv = st.text_input("LV function (LVEF %)", value=preop_lv_default)
            diabetes_index = 0
            if preop_diabetes_default == "Yes":
                diabetes_index = 1
            elif preop_diabetes_default == "No":
                diabetes_index = 2
            preop_diabetes = st.selectbox("Diabetes", ["", "Yes", "No"], index=diabetes_index)
        preop_submitted = st.form_submit_button("Save pre-op data")
        if preop_submitted:
            try:
                preop_row = {
                    "age_years": _parse_optional_float("Age (years)", preop_age),
                    "bmi": _parse_optional_float("BMI (kg/m2)", preop_bmi),
                    "frailty_score": _parse_optional_float("Frailty score (1-9)", preop_frailty),
                    "renal_function": _parse_optional_float("Renal function (eGFR ml/min)", preop_renal),
                    "lv_function": _parse_optional_float("LV function (LVEF %)", preop_lv),
                    "diabetes": _parse_yes_no(preop_diabetes),
                }
                save_preop(st.session_state.db_path, st.session_state.nhs_number, preop_row)
                st.success("Saved pre-op data.")
            except Exception as e:
                st.error(str(e))

    preop_ready = bool(preop_existing)
    if not preop_ready:
        st.info("Enter pre-op characteristics to enable hourly data entry.")

    st.subheader("Hourly data entry")
    with st.form("data_entry"):
        ts_col1, ts_col2 = st.columns(2)
        with ts_col1:
            ts_date = st.date_input("Date")
        with ts_col2:
            ts_time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            MAP = st.text_input("MAP (mmHg)", placeholder="e.g. 75")
            HR = st.text_input("HR (bpm)", placeholder="e.g. 90")
            RR = st.text_input("RR (/min)", placeholder="e.g. 18")
            SpO2 = st.text_input("SpO2 (%)", placeholder="e.g. 98")
            FiO2 = st.text_input("FiO2 (fraction)", placeholder="e.g. 0.35")
            oxygen_device = st.selectbox("Oxygen device", ["", "RA", "NC", "Mask", "Venturi", "HFNC", "NIV", "ETT"])
        with col_b:
            noradrenaline_mcgkgmin = st.text_input("Noradrenaline (mcg/kg/min)", placeholder="e.g. 0.02")
            adrenaline_mcgkgmin = st.text_input("Adrenaline (mcg/kg/min)", placeholder="e.g. 0.00")
            dobutamine_mcgkgmin = st.text_input("Dobutamine (mcg/kg/min)", placeholder="e.g. 2.5")
            milrinone_mcgkgmin = st.text_input("Milrinone (mcg/kg/min)", placeholder="e.g. 0.2")
            urine_output_ml_30min = st.text_input("Urine output (ml/30 min)", placeholder="e.g. 40")
            chest_drain_ml_30min = st.text_input("Chest drain (ml/30 min)", placeholder="e.g. 20")
        with col_c:
            lactate = st.text_input("Lactate", placeholder="e.g. 1.2")
            haemoglobin_gL = st.text_input("Haemoglobin (g/L)", placeholder="e.g. 105")
            creatinine_umolL = st.text_input("Creatinine (umol/L)", placeholder="e.g. 85")
            WCC_10e9L = st.text_input("WCC (10e9/L)", placeholder="e.g. 9.2")
            temperature_C = st.text_input("Temperature (C)", placeholder="e.g. 36.8")
            RASS = st.text_input("RASS", placeholder="e.g. -1")
        arterial_line_present = st.selectbox("Arterial line present", ["", "Yes", "No"])
        central_line_present = st.selectbox("Central line present", ["", "Yes", "No"])
        insulin_infusion = st.selectbox("Insulin infusion", ["", "Yes", "No"])
        rhythm = st.selectbox("Rhythm", ["", "Sinus", "AF", "Pacing required"])
        imaging_summary = st.text_area("Imaging summary (optional)")
        submitted = st.form_submit_button("Add hourly data")

        if submitted:
            if not preop_ready:
                st.error("Pre-op characteristics are required before saving hourly data.")
                st.stop()
            try:
                parsed_ts = _parse_timestamp(datetime.combine(ts_date, ts_time))
                row = {
                    "timestamp": parsed_ts,
                    "MAP": _parse_optional_float("MAP (mmHg)", MAP),
                    "HR": _parse_optional_float("HR (bpm)", HR),
                    "RR": _parse_optional_float("RR (/min)", RR),
                    "SpO2": _parse_optional_float("SpO2 (%)", SpO2),
                    "FiO2": _parse_optional_float("FiO2 (fraction)", FiO2),
                    "noradrenaline_mcgkgmin": _parse_optional_float("Noradrenaline (mcg/kg/min)", noradrenaline_mcgkgmin),
                    "adrenaline_mcgkgmin": _parse_optional_float("Adrenaline (mcg/kg/min)", adrenaline_mcgkgmin),
                    "dobutamine_mcgkgmin": _parse_optional_float("Dobutamine (mcg/kg/min)", dobutamine_mcgkgmin),
                    "milrinone_mcgkgmin": _parse_optional_float("Milrinone (mcg/kg/min)", milrinone_mcgkgmin),
                    "urine_output_ml_30min": _parse_optional_float("Urine output (ml/30 min)", urine_output_ml_30min),
                    "chest_drain_ml_30min": _parse_optional_float("Chest drain (ml/30 min)", chest_drain_ml_30min),
                    "lactate": _parse_optional_float("Lactate", lactate),
                    "haemoglobin_gL": _parse_optional_float("Haemoglobin (g/L)", haemoglobin_gL),
                    "creatinine_umolL": _parse_optional_float("Creatinine (umol/L)", creatinine_umolL),
                    "WCC_10e9L": _parse_optional_float("WCC (10e9/L)", WCC_10e9L),
                    "temperature_C": _parse_optional_float("Temperature (C)", temperature_C),
                    "RASS": _parse_optional_float("RASS", RASS),
                    "oxygen_device": oxygen_device or None,
                    "arterial_line_present": _parse_yes_no(arterial_line_present),
                    "central_line_present": _parse_yes_no(central_line_present),
                    "insulin_infusion": _parse_yes_no(insulin_infusion),
                    "pacing_active": 1.0 if rhythm == "Pacing required" else 0.0 if rhythm else None,
                    "rhythm": rhythm or None,
                    "imaging_summary": imaging_summary or None,
                }
                append_row(st.session_state.db_path, st.session_state.nhs_number, row)
                st.success("Saved hourly data.")
            except Exception as e:
                st.error(str(e))

    st.subheader("Latest readiness score")
    if st.button("Refresh score"):
        result = _score_from_db(st.session_state.db_path, cfg, model_path or None, hard_stops_only)
        st.session_state.last_score = result

    result = st.session_state.get("last_score")
    if result:
        if result["status"] == "insufficient_data":
            st.warning(
                f"Need 4h of data. Rows: {result.get('row_count', 0)}. "
                f"Span: {result.get('hours', 0):.2f}h "
                f"(min {result.get('min_timestamp')}, max {result.get('max_timestamp')})."
            )
        elif result["status"] == "no_data":
            st.info("No data yet.")
        elif result["status"] == "ok":
            dash = result.get("dashboard", {})
            if result.get("warning") == "hard_stops_only":
                st.warning("Hard-stops-only active: ML disabled.")
            elif result.get("warning") == "no_model_loaded_fail_closed":
                st.error("No model loaded: fail-closed RED. Provide a model to enable scoring.")
            traffic = dash.get("traffic_light")
            if traffic == "GREEN":
                st.success("GREEN: Ready for step-down (subject to clinician review).")
            elif traffic == "AMBER":
                st.warning("AMBER: Borderline. Review missing or concerning data.")
            elif traffic == "RED":
                st.error("RED: Not ready for step-down.")
            else:
                st.info("No score yet.")

            st.write(f"Recommendation: {dash.get('suggested_action', 'No recommendation available')}")

            reasons = dash.get("hard_stop_reason_summary") or ""
            signals = dash.get("signals") or ""
            if reasons:
                st.write(f"Hard stop reasons: {reasons}")
            if signals and signals != reasons:
                st.write(f"Signals: {signals}")
            if not dash.get("data_quality_ok_for_green", True):
                st.info("Data quality: missing or stale data prevents GREEN.")

    st.subheader("Operational summary")
    rows = load_rows(st.session_state.db_path, st.session_state.nhs_number)
    latest_row = latest_clinical_snapshot(rows)
    capacity = get_latest_capacity(OPS_DB_PATH) or {}
    bed_inventory = get_latest_bed_inventory(OPS_DB_PATH) or {}
    staffing = get_latest_staffing(OPS_DB_PATH) or {}
    capability = get_latest_capability(OPS_DB_PATH) or {}
    rules = get_latest_rules(OPS_DB_PATH) or {}
    procedure_ref = list_procedure_los(OPS_DB_PATH)
    procedure_groups = [row.get("procedure_group") for row in procedure_ref if row.get("procedure_group")] or DEFAULT_PROCEDURE_GROUPS

    current_ops = None
    encounter_id = None
    patient_id = st.session_state.get("pseudo_id")
    if latest_row:
        encounter_id = latest_row.get("encounter_id")
        for item in list_patient_operational_status(OPS_DB_PATH):
            if item.get("encounter_id") == encounter_id:
                current_ops = item
                break

    selected_group = st.selectbox(
        "Procedure group (for LOS context)",
        options=[""] + procedure_groups,
        index=([""] + procedure_groups).index(current_ops.get("procedure_group")) if current_ops and current_ops.get("procedure_group") in procedure_groups else 0,
    )
    if st.button("Save procedure context"):
        if encounter_id and patient_id:
            upsert_patient_operational_status(
                OPS_DB_PATH,
                {
                    "encounter_id": encounter_id,
                    "patient_id": patient_id,
                    "procedure_group": selected_group or None,
                    "readiness_status": current_ops.get("readiness_status") if current_ops else None,
                    "readiness_score": current_ops.get("readiness_score") if current_ops else None,
                    "destination_recommendation": current_ops.get("destination_recommendation") if current_ops else None,
                    "transfer_feasibility": current_ops.get("transfer_feasibility") if current_ops else None,
                    "operational_blockers": current_ops.get("operational_blockers") if current_ops else [],
                    "bed_priority_score": current_ops.get("bed_priority_score") if current_ops else None,
                },
                st.session_state.get("current_user"),
            )
            st.success("Saved procedure context.")

    readiness_status = "WAIT"
    readiness_score = None
    if result:
        if result.get("status") == "ok":
            dash = result.get("dashboard", {})
            readiness_status = dash.get("traffic_light") or "RED"
            readiness_score = dash.get("IRI")
        elif result.get("status") == "insufficient_data":
            readiness_status = "WAIT"
        elif result.get("status") == "no_data":
            readiness_status = "WAIT"

    destination = "Unknown"
    blockers: List[str] = []
    transfer_feasibility = "Uncertain"
    priority_score = None
    if latest_row:
        destination = compute_destination_recommendation(latest_row, rules, selected_group or None)
        blockers = compute_operational_blockers(latest_row, destination, capacity, bed_inventory, staffing, capability, rules)
        transfer_feasibility = compute_transfer_feasibility(readiness_status, blockers)
        hours_since_ready = None
        if current_ops and current_ops.get("first_ready_at"):
            try:
                first_ready = datetime.fromisoformat(current_ops.get("first_ready_at"))
                hours_since_ready = (datetime.utcnow() - first_ready).total_seconds() / 3600.0
            except Exception:
                hours_since_ready = None
        priority_score = bed_priority_score(
            readiness_status,
            readiness_score,
            hours_since_ready,
            transfer_feasibility,
            selected_group or None,
            OPS_DB_PATH,
        )
        if encounter_id and patient_id:
            upsert_patient_operational_status(
                OPS_DB_PATH,
                {
                    "encounter_id": encounter_id,
                    "patient_id": patient_id,
                    "procedure_group": selected_group or None,
                    "readiness_status": readiness_status,
                    "readiness_score": readiness_score,
                    "destination_recommendation": destination,
                    "transfer_feasibility": transfer_feasibility,
                    "operational_blockers": blockers,
                    "bed_priority_score": priority_score,
                },
                st.session_state.get("current_user"),
            )

    col_op1, col_op2, col_op3 = st.columns(3)
    with col_op1:
        st.markdown("**Readiness status**")
        if readiness_status == "GREEN":
            _pill("Ready", "ready")
        elif readiness_status == "AMBER":
            _pill("Borderline", "pending")
        elif readiness_status == "RED":
            _pill("Not ready", "blocked")
        else:
            _pill("Awaiting data", "info")
        st.write(f"Readiness score: {readiness_score if readiness_score is not None else '--'}")
    with col_op2:
        st.markdown("**Destination**")
        st.write(destination)
        st.markdown("**Transfer feasibility**")
        if transfer_feasibility == "Yes":
            _pill("Feasible", "ready")
        elif transfer_feasibility == "No":
            _pill("Blocked", "blocked")
        else:
            _pill("Uncertain", "pending")
    with col_op3:
        st.markdown("**Bed priority**")
        st.write(f"{priority_score:.1f}" if priority_score is not None else "--")

    if blockers:
        with st.expander("Operational blockers"):
            for item in blockers:
                st.write(f"- {item}")
    else:
        st.caption("No operational blockers recorded.")

    st.subheader("Raw data (latest encounter)")
    if rows:
        st.dataframe(_safe_display_df(pd.DataFrame(rows)))

if nav == "Operational management":
    st.header("Operational management")
    st.caption("Operational controls influence transfer feasibility, not physiological readiness.")
    admin_tabs = st.tabs([
        "Capacity Setup",
        "Staffing and Capability",
        "Procedure LOS Setup",
        "Theatre and Incoming Demand",
        "Bed Board",
        "Rules and Local Pathways",
        "Audit and Overrides",
    ])

with admin_tabs[0]:
    latest_capacity = get_latest_capacity(OPS_DB_PATH) or {}
    with st.form("capacity_form"):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            icu_beds = st.number_input("ICU beds", value=int(latest_capacity.get("icu_beds") or 0), min_value=0, step=1)
            hdu_beds = st.number_input("HDU beds", value=int(latest_capacity.get("hdu_beds") or 0), min_value=0, step=1)
            ward_beds = st.number_input("Ward beds", value=int(latest_capacity.get("ward_beds") or 0), min_value=0, step=1)
        with col_c2:
            telemetry_beds = st.number_input("Telemetry-capable ward beds", value=int(latest_capacity.get("telemetry_beds") or 0), min_value=0, step=1)
            isolation_beds = st.number_input("Isolation beds", value=int(latest_capacity.get("isolation_beds") or 0), min_value=0, step=1)
            surge_beds = st.number_input("Surge beds", value=int(latest_capacity.get("surge_beds") or 0), min_value=0, step=1)
        with col_c3:
            shift_label = st.selectbox("Shift", ["Day", "Night"], index=0 if latest_capacity.get("shift_label") != "Night" else 1)
        submitted = st.form_submit_button("Save capacity")
        if submitted:
            save_capacity(
                OPS_DB_PATH,
                {
                    "icu_beds": icu_beds,
                    "hdu_beds": hdu_beds,
                    "ward_beds": ward_beds,
                    "telemetry_beds": telemetry_beds,
                    "isolation_beds": isolation_beds,
                    "surge_beds": surge_beds,
                    "shift_label": shift_label,
                },
                st.session_state.get("current_user"),
            )
            st.success("Capacity updated.")
    if latest_capacity:
        st.caption(f"Last updated: {latest_capacity.get('updated_at')}")

with admin_tabs[1]:
    latest_staff = get_latest_staffing(OPS_DB_PATH) or {}
    latest_cap = get_latest_capability(OPS_DB_PATH) or {}
    with st.form("staffing_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            icu_nurse_available = st.number_input("ICU nurse availability", value=int(latest_staff.get("icu_nurse_available") or 0), min_value=0, step=1)
            hdu_nurse_available = st.number_input("HDU nurse availability", value=int(latest_staff.get("hdu_nurse_available") or 0), min_value=0, step=1)
            ward_nurse_available = st.number_input("Ward nurse availability", value=int(latest_staff.get("ward_nurse_available") or 0), min_value=0, step=1)
        with col_s2:
            ratio_icu = st.number_input("ICU nurse-to-patient ratio", value=float(latest_staff.get("ratio_icu") or 0), min_value=0.0, step=0.1)
            ratio_hdu = st.number_input("HDU nurse-to-patient ratio", value=float(latest_staff.get("ratio_hdu") or 0), min_value=0.0, step=0.1)
            ratio_ward = st.number_input("Ward nurse-to-patient ratio", value=float(latest_staff.get("ratio_ward") or 0), min_value=0.0, step=0.1)
        with col_s3:
            cardiac_trained = st.number_input("Cardiac-trained nurses available", value=int(latest_staff.get("cardiac_trained_nurses") or 0), min_value=0, step=1)
            telemetry_available = st.checkbox("Telemetry monitoring capacity available", value=bool(latest_staff.get("telemetry_available") or 0))
            outreach_available = st.checkbox("Outreach team available", value=bool(latest_staff.get("outreach_available") or 0))
            registrar_cover = st.checkbox("Cardiac registrar/resident overnight cover", value=bool(latest_staff.get("registrar_cover_available") or 0))
            physio_available = st.checkbox("Physiotherapy available today", value=bool(latest_staff.get("physio_available") or 0))
            resp_available = st.checkbox("Respiratory therapy support", value=bool(latest_staff.get("respiratory_support_available") or 0))
            dialysis_available = st.checkbox("Dialysis support available", value=bool(latest_staff.get("dialysis_support_available") or 0))
        shift_label = st.selectbox("Shift label", ["Day", "Night"], index=0 if latest_staff.get("shift_label") != "Night" else 1)
        notes = st.text_area("Staffing notes", value=latest_staff.get("notes") or "")
        submitted = st.form_submit_button("Save staffing status")
        if submitted:
            save_staffing(
                OPS_DB_PATH,
                {
                    "icu_nurse_available": icu_nurse_available,
                    "hdu_nurse_available": hdu_nurse_available,
                    "ward_nurse_available": ward_nurse_available,
                    "ratio_icu": ratio_icu,
                    "ratio_hdu": ratio_hdu,
                    "ratio_ward": ratio_ward,
                    "cardiac_trained_nurses": cardiac_trained,
                    "telemetry_available": 1 if telemetry_available else 0,
                    "outreach_available": 1 if outreach_available else 0,
                    "registrar_cover_available": 1 if registrar_cover else 0,
                    "physio_available": 1 if physio_available else 0,
                    "respiratory_support_available": 1 if resp_available else 0,
                    "dialysis_support_available": 1 if dialysis_available else 0,
                    "notes": notes,
                    "shift_label": shift_label,
                },
                st.session_state.get("current_user"),
            )
            st.success("Staffing status updated.")

    st.markdown("**Ward capability**")
    with st.form("capability_form"):
        can_manage_pacing = st.checkbox("Ward can manage pacing wires", value=bool(latest_cap.get("can_manage_pacing_wires") or 0))
        can_manage_drains = st.checkbox("Ward can manage chest drains", value=bool(latest_cap.get("can_manage_chest_drains") or 0))
        can_manage_oxygen = st.checkbox("Ward can manage low-dose oxygen", value=bool(latest_cap.get("can_manage_low_oxygen") or 0))
        can_manage_insulin = st.checkbox("Ward can manage insulin infusion", value=bool(latest_cap.get("can_manage_insulin_infusion") or 0))
        telemetry_monitoring = st.checkbox("Telemetry monitoring available on ward", value=bool(latest_cap.get("telemetry_monitoring_available") or 0))
        notes_cap = st.text_area("Capability notes", value=latest_cap.get("notes") or "")
        submitted = st.form_submit_button("Save ward capability")
        if submitted:
            save_capability(
                OPS_DB_PATH,
                {
                    "can_manage_pacing_wires": 1 if can_manage_pacing else 0,
                    "can_manage_chest_drains": 1 if can_manage_drains else 0,
                    "can_manage_low_oxygen": 1 if can_manage_oxygen else 0,
                    "can_manage_insulin_infusion": 1 if can_manage_insulin else 0,
                    "telemetry_monitoring_available": 1 if telemetry_monitoring else 0,
                    "notes": notes_cap,
                },
                st.session_state.get("current_user"),
            )
            st.success("Ward capability updated.")

with admin_tabs[2]:
    st.caption("Average ICU/HDU length of stay per procedure group.")
    los_rows = list_procedure_los(OPS_DB_PATH)
    los_df = pd.DataFrame(los_rows)
    if not los_df.empty:
        los_df = los_df[["procedure_group", "avg_icu_los_hours", "avg_hdu_los_hours", "comments", "last_reviewed"]]
    edited = _data_editor(_safe_editor_df(los_df), num_rows="dynamic", use_container_width=True)
    if st.button("Save LOS reference"):
        save_procedure_los(OPS_DB_PATH, edited.to_dict(orient="records"), st.session_state.get("current_user"))
        st.success("LOS reference updated.")

with admin_tabs[3]:
    st.caption("Capture expected incoming demand for the next 24–48 hours.")
    theatre_rows = list_theatre_schedule(OPS_DB_PATH)
    theatre_df = pd.DataFrame(theatre_rows)
    if not theatre_df.empty:
        theatre_df = theatre_df[["case_date", "procedure_group", "expected_arrival_time", "icu_need", "is_emergency", "notes"]]
    edited = _data_editor(_safe_editor_df(theatre_df), num_rows="dynamic", use_container_width=True)
    if st.button("Save theatre schedule"):
        save_theatre_schedule(OPS_DB_PATH, edited.to_dict(orient="records"), st.session_state.get("current_user"))
        st.success("Theatre schedule updated.")

with admin_tabs[4]:
    _render_bed_board(read_only=False)

with admin_tabs[5]:
    latest_rules = get_latest_rules(OPS_DB_PATH) or {}
    with st.form("rules_form"):
        min_telemetry = st.checkbox("Minimum telemetry availability required for certain patients", value=bool(latest_rules.get("min_telemetry_required") or 0))
        pacing_hdu = st.checkbox("Pacing wires require HDU rather than ward", value=bool(latest_rules.get("pacing_wires_require_hdu") or 0))
        drains_hdu = st.checkbox("Chest drains require HDU rather than ward", value=bool(latest_rules.get("chest_drains_require_hdu") or 0))
        ward_o2 = st.number_input("Ward exclusion if FiO2 exceeds", value=float(latest_rules.get("ward_oxygen_fio2_threshold") or 0.4), min_value=0.21, max_value=1.0, step=0.01)
        ward_vaso = st.checkbox("Ward exclusion if vasoactive support present", value=bool(latest_rules.get("ward_exclusion_vasoactive") or 0))
        ward_lines = st.checkbox("Ward exclusion if invasive lines remain", value=bool(latest_rules.get("ward_exclusion_invasive_lines") or 0))
        endocarditis_hdu = st.checkbox("Endocarditis patients default to HDU first", value=bool(latest_rules.get("endocarditis_to_hdu") or 0))
        notes = st.text_area("Local pathway notes", value=latest_rules.get("notes") or "")
        submitted = st.form_submit_button("Save rules")
        if submitted:
            save_transfer_rules(
                OPS_DB_PATH,
                {
                    "min_telemetry_required": 1 if min_telemetry else 0,
                    "pacing_wires_require_hdu": 1 if pacing_hdu else 0,
                    "chest_drains_require_hdu": 1 if drains_hdu else 0,
                    "ward_oxygen_fio2_threshold": ward_o2,
                    "ward_exclusion_vasoactive": 1 if ward_vaso else 0,
                    "ward_exclusion_invasive_lines": 1 if ward_lines else 0,
                    "endocarditis_to_hdu": 1 if endocarditis_hdu else 0,
                    "notes": notes,
                },
                st.session_state.get("current_user"),
            )
            st.success("Rules updated.")

with admin_tabs[6]:
    audit_rows = list_audit_log(OPS_DB_PATH, limit=200)
    if audit_rows:
        st.dataframe(_safe_display_df(pd.DataFrame(audit_rows)))
    else:
        st.write("No audit events yet.")
