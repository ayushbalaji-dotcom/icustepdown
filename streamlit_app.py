import os
from datetime import datetime

import pandas as pd
import streamlit as st

from icu_stepdown.config import load_config
from icu_stepdown.features import compute_features, compute_features_latest
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
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]) or pd.api.types.is_string_dtype(safe[col]):
            try:
                safe[col] = safe[col].astype("string[python]")
            except Exception:
                safe[col] = safe[col].astype(str)
    return safe


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


def _require_login() -> bool:
    st.sidebar.header("Login")
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    submit = st.sidebar.button("Sign in")
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if submit:
        expected_user = os.environ.get("ICU_APP_USER") or _secret_get("ICU_APP_USER")
        expected_pass = os.environ.get("ICU_APP_PASS") or _secret_get("ICU_APP_PASS")
        if not expected_user or not expected_pass:
            st.sidebar.error("Set ICU_APP_USER/ICU_APP_PASS or add .streamlit/secrets.toml.")
            st.session_state.authenticated = False
        elif user == expected_user and password == expected_pass:
            st.session_state.authenticated = True
        else:
            st.sidebar.error("Invalid credentials.")
            st.session_state.authenticated = False
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

st.title("ICU → HDU Step-Down Readiness")
st.caption("Decision support only. Conservative, fail-closed.")

cfg = load_config("configs/default.yaml")
model_path = st.sidebar.text_input("Model path", value="")
hard_stops_only = st.sidebar.checkbox("Hard-stops only (no ML)", value=not bool(model_path))

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

st.subheader("Raw data (latest encounter)")
rows = load_rows(st.session_state.db_path, st.session_state.nhs_number)
if rows:
    st.dataframe(_safe_display_df(pd.DataFrame(rows)))
