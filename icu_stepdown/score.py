from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .explain import compute_limiting_factor_and_signals
from .quality import QualityLogger


def _hard_stops(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    hs = {}
    hs["hard_stop_pressor_escalating"] = bool(row.get("pressor_on") == 1 and row.get("pressor_escalating") == 1)
    hs["hard_stop_oxygen_worsening"] = bool(
        (pd.notna(row.get("FiO2_slope_4h")) and row.get("FiO2_slope_4h") > cfg["hard_stops"]["fiO2_slope_per_hr"]) or
        (pd.notna(row.get("SpO2_time_ge_94")) and row.get("SpO2_time_ge_94") < cfg["hard_stops"]["spO2_time_ge_94"]) or
        (pd.notna(row.get("resp_support_level_slope")) and row.get("resp_support_level_slope") > cfg["hard_stops"]["resp_support_slope"])
    )
    hs["hard_stop_lactate_rising"] = bool(
        pd.notna(row.get("lactate_slope_4h")) and row.get("lactate_slope_4h") > cfg["hard_stops"]["lactate_slope_per_hr"] and
        (pd.isna(row.get("lactate_now")) or row.get("lactate_now") >= cfg["hard_stops"]["lactate_now_threshold"])
    )
    hs["hard_stop_bleeding"] = bool(
        (pd.notna(row.get("drain_sum_4h")) and row.get("drain_sum_4h") > cfg["hard_stops"]["drain_sum_4h"]) or
        (pd.notna(row.get("drain_slope")) and row.get("drain_slope") > 0) or
        (pd.notna(row.get("Hb_delta_6h")) and row.get("Hb_delta_6h") <= cfg["hard_stops"]["hb_drop_6h"])
    )
    hs["hard_stop_neuro"] = bool(
        pd.notna(row.get("RASS_now")) and (row.get("RASS_now") >= 2 or row.get("RASS_now") <= -4)
    )
    any_hs = any(hs.values())
    return any_hs, hs


def _data_quality_check(row: pd.Series, cfg: Dict[str, Any]) -> Tuple[bool, list]:
    reasons = []
    if row.get("pressor_missing_4h") == 1:
        reasons.append("pressor_missing_4h")
    if row.get("resp_missing_4h") == 1:
        reasons.append("resp_missing_4h")
    if row.get("map_missing_4h") == 1:
        reasons.append("map_missing_4h")
    if pd.isna(row.get("lactate_age_hours")) or row.get("lactate_age_hours") > cfg["thresholds"]["max_lactate_age_for_green"]:
        reasons.append("lactate_stale_or_missing")
    if pd.isna(row.get("Hb_age_hours")) or row.get("Hb_age_hours") > cfg["thresholds"]["max_hb_age_for_green"]:
        reasons.append("hb_stale_or_missing")
    if pd.isna(row.get("creatinine_age_hours")) or row.get("creatinine_age_hours") > cfg["thresholds"]["max_creatinine_age_for_green"]:
        reasons.append("creatinine_stale_or_missing")
    if pd.isna(row.get("WCC_age_hours")) or row.get("WCC_age_hours") > cfg["thresholds"]["max_wcc_age_for_green"]:
        reasons.append("wcc_stale_or_missing")
    return len(reasons) == 0, reasons


def _trajectory(prev_iri: float, iri: float) -> str:
    if pd.isna(prev_iri):
        return "→"
    delta = iri - prev_iri
    if delta > 1:
        return "↑"
    if delta < -1:
        return "↓"
    return "→"


def _deterioration_label(limiting_factor: str | Any, hard_stop_reason_summary: str | Any) -> str | None:
    reason = str(hard_stop_reason_summary or "")
    if "oxygen" in reason:
        return "Respiratory deterioration"
    if "pressor" in reason:
        return "Haemodynamic deterioration"
    if "lactate" in reason:
        return "Perfusion deterioration"
    if "bleeding" in reason:
        return "Bleeding deterioration"
    if "neuro" in reason:
        return "Neurological deterioration"

    mapping = {
        "Respiratory": "Respiratory deterioration",
        "Haemodynamic": "Haemodynamic deterioration",
        "Bleeding": "Bleeding deterioration",
        "Renal-Perfusion": "Perfusion deterioration",
        "Neuro-Infection": "Neurological deterioration",
        "Dependency": "Dependency-related deterioration",
        "Custom clinical": "Custom clinical deterioration",
    }
    return mapping.get(str(limiting_factor or ""))


def _trend_label(
    prev_iri: float,
    iri: float,
    trend_arrow: str,
    traffic_light: str,
    limiting_factor: str | Any,
    any_hard_stop: bool,
    hard_stop_reason_summary: str | Any,
    data_quality_ok: bool,
) -> str:
    deterioration = _deterioration_label(limiting_factor, hard_stop_reason_summary)
    if any_hard_stop and deterioration:
        return deterioration
    if deterioration:
        if trend_arrow == "↑" and traffic_light != "RED":
            return f"Improving {deterioration.lower()}"
        return deterioration
    if traffic_light == "GREEN" and data_quality_ok:
        if pd.isna(prev_iri) or trend_arrow == "→":
            return "Normal"
        if trend_arrow == "↑":
            return "Improving toward normal"
        return "Drifting from normal"
    if traffic_light == "AMBER" and not data_quality_ok:
        return "Incomplete data"
    if trend_arrow == "↓":
        return "Non-specific deterioration"
    if trend_arrow == "↑":
        return "Improving"
    return "Stable"


def score_features(features: pd.DataFrame, model_bundle: Dict[str, Any], cfg: Dict[str, Any], ql: QualityLogger, force_schema: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    feature_cols = model_bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in features.columns]
    if missing and not force_schema:
        ql.add("ERROR", "schema_mismatch_prevented_scoring", missing=missing)
        return _fail_closed_dashboard(features, ql)

    for col in missing:
        features[col] = np.nan

    means = pd.Series(model_bundle["training_means"]).fillna(0.0)
    X = features[feature_cols].fillna(means).fillna(0.0)
    calibrator = model_bundle["calibrator"]
    probs = calibrator.predict_proba(X)[:, 1]
    iri = (1 - probs) * 100

    scores = features[["patient_id", "encounter_id", "score_time"]].copy()
    scores["p_adverse"] = probs
    scores["IRI"] = iri

    # hard stops and data quality gating
    hard_stop_reasons = []
    any_hs_list = []
    data_quality_flags = []
    traffic = []
    green_thr = cfg["thresholds"]["green_iri"]
    amber_thr = cfg["thresholds"]["amber_iri"]

    for pos, row in enumerate(features.itertuples(index=False)):
        row = pd.Series(row._asdict())
        any_hs, hs_dict = _hard_stops(row, cfg)
        any_hs_list.append(any_hs)
        hard_stop_reasons.append(
            ",".join([k for k, v in hs_dict.items() if v])
        )
        dq_ok, dq_reasons = _data_quality_check(row, cfg)
        data_quality_flags.append(dq_ok)
        if dq_reasons:
            ql.add("WARN", "data_quality_gate", encounter_id=row.get("encounter_id"), reasons=dq_reasons)

        if any_hs:
            traffic.append("RED")
        elif iri[pos] >= green_thr and dq_ok:
            traffic.append("GREEN")
        elif iri[pos] >= green_thr and not dq_ok:
            traffic.append("AMBER")
            ql.add("WARN", "score_capped_due_to_data_quality", encounter_id=row.get("encounter_id"))
        elif iri[pos] >= amber_thr:
            traffic.append("AMBER")
        else:
            traffic.append("RED")

    scores["ANY_HARD_STOP"] = any_hs_list
    scores["hard_stop_reason_summary"] = hard_stop_reasons
    scores["data_quality_ok_for_green"] = data_quality_flags
    scores["traffic_light"] = traffic

    # trajectory
    scores = scores.sort_values(["patient_id", "encounter_id", "score_time"])
    scores["trend"] = "→"
    scores["trend_label"] = "Stable"
    prev_iri = None
    prev_key = None
    for idx, row in scores.iterrows():
        key = (row["patient_id"], row["encounter_id"])
        if key != prev_key:
            prev_iri = np.nan
            prev_key = key
        scores.at[idx, "trend"] = _trajectory(prev_iri, row["IRI"])
        prev_iri = row["IRI"]

    # explanations
    signals_df, limiting = compute_limiting_factor_and_signals(features, model_bundle, cfg)
    scores = scores.merge(limiting, on=["patient_id", "encounter_id", "score_time"], how="left")
    scores = scores.merge(signals_df, on=["patient_id", "encounter_id", "score_time"], how="left")

    prev_iri = np.nan
    prev_key = None
    for idx, row in scores.iterrows():
        key = (row["patient_id"], row["encounter_id"])
        if key != prev_key:
            prev_iri = np.nan
            prev_key = key
        scores.at[idx, "trend_label"] = _trend_label(
            prev_iri=prev_iri,
            iri=row["IRI"],
            trend_arrow=row["trend"],
            traffic_light=row["traffic_light"],
            limiting_factor=row.get("limiting_factor"),
            any_hard_stop=bool(row["ANY_HARD_STOP"]),
            hard_stop_reason_summary=row.get("hard_stop_reason_summary"),
            data_quality_ok=bool(row["data_quality_ok_for_green"]),
        )
        prev_iri = row["IRI"]

    scores["suggested_action"] = scores["traffic_light"].map({
        "GREEN": "Plan step-down: book HDU bed + prep structured handover",
        "AMBER": "Borderline: reassess in 2–4h (focus on limiting factor)",
        "RED": "Not ready: escalate attention; do not prompt step-down",
    })

    # dashboard: latest per encounter
    latest = scores.sort_values("score_time").groupby(["patient_id", "encounter_id"], as_index=False).tail(1)
    order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    latest = latest.sort_values(by=["traffic_light"], key=lambda s: s.map(order))
    dashboard = latest[[
        "patient_id",
        "encounter_id",
        "score_time",
        "traffic_light",
        "IRI",
        "trend",
        "trend_label",
        "limiting_factor",
        "suggested_action",
        "signals",
        "data_quality_ok_for_green",
        "ANY_HARD_STOP",
        "hard_stop_reason_summary",
    ]].rename(columns={"score_time": "last_score_time"})

    return scores, signals_df, dashboard, False


def score_hard_stops_only(features: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    scores = features[["patient_id", "encounter_id", "score_time"]].copy()
    scores["p_adverse"] = np.nan
    scores["IRI"] = np.nan

    hard_stop_reasons = []
    any_hs_list = []
    data_quality_flags = []
    traffic = []

    for row in features.itertuples(index=False):
        row = pd.Series(row._asdict())
        any_hs, hs_dict = _hard_stops(row, cfg)
        any_hs_list.append(any_hs)
        hard_stop_reasons.append(
            ",".join([k for k, v in hs_dict.items() if v])
        )
        dq_ok, dq_reasons = _data_quality_check(row, cfg)
        data_quality_flags.append(dq_ok)
        if dq_reasons:
            ql.add("WARN", "data_quality_gate", encounter_id=row.get("encounter_id"), reasons=dq_reasons)

        if any_hs:
            traffic.append("RED")
        elif dq_ok:
            traffic.append("GREEN")
        else:
            traffic.append("AMBER")
            ql.add("WARN", "score_capped_due_to_data_quality", encounter_id=row.get("encounter_id"))

    scores["ANY_HARD_STOP"] = any_hs_list
    scores["hard_stop_reason_summary"] = hard_stop_reasons
    scores["data_quality_ok_for_green"] = data_quality_flags
    scores["traffic_light"] = traffic

    scores["trend"] = "→"
    scores["trend_label"] = np.where(scores["ANY_HARD_STOP"], "Clinical deterioration", "Stable")
    scores["limiting_factor"] = np.where(scores["ANY_HARD_STOP"], "Hard stop triggered", "None")
    scores["signals"] = scores["hard_stop_reason_summary"].replace({"": "No hard stops triggered"})

    for idx, row in scores.iterrows():
        if row["ANY_HARD_STOP"]:
            scores.at[idx, "trend_label"] = _trend_label(
                prev_iri=np.nan,
                iri=np.nan,
                trend_arrow="→",
                traffic_light=row["traffic_light"],
                limiting_factor=row["limiting_factor"],
                any_hard_stop=bool(row["ANY_HARD_STOP"]),
                hard_stop_reason_summary=row["hard_stop_reason_summary"],
                data_quality_ok=bool(row["data_quality_ok_for_green"]),
            )
        elif row["traffic_light"] == "GREEN":
            scores.at[idx, "trend_label"] = "Normal"
        elif not row["data_quality_ok_for_green"]:
            scores.at[idx, "trend_label"] = "Incomplete data"

    scores["suggested_action"] = scores["traffic_light"].map({
        "GREEN": "Plan step-down: book HDU bed + prep structured handover",
        "AMBER": "Borderline: reassess in 2–4h (focus on missing data)",
        "RED": "Not ready: escalate attention; do not prompt step-down",
    })

    latest = scores.sort_values("score_time").groupby(["patient_id", "encounter_id"], as_index=False).tail(1)
    order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    latest = latest.sort_values(by=["traffic_light"], key=lambda s: s.map(order))
    dashboard = latest[[
        "patient_id",
        "encounter_id",
        "score_time",
        "traffic_light",
        "IRI",
        "trend",
        "trend_label",
        "limiting_factor",
        "suggested_action",
        "signals",
        "data_quality_ok_for_green",
        "ANY_HARD_STOP",
        "hard_stop_reason_summary",
    ]].rename(columns={"score_time": "last_score_time"})

    signals_df = scores[["patient_id", "encounter_id", "score_time", "signals"]].copy()
    return scores, signals_df, dashboard, False


def _fail_closed_dashboard(features: pd.DataFrame, ql: QualityLogger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    # Fail closed: no GREEN
    unique = features[["patient_id", "encounter_id"]].drop_duplicates()
    dashboard = unique.copy()
    dashboard["last_score_time"] = pd.NaT
    dashboard["traffic_light"] = "RED"
    dashboard["IRI"] = np.nan
    dashboard["trend"] = "→"
    dashboard["trend_label"] = "Unknown"
    dashboard["limiting_factor"] = "None"
    dashboard["suggested_action"] = "Not ready: escalate attention; do not prompt step-down"
    dashboard["signals"] = "Insufficient data to explain"
    dashboard["data_quality_ok_for_green"] = False
    dashboard["ANY_HARD_STOP"] = False
    dashboard["hard_stop_reason_summary"] = ""

    scores = features[["patient_id", "encounter_id", "score_time"]].copy()
    scores["p_adverse"] = np.nan
    scores["IRI"] = np.nan
    scores["ANY_HARD_STOP"] = False
    scores["hard_stop_reason_summary"] = ""
    scores["data_quality_ok_for_green"] = False
    scores["traffic_light"] = "RED"
    scores["trend"] = "→"
    scores["trend_label"] = "Unknown"
    scores["limiting_factor"] = "None"
    scores["suggested_action"] = "Not ready: escalate attention; do not prompt step-down"
    scores["signals"] = "Insufficient data to explain"

    signals_df = scores[["patient_id", "encounter_id", "score_time", "signals"]].copy()
    ql.add("ERROR", "fail_closed_scoring", reason="schema_mismatch_or_model_missing")
    return scores, signals_df, dashboard, True
