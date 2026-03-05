from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .quality import QualityLogger


def _slope(x: pd.Series, t: pd.Series) -> float:
    if x.dropna().shape[0] < 2:
        return np.nan
    mask = x.notna() & t.notna()
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    t = t[mask]
    # t in hours
    t_hours = (t - t.iloc[0]).dt.total_seconds() / 3600.0
    if np.isclose(t_hours.max(), 0):
        return np.nan
    coef = np.polyfit(t_hours, x.astype(float), 1)
    return float(coef[0])


def _time_in_range(series: pd.Series, threshold: float, min_points: int) -> float:
    valid = series.dropna()
    if valid.shape[0] < min_points:
        return np.nan
    return float((valid >= threshold).mean())


def _last_value(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def _mean(series: pd.Series, min_points: int) -> float:
    valid = series.dropna()
    if valid.shape[0] < min_points:
        return np.nan
    return float(valid.mean())


def _std(series: pd.Series, min_points: int) -> float:
    valid = series.dropna()
    if valid.shape[0] < min_points:
        return np.nan
    return float(valid.std(ddof=0))


def _min(series: pd.Series, min_points: int) -> float:
    valid = series.dropna()
    if valid.shape[0] < min_points:
        return np.nan
    return float(valid.min())


def _delta(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.shape[0] < 2:
        return np.nan
    return float(valid.iloc[-1] - valid.iloc[0])


def _age_hours(series: pd.Series, time_series: pd.Series, score_time: pd.Timestamp) -> float:
    mask = series.notna() & time_series.notna() & (time_series <= score_time)
    if mask.sum() == 0:
        return np.nan
    last_time = time_series[mask].iloc[-1]
    return float((score_time - last_time).total_seconds() / 3600.0)


def _ceil_to_4h(ts: pd.Timestamp) -> pd.Timestamp:
    base = ts.replace(minute=0, second=0, microsecond=0)
    hour = base.hour
    remainder = hour % 4
    if remainder == 0 and ts == base:
        return base
    delta_hours = (4 - remainder) % 4
    if delta_hours == 0:
        delta_hours = 4
    return base + timedelta(hours=delta_hours)


def _start_times(enc_df: pd.DataFrame, cfg: Dict[str, Any]) -> List[pd.Timestamp]:
    alignment = cfg.get("score_alignment", "encounter_anchor")
    interval = int(cfg.get("score_interval_hours", 4))
    t0 = enc_df["timestamp"].min()
    t1 = enc_df["timestamp"].max()
    if alignment == "wall_clock":
        start = _ceil_to_4h(t0)
    else:
        start = _ceil_to_4h(t0)
    times = []
    cur = start
    while cur <= t1:
        times.append(cur)
        cur += timedelta(hours=interval)
    return times


def _compute_feature_rows(
    enc_df: pd.DataFrame,
    score_times: List[pd.Timestamp],
    cfg: Dict[str, Any],
    patient_id: Any,
    encounter_id: Any,
) -> List[Dict[str, Any]]:
    min_points = int(cfg.get("min_points", 4))
    windows = cfg.get("windows_hours", {})
    vitals_h = int(windows.get("vitals", 4))
    drains_short = int(windows.get("drains_short", 4))
    drains_long = int(windows.get("drains_long", 8))
    labs_long = int(windows.get("labs_long", 24))
    hb_delta = int(windows.get("hb_delta", 6))

    rows = []
    for score_time in score_times:
        window_vitals = enc_df[(enc_df["timestamp"] > score_time - timedelta(hours=vitals_h)) & (enc_df["timestamp"] <= score_time)]
        window_drains_4 = enc_df[(enc_df["timestamp"] > score_time - timedelta(hours=drains_short)) & (enc_df["timestamp"] <= score_time)]
        window_drains_8 = enc_df[(enc_df["timestamp"] > score_time - timedelta(hours=drains_long)) & (enc_df["timestamp"] <= score_time)]
        window_labs_24 = enc_df[(enc_df["timestamp"] > score_time - timedelta(hours=labs_long)) & (enc_df["timestamp"] <= score_time)]
        window_hb_6 = enc_df[(enc_df["timestamp"] > score_time - timedelta(hours=hb_delta)) & (enc_df["timestamp"] <= score_time)]

        feats: Dict[str, Any] = {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "score_time": score_time,
        }

        # Haemodynamic
        feats["MAP_mean_4h"] = _mean(window_vitals["MAP"], min_points)
        feats["MAP_sd_4h"] = _std(window_vitals["MAP"], min_points)
        feats["MAP_min_4h"] = _min(window_vitals["MAP"], min_points)
        feats["MAP_slope_4h"] = _slope(window_vitals["MAP"], window_vitals["timestamp"])
        feats["MAP_time_ge_65"] = _time_in_range(window_vitals["MAP"], 65, min_points)

        feats["HR_mean_4h"] = _mean(window_vitals["HR"], min_points)
        feats["HR_sd_4h"] = _std(window_vitals["HR"], min_points)
        feats["HR_slope_4h"] = _slope(window_vitals["HR"], window_vitals["timestamp"])

        # Pressors
        pressor_series = window_vitals["total_pressor"]
        feats["pressor_on"] = np.nan
        feats["pressor_escalating"] = np.nan
        feats["pressor_delta_4h"] = np.nan
        if pressor_series.dropna().shape[0] >= 1:
            last_val = pressor_series.dropna().iloc[-1]
            feats["pressor_on"] = 1.0 if last_val > 0 else 0.0
        if pressor_series.dropna().shape[0] >= 2:
            first_val = pressor_series.dropna().iloc[0]
            last_val = pressor_series.dropna().iloc[-1]
            feats["pressor_escalating"] = 1.0 if last_val > first_val else 0.0
            feats["pressor_delta_4h"] = float((last_val - first_val) / max(abs(first_val), 1e-6))

        # pressor free hours
        pressor_hist = enc_df[(enc_df["timestamp"] <= score_time) & (enc_df["total_pressor"].notna())]
        if pressor_hist.empty:
            feats["pressor_free_hours"] = np.nan
        else:
            last_on = pressor_hist[pressor_hist["total_pressor"] > 0]
            if last_on.empty:
                feats["pressor_free_hours"] = float((score_time - pressor_hist["timestamp"].iloc[0]).total_seconds() / 3600.0)
            else:
                feats["pressor_free_hours"] = float((score_time - last_on["timestamp"].iloc[-1]).total_seconds() / 3600.0)

        # Respiratory
        feats["FiO2_now"] = _last_value(window_vitals["FiO2"])
        feats["FiO2_slope_4h"] = _slope(window_vitals["FiO2"], window_vitals["timestamp"])
        feats["SpO2_time_ge_94"] = _time_in_range(window_vitals["SpO2"], 94, min_points)
        feats["RR_mean_4h"] = _mean(window_vitals["RR"], min_points)
        feats["RR_slope_4h"] = _slope(window_vitals["RR"], window_vitals["timestamp"])
        feats["resp_support_level_now"] = _last_value(window_vitals["resp_support_level"])
        feats["resp_support_level_slope"] = _slope(window_vitals["resp_support_level"], window_vitals["timestamp"])

        # Extubated hours
        ett_hist = enc_df[(enc_df["timestamp"] <= score_time) & (enc_df["resp_support_level"] == 5)]
        if ett_hist.empty:
            feats["extubated_hours"] = np.nan
        else:
            feats["extubated_hours"] = float((score_time - ett_hist["timestamp"].iloc[-1]).total_seconds() / 3600.0)

        # Bleeding
        feats["drain_sum_4h"] = float(window_drains_4["chest_drain_ml_30min"].sum(min_count=1))
        feats["drain_sum_8h"] = float(window_drains_8["chest_drain_ml_30min"].sum(min_count=1))
        feats["drain_slope"] = _slope(window_drains_4["chest_drain_ml_30min"], window_drains_4["timestamp"])
        feats["Hb_delta_6h"] = _delta(window_hb_6["haemoglobin_gL"])

        # Renal/Perfusion
        feats["uop_sum_4h"] = float(window_vitals["urine_output_ml_30min"].sum(min_count=1))
        if pd.isna(feats["uop_sum_4h"]):
            feats["oliguria_flag"] = np.nan
        else:
            thresh = cfg["thresholds"]["oliguria_threshold_ml_per_4h"]
            feats["oliguria_flag"] = 1.0 if feats["uop_sum_4h"] < thresh else 0.0
        feats["lactate_now"] = _last_value(window_vitals["lactate"])
        feats["lactate_slope_4h"] = _slope(window_vitals["lactate"], window_vitals["timestamp"])
        feats["creatinine_delta_24h"] = _delta(window_labs_24["creatinine_umolL"])

        # Neuro/Infection
        feats["RASS_now"] = _last_value(window_vitals["RASS"])
        feats["temp_slope_4h"] = _slope(window_vitals["temperature_C"], window_vitals["timestamp"])
        feats["WCC_slope_24h"] = _slope(window_labs_24["WCC_10e9L"], window_labs_24["timestamp"])

        # Dependency
        feats["arterial_line_present_latest"] = _last_value(window_vitals["arterial_line_present"])
        feats["insulin_infusion_latest"] = _last_value(window_vitals["insulin_infusion"])

        # Recency
        feats["lactate_age_hours"] = _age_hours(enc_df["lactate"], enc_df["timestamp"], score_time)
        feats["Hb_age_hours"] = _age_hours(enc_df["haemoglobin_gL"], enc_df["timestamp"], score_time)
        feats["creatinine_age_hours"] = _age_hours(enc_df["creatinine_umolL"], enc_df["timestamp"], score_time)
        feats["WCC_age_hours"] = _age_hours(enc_df["WCC_10e9L"], enc_df["timestamp"], score_time)

        # Missingness flags
        feats["pressor_missing_4h"] = 1.0 if window_vitals["total_pressor"].dropna().empty else 0.0
        feats["resp_missing_4h"] = 1.0 if (window_vitals["FiO2"].dropna().empty and window_vitals["resp_support_level"].dropna().empty) else 0.0
        feats["map_missing_4h"] = 1.0 if window_vitals["MAP"].dropna().shape[0] < min_points else 0.0

        rows.append(feats)

    return rows


def compute_features(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    pressor_cols = [
        "noradrenaline_mcgkgmin",
        "adrenaline_mcgkgmin",
        "dobutamine_mcgkgmin",
        "milrinone_mcgkgmin",
    ]
    df = df.copy()
    df["total_pressor"] = df[pressor_cols].sum(axis=1, min_count=1)

    rows = []
    for (patient_id, encounter_id), enc_df in df.groupby(["patient_id", "encounter_id"]):
        enc_df = enc_df.sort_values("timestamp")
        score_times = _start_times(enc_df, cfg)
        if not score_times:
            continue
        rows.extend(_compute_feature_rows(enc_df, score_times, cfg, patient_id, encounter_id))

    feat_df = pd.DataFrame(rows)
    ql.add("INFO", "features_computed", count=int(feat_df.shape[0]))
    return feat_df


def compute_features_latest(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    pressor_cols = [
        "noradrenaline_mcgkgmin",
        "adrenaline_mcgkgmin",
        "dobutamine_mcgkgmin",
        "milrinone_mcgkgmin",
    ]
    df = df.copy()
    df["total_pressor"] = df[pressor_cols].sum(axis=1, min_count=1)

    rows = []
    for (patient_id, encounter_id), enc_df in df.groupby(["patient_id", "encounter_id"]):
        enc_df = enc_df.sort_values("timestamp")
        score_time = enc_df["timestamp"].max()
        if pd.isna(score_time):
            continue
        rows.extend(_compute_feature_rows(enc_df, [score_time], cfg, patient_id, encounter_id))

    feat_df = pd.DataFrame(rows)
    ql.add("INFO", "features_computed", count=int(feat_df.shape[0]), mode="latest")
    return feat_df
