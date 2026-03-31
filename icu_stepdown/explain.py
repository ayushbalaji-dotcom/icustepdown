from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .custom_features import custom_feature_columns_from_features, custom_feature_label


def _domain_groups(features: pd.DataFrame | None = None) -> Dict[str, List[str]]:
    groups = {
        "Haemodynamic": [
            "MAP_mean_4h",
            "MAP_sd_4h",
            "MAP_min_4h",
            "MAP_slope_4h",
            "MAP_time_ge_65",
            "HR_mean_4h",
            "HR_sd_4h",
            "HR_slope_4h",
            "pressor_on",
            "pressor_escalating",
            "pressor_delta_4h",
            "pressor_free_hours",
        ],
        "Respiratory": [
            "FiO2_now",
            "FiO2_slope_4h",
            "SpO2_time_ge_94",
            "RR_mean_4h",
            "RR_slope_4h",
            "resp_support_level_now",
            "resp_support_level_slope",
            "extubated_hours",
        ],
        "Bleeding": [
            "drain_sum_4h",
            "drain_sum_8h",
            "drain_slope",
            "Hb_delta_6h",
        ],
        "Renal-Perfusion": [
            "uop_sum_4h",
            "oliguria_flag",
            "lactate_now",
            "lactate_slope_4h",
            "creatinine_delta_24h",
        ],
        "Neuro-Infection": [
            "RASS_now",
            "temp_slope_4h",
            "WCC_slope_24h",
        ],
        "Dependency": [
            "arterial_line_present_latest",
            "insulin_infusion_latest",
        ],
    }
    if features is not None:
        custom_cols = custom_feature_columns_from_features(features)
        if custom_cols:
            groups["Custom clinical"] = custom_cols
    return groups


def _signal_templates() -> Dict[str, str]:
    return {
        "MAP_mean_4h": "MAP mean 4h {value:.1f} mmHg",
        "MAP_min_4h": "MAP min 4h {value:.1f} mmHg",
        "MAP_slope_4h": "MAP slope {value:.2f} per hr",
        "HR_mean_4h": "HR mean 4h {value:.1f} bpm",
        "pressor_on": "Pressor on {value:.0f}",
        "pressor_escalating": "Pressor escalating {value:.0f}",
        "FiO2_now": "FiO2 now {value:.2f}",
        "FiO2_slope_4h": "FiO2 slope {value:.3f} per hr",
        "SpO2_time_ge_94": "SpO2 time ≥94% {value:.2f}",
        "RR_mean_4h": "RR mean 4h {value:.1f} /min",
        "resp_support_level_now": "Resp support level {value:.0f}",
        "drain_sum_4h": "Drain sum 4h {value:.0f} ml",
        "Hb_delta_6h": "Hb delta 6h {value:.1f} g/L",
        "uop_sum_4h": "UOP sum 4h {value:.0f} ml",
        "lactate_now": "Lactate now {value:.2f}",
        "creatinine_delta_24h": "Creatinine delta 24h {value:.1f} umol/L",
        "RASS_now": "RASS now {value:.0f}",
        "temp_slope_4h": "Temp slope {value:.2f} C/hr",
        "WCC_slope_24h": "WCC slope {value:.2f} per hr",
        "arterial_line_present_latest": "Arterial line present {value:.0f}",
        "insulin_infusion_latest": "Insulin infusion {value:.0f}",
    }


def _custom_signal(feature_name: str, value: float) -> str:
    label = custom_feature_label(feature_name)
    if feature_name.endswith("_now"):
        return f"{label} now {value:.2f}"
    if feature_name.endswith("_mean_4h"):
        return f"{label} mean 4h {value:.2f}"
    if feature_name.endswith("_slope_4h"):
        return f"{label} slope {value:.3f} per hr"
    if feature_name.endswith("_missing_4h"):
        return f"{label} missing 4h {value:.0f}"
    return f"{label} {value:.2f}"


def _normal_signals(row: pd.Series, cfg: Dict[str, Any]) -> List[str]:
    signals: List[str] = []
    if pd.notna(row.get("MAP_time_ge_65")) and row.get("MAP_time_ge_65") >= 0.8:
        signals.append(f"MAP held >=65 for {row.get('MAP_time_ge_65') * 100:.0f}% of last 4h")
    elif pd.notna(row.get("MAP_mean_4h")) and row.get("MAP_mean_4h") >= 65:
        signals.append(f"MAP mean 4h {float(row.get('MAP_mean_4h')):.1f} mmHg")

    spo2_target = float(cfg.get("hard_stops", {}).get("spO2_time_ge_94", 0.8))
    if pd.notna(row.get("SpO2_time_ge_94")) and row.get("SpO2_time_ge_94") >= spo2_target:
        signals.append(f"SpO2 maintained >=94% for {row.get('SpO2_time_ge_94') * 100:.0f}% of last 4h")

    if pd.notna(row.get("pressor_on")) and float(row.get("pressor_on")) == 0:
        signals.append("No vasoactive support running")

    lactate_now_thr = float(cfg.get("hard_stops", {}).get("lactate_now_threshold", 2.0))
    if pd.notna(row.get("lactate_now")) and row.get("lactate_now") < lactate_now_thr:
        signals.append(f"Lactate {float(row.get('lactate_now')):.2f}")

    if pd.notna(row.get("FiO2_slope_4h")) and row.get("FiO2_slope_4h") <= 0:
        signals.append(f"FiO2 stable/improving ({float(row.get('FiO2_slope_4h')):.3f} per hr)")

    return signals[:3]


def compute_limiting_factor_and_signals(
    features: pd.DataFrame, model_bundle: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = model_bundle["feature_columns"]
    means = pd.Series(model_bundle["training_means"]).fillna(0.0)
    X = features[feature_cols].fillna(means).fillna(0.0)
    calibrator = model_bundle["calibrator"]
    base = calibrator.predict_proba(X)[:, 1]

    domain_groups = _domain_groups(features)
    drops = {}
    for domain, cols in domain_groups.items():
        X_occluded = X.copy()
        for col in cols:
            if col in X_occluded.columns:
                X_occluded[col] = means.get(col, np.nan)
        occluded = calibrator.predict_proba(X_occluded)[:, 1]
        drops[domain] = base - occluded

    limiting_factors = []
    signals = []
    template = _signal_templates()
    for i in range(len(features)):
        best_domain = "None"
        best_drop = 0.0
        for domain, diff in drops.items():
            drop = float(diff[i])
            if drop > best_drop:
                best_drop = drop
                best_domain = domain
        if best_drop < 1e-4:
            best_domain = "None"
        limiting_factors.append(best_domain)

        sigs = []
        if best_domain != "None":
            for feat in domain_groups[best_domain]:
                if feat in features.columns and pd.notna(features.iloc[i][feat]):
                    tpl = template.get(feat)
                    if tpl:
                        sigs.append(tpl.format(value=float(features.iloc[i][feat])))
                    elif feat.startswith("custom_"):
                        sigs.append(_custom_signal(feat, float(features.iloc[i][feat])))
                if len(sigs) >= 3:
                    break
        else:
            sigs = _normal_signals(features.iloc[i], cfg)
        if not sigs:
            sigs = ["Insufficient data to explain"]
        signals.append("; ".join(sigs))

    limiting_df = features[["patient_id", "encounter_id", "score_time"]].copy()
    limiting_df["limiting_factor"] = limiting_factors

    signals_df = features[["patient_id", "encounter_id", "score_time"]].copy()
    signals_df["signals"] = signals

    return signals_df, limiting_df
