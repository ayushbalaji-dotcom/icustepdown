from typing import Any, Dict

import numpy as np
import pandas as pd

from .custom_features import resolve_feature_schema


class BaselineCalibrator:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        baseline_cfg = cfg.get("baseline", {}) or {}
        self.base_iri = float(baseline_cfg.get("base_iri", 80))
        self.penalties = baseline_cfg.get("penalties", {}) or {}
        self.hard_stops = cfg.get("hard_stops", {}) or {}
        self.thresholds = cfg.get("thresholds", {}) or {}

    def _penalty(self, key: str, default: float) -> float:
        val = self.penalties.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        iri = np.full(len(X), self.base_iri, dtype=float)

        if "pressor_on" in X.columns:
            mask = X["pressor_on"].notna() & (X["pressor_on"] >= 1)
            iri[mask] -= self._penalty("pressor_on", 20)

        if "pressor_escalating" in X.columns:
            mask = X["pressor_escalating"].notna() & (X["pressor_escalating"] >= 1)
            iri[mask] -= self._penalty("pressor_escalating", 10)

        fio2_thr = self.hard_stops.get("fiO2_slope_per_hr")
        if fio2_thr is not None and "FiO2_slope_4h" in X.columns:
            mask = X["FiO2_slope_4h"].notna() & (X["FiO2_slope_4h"] > fio2_thr)
            iri[mask] -= self._penalty("fio2_slope_worsening", 15)

        spo2_thr = self.hard_stops.get("spO2_time_ge_94")
        if spo2_thr is not None and "SpO2_time_ge_94" in X.columns:
            mask = X["SpO2_time_ge_94"].notna() & (X["SpO2_time_ge_94"] < spo2_thr)
            iri[mask] -= self._penalty("spo2_time_low", 10)

        resp_thr = self.hard_stops.get("resp_support_slope")
        if resp_thr is not None and "resp_support_level_slope" in X.columns:
            mask = X["resp_support_level_slope"].notna() & (X["resp_support_level_slope"] > resp_thr)
            iri[mask] -= self._penalty("resp_support_worsening", 10)

        lactate_slope_thr = self.hard_stops.get("lactate_slope_per_hr")
        if lactate_slope_thr is not None and "lactate_slope_4h" in X.columns:
            mask = X["lactate_slope_4h"].notna() & (X["lactate_slope_4h"] > lactate_slope_thr)
            iri[mask] -= self._penalty("lactate_slope_worsening", 15)

        lactate_now_thr = self.hard_stops.get("lactate_now_threshold")
        if lactate_now_thr is not None and "lactate_now" in X.columns:
            mask = X["lactate_now"].notna() & (X["lactate_now"] >= lactate_now_thr)
            iri[mask] -= self._penalty("lactate_now_high", 10)

        drain_thr = self.hard_stops.get("drain_sum_4h")
        if drain_thr is not None and "drain_sum_4h" in X.columns:
            mask = X["drain_sum_4h"].notna() & (X["drain_sum_4h"] > drain_thr)
            iri[mask] -= self._penalty("drain_high", 15)

        hb_drop_thr = self.hard_stops.get("hb_drop_6h")
        if hb_drop_thr is not None and "Hb_delta_6h" in X.columns:
            mask = X["Hb_delta_6h"].notna() & (X["Hb_delta_6h"] <= hb_drop_thr)
            iri[mask] -= self._penalty("hb_drop", 10)

        if "RASS_now" in X.columns:
            mask = X["RASS_now"].notna() & ((X["RASS_now"] >= 2) | (X["RASS_now"] <= -4))
            iri[mask] -= self._penalty("rass_extreme", 10)

        oliguria_thr = self.thresholds.get("oliguria_threshold_ml_per_4h")
        if oliguria_thr is not None and "uop_sum_4h" in X.columns:
            mask = X["uop_sum_4h"].notna() & (X["uop_sum_4h"] < oliguria_thr)
            iri[mask] -= self._penalty("oliguria", 10)

        iri = np.clip(iri, 0, 100)
        p_adverse = 1.0 - (iri / 100.0)
        return np.column_stack([1.0 - p_adverse, p_adverse])


def build_baseline_bundle(features: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    feature_cols = resolve_feature_schema(cfg, features_df=features)
    data = features.copy()
    for col in feature_cols:
        if col not in data.columns:
            data[col] = np.nan
    training_means = data[feature_cols].mean().fillna(0.0)
    return {
        "base_model": None,
        "calibrator": BaselineCalibrator(cfg),
        "feature_columns": feature_cols,
        "training_means": training_means.to_dict(),
        "metrics": {"calibration_method": "baseline"},
        "config": cfg,
    }
