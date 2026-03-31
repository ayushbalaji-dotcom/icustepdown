import numpy as np
import pandas as pd

from icu_stepdown.quality import QualityLogger
from icu_stepdown.score import score_features


class RespiratoryTrendCalibrator:
    def predict_proba(self, X):
        risk = (
            X["FiO2_slope_4h"].fillna(-0.01).clip(lower=0) * 20
            + (0.98 - X["SpO2_time_ge_94"].fillna(0.98)).clip(lower=0) * 2
            + X["resp_support_level_slope"].fillna(0).clip(lower=0) * 0.15
        )
        probs = np.clip(0.05 + risk, 0.05, 0.9)
        return np.column_stack([1.0 - probs, probs])


def test_trend_labels_capture_normal_and_specific_deterioration(config):
    ql = QualityLogger()
    features = pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "encounter_id": ["e1", "e1"],
        "score_time": pd.to_datetime(["2025-01-01 04:00", "2025-01-01 08:00"]),
        "MAP_mean_4h": [76.0, 74.0],
        "MAP_time_ge_65": [1.0, 1.0],
        "FiO2_slope_4h": [-0.01, 0.01],
        "SpO2_time_ge_94": [0.98, 0.85],
        "RR_mean_4h": [16.0, 24.0],
        "resp_support_level_now": [1.0, 3.0],
        "resp_support_level_slope": [0.0, 1.0],
        "pressor_on": [0.0, 0.0],
        "pressor_escalating": [0.0, 0.0],
        "drain_sum_4h": [10.0, 10.0],
        "drain_slope": [0.0, 0.0],
        "Hb_delta_6h": [0.0, 0.0],
        "lactate_now": [1.1, 1.3],
        "lactate_slope_4h": [-0.02, 0.0],
        "RASS_now": [0.0, 0.0],
        "pressor_missing_4h": [0.0, 0.0],
        "resp_missing_4h": [0.0, 0.0],
        "map_missing_4h": [0.0, 0.0],
        "lactate_age_hours": [1.0, 1.0],
        "Hb_age_hours": [1.0, 1.0],
        "creatinine_age_hours": [1.0, 1.0],
        "WCC_age_hours": [1.0, 1.0],
    })
    means = {c: 0 for c in config["feature_schema"]}
    means.update({
        "FiO2_slope_4h": -0.01,
        "SpO2_time_ge_94": 0.98,
        "resp_support_level_slope": 0.0,
        "pressor_on": 0.0,
        "lactate_now": 1.0,
    })
    model_bundle = {
        "feature_columns": config["feature_schema"],
        "training_means": means,
        "calibrator": RespiratoryTrendCalibrator(),
    }

    scores, _, dashboard, _ = score_features(features, model_bundle, config, ql, force_schema=True)

    assert scores.iloc[0]["trend_label"] == "Normal"
    assert "Insufficient data" not in scores.iloc[0]["signals"]
    assert scores.iloc[1]["trend_label"] == "Respiratory deterioration"
    assert scores.iloc[1]["limiting_factor"] == "Respiratory"
    assert dashboard.iloc[0]["trend_label"] == "Respiratory deterioration"
