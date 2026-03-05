import pandas as pd
import numpy as np

from icu_stepdown.score import score_features
from icu_stepdown.quality import QualityLogger


class DummyCalibrator:
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 0.05
        probs[:, 0] = 0.95
        return probs


def test_hard_stop_forces_red(config):
    ql = QualityLogger()
    features = pd.DataFrame({
        "patient_id": ["p1"],
        "encounter_id": ["e1"],
        "score_time": pd.to_datetime(["2025-01-01 04:00"]),
        "FiO2_slope_4h": [0.02],
        "SpO2_time_ge_94": [0.9],
        "resp_support_level_slope": [0.0],
        "pressor_on": [0],
        "pressor_escalating": [0],
        "drain_sum_4h": [0],
        "drain_slope": [0],
        "Hb_delta_6h": [0],
        "lactate_slope_4h": [0],
        "lactate_now": [1.2],
        "RASS_now": [0],
        "pressor_missing_4h": [0],
        "resp_missing_4h": [0],
        "map_missing_4h": [0],
        "lactate_age_hours": [1],
        "Hb_age_hours": [1],
        "creatinine_age_hours": [1],
        "WCC_age_hours": [1],
    })
    model_bundle = {
        "feature_columns": config["feature_schema"],
        "training_means": {c: 0 for c in config["feature_schema"]},
        "calibrator": DummyCalibrator(),
    }
    scores, _, _, _ = score_features(features, model_bundle, config, ql, force_schema=True)
    assert scores.iloc[0]["traffic_light"] == "RED"

