import pandas as pd

from icu_stepdown.score import score_features
from icu_stepdown.quality import QualityLogger


class DummyCalibrator:
    def predict_proba(self, X):
        return pd.DataFrame([[0.9, 0.1]] * len(X)).values


def test_fail_closed_schema_mismatch(config):
    ql = QualityLogger()
    features = pd.DataFrame({
        "patient_id": ["p1"],
        "encounter_id": ["e1"],
        "score_time": pd.to_datetime(["2025-01-01 04:00"]),
    })
    model_bundle = {
        "feature_columns": ["MAP_mean_4h"],
        "training_means": {"MAP_mean_4h": 75},
        "calibrator": DummyCalibrator(),
    }
    scores, _, dashboard, fail_closed = score_features(features, model_bundle, config, ql, force_schema=False)
    assert fail_closed is True
    assert (scores["traffic_light"] == "RED").all()
    assert (dashboard["traffic_light"] == "RED").all()

