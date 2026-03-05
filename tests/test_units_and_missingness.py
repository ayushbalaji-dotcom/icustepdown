import numpy as np
import pandas as pd

from icu_stepdown.preprocess import preprocess
from icu_stepdown.features import compute_features
from icu_stepdown.score import score_features
from icu_stepdown.quality import QualityLogger
from icu_stepdown.schema import validate_raw


class DummyCalibrator:
    def predict_proba(self, X):
        # low risk
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 0.1
        probs[:, 0] = 0.9
        return probs


def test_units_and_missingness(config):
    ql = QualityLogger()
    df = pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "encounter_id": ["e1", "e1"],
        "timestamp": ["2025-01-01 00:00", "2025-01-01 01:00"],
        "FiO2": [40, 50],
        "SpO2": [0.95, 0.96],
        "temperature_C": [100.0, 101.0],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df, _ = validate_raw(df, config, ql)
    df = preprocess(df, config, ql)
    assert df["FiO2"].max() <= 1.0
    assert df["SpO2"].max() > 90
    assert df["temperature_C"].max() < 40

    df.loc[0, "noradrenaline_mcgkgmin"] = -0.1
    df = preprocess(df, config, ql)
    assert pd.isna(df.loc[0, "noradrenaline_mcgkgmin"])

    # Missing labs should cap green
    df["MAP"] = np.nan
    df["RR"] = 20
    df["HR"] = 80
    df["oxygen_device"] = "NC"
    df["lactate"] = np.nan
    df["haemoglobin_gL"] = np.nan
    df["creatinine_umolL"] = np.nan
    df["WCC_10e9L"] = np.nan
    df = preprocess(df, config, ql)
    feat = compute_features(df, config, ql)

    # All-NaN pressors -> pressor_on NaN
    assert feat["pressor_on"].isna().all()

    model_bundle = {
        "feature_columns": config["feature_schema"],
        "training_means": pd.Series(feat[config["feature_schema"]].mean()).to_dict(),
        "calibrator": DummyCalibrator(),
    }
    scores, _, _, _ = score_features(feat, model_bundle, config, ql)
    assert (scores["traffic_light"] != "GREEN").all()


def test_stale_labs_cap_amber(config):
    ql = QualityLogger()
    features = pd.DataFrame({
        "patient_id": ["p1"],
        "encounter_id": ["e1"],
        "score_time": pd.to_datetime(["2025-01-01 04:00"]),
        "pressor_missing_4h": [0],
        "resp_missing_4h": [0],
        "map_missing_4h": [0],
        "lactate_age_hours": [99],
        "Hb_age_hours": [99],
        "creatinine_age_hours": [99],
        "WCC_age_hours": [99],
    })
    model_bundle = {
        "feature_columns": config["feature_schema"],
        "training_means": {c: 0 for c in config["feature_schema"]},
        "calibrator": DummyCalibrator(),
    }
    scores, _, _, _ = score_features(features, model_bundle, config, ql, force_schema=True)
    assert (scores["traffic_light"] != "GREEN").all()
