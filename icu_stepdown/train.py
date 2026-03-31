import pickle
from datetime import timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from .custom_features import resolve_feature_schema
from .quality import QualityLogger
from .split import split_encounters, calibration_split


class _NoCalibrator:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _restore_training_class_coverage(
    data: pd.DataFrame,
    train_enc: list[str],
    cal_enc: list[str],
    test_enc: list[str],
    ql: QualityLogger,
) -> tuple[list[str], list[str], list[str]]:
    encounter_labels = data.groupby("encounter_id")["ADVERSE_EVENT"].max().astype(int).to_dict()
    observed_classes = {int(label) for label in encounter_labels.values()}
    if len(observed_classes) < 2:
        return train_enc, cal_enc, test_enc

    train_enc = list(dict.fromkeys(train_enc))
    cal_enc = list(dict.fromkeys(cal_enc))
    test_enc = list(dict.fromkeys(test_enc))

    def train_classes() -> set[int]:
        return {int(encounter_labels[e]) for e in train_enc if e in encounter_labels}

    if len(train_classes()) >= 2:
        return train_enc, cal_enc, test_enc

    ql.add("WARN", "expanded_training_split_for_class_coverage")
    for needed_class in sorted(observed_classes - train_classes()):
        moved = False
        for split_name, split_values in (("calibration", cal_enc), ("test", test_enc)):
            for encounter_id in list(split_values):
                if encounter_labels.get(encounter_id) == needed_class:
                    split_values.remove(encounter_id)
                    train_enc.append(encounter_id)
                    ql.add(
                        "WARN",
                        "encounter_moved_to_training",
                        encounter_id=encounter_id,
                        from_split=split_name,
                        reason="class_coverage",
                    )
                    moved = True
                    break
            if moved:
                break

    return list(dict.fromkeys(train_enc)), cal_enc, test_enc


def _select_training_rows(features: pd.DataFrame, outcomes: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    outcomes = outcomes.copy()
    outcomes["icu_discharge_time"] = pd.to_datetime(outcomes["icu_discharge_time"], errors="coerce")
    merged = features.merge(
        outcomes[["patient_id", "encounter_id", "icu_discharge_time", "ADVERSE_EVENT", "censored"]],
        on=["patient_id", "encounter_id"],
        how="inner",
    )
    merged = merged[merged["censored"] == 0].copy()
    merged["score_time"] = pd.to_datetime(merged["score_time"])
    merged["within_window"] = (
        merged["score_time"] <= merged["icu_discharge_time"]
    ) & (
        merged["score_time"] >= merged["icu_discharge_time"] - timedelta(hours=lookback_hours)
    )
    return merged[merged["within_window"]].copy()


def _build_model(cfg: Dict[str, Any], ql: QualityLogger) -> tuple[Any, str]:
    backend = str(cfg.get("model", {}).get("backend", "gaussian_nb")).strip().lower()
    random_state = cfg["model"]["random_state"]

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier

            params = cfg["xgboost_params"].copy()
            return XGBClassifier(**params, random_state=random_state), "xgboost"
        except Exception as exc:
            ql.add("WARN", "xgboost_unavailable_fallback", reason=str(exc))

    if backend == "gaussian_nb":
        return GaussianNB(), "gaussian_nb"

    ql.add("WARN", "unknown_model_backend_fallback", requested_backend=backend)
    _ = random_state  # keep signature aligned for alternate backends
    return GaussianNB(), "gaussian_nb"


def train_model(features: pd.DataFrame, outcomes: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lookback_hours = int(cfg["training"]["lookback_hours"])
    data = _select_training_rows(features, outcomes, lookback_hours)

    train_enc, test_enc = split_encounters(outcomes[outcomes["censored"] == 0], cfg["training"]["test_frac"])
    train_enc, cal_enc = calibration_split(train_enc, cfg["training"]["calib_holdout_frac"])
    train_enc, cal_enc, test_enc = _restore_training_class_coverage(data, train_enc, cal_enc, test_enc, ql)

    feature_cols = resolve_feature_schema(cfg, features_df=features)
    for col in feature_cols:
        if col not in data.columns:
            data[col] = np.nan

    train_rows = data[data["encounter_id"].isin(train_enc)]
    cal_rows = data[data["encounter_id"].isin(cal_enc)]
    test_rows = data[data["encounter_id"].isin(test_enc)]

    X_train = train_rows[feature_cols]
    y_train = train_rows["ADVERSE_EVENT"].astype(int)
    X_cal = cal_rows[feature_cols]
    y_cal = cal_rows["ADVERSE_EVENT"].astype(int) if not cal_rows.empty else pd.Series(dtype=int)
    X_test = test_rows[feature_cols]
    y_test = test_rows["ADVERSE_EVENT"].astype(int)

    training_means = X_train.mean().fillna(0.0)
    X_train_filled = X_train.fillna(training_means)
    X_cal_filled = X_cal.fillna(training_means)
    X_test_filled = X_test.fillna(training_means)
    X_train_filled = X_train_filled.fillna(0.0)
    X_cal_filled = X_cal_filled.fillna(0.0)
    X_test_filled = X_test_filled.fillna(0.0)

    if y_train.nunique() < 2:
        raise ValueError("Training data must include both outcome classes")

    model, backend = _build_model(cfg, ql)
    model.fit(X_train_filled, y_train)

    calibration_method = cfg["model"].get("calibration", "isotonic")
    min_cal = int(cfg["model"].get("min_calibration_samples", 200))
    if len(X_cal_filled) < min_cal or y_cal.nunique() < 2:
        calibration_method = "sigmoid"
        ql.add("WARN", "calibration_fallback_sigmoid", reason="insufficient_calibration_samples_or_classes")

    if len(X_cal_filled) == 0 or y_cal.nunique() < 2:
        calibrator = _NoCalibrator(model)
        calibration_method = "uncalibrated"
        ql.add("WARN", "calibration_fallback_uncalibrated", reason="empty_or_single_class_calibration")
    else:
        calibrator = CalibratedClassifierCV(model, method=calibration_method, cv="prefit")
        calibrator.fit(X_cal_filled, y_cal)

    metrics: Dict[str, Any] = {
        "model_backend": backend,
        "calibration_method": calibration_method,
        "train_rows": int(len(train_rows)),
        "calibration_rows": int(len(cal_rows)),
        "test_rows": int(len(test_rows)),
        "train_encounters": int(train_rows["encounter_id"].nunique()) if not train_rows.empty else 0,
        "calibration_encounters": int(cal_rows["encounter_id"].nunique()) if not cal_rows.empty else 0,
        "test_encounters": int(test_rows["encounter_id"].nunique()) if not test_rows.empty else 0,
    }
    if len(y_test.unique()) > 1:
        preds = calibrator.predict_proba(X_test_filled)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, preds))
        metrics["brier"] = float(brier_score_loss(y_test, preds))
    else:
        metrics["roc_auc"] = None
        metrics["brier"] = None
        ql.add("WARN", "insufficient_test_size", count=int(len(y_test)))

    # operating points
    green_thr = cfg["thresholds"]["green_iri"]
    amber_thr = cfg["thresholds"]["amber_iri"]
    if len(y_test) > 0:
        preds = calibrator.predict_proba(X_test_filled)[:, 1]
        iri = (1 - preds) * 100
        metrics["green_metrics"] = _binary_metrics(y_test, iri >= green_thr)
        metrics["amber_metrics"] = _binary_metrics(y_test, iri >= amber_thr)

    bundle = {
        "base_model": model,
        "calibrator": calibrator,
        "feature_columns": feature_cols,
        "training_means": training_means.to_dict(),
        "metrics": metrics,
        "config": cfg,
    }
    return bundle, metrics


def _binary_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
    npv = tn / (tn + fn) if tn + fn > 0 else np.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def save_model_bundle(bundle: Dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
