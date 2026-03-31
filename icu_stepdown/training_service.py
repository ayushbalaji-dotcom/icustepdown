import os
import re
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from .config import load_config
from .custom_features import custom_raw_columns, custom_raw_columns_from_features
from .features import compute_features
from .io_excel import read_excel_sheets, write_excel_preserve
from .labels import build_labels
from .model_store import models_dir, set_active_model
from .preprocess import preprocess
from .quality import QualityLogger
from .schema import validate_outcomes, validate_raw
from .score import score_features
from .train import save_model_bundle, train_model


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or "icu_stepdown_model"


def _artifact_paths(input_path: str, output_dir: str, model_name: str | None) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_name = model_name or os.path.splitext(os.path.basename(input_path))[0]
    stem = _safe_stem(base_name)
    model_path = os.path.join(output_dir, f"{stem}_{timestamp}.pkl")
    report_path = os.path.join(output_dir, f"{stem}_{timestamp}_training_report.xlsx")
    return model_path, report_path


def _load_training_inputs(input_path: str, cfg: Dict[str, Any], ql: QualityLogger) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sheets = read_excel_sheets(input_path)
    if "outcomes" not in sheets:
        raise ValueError("Missing required sheet: outcomes")

    source_meta: Dict[str, Any] = {"input_sheets": sorted(sheets.keys())}
    if "raw_icu_data" in sheets:
        raw = sheets["raw_icu_data"]
        raw, _ = validate_raw(raw, cfg, ql)
        raw = preprocess(raw, cfg, ql)
        features = compute_features(raw, cfg, ql)
        source_meta["raw_row_count"] = int(len(raw))
        source_meta["custom_clinical_variables"] = custom_raw_columns(raw)
    elif "features_4h" in sheets:
        features = sheets["features_4h"].copy()
        source_meta["custom_clinical_variables"] = custom_raw_columns_from_features(features)
    else:
        raise ValueError("Training workbook must contain raw_icu_data or features_4h")

    outcomes = sheets["outcomes"].copy()
    validate_outcomes(outcomes, cfg)
    outcomes = build_labels(outcomes, cfg, ql)
    source_meta["feature_row_count"] = int(len(features))
    source_meta["encounter_count"] = int(outcomes["encounter_id"].nunique())
    return features, outcomes, source_meta


def _summary(
    input_path: str,
    model_path: str,
    report_path: str,
    metrics: Dict[str, Any],
    outcomes: pd.DataFrame,
    dashboard: pd.DataFrame,
    ql: QualityLogger,
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    usable = outcomes[outcomes["censored"] == 0].copy()
    trend_counts = {}
    if not dashboard.empty and "trend_label" in dashboard.columns:
        trend_counts = {str(k): int(v) for k, v in dashboard["trend_label"].value_counts(dropna=False).to_dict().items()}
    traffic_counts = {}
    if not dashboard.empty and "traffic_light" in dashboard.columns:
        traffic_counts = {str(k): int(v) for k, v in dashboard["traffic_light"].value_counts(dropna=False).to_dict().items()}

    return {
        "source_file": os.path.basename(input_path),
        "model_path": os.path.abspath(model_path),
        "report_path": os.path.abspath(report_path),
        "encounter_count": int(outcomes["encounter_id"].nunique()),
        "trainable_encounters": int((outcomes["censored"] == 0).sum()),
        "censored_encounters": int(outcomes["censored"].sum()),
        "positive_outcomes": int(usable["ADVERSE_EVENT"].sum()) if not usable.empty else 0,
        "negative_outcomes": int((usable["ADVERSE_EVENT"] == 0).sum()) if not usable.empty else 0,
        "traffic_counts": traffic_counts,
        "trend_label_counts": trend_counts,
        "quality_events": int(len(ql.entries)),
        "rejected_rows": int(len(ql.rejected_rows)),
        "metrics": metrics,
        **source_meta,
    }


def train_workbook(
    input_path: str,
    *,
    config_path: str = "configs/default.yaml",
    output_dir: str | None = None,
    model_name: str | None = None,
    activate: bool = True,
    base_dir: str = "database",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    ql = QualityLogger()
    features, outcomes, source_meta = _load_training_inputs(input_path, cfg, ql)
    bundle, metrics = train_model(features, outcomes, cfg, ql)

    artifact_dir = output_dir or models_dir(base_dir)
    model_path, report_path = _artifact_paths(input_path, artifact_dir, model_name)
    save_model_bundle(bundle, model_path)

    scores, signals, dashboard, _ = score_features(features, bundle, cfg, ql, force_schema=True)
    write_excel_preserve(
        input_path,
        report_path,
        {
            "features_4h": features,
            "scores_4h": scores,
            "signals_explained": signals,
            "dashboard": dashboard,
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        },
    )

    summary = _summary(input_path, model_path, report_path, metrics, outcomes, dashboard, ql, source_meta)
    active_model = None
    if activate:
        active_model = set_active_model(
            model_path,
            base_dir=base_dir,
            metrics=metrics,
            metadata={
                "report_path": os.path.abspath(report_path),
                "source_file": os.path.basename(input_path),
                "summary": summary,
                "custom_clinical_variables": source_meta.get("custom_clinical_variables", []),
            },
        )

    return {
        "model_path": os.path.abspath(model_path),
        "report_path": os.path.abspath(report_path),
        "metrics": metrics,
        "summary": summary,
        "quality_log": ql.to_dataframe(),
        "rejected_rows": ql.rejected_to_dataframe(),
        "dashboard": dashboard,
        "scores": scores,
        "signals": signals,
        "features": features,
        "active_model": active_model,
    }
