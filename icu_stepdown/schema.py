from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .quality import QualityLogger


EXCEL_EPOCH = "1899-12-30"


def _parse_timestamp_series(series: pd.Series, ql: QualityLogger) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    # Normalize to timezone-naive
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
    except Exception:
        pass
    # Handle Excel serial dates for numeric values that failed
    mask = parsed.isna() & series.notna()
    if mask.any():
        numeric = pd.to_numeric(series[mask], errors="coerce")
        serial_mask = numeric.notna()
        if serial_mask.any():
            parsed.loc[mask[mask].index[serial_mask]] = pd.to_datetime(
                numeric[serial_mask], unit="d", origin=EXCEL_EPOCH, errors="coerce"
            )
    return parsed


def validate_raw(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> Tuple[pd.DataFrame, bool]:
    required = cfg["columns"]["required"]
    optional_num = cfg["columns"]["optional_numeric"]
    optional_cat = cfg["columns"]["optional_categorical"]

    for col in required:
        if col not in df.columns:
            if col == "encounter_id":
                raise ValueError("Missing required column: encounter_id")
            raise ValueError(f"Missing required column: {col}")

    for col in optional_num + optional_cat:
        if col not in df.columns:
            df[col] = np.nan

    df["timestamp"] = _parse_timestamp_series(df["timestamp"], ql)

    # Reject rows with missing timestamp
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        for _, row in df[bad_ts].iterrows():
            ql.reject_row(row.to_dict(), "timestamp_parse_failed")
        ql.add("WARN", "timestamp_parse_failed", count=int(bad_ts.sum()))
    df = df[~bad_ts].copy()

    return df, True


def validate_outcomes(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    required = cfg["outcomes_columns"]["required"]
    for col in required:
        if col not in df.columns:
            if col == "encounter_id":
                raise ValueError("Missing required outcomes column: encounter_id")
            raise ValueError(f"Missing required outcomes column: {col}")
