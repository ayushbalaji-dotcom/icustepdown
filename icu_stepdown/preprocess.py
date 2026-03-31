from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .quality import QualityLogger


def _clamp_plausible(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    ranges = cfg.get("plausible_ranges", {})
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                df.loc[bad, col] = np.nan
                ql.add("WARN", "value_out_of_range", column=col, count=int(bad.sum()))
    # Negative doses/outputs -> NaN
    for col in df.columns:
        if col.endswith("_mcgkgmin") or col.endswith("_ml_30min"):
            neg = df[col] < 0
            if neg.any():
                df.loc[neg, col] = np.nan
                ql.add("WARN", "negative_value", column=col, count=int(neg.sum()))
    return df


def _unit_normalize(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    unit_autofix = bool(cfg.get("unit_autofix", True))
    if "FiO2" in df.columns:
        mask = df["FiO2"].notna() & (df["FiO2"] > 1.2) & (df["FiO2"] <= 100)
        if mask.any():
            if unit_autofix:
                df.loc[mask, "FiO2"] = df.loc[mask, "FiO2"] / 100.0
                ql.add("INFO", "unit_fix_fio2", count=int(mask.sum()))
            else:
                df.loc[mask, "FiO2"] = np.nan
                ql.add("WARN", "unit_suspected_fio2", count=int(mask.sum()))
    if "SpO2" in df.columns:
        mask = df["SpO2"].notna() & (df["SpO2"] <= 1.2)
        if mask.any():
            if unit_autofix:
                df.loc[mask, "SpO2"] = df.loc[mask, "SpO2"] * 100.0
                ql.add("INFO", "unit_fix_spo2", count=int(mask.sum()))
            else:
                df.loc[mask, "SpO2"] = np.nan
                ql.add("WARN", "unit_suspected_spo2", count=int(mask.sum()))
    if "temperature_C" in df.columns:
        mask = df["temperature_C"].notna() & (df["temperature_C"] > 45) & (df["temperature_C"] <= 110)
        if mask.any():
            if unit_autofix:
                df.loc[mask, "temperature_C"] = (df.loc[mask, "temperature_C"] - 32) * 5.0 / 9.0
                ql.add("INFO", "unit_fix_temp_f", count=int(mask.sum()))
            else:
                df.loc[mask, "temperature_C"] = np.nan
                ql.add("WARN", "unit_suspected_temp", count=int(mask.sum()))
    return df


def _map_oxygen_device(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    mapping = {k.upper(): v for k, v in cfg.get("oxygen_device_map", {}).items()}
    if "oxygen_device" in df.columns:
        def map_val(x: Any) -> Any:
            if pd.isna(x):
                return np.nan
            key = str(x).strip().upper()
            return mapping.get(key, np.nan)

        df["resp_support_level"] = df["oxygen_device"].apply(map_val)
        unknown = df["oxygen_device"].notna() & df["resp_support_level"].isna()
        if unknown.any():
            ql.add("WARN", "unknown_oxygen_device", count=int(unknown.sum()))
    else:
        df["resp_support_level"] = np.nan
    return df


def _coerce_custom_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if str(col).startswith("custom_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dedupe_and_sort(df: pd.DataFrame, ql: QualityLogger) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "encounter_id", "timestamp"])
    # merge duplicates by last non-null wins
    key_cols = ["patient_id", "encounter_id", "timestamp"]
    if df.duplicated(key_cols).any():
        ql.add("INFO", "duplicates_found", count=int(df.duplicated(key_cols).sum()))
        df = (
            df.groupby(key_cols, as_index=False)
            .agg(lambda s: s.dropna().iloc[-1] if s.dropna().size > 0 else np.nan)
        )
    return df


def preprocess(df: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    df = dedupe_and_sort(df, ql)
    df = _coerce_custom_columns(df)
    df = _unit_normalize(df, cfg, ql)
    df = _clamp_plausible(df, cfg, ql)
    df = _map_oxygen_device(df, cfg, ql)
    return df

