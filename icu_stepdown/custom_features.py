import json
import re
from typing import Any, Dict, Iterable, List

import pandas as pd


CUSTOM_PREFIX = "custom_"
CUSTOM_FEATURE_SUFFIXES = ("now", "mean_4h", "slope_4h", "missing_4h")
_CUSTOM_FEATURE_RE = re.compile(r"^(custom_[a-z0-9_]+)_(now|mean_4h|slope_4h|missing_4h)$")


def normalize_custom_variable_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not slug:
        raise ValueError("Variable name is required")
    if not slug.startswith(CUSTOM_PREFIX):
        slug = f"{CUSTOM_PREFIX}{slug}"
    return slug


def custom_raw_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if not str(col).startswith(CUSTOM_PREFIX):
            continue
        if _CUSTOM_FEATURE_RE.match(str(col)):
            continue
        cols.append(str(col))
    return sorted(dict.fromkeys(cols))


def custom_feature_columns_from_raw_columns(raw_columns: Iterable[str]) -> List[str]:
    columns: List[str] = []
    for raw_col in raw_columns:
        base = str(raw_col)
        columns.extend([
            f"{base}_now",
            f"{base}_mean_4h",
            f"{base}_slope_4h",
            f"{base}_missing_4h",
        ])
    return columns


def custom_feature_columns_from_features(df: pd.DataFrame) -> List[str]:
    cols = [str(col) for col in df.columns if _CUSTOM_FEATURE_RE.match(str(col))]
    return sorted(dict.fromkeys(cols))


def custom_raw_columns_from_features(df: pd.DataFrame) -> List[str]:
    roots = []
    for col in custom_feature_columns_from_features(df):
        match = _CUSTOM_FEATURE_RE.match(col)
        if match:
            roots.append(match.group(1))
    return sorted(dict.fromkeys(roots))


def resolve_feature_schema(cfg: Dict[str, Any], *, raw_df: pd.DataFrame | None = None, features_df: pd.DataFrame | None = None) -> List[str]:
    base = list(cfg.get("feature_schema", []) or [])
    extras: List[str] = []
    if raw_df is not None and not raw_df.empty:
        extras.extend(custom_feature_columns_from_raw_columns(custom_raw_columns(raw_df)))
    if features_df is not None and not features_df.empty:
        extras.extend(custom_feature_columns_from_features(features_df))

    seen = set()
    ordered: List[str] = []
    for col in base + sorted(dict.fromkeys(extras)):
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def expand_custom_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "custom_data" not in df.columns:
        return df

    expanded = df.copy()
    custom_values: Dict[str, List[Any]] = {}
    for payload in expanded["custom_data"]:
        parsed: Dict[str, Any] = {}
        if isinstance(payload, dict):
            parsed = payload
        elif isinstance(payload, str) and payload.strip():
            try:
                loaded = json.loads(payload)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}
        for key in parsed:
            custom_values.setdefault(key, [])

    if not custom_values:
        return expanded

    for key in custom_values:
        expanded[key] = pd.NA

    for idx, payload in expanded["custom_data"].items():
        parsed: Dict[str, Any] = {}
        if isinstance(payload, dict):
            parsed = payload
        elif isinstance(payload, str) and payload.strip():
            try:
                loaded = json.loads(payload)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}
        for key, value in parsed.items():
            expanded.at[idx, key] = value

    return expanded


def custom_feature_label(name: str) -> str:
    text = str(name)
    match = _CUSTOM_FEATURE_RE.match(text)
    base = match.group(1) if match else text
    if base.startswith(CUSTOM_PREFIX):
        base = base[len(CUSTOM_PREFIX):]
    return base.replace("_", " ").strip().title()
