from typing import Any, Dict

import pandas as pd

from .quality import QualityLogger


def build_labels(outcomes: pd.DataFrame, cfg: Dict[str, Any], ql: QualityLogger) -> pd.DataFrame:
    outcomes = outcomes.copy()
    # Validate required columns
    required = cfg["outcomes_columns"]["required"]
    for col in required:
        if col not in outcomes.columns:
            if col == "encounter_id":
                raise ValueError("Missing required outcomes column: encounter_id")
            raise ValueError(f"Missing required outcomes column: {col}")

    if "ADVERSE_EVENT" in outcomes.columns:
        outcomes["ADVERSE_EVENT"] = outcomes["ADVERSE_EVENT"].fillna(0).astype(int)
    else:
        flags = cfg["outcomes_columns"]["component_flags"]
        present_flags = [f for f in flags if f in outcomes.columns]
        if not present_flags:
            raise ValueError("No outcome flags available to build ADVERSE_EVENT")
        outcomes["ADVERSE_EVENT"] = outcomes[present_flags].fillna(0).astype(int).max(axis=1)

    if "planned_readmission_48h" in outcomes.columns:
        mask = outcomes["planned_readmission_48h"] == 1
        if mask.any():
            outcomes.loc[mask, "ADVERSE_EVENT"] = 0
            ql.add("WARN", "planned_readmission_excluded", count=int(mask.sum()))

    # Censoring
    censored = pd.Series(False, index=outcomes.index)
    if "no_stepdown" in outcomes.columns:
        censored = censored | (outcomes["no_stepdown"] == 1)
    censored = censored | outcomes["icu_discharge_time"].isna()

    if censored.any():
        ql.add("INFO", "censored_encounters", count=int(censored.sum()))

    outcomes["censored"] = censored.astype(int)
    return outcomes


