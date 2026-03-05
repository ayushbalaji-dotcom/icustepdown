from typing import Dict, List, Tuple

import pandas as pd


def split_encounters(outcomes: pd.DataFrame, test_frac: float) -> Tuple[List[str], List[str]]:
    # outcomes must have patient_id, encounter_id, icu_discharge_time
    df = outcomes.sort_values("icu_discharge_time")
    encounter_ids = df["encounter_id"].tolist()
    n_test = max(1, int(len(encounter_ids) * test_frac)) if encounter_ids else 0
    test_encounters = set(encounter_ids[-n_test:])

    # enforce patient-level separation
    test_patients = set(df[df["encounter_id"].isin(test_encounters)]["patient_id"].unique())
    # move all encounters of test patients to test
    test_encounters = set(df[df["patient_id"].isin(test_patients)]["encounter_id"].unique())

    train_encounters = [e for e in encounter_ids if e not in test_encounters]
    test_encounters = [e for e in encounter_ids if e in test_encounters]
    return train_encounters, test_encounters


def calibration_split(train_encounters: List[str], frac: float) -> Tuple[List[str], List[str]]:
    if len(train_encounters) < 2:
        return train_encounters, []
    n_cal = max(1, int(len(train_encounters) * frac)) if train_encounters else 0
    cal = train_encounters[-n_cal:]
    train = train_encounters[:-n_cal] if n_cal > 0 else train_encounters
    return train, cal

