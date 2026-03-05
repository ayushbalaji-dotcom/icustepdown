import pandas as pd

from icu_stepdown.split import split_encounters, calibration_split


def test_leakage_split():
    outcomes = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p3"],
        "encounter_id": ["e1", "e2", "e3", "e4"],
        "icu_discharge_time": pd.to_datetime([
            "2025-01-01 10:00",
            "2025-01-02 10:00",
            "2025-01-03 10:00",
            "2025-01-04 10:00",
        ]),
    })
    train_enc, test_enc = split_encounters(outcomes, 0.25)
    # patient p1 has two encounters, should be fully in test or train
    in_train = any(e in train_enc for e in ["e1", "e2"])
    in_test = any(e in test_enc for e in ["e1", "e2"])
    assert not (in_train and in_test)

    train_only, cal = calibration_split(train_enc, 0.5)
    assert set(train_only).isdisjoint(set(cal))

