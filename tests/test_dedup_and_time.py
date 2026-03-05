import pandas as pd

from icu_stepdown.schema import validate_raw
from icu_stepdown.preprocess import preprocess
from icu_stepdown.quality import QualityLogger


def test_dedup_and_time_parsing(config):
    ql = QualityLogger()
    # Excel serial date for 2025-01-01 is 45658
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1"],
        "encounter_id": ["e1", "e1", "e1"],
        "timestamp": [45658, 45658, "2025-01-01T00:00:00+00:00"],
        "MAP": [70, None, 75],
        "HR": [80, 85, 90],
    })
    df, _ = validate_raw(df, config, ql)
    df = preprocess(df, config, ql)
    # Dedupe: last non-null wins for HR/MAP
    assert df.shape[0] == 1
    assert df.iloc[0]["MAP"] == 75
    assert df.iloc[0]["HR"] == 90
