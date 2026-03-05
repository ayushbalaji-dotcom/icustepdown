import pandas as pd
import pytest

from icu_stepdown.schema import validate_raw
from icu_stepdown.quality import QualityLogger


def test_missing_encounter_id_refused(config):
    ql = QualityLogger()
    df = pd.DataFrame({
        "patient_id": ["p1"],
        "timestamp": ["2025-01-01 00:00"],
    })
    with pytest.raises(ValueError):
        validate_raw(df, config, ql)

