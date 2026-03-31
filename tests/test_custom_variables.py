from datetime import datetime

import numpy as np
import pandas as pd

from icu_stepdown.model_store import active_model_record
from icu_stepdown.ops_store import (
    get_latest_custom_variable_values,
    init_ops_db,
    list_custom_variable_definitions,
    save_custom_variable_definitions,
    save_custom_variable_values,
)
from icu_stepdown.patient_store import append_row, load_rows
from icu_stepdown.train import load_model_bundle
from icu_stepdown.training_service import train_workbook
from tests.synthetic_data import make_raw_icu_data


def test_custom_variable_definitions_and_values_round_trip(tmp_path):
    db = tmp_path / "ops.sqlite"
    init_ops_db(str(db))

    save_custom_variable_definitions(
        str(db),
        "ward",
        [
            {"label": "Male bay beds", "data_type": "number", "unit": "beds", "active": True},
            {"label": "Female bay open", "data_type": "boolean", "active": True},
            {"label": "Protected isolation bay", "data_type": "text", "active": False},
        ],
        user="tester",
    )

    definitions = list_custom_variable_definitions(str(db), "ward")
    assert [item["slug"] for item in definitions] == [
        "custom_female_bay_open",
        "custom_male_bay_beds",
        "custom_protected_isolation_bay",
    ]
    active = list_custom_variable_definitions(str(db), "ward", active_only=True)
    assert [item["slug"] for item in active] == [
        "custom_female_bay_open",
        "custom_male_bay_beds",
    ]

    save_custom_variable_values(
        str(db),
        "ward",
        {
            "custom_male_bay_beds": 18.0,
            "custom_female_bay_open": 1.0,
            "custom_protected_isolation_bay": "Closed",
        },
        user="tester",
    )
    values = get_latest_custom_variable_values(str(db), "ward")
    assert values["custom_male_bay_beds"] == 18.0
    assert values["custom_female_bay_open"] == 1.0
    assert values["custom_protected_isolation_bay"] == "Closed"


def test_patient_store_round_trips_custom_clinical_values(tmp_path):
    db = tmp_path / "patient.sqlite"
    append_row(
        str(db),
        "1234567890",
        {
            "timestamp": datetime(2025, 1, 1, 8, 0).isoformat(),
            "MAP": 75.0,
            "custom_lung_ultrasound_score": 3.0,
            "custom_secretions_high": 1.0,
        },
    )

    rows = load_rows(str(db), "1234567890")
    assert len(rows) == 1
    assert rows[0]["custom_lung_ultrasound_score"] == 3.0
    assert rows[0]["custom_secretions_high"] == 1.0


def _encounter(patient_idx: int, adverse: bool) -> pd.DataFrame:
    raw = make_raw_icu_data(
        start=pd.Timestamp(2025, 1, patient_idx, 0, 0),
        patient_id=f"p{patient_idx}",
        encounter_id=f"e{patient_idx}",
    )
    n_rows = len(raw)
    if adverse:
        raw["FiO2"] = np.linspace(0.45, 0.7, n_rows)
        raw["SpO2"] = np.linspace(96, 91, n_rows)
        raw["custom_lung_ultrasound_score"] = np.linspace(6, 10, n_rows)
        raw["custom_secretions_high"] = np.ones(n_rows)
    else:
        raw["custom_lung_ultrasound_score"] = np.linspace(1, 3, n_rows)
        raw["custom_secretions_high"] = np.zeros(n_rows)
    return raw


def test_training_workbook_includes_custom_clinical_features(tmp_path):
    raw = pd.concat(
        [_encounter(idx, adverse=idx % 2 == 0) for idx in range(1, 7)],
        ignore_index=True,
    )
    outcomes = pd.DataFrame({
        "patient_id": [f"p{idx}" for idx in range(1, 7)],
        "encounter_id": [f"e{idx}" for idx in range(1, 7)],
        "icu_discharge_time": [pd.Timestamp(2025, 1, idx, 12, 0) for idx in range(1, 7)],
        "ADVERSE_EVENT": [0, 1, 0, 1, 0, 1],
    })

    input_path = tmp_path / "custom_training_input.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="raw_icu_data", index=False)
        outcomes.to_excel(writer, sheet_name="outcomes", index=False)

    result = train_workbook(
        str(input_path),
        config_path="configs/default.yaml",
        base_dir=str(tmp_path),
        model_name="custom_feature_model",
        activate=True,
    )

    bundle = load_model_bundle(result["model_path"])
    assert "custom_lung_ultrasound_score_now" in bundle["feature_columns"]
    assert "custom_secretions_high_missing_4h" in bundle["feature_columns"]

    active = active_model_record(str(tmp_path))
    assert sorted(active["custom_clinical_variables"]) == [
        "custom_lung_ultrasound_score",
        "custom_secretions_high",
    ]
