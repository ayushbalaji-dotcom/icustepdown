import os

import numpy as np
import pandas as pd

from icu_stepdown.model_store import active_model_record
from icu_stepdown.training_service import train_workbook
from tests.synthetic_data import make_raw_icu_data


def _encounter(patient_idx: int, adverse: bool) -> pd.DataFrame:
    raw = make_raw_icu_data(
        start=pd.Timestamp(2025, 1, patient_idx, 0, 0),
        patient_id=f"p{patient_idx}",
        encounter_id=f"e{patient_idx}",
    )
    if adverse:
        n_rows = len(raw)
        raw["FiO2"] = np.linspace(0.4, 0.65, n_rows)
        raw["SpO2"] = np.linspace(96, 92, n_rows)
        raw["noradrenaline_mcgkgmin"] = np.linspace(0.02, 0.06, n_rows)
        raw["urine_output_ml_30min"] = np.linspace(45, 20, n_rows)
        raw["lactate"] = np.linspace(1.6, 2.6, n_rows)
        raw["oxygen_device"] = ["HFNC"] * n_rows
    return raw


def test_train_workbook_activates_model(tmp_path):
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

    input_path = tmp_path / "training_input.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="raw_icu_data", index=False)
        outcomes.to_excel(writer, sheet_name="outcomes", index=False)

    result = train_workbook(
        str(input_path),
        config_path="configs/default.yaml",
        base_dir=str(tmp_path),
        model_name="integration_test_model",
        activate=True,
    )

    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["report_path"])
    assert result["summary"]["positive_outcomes"] == 3
    assert result["summary"]["negative_outcomes"] == 3
    assert result["summary"]["metrics"]["train_rows"] > 0

    active = active_model_record(str(tmp_path))
    assert active["available"] is True
    assert active["model_path"] == os.path.abspath(result["model_path"])
