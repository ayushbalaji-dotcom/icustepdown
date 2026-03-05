from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


def make_raw_icu_data(n_points: int = 20, start: datetime | None = None, patient_id: str = "p1", encounter_id: str = "e1") -> pd.DataFrame:
    start = start or datetime(2025, 1, 1, 0, 0, 0)
    times = [start]
    for i in range(1, n_points):
        times.append(times[-1] + timedelta(minutes=17 + (i % 2) * 26))

    df = pd.DataFrame({
        "patient_id": [patient_id] * n_points,
        "encounter_id": [encounter_id] * n_points,
        "timestamp": times,
        "MAP": np.linspace(75, 80, n_points),
        "HR": np.linspace(90, 85, n_points),
        "RR": np.linspace(18, 16, n_points),
        "SpO2": np.linspace(0.96, 0.98, n_points),
        "FiO2": np.linspace(0.4, 0.35, n_points),
        "noradrenaline_mcgkgmin": np.zeros(n_points),
        "adrenaline_mcgkgmin": np.zeros(n_points),
        "dobutamine_mcgkgmin": np.zeros(n_points),
        "milrinone_mcgkgmin": np.zeros(n_points),
        "urine_output_ml_30min": np.linspace(60, 80, n_points),
        "chest_drain_ml_30min": np.linspace(10, 5, n_points),
        "lactate": np.linspace(1.2, 1.0, n_points),
        "haemoglobin_gL": np.linspace(110, 112, n_points),
        "creatinine_umolL": np.linspace(100, 95, n_points),
        "WCC_10e9L": np.linspace(9, 8.5, n_points),
        "temperature_C": np.linspace(37.2, 37.0, n_points),
        "RASS": np.zeros(n_points),
        "oxygen_device": ["NC"] * n_points,
        "arterial_line_present": [1] * n_points,
        "central_line_present": [1] * n_points,
        "insulin_infusion": [0] * n_points,
        "pacing_active": [0] * n_points,
    })
    return df


def make_outcomes(discharge_time: datetime | None = None, adverse: int = 0) -> pd.DataFrame:
    discharge_time = discharge_time or datetime(2025, 1, 1, 12, 0, 0)
    return pd.DataFrame({
        "patient_id": ["p1"],
        "encounter_id": ["e1"],
        "icu_discharge_time": [discharge_time],
        "ADVERSE_EVENT": [adverse],
    })


def make_two_patient_outcomes() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "encounter_id": ["e1", "e2"],
        "icu_discharge_time": [
            datetime(2025, 1, 1, 12, 0, 0),
            datetime(2025, 1, 2, 12, 0, 0),
        ],
        "ADVERSE_EVENT": [0, 1],
    })

