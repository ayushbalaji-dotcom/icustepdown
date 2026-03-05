import os
import tempfile

import pandas as pd

from icu_stepdown.config import load_config
from icu_stepdown.features import compute_features
from icu_stepdown.io_excel import write_excel_preserve, read_excel_sheets
from icu_stepdown.labels import build_labels
from icu_stepdown.preprocess import preprocess
from icu_stepdown.quality import QualityLogger
from icu_stepdown.schema import validate_raw, validate_outcomes
from icu_stepdown.score import score_features
from icu_stepdown.train import train_model
from tests.synthetic_data import make_raw_icu_data


def test_end_to_end():
    cfg = load_config("configs/default.yaml")
    ql = QualityLogger()
    raw1 = make_raw_icu_data(patient_id="p1", encounter_id="e1")
    raw2 = make_raw_icu_data(start=pd.Timestamp("2025-01-02 00:00"), patient_id="p2", encounter_id="e2")
    raw = pd.concat([raw1, raw2], ignore_index=True)
    outcomes = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "encounter_id": ["e1", "e2"],
        "icu_discharge_time": [pd.Timestamp("2025-01-01 12:00"), pd.Timestamp("2025-01-02 12:00")],
        "ADVERSE_EVENT": [0, 1],
    })

    raw, _ = validate_raw(raw, cfg, ql)
    raw = preprocess(raw, cfg, ql)
    feat = compute_features(raw, cfg, ql)
    validate_outcomes(outcomes, cfg)
    outcomes = build_labels(outcomes, cfg, ql)

    bundle, _ = train_model(feat, outcomes, cfg, ql)
    scores, signals, dashboard, _ = score_features(feat, bundle, cfg, ql, force_schema=True)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.xlsx")
        out_path = os.path.join(tmp, "output.xlsx")
        with pd.ExcelWriter(in_path, engine="openpyxl") as writer:
            raw.to_excel(writer, sheet_name="raw_icu_data", index=False)
            outcomes.to_excel(writer, sheet_name="outcomes", index=False)

        write_excel_preserve(in_path, out_path, {
            "features_4h": feat,
            "scores_4h": scores,
            "signals_explained": signals,
            "dashboard": dashboard,
            "quality_log": ql.to_dataframe(),
            "rejected_rows": ql.rejected_to_dataframe(),
        })
        sheets = read_excel_sheets(out_path)
        for name in ["features_4h", "scores_4h", "signals_explained", "dashboard", "quality_log", "rejected_rows"]:
            assert name in sheets
