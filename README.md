# ICU→HDU Step-Down Readiness (Decision Support Backend)

This repo provides a production-grade, backend-only pipeline for ICU→HDU step-down readiness decision support. It **does not** automate transfer decisions. It outputs a traffic light, trajectory, and explanations with a conservative, fail-closed posture.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Commands

```bash
icu-stepdown validate --input in.xlsx --output validated.xlsx
icu-stepdown features --input in.xlsx --output with_features.xlsx
icu-stepdown train --input with_features.xlsx --model-out model.pkl
icu-stepdown score --input with_features.xlsx --model-in model.pkl --output scored.xlsx
icu-stepdown run-all --input in.xlsx --output scored.xlsx --model-out model.pkl
icu-stepdown serve --host 127.0.0.1 --port 8000 --model model.pkl --db-path database/icu_stepdown.sqlite
icu-stepdown score --input with_features.xlsx --output scored.xlsx --baseline
icu-stepdown serve --host 127.0.0.1 --port 8000 --db-path database/icu_stepdown.sqlite --baseline
```

If the console script is not installed, use:

```bash
python -m icu_stepdown validate --input in.xlsx --output validated.xlsx
```

Flags:
- `--config configs/default.yaml`
- `--debug` (prints stack traces)
- `--force-schema` (allows scoring if feature schema mismatch; still fail-closed on missing core fields)
- `serve` uses SQLite at `database/icu_stepdown.sqlite` by default.

## Excel Workbook Contract

Input workbook must contain:
- `raw_icu_data` (required)
- `outcomes` (required for training; optional for scoring-only)

`raw_icu_data` required columns:
- `patient_id` (string)
- `encounter_id` (string) **required; scoring/training will refuse without it**
- `timestamp` (datetime)

Optional numeric columns (created as NaN if missing):
- `MAP`, `HR`, `RR`, `SpO2`, `FiO2`
- `noradrenaline_mcgkgmin`, `adrenaline_mcgkgmin`, `dobutamine_mcgkgmin`, `milrinone_mcgkgmin`
- `urine_output_ml_30min`, `chest_drain_ml_30min`, `lactate`, `haemoglobin_gL`, `creatinine_umolL`, `WCC_10e9L`, `temperature_C`, `RASS`

Optional categorical/binary:
- `oxygen_device` (`ETT`,`NIV`,`HFNC`,`Venturi`,`Mask`,`NC`,`RA` or local variants)
- `arterial_line_present`, `central_line_present`, `insulin_infusion`, `pacing_active`

`outcomes` required columns:
- `patient_id`, `encounter_id`, `icu_discharge_time`
- Either `ADVERSE_EVENT` or any subset of component flags (`readmitted_within_48h`, `MET_call_within_48h`, `pressor_restart_48h`, `reintubation_or_NIV_escalation_48h`, `death_48h`)
- Optional: `planned_readmission_48h`, `no_stepdown`, `reason_no_stepdown`

## Outputs

Always written to the output workbook (preserving existing sheets):
- `features_4h`, `scores_4h`, `signals_explained`, `dashboard`, `quality_log`, `rejected_rows`, `run_manifest` (sheet or JSON alongside)

## Frontend (Local)

Run a lightweight local UI:

```bash
icu-stepdown serve --host 127.0.0.1 --port 8000 --model model.pkl --db-path database/icu_stepdown.sqlite
```

The UI stores hourly entries per patient in SQLite at `database/icu_stepdown.sqlite`.
Scores are produced only after 4 hours of data are present.

## Streamlit Hosting

Run with Streamlit:

```bash
streamlit run streamlit_app.py
```

Set login credentials:

```bash
export ICU_APP_USER="your_user"
export ICU_APP_PASS="your_pass"
```

The Streamlit app creates a per-patient subdirectory under `database/<NHS_NUMBER>/icu_stepdown.sqlite`.

## Hard Stops vs ML

Hard-stop safety rules **override ML**. If any hard stop is true, the traffic light is **RED** regardless of model output. Unknown or stale data is never allowed to produce GREEN (fail-closed gating).

## Baseline (No-ML) Beta Test

Use the baseline heuristic model for pre-training validation:

```bash
icu-stepdown score --input with_features.xlsx --output scored.xlsx --baseline
icu-stepdown serve --host 127.0.0.1 --port 8000 --db-path database/icu_stepdown.sqlite --baseline
```

Baseline penalties live under `baseline:` in `configs/default.yaml` and use the same thresholds and hard-stop rules.

## IRI Meaning & Calibration

`p_adverse` is a calibrated probability of 48h deterioration after step-down. `IRI = (1 - p_adverse) * 100` is a readiness index on 0–100. Calibration uses isotonic regression when possible, with sigmoid fallback if the calibration holdout is too small.

## Governance Disclaimer

This tool provides decision support **only**. It does not automate transfers. Consultant sign-off is required. Thresholds are conservative. Track 48h readmission and deterioration as the safety outcome.

## Training Policy

The ML model is trained on objective post-step-down outcomes (e.g., deterioration within 48h). It does not use consultant decision data or operational signals (bed/staffing/ward timing) to avoid behavior cloning and leakage.

## Tuning

All thresholds, ranges, and mapping rules live in `configs/default.yaml`. Adjust only with audit and clinical governance.

## Drift Policy

Model bundles are versioned and locked. There is **no silent drift**. Retraining requires the explicit `train` or `run-all` command and generates audit artifacts (run manifest + quality log).
# icustepdown
