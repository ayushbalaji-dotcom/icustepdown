# ICU->HDU Step-Down Readiness Algorithm Protocol (Draft)

**Version:** 0.1  
**Date:** 2026-03-04  
**Owner:** ICU Step-Down Project Team  
**Status:** Draft for review

## Purpose
Provide structured decision support for ICU->HDU step-down readiness. The tool highlights safety "hard stops," summarizes risk, and supports clinician review. It does not automate decisions.

## Scope
Applies to adult ICU encounters where step-down is being considered. Use alongside standard clinical assessment and local policy. Consultant sign-off remains required.

## Inputs and Required Data
### Minimum to Run Hard-Stops-Only Mode
Required identifiers:
- `patient_id`
- `encounter_id`
- `timestamp`

Required or strongly recommended clinical measures for minimum safe scoring:
- Blood pressure: `MAP` (mmHg)
- Heart rate: `HR`
- Pressor infusions: `noradrenaline_mcgkgmin`, `adrenaline_mcgkgmin`, `dobutamine_mcgkgmin`, `milrinone_mcgkgmin`
- Respiratory: `FiO2`, `SpO2`, `oxygen_device`
- Lactate: `lactate`
- Bleeding: `chest_drain_ml_30min`, `haemoglobin_gL`
- Neuro: `RASS`
- Pacing status: `pacing_active`

### Full Model Feature Set (for ML training/scoring)
All fields listed in `configs/default.yaml` under `columns.required`, `columns.optional_numeric`, and `columns.optional_categorical`.

### Outcomes for Training (if/when ML is trained)
Required outcome fields per encounter from `configs/default.yaml` under `outcomes_columns`, including:
- `icu_discharge_time`
- `ADVERSE_EVENT` or component outcomes (e.g., readmission, MET call, reintubation/NIV escalation, death within 48h)

## Data Preprocessing
All incoming data is standardized before feature generation:
- Unit normalization (e.g., FiO2 as fraction, SpO2 as %, temperature in C).
- Plausible range checks with out-of-range values set to missing.
- Oxygen device mapping to a respiratory support level.
- Deduplication by `(patient_id, encounter_id, timestamp)` with last non-null wins.

## Feature Generation
Features are calculated over rolling windows (default 4h, with some 6-24h windows) at aligned score times:
- Haemodynamic, respiratory, bleeding, renal/perfusion, neuro/infection, device dependency, recency, and missingness flags.
- Full list is documented in `docs/feature_spec.md`.

## Hard-Stop Safety Rules (Override All Scoring)
The following rules are evaluated at each score time. Any positive rule produces a **RED** decision:
- Pressors escalating: `pressor_on == 1` and `pressor_escalating == 1`
- Oxygen worsening: FiO2 slope rising, SpO2 time >=94% below threshold, or respiratory support level increasing
- Lactate rising: lactate slope rising **and** lactate above threshold
- Bleeding: drain sum high, drain slope rising, or haemoglobin drop beyond threshold
- Neuro: RASS >=2 or <=-4

Default thresholds are configured in `configs/default.yaml` under `hard_stops`:
- `fiO2_slope_per_hr`: 0.01
- `spO2_time_ge_94`: 0.80
- `resp_support_slope`: 0
- `lactate_slope_per_hr`: 0.05
- `lactate_now_threshold`: 2.0
- `drain_sum_4h`: 200
- `hb_drop_6h`: -10

## Scoring Logic
### Standard ML Mode
- ML model predicts adverse-event probability.
- IRI (Inverse Risk Index) = `(1 - p_adverse) * 100`.
Traffic light rules (ML mode):
- **GREEN** if IRI >= green threshold and data quality OK
- **AMBER** if IRI >= amber threshold or if data quality prevents GREEN
- **RED** otherwise
- Any hard stop overrides to **RED**.

### Hard-Stops-Only Mode (No ML)
- If any hard stop is triggered -> **RED**
- If no hard stop and data quality OK -> **GREEN**
- If no hard stop but data quality not OK -> **AMBER**
- IRI is not calculated in this mode.

Default thresholds are configured in `configs/default.yaml` under `thresholds`:
- `green_iri`: 70
- `amber_iri`: 50
- Lab recency limits for GREEN (e.g., lactate <=8h, Hb <=12h)

## Output
Each score produces:
- `traffic_light` (GREEN/AMBER/RED)
- `hard_stop_reason_summary`
- `data_quality_ok_for_green`
- `trend` (for ML mode)
- `suggested_action`

Latest score per encounter is shown on the dashboard.

## Data Quality and Fail-Closed Behavior
- Missing or stale data prevents **GREEN**.
- If scoring fails (schema mismatch or model missing), output is fail-closed **RED**.

## Governance and Audit
Each run produces:
- Quality log and rejected rows
- Run manifest (config hash, schema hash, input/output hashes, calibration method)
- Model bundle versioning with explicit retraining

## Limitations
This tool provides decision support only and must not replace clinician judgment. It is not validated for pediatric or non-ICU use without explicit local approval.

## Change Control
Thresholds, features, and model updates require documented clinical review, versioning, and re-validation prior to deployment.
