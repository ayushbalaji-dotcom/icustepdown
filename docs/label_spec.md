# Label Specification: 48h Deterioration After Step-Down

## Definition
Training label is “deterioration within 48h after ICU step-down”.

## Rules (Deterministic)
- If `ADVERSE_EVENT` exists in `outcomes`, use it directly (0/1).
- Else derive `ADVERSE_EVENT` as logical OR across available component flags:
  - `readmitted_within_48h`
  - `MET_call_within_48h`
  - `pressor_restart_48h`
  - `reintubation_or_NIV_escalation_48h`
  - `death_48h`
- If `planned_readmission_48h == 1`, treat as **non-adverse** and set `ADVERSE_EVENT = 0`. A warning is logged.
- If `no_stepdown == 1` or `icu_discharge_time` is missing, mark as censored and exclude from training.

## Pressor Restart Clarification
If only a pressor dose series is available, `pressor_restart_48h` should mean pressor-on becomes >0 and sustained for ≥60 minutes or requires escalation. If the dataset only provides a binary flag, use it as-is and document the limitation.

## Assumptions
- All label derivation is deterministic and logged.
- Censored encounters are excluded from training.

