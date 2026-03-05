# Feature Specification (q4h Cognitive Compression)

All features are computed at 4-hour score times per encounter. Irregular sampling is supported; slopes use real elapsed time.

## Haemodynamic
- `MAP_mean_4h`, `MAP_sd_4h`, `MAP_min_4h` over last 4h
- `MAP_slope_4h` (per hour)
- `MAP_time_ge_65` (fraction of points ≥65)
- `HR_mean_4h`, `HR_sd_4h`, `HR_slope_4h`

## Pressors
- `pressor_on` (last total pressor >0 in last 4h)
- `pressor_escalating` (last > first in 4h window)
- `pressor_delta_4h` ((last-first)/max(abs(first),1e-6))
- `pressor_free_hours` (time since last pressor >0; NaN if no history)

## Respiratory
- `FiO2_now` (last value), `FiO2_slope_4h`
- `SpO2_time_ge_94` (fraction of points ≥94)
- `RR_mean_4h`, `RR_slope_4h`
- `resp_support_level_now`, `resp_support_level_slope`
- `extubated_hours` (time since last ETT)

## Bleeding
- `drain_sum_4h`, `drain_sum_8h`, `drain_slope`
- `Hb_delta_6h`

## Renal/Perfusion
- `uop_sum_4h`, `oliguria_flag`
- `lactate_now`, `lactate_slope_4h`
- `creatinine_delta_24h`

## Neuro/Infection
- `RASS_now`, `temp_slope_4h`, `WCC_slope_24h`

## Dependency
- `arterial_line_present_latest`, `insulin_infusion_latest`

## Recency & Missingness
- `lactate_age_hours`, `Hb_age_hours`, `creatinine_age_hours`, `WCC_age_hours`
- `pressor_missing_4h`, `resp_missing_4h`, `map_missing_4h`

