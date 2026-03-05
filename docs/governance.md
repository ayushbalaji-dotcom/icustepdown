# Governance and Safety

## Decision Support Only
This tool provides decision support outputs only. It does not automate ICU→HDU step-down or discharge. Consultant sign-off is required.

## Hard Stops and Fail-Closed
Hard-stop safety rules always override ML predictions. Unknown, stale, or inconsistent data is never allowed to produce GREEN. If scoring fails, the dashboard is produced with all encounters marked RED.

## Audit & Drift
Each run produces audit artifacts:
- quality log and rejected rows
- run manifest (config hash, schema hash, input/output hashes, calibration method, counts)

Model bundles are versioned and locked. Retraining is explicit and produces audit records. There is no silent drift.

## Monitoring
Track 48h readmission and deterioration outcomes as safety endpoints. Review any post-step-down events and re-calibrate thresholds when clinically justified.

