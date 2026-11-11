# ADR 0003: LACE plus an under-65 flag as the agent's risk score

**Status:** accepted (2026-10-25)

## Context
In Stage 1 (a primary care Medicare Advantage population, payer encounter notifications, about 900 index stays),
neither logistic regression nor XGBoost beat LACE in patient-grouped nested CV. LACE run through a logistic model
was also the best calibrated. Readmission risk was U-shaped in age, highest under 65, among beneficiaries eligible
through disability or end-stage kidney disease.

## Decision
The `score` node computes LACE (van Walraven 2010; Charlson from conditions) plus an under-65 flag. All of it is
deterministic and explainable. An ML model replaces LACE only after it beats LACE on discrimination and calibration
with patient-grouped validation on richer data (outpatient EHR features).

## Consequences
The agent's value doesn't depend on a fragile model. The score is easy to explain to coordinators and auditors.

## Update (2026-11-10)
- **Outside the paper:** the Stage 1 analysis in the Context is outside the paper
  ([paper/manuscript.md](../../paper/manuscript.md); doi:10.5281/zenodo.24090449), which reports no result from
  it. The paper's rationale for LACE is that it is a published, transparent score that needs no local training
  data, and it cites the analysis only as informing the provisional lower threshold (Section 2.1 and Table 1).
- **The flag is Medicare coverage at an age under 65** (paper Section 2.1 and Table 1), not age alone. In the
  code, `DischargeEvent.under_65` is `medicare and age < 65` (`src/ctagent/models.py`).
