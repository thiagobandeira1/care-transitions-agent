# ADR 0005: Provisional risk tiers; Synthea is never used to judge prediction

**Status:** accepted (2026-10-25)

## Context
The agent needs a high/standard tier to decide visit timing and check-ins. In the Stage 1 payer data, LACE >= 10
flagged about 11% of discharges with a PPV of 0.45, and Medicare patients under 65 had about double the
readmission rate of those aged 65-84.

Synthea generates readmissions from scripted disease modules, not from the mechanisms LACE captures. On this
Synthea cohort (743 Medicare index stays, 6.9% readmitted), LACE scores an AUROC of 0.41.

## Decision
- **Tiers (configurable `PlanRules`, provisional):** high if LACE >= 10, or if the patient is under 65 and
  LACE >= 7. High tier means a visit within 3 days, a TCM 99496 candidate (face-to-face within 7 days), and
  check-ins on days 10 and 20. Standard means a visit within 7 days and a 99495 candidate (14 days). Every patient
  gets the TCM contact deadline of 2 business days.
- **These thresholds are calibrated on a practice's own outcomes during a pilot** (Stage 2 at MMC), not on Synthea.
- **Synthea is used only to demonstrate and test the workflow:** data plumbing, rules, drafting, human review and
  tracking. No discrimination or calibration result on Synthea is reported as evidence about risk prediction.

## Consequences
The demo shows realistic mechanics without implying clinical validity. Model claims rest on the MMC payer-feed
analysis (Stage 1) and, for extraction, on MIMIC-IV-Note.

## Update (2026-11-09)
- **The tiers were evaluated on MIMIC-IV, not Synthea** ([the paper](../../paper/manuscript.md), Section 3.1
  and Table 3; preprint doi:10.5281/zenodo.24090449). On 64,036 stays, against 30-day unplanned readmission (an
  admission-type proxy), LACE scored an AUROC of 0.638, and the high tier flagged 38.2% of stays, readmitted at
  28.3% against 14.3% for the standard tier.
- **The pilot of the full agent at MMC was dropped on 2026-11-02**, so the thresholds were not calibrated in a
  pilot as planned above.
- **The thresholds stay provisional** and are not calibrated on any practice's outcomes (paper Section 4.6).
  They are to be calibrated on each practice's own outcomes before use.
- **Correction to Consequences:** model claims in the paper rest on MIMIC-IV, for the tiers as well as for
  extraction. The payer-feed analysis informed the provisional lower threshold and is outside the paper
  (Section 2.1). As in paper Table 1, that threshold is LACE >= 7 with Medicare coverage under 65.
