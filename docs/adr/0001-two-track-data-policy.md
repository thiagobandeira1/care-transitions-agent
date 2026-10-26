# ADR 0001: Two-track data, MIMIC local-only and Synthea public

**Status:** accepted (2026-10-25)

## Context
The extraction model needs real clinical prose (MIMIC-IV-Note), but the PhysioNet credentialed DUA forbids sharing
the data and sending it to third-party services, cloud LLMs included. The project also has to be publicly
demonstrable.

## Decision
- **MIMIC track (private, local):** training and evaluation run on this PC only. Data paths come from `.env` and
  must resolve outside the repo (`assert_outside_repo`). Only code, a model card and aggregate metrics are
  committed. Claude (a cloud model) writes code and reads aggregates, never rows or note text.
- **Synthea track (public):** demos, fixtures, screenshots and the committed eval sets use Synthea patients and
  synthetic discharge summaries generated from them.

## Consequences
- The public benchmark (synthetic notes) and the private benchmark (MIMIC) are reported separately. Synthetic
  notes are easier, so they are never presented as clinical validity.
- CI never sees MIMIC. Tests that need it are marked `mimic` and run locally only.
