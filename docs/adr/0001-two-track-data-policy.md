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

## Exception (2026-11-01)
The 200 gold-set notes were also labeled by Claude Opus 5.5, through Anthropic's Claude browser extension under
the author's account with model training turned off, to make a second, model-made reference
([docs/ANNOTATION.md](../ANNOTATION.md)). PhysioNet's guidance on online services
(https://physionet.org/news/post/gpt-responsible-use/) lists Anthropic Claude among the services it suggests for
credentialed data. This is the only step in which MIMIC note text left the local machine. Every model that is
trained or evaluated still runs locally, and Claude Code still reads aggregates only.
