# ADR 0001: Two-track data, MIMIC local-only and Synthea public

**Status:** accepted (2026-10-25); amended 2026-11-09 to record a one-time exception made on 2026-11-01
(below)

## Context
The extraction model needs real clinical prose (MIMIC-IV-Note), but the PhysioNet credentialed DUA prohibits
sharing access to the data with third parties. PhysioNet's guidance on online services (18 April 2023, linked
under Exception below) says this includes sending the data through third-party companies' APIs or using it on
online chat platforms, and it names a few services it suggests instead, Anthropic Claude among them. The project
also has to be publicly demonstrable.

(Reworded 2026-11-09 to match that guidance. The 2026-10-25 text said the DUA "forbids sharing the data and
sending it to third-party services, cloud LLMs included.")

## Decision
- **MIMIC track (private, local):** training and evaluation run on this PC only. Data paths come from `.env` and
  must resolve outside the repo (`assert_outside_repo`). Only code, a model card and aggregate metrics are
  committed. Claude Code (the coding assistant, which runs on a cloud model) writes code and reads aggregates,
  never rows or note text.
- **Synthea track (public):** demos, fixtures, screenshots and the committed eval sets use Synthea patients and
  synthetic discharge summaries generated from them.

## Consequences
- The public benchmark (synthetic notes) and the private benchmark (MIMIC) are reported separately. Synthetic
  notes are easier, so they are never presented as clinical validity.
- CI never sees MIMIC. Tests that need it are marked `mimic` and run locally only.

## Exception (2026-11-01)
On 1 November 2026, Claude Opus 5.5 (Anthropic, a cloud service), operated through Anthropic's Claude browser
extension under the author's account with model training ("Help improve Claude") turned off, read the 200 gold-set
notes in the local annotation screen and corrected their pre-filled labels. This made `claude.json`, a second,
model-made reference (the paper's language-model-corrected reference, Section 2.4;
[docs/ANNOTATION.md](../ANNOTATION.md)). PhysioNet's guidance on online services
(https://physionet.org/news/post/gpt-responsible-use/) lists Anthropic Claude among the services it suggests for
credentialed data, noting that Claude does not train on prompts or responses by default and does not routinely
have humans review data. This is the one exception to the MIMIC track and the only step in which MIMIC note text
left the local machine. It is not a precedent: every other step, including every model that is trained or
evaluated, still runs locally, and Claude Code still reads aggregates only.

This section was added on 2026-11-09. At the time of the step, this ADR and the project's other written data
rules (rule 1 in `SPEC.md` and `CLAUDE.md`, and `docs/ANNOTATION.md`) said note text never goes to a cloud
model, so the step departed from the project's own rule at the time.
