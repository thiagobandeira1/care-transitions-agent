# CLAUDE.md: care-transitions-agent

Stage 4 of Thiago's Readmission-Agent roadmap: a local discharge-summary extraction model (paper 2) and a
LangGraph care-transitions agent demonstrated on Synthea. Read `SPEC.md` first. Keep the work rigorous, honest about negative
results, and clearly Thiago's own.

## Data rules (override everything; if a task seems to need breaking one, stop and ask Thiago)
1. **MIMIC data never enters git and never reaches a cloud LLM, including you.** Never open, cat, head, print or
   screenshot MIMIC rows or note text, and never paste it into a prompt. You write scripts. Anything that reads note
   text runs locally (Ollama, transformers or Unsloth). Scripts print aggregates only: counts, token-length
   distributions, metrics.
2. MIMIC lives outside this repo. The paths come from `.env` (`MIMIC_NOTE_DIR`, `MIMIC_HOSP_DIR`), which is never
   committed.
3. Only Synthea-derived or generated synthetic data may be committed, under `synthetic/samples/`. You may read
   synthetic data freely.
4. MIMIC-trained adapters, merged weights and GGUF files are never committed or uploaded.
5. LangSmith and LangGraph Platform tracing stay off; `ctagent.privacy.assert_no_cloud_tracing()` enforces it at
   startup.
6. No MMC data here. MMC work (stages 1-3) happens in Thiago's work session.

## Environment
- Python 3.12 via uv (`uv sync --group dev`). RTX 4080 16 GB, i9-14900K, 64 GB RAM.
- Ollama at http://localhost:11434. Check the Ollama library for current tags before pulling.
- MIMIC on this PC: MIMIC-IV v3.1 at `Documents\Coursework\Capstone Project\Dataset\mimic-iv-3.1 - Full\mimic-iv-3.1`
  (`hosp`, `icu`). MIMIC-IV-Note v2.2 is being downloaded by Thiago.
- Synthea: `..\_synthea\output\csv` (3,726 patients, Florida, ages 65-95).

## Conventions
Follow the portfolio pattern (see `../hedis-care-gap-agent`): `src/` layout, ruff, mypy strict, pytest with a
coverage floor, keyless CI on Ubuntu and Windows, ADRs, feature branches and PRs. Publishing to GitHub needs
Thiago's OK.
