# care-transitions-agent

Local, privacy-preserving AI for the 30 days after a hospital discharge, built for primary care practices and MSOs.

- **Extraction model:** a 7-8B open model fine-tuned with QLoRA to turn a discharge summary into structured JSON
  (diagnoses, medication changes, follow-up, pending tests, red flags).
- **Care-transitions agent:** a fixed LangGraph workflow. It scores readmission risk with deterministic rules,
  drafts the coordinator call script and provider brief with a local LLM, and waits for human approval.

**Status:** early development (M0). See [SPEC.md](SPEC.md) and [docs/adr](docs/adr).

**Data policy:** public demos use Synthea synthetic patients only. Model training uses MIMIC-IV-Note locally under
the PhysioNet credentialed license. No MIMIC data, note text or MIMIC-trained weights are in this repository. No
patient data ever goes to a cloud service. See [ADR 0001](docs/adr/0001-two-track-data-policy.md).

```
uv sync --group dev
uv run pytest
```
