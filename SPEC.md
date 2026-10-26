# SPEC: care-transitions-agent

A local, privacy-preserving AI system for the 30 days after a hospital discharge, built for primary care practices
and MSOs. It has 2 parts:

- **A. Discharge-summary extraction model (paper 2).** A 7-8B open model, fine-tuned with QLoRA on MIMIC-IV-Note,
  that turns a discharge summary into structured JSON.
- **B. Care-transitions agent.** A LangGraph workflow with a fixed order of steps. It scores readmission risk,
  plans follow-up with deterministic rules, drafts a coordinator call script and a provider brief with a local LLM,
  and waits for human approval. It is demonstrated publicly on Synthea patients only.

This is Stage 4 of the Readmission-Agent roadmap. Stages 1-3 (MMC payer feed, KPI dashboard pilot) live at MMC and
are out of scope here. This repo never touches MMC data.

## 1. Non-negotiable data rules
1. **MIMIC data never enters git** (no rows, note text, examples, fixtures, screenshots or notebook outputs) and
   **never goes to a cloud LLM, including Claude.** Anything that reads MIMIC text runs locally (Ollama,
   transformers or Unsloth). Claude writes the code and sees aggregate metrics only.
2. **Public artifacts are Synthea-only.** That covers demos, fixtures, screenshots, the README and the eval sets
   that get committed. Committable fixtures live under `synthetic/samples/`.
3. **MIMIC-trained weights are not redistributed.** Adapters and merged or GGUF models stay local. Sharing them would
   go through PhysioNet under the same credentialed license.
4. **No cloud tracing.** LangSmith and LangGraph Platform stay off, and the code refuses to start if tracing is on.
5. Every agent output is a suggestion. A human approves before anything reaches a patient or an EHR.

## 2. Part A: extraction model

**Output schema (Pydantic, versioned):**
- `principal_diagnosis`: text, with an optional ICD-10 code
- `secondary_diagnoses`: a list of text, each with an optional code
- `medication_changes`: a list of `{name, change: started | stopped | dose_changed | continued, detail}`
- `follow_up`: a list of `{with_whom, timeframe, reason}`
- `pending_tests`: a list
- `red_flags`: a list of symptoms that should prompt a call or an ER visit

**Data** (local only; requires MIMIC-IV-Note v2.2 and MIMIC-IV v3.1 `hosp`):
- **Cohort:** discharge summaries linked to admissions. Split by `subject_id` into train, validation and test
  (70/10/20, fixed seed), so no patient appears in two splits.
- **Input:** the note, section-aware and trimmed to a token budget of about 4k. Keep these sections: discharge
  diagnosis, medications on admission, discharge medications, discharge instructions, follow-up, and pending
  results. Report how often trimming happens.
- **Silver labels**, pre-specified and each source documented:
  - diagnoses from `diagnoses_icd` (seq_num 1 = principal)
  - medication changes from rule-based comparison of the "Medications on Admission" and "Discharge Medications"
    sections, with name normalization
  - follow-up, pending tests and red flags from rule-based section parsing, plus an optional local teacher LLM.
    The teacher is a larger Ollama model running locally, and its agreement with gold is reported.
- **Gold set:** about 200 test-split notes, annotated by Thiago under a written guideline, with a second annotator
  (a co-author) on a 50-note overlap to measure inter-annotator agreement (kappa / F1). Annotation files stay local.

**Models:**
- Base: Qwen3-8B or Llama-3.1-8B-Instruct, from Hugging Face weights, chosen by a small pilot on validation only.
- Baselines: the base model zero-shot and few-shot (k=3, fixed examples from train), with constrained JSON
  decoding.
- Fine-tuned: QLoRA with Unsloth or TRL + PEFT, 4-bit, sized for 16 GB (RTX 4080). Hyperparameters are tuned on
  validation only.

**Evaluation** (test split only; patient-bootstrap 95% CIs):
- Field-level precision, recall and F1 for each schema field (exact and relaxed match), on gold and on silver.
- Medication-change accuracy and macro-F1 over the 4 change types.
- Schema-valid output rate, hallucination rate (extracted medications not mentioned in the note), and latency on GPU
  and CPU.
- **Core result for paper 2:** base zero-shot vs few-shot vs fine-tuned.

**Deploy:** merge the adapter, convert to GGUF (llama.cpp), and load into Ollama with a Modelfile, then plug into
Part B's `extract` node.

## 3. Part B: care-transitions agent

**Data (public track):** Synthea patients (Florida, ages 65-95, about 3,700 people), already generated in
`../_synthea/output/csv`.
- An inpatient encounter is a discharge event.
- Conditions, medications and observations stand in for the EHR context.
- **Synthetic discharge summaries** are generated from each encounter's structured data by a local LLM, so the
  ground truth is known by construction. That gives a public, committable extraction benchmark next to the private
  MIMIC one.

**Graph** (LangGraph, fixed order; the LLM never chooses tools):
1. `ingest`: load and normalize the discharge event
2. `score`: deterministic. LACE (LOS, acute via an ER encounter in the prior day, Charlson from a curated SNOMED
   condition list, ER visits in the prior 6 months) plus an under-65 flag
3. `ehr_context`: problem list and active medications as of discharge
4. `extract`: the Part A model, when a summary exists
5. `plan`: rules set the tier (high = a visit within 2-3 days and check-ins around days 10 and 20) and the TCM
   clocks (99495/99496: contact within 2 business days; a face-to-face visit within 14 or 7 days). The LLM only
   writes the explanation.
6. `draft`: the LLM writes the coordinator call script and a one-page provider brief
7. `human_review`: an `interrupt`, with state in a local SQLite checkpointer
8. `track`: outcomes (contacted, scheduled, attended, readmitted) go to local SQLite

**LLM nodes:** each output is validated by Pydantic and retried once on failure. After that, the case is flagged for
human handling.

**Grounding check (deterministic):** every medication or diagnosis named in a draft must exist in the patient's
context. Otherwise the draft is flagged.

**30-day follow-up:** no graph stays open for 30 days. A daily job (Windows Task Scheduler) reads the status
database and starts short runs only for check-ins that are due.

**UI:** a Streamlit coordinator queue to approve, edit or reject drafts and record outcomes.

**Evals (public, Synthea):**
- schema-valid rate, grounding-violation rate, retry and flag rates
- presence of the required elements (TCM deadline, medication changes, red flags)
- latency per discharge on GPU vs **CPU only**. This measures whether a practice without a GPU, like MMC, can run
  the morning batch.

## 4. Engineering
- Python 3.12, uv, hatchling `src/` layout, ruff, mypy strict, pytest with a coverage floor, and GitHub Actions on
  Ubuntu and Windows.
- CI is keyless and modelless: LLM nodes run under fakes in CI, and real-model tests are marked and skipped.
- A CI guard fails on tracked data files, secrets, and MIMIC references inside `synthetic/` or `tests/`.
- ADRs in `docs/adr/`. Feature branches and PRs. Conventional commits.
- Pin LangChain and LangGraph versions, and check the current docs before writing graph code (LangGraph 1.x
  interrupt return format).

## 5. Milestones
| # | Milestone | Needs |
|---|---|---|
| M0 | Spec, ADRs, hardened skeleton, privacy guard | none |
| M1 | Synthea discharge cohort, LACE + rules (deterministic), tests | Synthea (present) |
| M2 | LangGraph skeleton with fake LLM, interrupt + SQLite checkpointer, tracking DB, daily check-in job | none |
| M3 | Ollama drafting, Pydantic validation, grounding check, GPU vs CPU latency benchmark | Ollama installed |
| M4 | Synthetic discharge summaries + public extraction benchmark | Ollama |
| M5 | MIMIC-IV-Note prep: cohort, splits, sections, silver labels, gold-set protocol and tool | MIMIC-IV-Note download |
| M6 | Baselines (zero-shot / few-shot) on MIMIC test and the synthetic benchmark | M5 |
| M7 | QLoRA fine-tune + evaluation (paper 2 core result) | M5, GPU, PyTorch + Unsloth |
| M8 | GGUF export, Ollama Modelfile, plugged into `extract` | M7, llama.cpp |
| M9 | Streamlit queue, demo video, README, public release (with Thiago's approval) | M1-M8 |

## 6. Done means
- Every milestone ships with tests, green CI, and an ADR for any decision that would surprise a reviewer.
- Paper 2 has a results table (base vs few-shot vs fine-tuned, with CIs) and a model card.
- A public repo with a Synthea demo runs end to end with `uv sync` and Ollama.
