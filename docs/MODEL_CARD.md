# Model card: ctagent-extractor v0 (silver-only fine-tune)

_2026-10-27. The weights are not distributed. They were trained on MIMIC-IV-Note, whose data use agreement
forbids sharing data or derived models outside credentialed access. They stay on the training machine._

## What it is
Qwen3-8B fine-tuned with QLoRA to turn a hospital discharge summary into structured JSON: principal and
secondary diagnoses, medication changes (started, stopped, dose changed, continued), follow-up appointments,
pending tests and red-flag symptoms. It is served locally by Ollama as a Q4_K_M GGUF, with the same prompt as
the zero-shot extractor (`src/ctagent/extractor.py`), and is meant to feed the agent's `extract` node.

## Training
- **Data:** 8,851 MIMIC-IV-Note discharge summaries from the train split (split by patient), drawn at random
  from 10,000. Notes over 3,072 tokens (prompt plus answer) were dropped, not truncated (11%).
- **Labels:** rule-based silver labels (`src/ctagent/notes.py`, `src/ctagent/mimic.py`):
  - diagnoses: from the note's Discharge Diagnosis section
  - medication changes: admission vs discharge medication lists
  - follow-up, pending tests and red flags: simple section and phrase rules
  - MIMIC redacts Followup Instructions, so silver has no follow-ups.
- **Method:** 4-bit NF4 base with LoRA rank 16 (alpha 32) on all linear layers, and loss on the answer only.
  1 epoch, learning rate 2e-4 (cosine, 3% warm-up), effective batch 8. Took 4.4 hours on one RTX 4080 (16 GB,
  Windows). Final eval loss was 0.0136 on 264 validation notes. Code: `scripts/train_qlora.py`.
- **Export:** adapter merged into the bf16 base, converted and quantized to Q4_K_M with llama.cpp
  (`scripts/export_ollama.py`).

## Evaluation (micro F1 with token-overlap matching, 95% bootstrap CI)

| Benchmark | Base 8b, same export (control) | Base 8b, few-shot | **Fine-tuned** |
|---|---|---|---|
| MIMIC, 200 gold-set notes, against silver | 0.563 (0.529-0.598) | 0.630 (0.594-0.664) | **0.958** (0.941-0.972) |
| Synthetic, 107 notes with known answers | 0.949 (0.937-0.961) | 0.965 (0.955-0.974) | **0.788** (0.768-0.812) |

Per-field results are in `PROGRESS.md` (M8) and `evals/results/`. Scores against the hand-annotated gold set are
pending annotation.

## Limitations
- **It copies the silver labels, gaps included.** It never extracts follow-up appointments, and it misses most
  pending tests and many red flags. On the synthetic notes, follow-up F1 is 0.06, against 0.77 for the base
  model. Do not use it for those fields.
- **The MIMIC scores are against silver,** the labels it was trained to imitate. They measure agreement with
  the rules, not clinical accuracy.
- **One hospital (Beth Israel Deaconess, 2008-2019), English, adult inpatients.** It has not been tested on
  other note styles.

## Intended use
Research only: the silver-only arm of paper 2's comparison (few-shot vs fine-tuned, evaluated on gold). It is
not for clinical use. In the agent, every extraction is a suggestion that a coordinator or provider reviews
before it reaches a patient or the EHR. The few-shot extractor remains the default until a fine-tuned model
beats it on gold.
