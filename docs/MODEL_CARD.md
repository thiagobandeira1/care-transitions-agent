# Model card: ctagent-extractor (fine-tuned discharge-summary extractor)

_2026-10-27. The weights are not distributed. They were trained on MIMIC-IV-Note, whose data use agreement
forbids sharing data or derived models outside credentialed access. They stay on the training machine._

Four variants exist, all Qwen3-8B with the same method. They differ only in the training labels and the number
of notes:

| Ollama name | Labels | Train notes | Role in paper 2 |
|---|---|---|---|
| `ctagent-extractor` (v0, M7) | silver | 8,851 | silver-only arm |
| `ctagent-extractor-s2500` | silver | 2,500 | control for c2500 (same notes) |
| `ctagent-extractor-c1250` | silver + teacher | 1,250 (the first half of c2500's) | learning curve |
| **`ctagent-extractor-c2500`** | silver + teacher | 2,500 | **candidate fine-tuned arm (option C)** |

## What it is
Qwen3-8B fine-tuned with QLoRA to turn a hospital discharge summary into structured JSON: principal and
secondary diagnoses, medication changes (started, stopped, dose changed, continued), follow-up appointments,
pending tests and red-flag symptoms. It is served locally by Ollama as a Q4_K_M GGUF, with the same prompt as
the zero-shot extractor (`src/ctagent/extractor.py`), and is meant to feed the agent's `extract` node.

## Training
- **Data:** MIMIC-IV-Note discharge summaries from the train split (split by patient).
  - v0 used 8,851 notes drawn at random from 10,000.
  - The option C models use the first 2,500 (or 1,250) notes of the same shuffled order.
  - Notes over 3,072 tokens (prompt plus answer) were dropped, not truncated: 11% for v0, under 0.1% for
    option C, which skipped long notes before labeling.
- **Labels:**
  - **Silver** (`src/ctagent/notes.py`, `src/ctagent/mimic.py`):
    - diagnoses: from the note's Discharge Diagnosis section
    - medication changes: admission vs discharge medication lists
    - follow-up, pending tests and red flags: simple section and phrase rules
    - MIMIC redacts Followup Instructions, so silver has no follow-ups.
  - **Silver + teacher (option C):** follow-up, pending tests and red flags come from qwen3:14b with 3 train
    examples in the prompt (`scripts/teacher_label.py`). Diagnoses and medication changes stay silver.
- **Method:** 4-bit NF4 base with LoRA rank 16 (alpha 32) on all linear layers, and loss on the answer only.
  1 epoch, learning rate 2e-4 (cosine, 3% warm-up), effective batch 8, on one RTX 4080 (16 GB, Windows).
  Training time: 4.4 hours for v0, 74 minutes for c2500. Code: `scripts/train_qlora.py`.
- **Export:** adapter merged into the bf16 base, converted and quantized to Q4_K_M with llama.cpp
  (`scripts/export_ollama.py`).

## Evaluation (micro F1 with token-overlap matching, 95% bootstrap CI)

| Benchmark | Base 8b, same export (control) | Base 8b, few-shot | 14b few-shot (teacher) | v0 (silver, 8,851) | **c2500** |
|---|---|---|---|---|---|
| Synthetic, 107 notes with known answers | 0.949 (0.937-0.961) | 0.965 (0.955-0.974) | 0.979 (0.970-0.987) | 0.788 (0.768-0.812) | **0.948** (0.934-0.961) |
| MIMIC, 200 gold-set notes, against silver | 0.563 (0.529-0.598) | 0.630 (0.594-0.664) | 0.631 (0.597-0.665) | 0.958 (0.941-0.972) | **0.778** (0.705-0.830) |

c2500's MIMIC score is 0.808 with exact duplicate list items removed (Limitations). Per-field results, the
controls and the learning curve are in `PROGRESS.md` (Findings 1 and 2) and `evals/results/`. Scores against
the hand-annotated gold set are pending annotation.

## Limitations
- **Repetition loops (option C models).** On a few MIMIC notes the 2,500-note models repeat one list item
  hundreds of times. The output still parses as valid JSON. Drop exact duplicate items before using the output.
- **v0 copies the silver labels, gaps included.** It never extracts follow-up appointments, and it misses most
  pending tests and many red flags (synthetic follow-up F1 0.06, against 0.77 for the base model). c2500 fixes
  this on synthetic notes (follow-up 0.80, pending tests 0.91, red flags 0.98).
- **The MIMIC scores are against silver.** On the weak fields c2500 deliberately disagrees with silver, so its
  MIMIC score there measures disagreement with the rules, not accuracy. Only the gold set says which is right.
- **A student trails its teacher.** On the weak fields c2500 can at best match the 14b few-shot teacher.
- **One hospital (Beth Israel Deaconess, 2008-2019), English, adult inpatients.** It has not been tested on
  other note styles.

## Intended use
Research only: the fine-tuned arms of paper 2's comparison (few-shot vs fine-tuned, evaluated on gold). It is
not for clinical use. In the agent, every extraction is a suggestion that a coordinator or provider reviews
before it reaches a patient or the EHR. The few-shot extractor remains the default until a fine-tuned model
beats it on gold.
