# Model card: ctagent-extractor (fine-tuned discharge-summary extractor)

## Update (2026-11-09)
These models are the fine-tuned arms of the paper, published as a preprint on Zenodo: *From discharge alerts to
follow-up work: design and retrospective evaluation of a local, human-in-the-loop AI agent for primary care*,
doi:[10.5281/zenodo.24090449](https://doi.org/10.5281/zenodo.24090449), which always resolves to the latest
version (the figures in this update are from version 1.3). Source: [paper/manuscript.md](../paper/manuscript.md).
The weights are still not distributed.

**Names.** The paper's names for these models:

| In the paper | Earlier names (this card, `PROGRESS.md`) |
|---|---|
| Silver-8851 | M7, v0, `ctagent-extractor` |
| Silver-2500 | s2500, `ctagent-extractor-s2500` |
| Teacher-1250 | c1250, `ctagent-extractor-c1250` |
| Teacher-2500 | c2500, `ctagent-extractor-c2500` |
| Base-8B-local | `qwen3-8b-local`, the untouched base model through the same GGUF conversion (the control) |
| Qwen3-8B, zero-shot and few-shot | `qwen3:8b`, the Ollama library build |

**Gold-set results**, pending in the snapshot below (paper Section 3.2; micro F1, relaxed matching):
- **Diagnoses, author-corrected reference** (51 notes, Table 5): the fine-tunes score 0.93-0.94, about the rules'
  own 0.93, against 0.69-0.72 for the prompted models zero-shot and 0.81-0.85 few-shot. The gap is in secondary
  diagnoses: the prompted models list many that the reference does not, while the fine-tunes keep to the
  Discharge Diagnosis list, as their labels do.
- **All six fields, language-model-corrected reference** (200 notes, Table 6): the prompted models score
  0.70-0.72 zero-shot and 0.67-0.68 few-shot, and the fine-tunes 0.58-0.60, near the rules (0.58). On medication
  changes the fine-tunes stay at the rules' level (F1 0.61-0.63, rules 0.64, prompted models 0.73-0.78) and
  repeat most of the rules' potential false `started` drugs (Table 7). On follow-up the silver fine-tunes list
  none, and the teacher fine-tunes recover part (0.30, against 0.40-0.49 for the prompted models zero-shot).
- Teacher-2500 (c2500), the candidate arm in the snapshot, scores 0.935 and 0.588 against these two
  references.
- Against silver (the snapshot's MIMIC row), Silver-8851 looked far better than the prompted models (0.959
  against 0.57-0.64). Against references other than its training labels, it is no better than the rules.
- Both references started from the silver labels, so neither is independent of the rules. The diagnosis
  reference is one annotator's quick pass, with no agreement measure yet. The all-field reference is model-made,
  by Claude Opus 5.5 (a cloud service; the one exception in [ADR 0001](adr/0001-two-track-data-policy.md)), and
  its accuracy is unmeasured.

Per-model rows are in paper Tables 5 and 6, `PROGRESS.md` (Finding 3) and `evals/results/`.

**Intended use now** (paper Sections 4.2 and 5, and Code availability): research only, not for clinical use.
A prompted base model is the preferred default extractor for the agent. A fine-tuned model is worth using only
for fields whose labels were checked against a reference other than the training labels: here, the
section-based diagnoses, where it is also about twice as fast.

---

_The rest of this card is its 2026-10-28 snapshot, from before the gold-set results. Where it differs from the
update above, the update applies._

_2026-10-28. The weights are not distributed. They were trained on MIMIC-IV-Note, whose data use agreement
forbids sharing data or derived models outside credentialed access. They stay on the training machine._

Four variants exist, all Qwen3-8B with the same method. They differ only in the training labels and the number
of notes:

| Ollama name | Labels | Train notes | Role in the paper |
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
| Synthetic, 107 notes with known answers | 0.948 (0.936-0.959) | 0.966 (0.956-0.974) | 0.979 (0.969-0.987) | 0.789 (0.766-0.811) | **0.948** (0.934-0.961) |
| MIMIC, 200 gold-set notes, against silver | 0.567 (0.532-0.602) | 0.638 (0.602-0.672) | 0.632 (0.597-0.666) | 0.959 (0.942-0.973) | **0.808** (0.782-0.834) |

All scores use extraction schema 1.3, which drops repeated list items (Limitations). The synthetic row's CIs are
paper Table 4's (a patient bootstrap, 1,000 resamples). Per-field results, the
controls and the learning curve are in `PROGRESS.md` (Findings 1 and 2) and `evals/results/`. Scores against
the hand-annotated gold set are pending annotation.

## Limitations
- **Repetition loops (option C models).** On a few MIMIC notes the 2,500-note models repeat one list item
  hundreds of times, in output that still parses as valid JSON. The extraction schema (1.3) drops the repeats
  when the output is validated, so scores and the agent are not affected. A loop still costs generation time,
  up to the answer cap.
- **v0 copies the silver labels, gaps included.** It never extracts follow-up appointments, and it misses most
  pending tests and many red flags (synthetic follow-up F1 0.06, against 0.77 for the base model). c2500 fixes
  this on synthetic notes (follow-up 0.80, pending tests 0.91, red flags 0.98).
- **The MIMIC scores are against silver.** On the weak fields c2500 deliberately disagrees with silver, so its
  MIMIC score there measures disagreement with the rules, not accuracy. Only the gold set says which is right.
- **A student trails its teacher.** On the weak fields c2500 can at best match the 14b few-shot teacher.
- **One hospital (Beth Israel Deaconess, 2008-2019), English, adult inpatients.** It has not been tested on
  other note styles.

## Intended use
Research only: the fine-tuned arms of the paper's comparison (few-shot vs fine-tuned, evaluated on gold). It is
not for clinical use. In the agent, every extraction is a suggestion that a coordinator or provider reviews
before it reaches a patient or the EHR. A prompted base model is the default extractor (see Intended use now,
above).
