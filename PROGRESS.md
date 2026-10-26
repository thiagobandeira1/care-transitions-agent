# Progress: care-transitions-agent (Stage 4)

_Last updated: 2026-10-26, ~6 AM, after the overnight runs. Local commits only; nothing is pushed to GitHub
(publishing needs Thiago's OK). Every number below is from synthetic data (Synthea and notes generated from it).
None of it is MIMIC or MMC data._

## Where things stand
| Milestone | Status |
|---|---|
| M0 spec, ADRs, hardened skeleton, privacy guards | ✅ done |
| M1 Synthea cohort, LACE + under-65, TCM rules | ✅ done (743 Medicare index stays) |
| M2 LangGraph workflow, human review, SQLite tracking, daily check-ins | ✅ done |
| M3 real local drafting + GPU vs CPU benchmark | ✅ done (below) |
| M4 public synthetic extraction benchmark | ✅ done: 150 notes (43 train / 107 test), rules vs zero-/few-shot 8b and 14b |
| M5 MIMIC-IV-Note prep code + gold-annotation tooling | ✅ code done; runs once the download finishes |
| M6 baselines on MIMIC (zero-/few-shot) | waits for the download and the gold set |
| M7 QLoRA fine-tune script + training environment | ✅ pipeline verified end to end (small model, synthetic); real run waits for MIMIC |
| M8 GGUF/Ollama export, M9 UI and demo | export script written, not run; UI not smoke-tested yet |

## Measured results

### M3: drafting (3 LLM calls per discharge: plan explanation, call script, provider brief)
RTX 4080 vs i9-14900K CPU-only, qwen3 at Ollama's default 4-bit quantization, Synthea discharges, seed 7.

| Model | Device | n | Mean s / discharge | Median | Max | Schema failures | Discharges with grounding flags |
|---|---|---|---|---|---|---|---|
| qwen3:14b | GPU | 30 | 19.0 | 16.4 | 38.3 | 0 | 0 |
| qwen3:8b | GPU | 30 | 21.9 | 22.5 | 45.8 | 1 | 3 (5 flags) |
| qwen3:14b | CPU | 3 | 166.5 | 172.1 | 200.0 | 0 | 0 |
| qwen3:8b | CPU | 5 | 195.4 | 225.4 | 258.7 | 0 | 2* |

- A morning batch of 15 discharges takes about 5 minutes on the GPU and about 45 minutes on CPU only. CPU is slow
  but workable as an overnight batch at a practice without a GPU.
- The CPU rows are small samples of different discharges; read them as "about 3 minutes per discharge", not as
  14b being faster than 8b.
- Every 8b GPU flag was checked by hand. All were real errors, no false alarms: a garbled condition name
  ("Myocard, "), three diagnoses listed as medications in a call script, and a misspelled drug ("clopidoogrel").
  The schema failure was a call script that failed validation twice and went to human handling, as designed.
  Its rejected text also had a placeholder ("[Your Name]") and invented a family relationship, which the
  grounding check (medication and condition names only) would not catch. That is why every draft is reviewed.
- The first GPU runs flagged 7/30 (14b) and 10/30 (8b) discharges. Most of those were false alarms from
  Synthea's double-spaced drug names; the check now normalizes spacing and punctuation (with regression tests).
- \*8b CPU ran before that fix; its 2 flags were not re-checked.
- Recommendation: **qwen3:14b for drafting** (no flags, no failures, slightly faster on the GPU).

### M4: extraction on the synthetic benchmark (107 test notes)
Notes were written by qwen3:14b from Synthea records, and a note was kept only if it mentioned every reference
fact. F1 with relaxed matching (token overlap ≥ 0.5); "micro" pools all fields; 95% CI from a 2,000-sample
bootstrap.

| Mode | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|
| Rules (no LLM) | 1.000 | 0.963 | 0.907 | 0.159 | 0.452 | 0.037 | **0.696** (0.675-0.716) | 0.654 | 0.0 |
| qwen3:8b zero-shot | 0.991 | 0.993 | 0.269 | 0.784 | 0.965 | 0.988 | **0.761** (0.738-0.783) | 0.697 | 3.2 |
| qwen3:14b zero-shot | 0.850 | 0.939 | 0.931 | 0.829 | 0.965 | 0.994 | **0.929** (0.910-0.946) | 0.861 | 6.2 |
| qwen3:8b few-shot (3) | 0.907 | 0.973 | 0.974 | 0.865 | 0.965 | 0.994 | **0.961** (0.949-0.972) | 0.943 | 5.5 |
| qwen3:14b few-shot (3) | 0.944 | 0.983 | 0.985 | 0.967 | 0.950 | 0.994 | **0.981** (0.972-0.988) | 0.972 | 6.4 |

No schema failures in any mode. Medication change-type accuracy (among matched drugs) is 0.95 for rules and
0.99-1.00 for every LLM mode.

What this shows, and what it does not:
- **The 8B model's zero-shot gap is a convention problem.** It finds medications (precision 0.87) but lists only
  drugs that changed, leaving out the continued ones the schema asks for (recall 0.16). Three examples fix it.
  Fine-tuning should teach the convention without spending prompt length on examples, so this is the gap paper 2
  measures on MIMIC.
- **Rules are good only where the note is a clean list.** Diagnoses and medication lists are easy; follow-up,
  pending tests and red flags in prose are not. That is the case for an LLM.
- **This benchmark is easier than real notes**, and few-shot is near its ceiling (0.96-0.98). The notes are short
  and clean, and they were written by qwen3:14b, which may favor 14b. It tests the pipeline and the scoring; the
  paper's comparison (zero-/few-shot vs fine-tuned) needs MIMIC and the hand-checked gold set.
- Two benchmark bugs were found and fixed before these numbers (both in git history, with tests):
  1. The section parser only knew MIMIC's plain "Header:" style and missed the generated notes' Markdown headers,
     so the rules scored 0.
  2. The JSON schema made only the principal diagnosis required, and constrained decoding let zero-shot models
     skip whole fields (follow-up and pending tests scored 0.000). Every key is now required (schema 1.1).
  The 150 notes were generated on CPU by accident (a model left loaded by the CPU benchmark was reused); the notes
  are the same as GPU output would be, and the benchmark now unloads CPU-run models.

### M7: QLoRA pipeline test
`scripts/train_qlora.py --source synthetic --base Qwen/Qwen3-0.6B --max-train 40 --max-eval 10` in `.venv-train`
(torch 2.11 + CUDA 12.8, TRL 1.16, PEFT 0.21, bitsandbytes 0.50): 4-bit NF4 base, LoRA rank 16 on all linear layers,
loss on the JSON answer only. It ran end to end: 18 s of training, eval loss 0.375, eval token accuracy 0.93, adapter
written outside the repo. Qwen3-8B base weights (16 GB, SHA-256 verified against Hugging Face) are at
`C:\Users\Thiago\models\Qwen3-8B`, ready for the real run (`--source mimic --base C:\Users\Thiago\models\Qwen3-8B`).

## What needs you
1. **MIMIC-IV-Note download:** `discharge_detail.csv.gz` is complete. `discharge.csv.gz` (about 1.1 GB) was at
   about 465 MB at 6 AM, at ~20 KB/s, so expect early afternoon. The script in the terminal resumes by itself after
   drops. When it finishes, I verify the SHA-256 against PhysioNet's list and run `mimic-prep` (aggregates only).
2. **Gold set** (paper 2's reference): once prep has run, `uv run python -m ctagent gold-export --n 200
   --overlap 50`, then annotate following `docs/ANNOTATION.md`, ideally with an FIU co-author for the 50-note
   overlap.
3. **Publishing:** the repo is local only. Say the word and I'll create the public GitHub repo. It contains no
   MIMIC data and nothing from MMC; the CI guard checks this.

## How to run
```
uv sync --group dev
uv run pytest
uv run python -m ctagent cohort
uv run python -m ctagent bench --model qwen3:14b --n 30          # GPU; add --cpu for CPU-only
uv run python -m ctagent make-benchmark --n 150
uv run python -m ctagent extract-bench --mode few --model qwen3:8b
uv run python -m ctagent mimic-prep                               # after the MIMIC download
```
