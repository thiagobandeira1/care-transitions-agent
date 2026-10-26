# Progress: care-transitions-agent (Stage 4)

_Last updated: 2026-10-26, ~9 AM, after the morning re-runs. Local commits only; nothing is pushed to GitHub
(publishing needs Thiago's OK). Every number below is from synthetic data (Synthea and notes generated from it).
None of it is MIMIC or MMC data._

## Where things stand
| Milestone | Status |
|---|---|
| M0 spec, ADRs, hardened skeleton, privacy guards | ✅ done |
| M1 Synthea cohort, LACE + under-65, TCM rules | ✅ done (743 Medicare index stays) |
| M2 LangGraph workflow, human review, SQLite tracking, daily check-ins | ✅ done |
| M3 real local drafting + GPU vs CPU benchmark | ✅ done (below), re-measured after the prompt fix |
| M4 public synthetic extraction benchmark | ✅ done: 150 notes (43 train / 107 test), rules vs zero-/few-shot 8b and 14b |
| M5 MIMIC-IV-Note prep code + gold-annotation tooling | ✅ code done; runs once the download finishes |
| M6 baselines on MIMIC (zero-/few-shot) | waits for the download and the gold set |
| M7 QLoRA fine-tune script + training environment | ✅ pipeline verified end to end (small model, synthetic); real run waits for MIMIC |
| M8 GGUF/Ollama export | script written, not run (needs the M7 adapter) |
| M9 review screen and demo | ✅ smoke-tested on 5 synthetic discharges (queue, drafts, approve) |

## The main finding this morning: the model never saw the field instructions
Each LLM call returns JSON that must match a schema, and the schema's field descriptions held the
instructions ("2-3 plain sentences", "every medication, continued ones too"). Ollama uses the schema only to
constrain the output format; the model never reads the descriptions. So the instructions never reached it.

How it showed up, and how it was found:
1. **Greedy decoding (overnight run):** plan explanations, meant to be 2-3 sentences, repeated themselves up
   to the 1,500-token cap (about 8,000 characters). The JSON still parsed, so nothing flagged it.
2. **Sampling, the first fix:** the Qwen3 model card warns that greedy decoding can cause endless repetition,
   so drafting switched to its recommended sampling settings, with a check that rejects runaway text.
   Failures went *up*: 8b failed 26 of 30 plan explanations and 14b failed 7 of 30. Raw outputs showed half
   still looped and the rest ran to 9 sentences.
3. **The real fix:** `field_guide()` writes every field's name, type, allowed values and description into the
   prompt. 8b then produced 12 of 12 valid explanations of about 250 characters, under sampling *and* greedy
   decoding. The extraction prompt had the same gap.

Results after the fix: no drafting schema failures for either model (below), and 8b zero-shot extraction went
from 0.761 to 0.947 micro F1. ADR 0007 records the decision.

## Measured results

### M3: drafting (3 LLM calls per discharge: plan explanation, call script, provider brief)
RTX 4080 vs i9-14900K CPU-only, qwen3 at Ollama's default 4-bit quantization, Synthea discharges, seed 7,
Qwen3 non-thinking sampling with field guidance in the prompt.

| Model | Device | n | Mean s / discharge | Median | Max | Schema failures | Discharges with grounding flags |
|---|---|---|---|---|---|---|---|
| qwen3:14b | GPU | 30 | 15.8 | 15.5 | 23.1 | 0 | 0 |
| qwen3:8b | GPU | 30 | 6.2 | 5.5 | 10.9 | 0 | 1 |
| qwen3:14b | CPU | 5 | 186.3 | 187.4 | 279.0 | 0 | 0 |
| qwen3:8b | CPU | 5 | 69.2 | 67.7 | 96.7 | 0 | 0 |

- A morning batch of 15 discharges takes about 4 minutes with 14b on the GPU, or under 2 minutes with 8b.
  On CPU only it takes about 47 minutes (14b) or 17 minutes (8b), which is workable overnight at a practice
  without a GPU. The CPU rows are 5 discharges each, so treat them as rough.
- The one 8b flag is a real error: a misspelled condition ("Pneumoma"). The grounding check has had no false
  alarms since it was fixed to normalize spacing and punctuation.
- Earlier numbers (in git history) are superseded: the greedy run's timings (14b 19.0 s, 8b 21.9 s) were
  inflated by looping, and the sampling-only run failed as described above.
- Recommendation: **qwen3:14b on the GPU for drafting** (no flags in 30, 4 minutes per batch). qwen3:8b is
  the CPU-only option: 2.7x faster, with every draft still reviewed by a person.
- Plan explanations in the saved samples are now 195-353 characters.

### M4: extraction on the synthetic benchmark (107 test notes)
Notes were written by qwen3:14b from Synthea records, and a note was kept only if it mentioned every reference
fact. F1 with relaxed matching (token overlap ≥ 0.5); "micro" pools all fields; 95% CI from a 2,000-sample
bootstrap. Greedy decoding, extraction schema 1.2 (field guidance in the prompt).

| Mode | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|
| Rules (no LLM) | 1.000 | 0.963 | 0.907 | 0.159 | 0.452 | 0.037 | **0.696** (0.675-0.716) | 0.654 | 0.0 |
| qwen3:8b zero-shot | 0.953 | 0.986 | 0.954 | 0.776 | 0.944 | 0.994 | **0.947** (0.933-0.959) | 0.881 | 5.4 |
| qwen3:14b zero-shot | 0.849 | 0.960 | 0.974 | 0.796 | 0.965 | 0.994 | **0.947** (0.935-0.958) | 0.886 | 9.0 |
| qwen3:8b few-shot (3) | 0.925 | 0.979 | 0.983 | 0.856 | 0.957 | 0.994 | **0.965** (0.955-0.974) | 0.941 | 5.8 |
| qwen3:14b few-shot (3) | 0.935 | 0.981 | 0.981 | 0.981 | 0.943 | 0.994 | **0.979** (0.970-0.987) | 0.969 | 9.5 |

No schema failures in any mode. Medication change-type accuracy (among matched drugs) is 0.95 for rules and
0.99-1.00 for every LLM mode. Timings are from one sequential run each and vary between runs (14b took
6.2-6.4 s per note in the earlier run).

What this shows, and what it does not:
- **Stating the conventions closes most of the zero-shot gap.** Before the fix, 8b zero-shot left out
  continued medications (medication F1 0.269, micro 0.761). With the field guidance it scores 0.954 and
  0.947. For paper 2, the zero-shot baseline must include the field guidance, or the comparison would
  overstate what fine-tuning adds.
- **What zero-shot still misses:** follow-up appointments (0.78-0.80; three examples raise 14b to 0.98) and,
  for 14b, the principal diagnosis (0.849). Those are the fields to watch on MIMIC.
- **Rules are good only where the note is a clean list.** Diagnoses and medication lists are easy; follow-up,
  pending tests and red flags in prose are not. That is the case for an LLM.
- **This benchmark is easier than real notes**, and every LLM mode is at 0.95-0.98. The notes are short and
  clean, and they were written by qwen3:14b, which may favor 14b. It tests the pipeline and the scoring; the
  paper's comparison (zero-/few-shot vs fine-tuned) needs MIMIC and the hand-checked gold set, where notes
  are long and messy.
- Three prompt or benchmark bugs were found and fixed before these numbers (all in git history, with tests):
  1. The section parser only knew MIMIC's plain "Header:" style and missed the generated notes' Markdown
     headers, so the rules scored 0.
  2. The JSON schema made only the principal diagnosis required, and constrained decoding let zero-shot models
     skip whole fields (follow-up and pending tests scored 0.000). Every key is now required (schema 1.1).
  3. The field descriptions never reached the model (above). The prompt now includes them (schema 1.2). The
     earlier numbers (8b zero 0.761, 14b zero 0.929, 8b few 0.961, 14b few 0.981) are in git history.
  The 150 notes were generated on CPU by accident (a model left loaded by the CPU benchmark was reused); the notes
  are the same as GPU output would be, and the benchmark now unloads CPU-run models.

### M7: QLoRA pipeline test
`scripts/train_qlora.py --source synthetic --base Qwen/Qwen3-0.6B --max-train 40 --max-eval 10` in `.venv-train`
(torch 2.11 + CUDA 12.8, TRL 1.16, PEFT 0.21, bitsandbytes 0.50): 4-bit NF4 base, LoRA rank 16 on all linear layers,
loss on the JSON answer only. It ran end to end: 18 s of training, eval loss 0.375, eval token accuracy 0.93, adapter
written outside the repo. Qwen3-8B base weights (16 GB, SHA-256 verified against Hugging Face) are at
`C:\Users\Thiago\models\Qwen3-8B`, ready for the real run (`--source mimic --base C:\Users\Thiago\models\Qwen3-8B`).
The training prompt reuses the zero-shot extractor's prompt, so it now includes the field guidance too.

### M9: review screen smoke test
`demo --n 5` processed 5 synthetic discharges to the review pause (1 high tier, 4 standard). In the Streamlit
review screen the queue, plan, call script and provider brief displayed correctly, and approving the high-tier
discharge recorded the reviewer and time and scheduled its two check-ins (days 10 and 20).

Product notes from reading the drafts (for the pilot, not bugs):
- Call scripts use placeholders such as "[Your Name]" and "[Practice Name]". A practice profile (practice and
  coordinator names) should go into the prompt before the pilot.
- The call script's medication review lists continued medications even when nothing changed, although the
  field asks for changes only. Whether the coordinator call should cover all active medications or only
  changes is a workflow choice for Thiago.

## What needs you
1. **MIMIC-IV-Note download:** `discharge_detail.csv.gz` is complete. `discharge.csv.gz` (about 1.1 GB) was at
   about 680 MB at 8:45 AM, at ~20 KB/s with frequent stalls, so expect mid-to-late afternoon. The script in the
   terminal resumes by itself after drops. When it finishes, I verify the SHA-256 against PhysioNet's list and run
   `mimic-prep` (aggregates only).
2. **Gold set** (paper 2's reference): once prep has run, `uv run python -m ctagent gold-export --n 200
   --overlap 50`, then annotate following `docs/ANNOTATION.md`, ideally with an FIU co-author for the 50-note
   overlap.
3. **Publishing:** the repo stays local until the MIMIC baselines (M6) are in (Thiago, 2026-10-26). Then I'll
   create the public GitHub repo. It contains no MIMIC data and nothing from MMC; the CI guard checks this.

## How to run
```
uv sync --group dev
uv run pytest
uv run python -m ctagent cohort
uv run python -m ctagent bench --model qwen3:14b --n 30          # GPU; add --cpu for CPU-only
uv run python -m ctagent make-benchmark --n 150
uv run python -m ctagent extract-bench --mode few --model qwen3:8b
uv run python -m ctagent demo --n 5                               # morning batch into data/demo
uv run streamlit run app/review_app.py -- --workspace data/demo   # review screen
uv run python -m ctagent mimic-prep                               # after the MIMIC download
```
