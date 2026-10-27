# Progress: care-transitions-agent (Stage 4)

_Last updated: 2026-10-26, ~8:40 PM, after the MIMIC baselines (M6). Local commits only; nothing is pushed to
GitHub yet. The M5 and M6 sections report MIMIC aggregates only (counts, rates, length quantiles, metrics). Every
other number is from synthetic data (Synthea and notes generated from it). None of it is MMC data._

## Where things stand
| Milestone | Status |
|---|---|
| M0 spec, ADRs, hardened skeleton, privacy guards | ✅ done |
| M1 Synthea cohort, LACE + under-65, TCM rules | ✅ done (743 Medicare index stays) |
| M2 LangGraph workflow, human review, SQLite tracking, daily check-ins | ✅ done |
| M3 real local drafting + GPU vs CPU benchmark | ✅ done (below), re-measured after the prompt fix |
| M4 public synthetic extraction benchmark | ✅ done: 150 notes (43 train / 107 test), rules vs zero-/few-shot 8b and 14b |
| M5 MIMIC-IV-Note prep code + gold-annotation tooling | ✅ done: download verified, prep run, 200-note gold set exported |
| M6 baselines on MIMIC (zero-/few-shot) | ✅ done against silver labels (below); the gold-label run waits for annotation |
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
  without a GPU. The CPU rows are 5 discharges each, so treat them as rough. Thiago's call (2026-10-26): the
  GPU is the priority and these CPU numbers are enough, so no larger CPU run is planned.
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

### M5: MIMIC-IV-Note prep (aggregates only)
Download: `discharge.csv.gz` (1,139 MB) and `discharge_detail.csv.gz` pass `gzip -t` and match PhysioNet's
SHA256SUMS.txt for MIMIC-IV-Note 2.2. Diagnosis codes come from MIMIC-IV 3.1 `hosp`. `mimic-prep` took 11.5 minutes
and wrote train/validation/test parquet files to `MIMIC_WORK_DIR`, outside the repo.

| | Notes | Patients |
|---|---|---|
| Train | 232,305 | 102,140 |
| Validation | 33,068 | 14,591 |
| Test | 66,420 | 29,183 |
| **Total** | **331,793** | **145,914** |

- Split by patient (70/10/20, seed 13), so no patient is in two splits.
- Sections found in 88-100% of notes (hospital course is the lowest, 0.88). A "Pending" header is almost never
  there (0.2%), so pending tests have to come from the text itself.
- Model input (the extraction sections, about 4 characters per token): median about 1,050 tokens, 90th
  percentile 1,880, 99th 2,940. Only 0.3% of inputs hit the 14,000-character cap. That fits Ollama's 8,192-token
  context and a ~4k training length.
- 99.9% of notes have a coded principal diagnosis, but only 36.9% are ICD-10 (the rest are ICD-9, before
  October 2015). Diagnoses are therefore scored on the code's text, not the ICD-10 code.
- Silver labels average 11.3 medication entries per note (continued medications included).
- Gold set: 200 test notes exported to `MIMIC_WORK_DIR/gold/` with the model input and the silver labels as a
  pre-fill; 50 are marked for a second annotator. `gold-check` reports progress and agreement.

### M6: zero- and few-shot baselines on MIMIC (200 gold-set notes, silver labels)
Same extractor and prompt as M4 (field guidance, greedy decoding), with a 16,384-token context and a
3,000-token answer cap. The notes are the 200 gold-set notes from the test split. Few-shot examples are 3 short
train notes, one with pending tests. The gold set is not annotated yet, so scoring is against the rule-based
silver labels. **These numbers are preliminary:** silver labels are a weak reference for every field except
medications (below). `mimic-bench --ref gold` reruns the same comparison once the gold set exists.

| Mode | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b zero-shot | 0.080 | 0.071 | 0.621 | 0.000 | 0.048 | 0.066 | **0.378** (0.342-0.414) | 0.296 | 0.72 | 7.2* |
| qwen3:14b zero-shot | 0.075 | 0.082 | 0.670 | 0.000 | 0.043 | 0.069 | **0.403** (0.375-0.432) | 0.315 | 0.76 | 12.3 |
| qwen3:8b few-shot (3) | 0.065 | 0.112 | 0.707 | 0.000 | 0.090 | 0.084 | **0.434** (0.404-0.464) | 0.339 | 0.78 | 9.1 |
| qwen3:14b few-shot (3) | 0.085 | 0.123 | 0.716 | 0.000 | 0.069 | 0.112 | **0.444** (0.417-0.470) | 0.343 | 0.79 | 14.7 |

One schema failure (8b zero-shot, after the retry); none in the other runs. \*Mean over notes 91-200: the first
90 shared the GPU with a stale run that was stopped (the full-run mean, 11.7 s, is in the results file).

What this shows:
- **Medications are the one field the silver labels measure fairly.** Silver medication changes are a diff of
  the admission and discharge lists, which is the task the model does. Real notes are much harder than the
  synthetic ones: F1 0.62-0.72 here vs 0.95-0.98 in M4, and change-type accuracy 0.72-0.79 vs 0.99-1.00. Three
  examples help both models (8b 0.621 → 0.707, CIs 0.570-0.669 vs 0.662-0.750). This is the field with the most
  room for fine-tuning and the clearest reference.
- **8b vs 14b:** 14b is ahead in every mode, but every CI overlaps. 8b few-shot is within the CI of 14b few-shot
  at 9.1 vs 14.7 s per note, which supports fine-tuning the 8b model.
- **Most of the low scores in other fields reflect the silver labels, not the models:**
  - Diagnoses: silver holds the hospital's billing-code titles (ICD-9 or ICD-10). The models copy the wording in
    Discharge Diagnosis. Billing codes also list more conditions: 6.8 secondary diagnoses per note vs 2.7-4.0
    from the models.
  - Follow-up: MIMIC redacts Followup Instructions to `___`, so silver has none. The models find about one
    follow-up per note elsewhere in the input, and every one counts as a false positive.
  - Red flags: the rules only take a list after phrases like "call your doctor", and 92.5% of notes get none.
    The models list 2.9-3.8 per note.
  - Pending tests: silver takes any sentence containing "pending" (0.41 per note). The models give 0.46-0.83.
- **Few-shot examples teach the silver conventions as well as the task.** None of the 3 examples has a
  follow-up (they are redacted). With them, the share of notes with no follow-up rises from 10% to 41% (14b) and
  from 47% to 55% (8b), and pending tests fall to silver's level (0.46-0.47 per note). Part of the few-shot gain
  against silver may come from matching conventions; the gold set will tell.

Two bugs were found and fixed before these numbers (both with tests):
1. Silver follow-ups were `___` placeholders, because MIMIC redacts Followup Instructions (198 of 200 gold-set
   notes). They are now dropped, prep and the gold export were rerun, and `docs/ANNOTATION.md` tells annotators
   to record follow-ups stated elsewhere.
2. After that fix no train note had a silver follow-up. The few-shot picker required one, so the first few-shot
   runs went ahead with 0 examples (14b scored the same as zero-shot). The picker no longer requires a
   follow-up, and a few-shot run with fewer than 3 examples now stops with an error.

**Decision before M7 (the real fine-tune).** Training on these silver labels as they are would teach the model
the silver conventions: billing-code titles for diagnoses, no follow-ups, and the rules' narrow red flags and
pending tests. It would score well against silver and worse against gold. Options:
- **A. Train on silver as is.** Simplest. The paper reports silver and gold scores side by side, and the gold
  scores for diagnoses, follow-up and red flags will be limited by the labels.
- **B. Align silver with the gold guideline first (recommended).** Take diagnoses from the note's Discharge
  Diagnosis section (the rule already exists for notes without codes) instead of billing-code titles. Keep
  medication changes as they are. Follow-up, red flags and pending tests stay rule-based and weak, and the paper
  shows per-field gold results so this is visible. About an hour of work: change prep, rerun it (12 minutes),
  re-export the gold set and rescore the saved predictions against the new silver labels (no new GPU runs).
- **C. Teacher labels.** Use qwen3:14b few-shot output as the labels for follow-up, red flags and pending tests
  (local, so the DUA allows it). Likely the best gold scores, but it is distillation: the comparison becomes
  "fine-tuned 8b vs its 14b teacher", and the teacher's errors become training targets. Worth discussing with
  the FIU co-authors before choosing it.

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
1. **MIMIC-IV-Note download:** done and verified (M5 above).
2. **Gold set** (paper 2's reference): exported. Annotate the 200 folders in `MIMIC_WORK_DIR/gold/` following
   `docs/ANNOTATION.md`, ideally with an FIU co-author for the 50-note overlap. Run `uv run python -m ctagent
   gold-check` to see progress and agreement.
3. **Publishing:** the repo was to stay local until the MIMIC baselines (M6) were in (Thiago, 2026-10-26). They
   are in. On your OK I'll create the public GitHub repo. It contains no MIMIC data and nothing from MMC (only
   aggregate metrics in `evals/results/`); the CI guard checks this.
4. **Silver labels for M7:** choose A, B or C in the M6 section above. I recommend B.

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
uv run python -m ctagent gold-export --n 200 --overlap 50
uv run python -m ctagent mimic-bench --mode few --model qwen3:8b   # add --ref gold once annotated
```
