# Progress: care-transitions-agent (Stage 4)

_Last updated: 2026-10-28, ~12:30 AM, after repeated list items started being dropped (extraction schema 1.3)
and every model was scored again. Published on 2026-10-28 at
https://github.com/thiagobandeira1/care-transitions-agent (code, docs and aggregate metrics only). The M5 to M8 sections report MIMIC aggregates only (counts, rates,
length quantiles, metrics). Every other number is from synthetic data (Synthea
and notes generated from it). None of it is MMC data._

## Where things stand
| Milestone | Status |
|---|---|
| M0 spec, ADRs, hardened skeleton, privacy guards | ✅ done |
| M1 Synthea cohort, LACE + under-65, TCM rules | ✅ done (743 Medicare index stays) |
| M2 LangGraph workflow, human review, SQLite tracking, daily check-ins | ✅ done |
| M3 real local drafting + GPU vs CPU benchmark | ✅ done (below), re-measured after the prompt fix |
| M4 public synthetic extraction benchmark | ✅ done: 150 notes (43 train / 107 test), rules vs zero-/few-shot 8b and 14b |
| M5 MIMIC-IV-Note prep code + gold-annotation tooling | ✅ done: download verified, prep run, 200-note gold set exported |
| M6 baselines on MIMIC (zero-/few-shot) | ✅ done against silver labels, with diagnoses now taken from the note (below); the gold-label run waits for annotation |
| M7 QLoRA fine-tune | ✅ done: Qwen3-8B, 1 epoch on 8,851 MIMIC train notes with silver labels, 4.4 h, eval loss 0.0136 |
| M8 GGUF/Ollama export + evaluation | ✅ done: Q4_K_M GGUF via llama.cpp, in Ollama as `ctagent-extractor`; scored on 200 MIMIC and 107 synthetic notes with a matched control (below) |
| M9 review screen and demo | ✅ smoke-tested on 5 synthetic discharges (queue, drafts, approve) |
| Option C: teacher labels for the weak fields | ✅ done: 3 fine-tunes (c2500, s2500 control, c1250) on 14b-labeled notes; weak fields recovered on synthetic; repetition loops found on MIMIC and fixed by dropping repeated list items (Finding 2) |

## Finding 1 (M7): the fine-tuned model copies its training labels, blind spots included
The fine-tune worked mechanically: no schema failures, 4.5 s per note (about half the base model's time), and
0.959 micro F1 against the silver labels on the 200 gold-set notes, against 0.567 for the same base model
exported the same way. But silver is what it learned to reproduce, gaps included:
- **Silver has no follow-ups** (MIMIC redacts that section), so the fine-tuned model never outputs one, in 0 of
  200 notes. The base model found about one per note.
- **On the public synthetic notes,** where every field has a known answer, micro F1 fell from 0.948 (control)
  to 0.789. Follow-up fell from 0.771 to 0.057, pending tests from 0.958 to 0.604, and red flags from 0.994 to
  0.679. Only the principal diagnosis improved (0.939 → 1.000).
- **Diagnoses and medications are where silver is good,** and there the fine-tune lifted the 8b model from
  0.63-0.92 to 0.98-1.00 against silver. If the gold set confirms silver on those fields, that part is a real
  gain.

So this model should not replace the few-shot extractor in the agent. The training labels for follow-up,
pending tests and red flags have to improve first (option C, Finding 2). This run stays in paper 2 as the
silver-only arm, and ADR 0006 records the finding.

## Finding 2 (option C, 2026-10-27): teacher labels bring the weak fields back
**Synthetic benchmark (107 test notes, every field has a known answer).** The three option C models, with M7 and
the base models for reference:

| Model | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b-local zero-shot (control) | 0.939 | 0.982 | 0.961 | 0.771 | 0.958 | 0.994 | **0.948** (0.935-0.960) | 0.882 | 1.00 | 5.4 |
| qwen3:8b few-shot (3) | 0.925 | 0.979 | 0.983 | 0.856 | 0.972 | 0.994 | **0.966** (0.956-0.974) | 0.941 | 1.00 | 5.8 |
| qwen3:14b few-shot (3), the teacher | 0.935 | 0.981 | 0.981 | 0.981 | 0.928 | 0.994 | **0.979** (0.970-0.987) | 0.969 | 0.99 | 9.4 |
| M7: silver, 8,851 notes | 1.000 | 0.964 | 0.917 | 0.057 | 0.604 | 0.679 | **0.789** (0.766-0.812) | 0.758 | 0.96 | 3.8 |
| s2500: silver, 2,500 notes (control) | 1.000 | 0.963 | 0.942 | 0.190 | 0.609 | 0.902 | **0.842** (0.822-0.861) | 0.800 | 0.98 | 4.0 |
| c1250: silver + teacher, 1,250 notes | 1.000 | 0.995 | 0.975 | 0.752 | 0.892 | 0.960 | **0.949** (0.938-0.960) | 0.879 | 0.99 | 3.8 |
| **c2500: silver + teacher, 2,500 notes** | 1.000 | 0.995 | 0.949 | 0.797 | 0.905 | 0.978 | **0.948** (0.934-0.961) | 0.875 | 0.98 | 3.9 |

Every row was run again with extraction schema 1.3, which drops repeated list items (MIMIC, below). The
synthetic predictions from the first runs were not saved, so this was a rerun, not a rescore. c1250, c2500,
the rules and the library 8b and 14b zero-shot runs (M4) reproduced their first scores exactly. The other rows moved by at
most 0.015 on one field and 0.001 in micro F1. The fine-tunes were about 0.3 s per note slower, because the
MIMIC rescore was using the CPU at the same time.

- **The label source is the cause.** On the same 2,500 notes, teacher labels give 0.948 and silver gives 0.842.
  Follow-up rises from 0.19 to 0.80, pending tests from 0.61 to 0.91 and red flags from 0.90 to 0.98.
- **c2500 matches the base model overall** (0.948 for both) and keeps the fine-tune's edge on diagnoses
  (principal 1.000 vs 0.939, secondary 0.995 vs 0.982). It stays below 8b few-shot (0.966) and its 14b teacher
  (0.979), mostly on follow-up (0.80 vs 0.86 and 0.98). A student is expected to trail its teacher.
- **More notes help little.** c1250 → c2500 leaves micro F1 unchanged (0.949 vs 0.948). The weak fields gain
  2-5 points (follow-up 0.75 → 0.80, red flags 0.96 → 0.98). Labeling 2,500 more notes (about 8 GPU hours)
  is not worth it before the gold set says whether the teacher is right on MIMIC.
- **It is fast:** 3.9 s per synthetic note and 4.9 s per MIMIC note, 2.4-2.8 times faster than the 14b
  teacher (9.4 and 13.5 s), with no examples in the prompt.

**MIMIC (200 gold-set notes, against silver).** Scored with repeated list items dropped (schema 1.3, below):

| Model | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:14b few-shot (3), the teacher | 0.932 | 0.776 | 0.685 | 0.000 | 0.094 | 0.123 | **0.632** (0.597-0.666) | 0.552 | 0.78 | 13.5 |
| M7: silver, 8,851 notes | 0.995 | 0.997 | 0.980 | 0.000 | 0.518 | 0.669 | **0.959** (0.942-0.973) | 0.956 | 0.99 | 4.5 |
| s2500: silver, 2,500 notes | 0.995 | 0.993 | 0.926 | 0.000 | 0.331 | 0.449 | **0.890** (0.865-0.915) | 0.881 | 0.96 | 4.7 |
| c1250: silver + teacher, 1,250 notes | 0.982 | 0.954 | 0.864 | 0.000 | 0.069 | 0.109 | **0.776** (0.745-0.804) | 0.751 | 0.92 | 5.3 |
| **c2500: silver + teacher, 2,500 notes** | 0.987 | 0.967 | 0.912 | 0.000 | 0.099 | 0.098 | **0.808** (0.782-0.834) | 0.791 | 0.95 | 4.9 |

c1250 and c2500 each had one schema failure (unparseable output) out of 200.

**Repetition loops, found and fixed.** On a few notes the 2,500-note models repeated one item many times. One
s2500 answer listed the same pending test 962 times, and c1250 and c2500 each listed one secondary diagnosis
249 times. The JSON still parsed, so nothing flagged it. A local script found them by counting items in the
saved predictions (counts only, no text). M7, trained on 8,851 notes, repeated at most one item. The base 8b
zero-shot also repeats (189 duplicate red flags), so this is not new to fine-tuning, but the smaller fine-tunes
do it more.

A list field is a set, so a repeated item is never right. Extraction schema 1.3 drops repeats (ignoring case
and spacing) whenever an extraction is validated: for every model, in the agent and in the benchmarks, and in
the reference labels too. The JSON schema and the prompt did not change, so the fine-tuned models still see
their training prompt. The saved MIMIC predictions were rescored without calling the models (`--rescore`).
Micro F1 before → after:
- s2500: 0.664 → 0.890
- c1250: 0.688 → 0.776
- c2500: 0.778 → 0.808
- 8b zero-shot: 0.546 → 0.566
- every other model: up by 0.001-0.008

A loop still costs generation time up to the 3,000-token answer cap. A cap on items per list would bound that,
but it is not needed for correctness.

- **Option C did not cost the diagnoses.** c2500 keeps silver-level principal and secondary diagnoses (0.987
  and 0.967). Medication changes (0.912) are close to the silver control on the same notes (0.926). M7's 0.980
  suggests medications are where more notes help.
- **On the weak fields c2500 learned its teacher, as intended.** It agrees with silver about as little as the
  14b teacher does (pending tests 0.099 vs 0.094, red flags 0.098 vs 0.123). It lists 126 follow-ups on the 200
  notes, where silver has none. So MIMIC micro F1 against silver no longer ranks the arms on those fields. Only
  the gold set can say whether the teacher or the rules are right.
- **Without the loops, the silver control looks strong against silver** (s2500 0.890, M7 0.959). That is the
  imitation effect from Finding 1, not evidence that silver labels are better.

**Next step for the model:** c2500 is the candidate fine-tuned arm. Until gold says the teacher is right on
MIMIC, the few-shot extractor stays the agent's default.

### How option C was run
There were no co-authors to wait for (Thiago, 2026-10-27), so it went ahead right away.
`scripts/teacher_label.py` has qwen3:14b, with the M6 few-shot prompt (3 train examples, seed 5), label
follow-up, pending tests and red flags. Diagnoses and medication changes stay silver. Then three fine-tunes
use the same notes:

| Run | Labels | Train notes | Question it answers |
|---|---|---|---|
| `ctagent-extractor-c2500` | silver + teacher | 2,500 | Does option C bring the weak fields back? |
| `ctagent-extractor-s2500` | silver only | the same 2,500 | Control: only the label source differs |
| `ctagent-extractor-c1250` | silver + teacher | first 1,250 of them | Learning curve: would more notes help? |

The teacher also labels 300 validation notes for the eval loss. Each model is exported like M8 and scored on
the 107 synthetic notes and the 200 MIMIC gold-set notes (against silver).

**What it took:**
- **Labeling:** 2,500 train and 300 validation notes, 11.2 s per note, 1 failure (skipped), about 9 GPU hours
  including a restart.
- **Teacher label rates on the 2,500 train notes:** follow-ups in 40% of notes (0.66 per note), pending tests
  in 19% (0.30), red flags in 74% (2.2).
- **Training:**
  - c2500: 74 min, 2,498 examples (2 over 3,072 tokens)
  - s2500: 71 min, 2,500 examples
  - c1250: 37 min, 1,249 examples
  - Peak GPU use was 10.5 GB.
  - Final eval loss: 0.024 (c2500), 0.019 (s2500), 0.031 (c1250). The c and s losses are not comparable,
    since their validation labels differ.
- **Export and scoring:** about 25 minutes per model.

**Why 2,500 notes:**
- **Enough examples of the rarest field.** On the 200 gold-set notes the 14b teacher lists follow-ups in 46%
  of notes, pending tests in 24% and red flags in 78%. At 2,500 notes that's about 600 notes with pending
  tests (the rarest field), 1,100 with follow-ups and 1,900 with red flags.
- **The first fine-tune flattened early.** Validation loss was 0.0233 at 1,600 examples, 0.0175 at 3,200,
  0.0153 at 4,800 and 0.0136 at 8,000. Most of the drop came before 5,000 examples.
- **The teacher sets the ceiling.** On the weak fields a student learns its teacher, mistakes included. More
  notes from the same teacher won't make it better than the teacher.
- **It fits in one run.** Notes that fit the 3,072-token training limit took about 6 s each in the smoke test
  (the 13.5 s M6 figure was for full-length gold-set notes), so 2,800 notes is roughly 5-8 hours. Training
  each 2,500-note model takes about 1.3 hours.
- **It can grow.** Labels are appended note by note in a fixed order, and a smaller run uses the first notes.
  If c1250 → c2500 still improves the weak fields, rerunning with `--n-train 5000` adds notes without
  relabeling any.

**Design details:**
- Train notes follow the training script's shuffled order (seed 3), so they are a subset of the first
  fine-tune's 8,851.
- Notes whose training example would pass 3,072 tokens are skipped before labeling, with 192 tokens of margin
  for the teacher's longer answers, so no GPU time goes to notes the trainer would drop.
- The 3 few-shot example notes are skipped.
- A stopped run resumes with the same command.

**Caveat:** MIMIC redacts the Followup Instructions section, so the follow-ups the teacher finds come from
other sections, such as the discharge instructions. On the weak fields the teacher and silver disagree a lot:
pending tests precision 0.10 and recall 0.09, red flags 0.08 and 0.33 (M6, 14b few-shot vs silver). Only the
gold set can say which one is right.

## Earlier finding (2026-10-26): the model never saw the field instructions
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
bootstrap. Greedy decoding, extraction schema 1.3 (field guidance in the prompt; repeated list items dropped,
Finding 2).

| Mode | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|
| Rules (no LLM) | 1.000 | 0.963 | 0.907 | 0.159 | 0.452 | 0.037 | **0.696** (0.675-0.716) | 0.654 | 0.0 |
| qwen3:8b zero-shot | 0.953 | 0.986 | 0.954 | 0.776 | 0.944 | 0.994 | **0.947** (0.933-0.959) | 0.881 | 5.4 |
| qwen3:14b zero-shot | 0.849 | 0.960 | 0.974 | 0.796 | 0.965 | 0.994 | **0.947** (0.935-0.958) | 0.886 | 9.0 |
| qwen3:8b few-shot (3) | 0.925 | 0.979 | 0.983 | 0.856 | 0.972 | 0.994 | **0.966** (0.956-0.974) | 0.941 | 5.8 |
| qwen3:14b few-shot (3) | 0.935 | 0.981 | 0.981 | 0.981 | 0.928 | 0.994 | **0.979** (0.970-0.987) | 0.969 | 9.4 |

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
  percentile 1,880, 99th 2,940. Only 0.3% of inputs hit the 14,000-character cap.
- Silver diagnoses come from the note's Discharge Diagnosis section, the same convention as the gold set
  (option B, below). A principal diagnosis is found in 95.9% of notes, with 2.4 secondary diagnoses per note.
  Billing codes are no longer used for labels: 99.9% of notes have one, but only 36.9% are ICD-10 (the rest
  are ICD-9, before October 2015), and their titles are not the note's wording.
- Silver labels average 11.3 medication entries per note (continued medications included).
- Gold set: 200 test notes exported to `MIMIC_WORK_DIR/gold/` with the model input and the silver labels as a
  pre-fill; 50 are marked for a second annotator. `gold-check` reports progress and agreement.

### M6: zero- and few-shot baselines on MIMIC (200 gold-set notes, silver labels)
Same extractor and prompt as M4 (field guidance, greedy decoding), with a 16,384-token context and a
3,000-token answer cap. The notes are the 200 gold-set notes from the test split. Few-shot examples are 3 short
train notes, one with pending tests. The gold set is not annotated yet, so scoring is against the rule-based
silver labels, with diagnoses taken from the note's Discharge Diagnosis section (option B, below). **These
numbers are preliminary:** silver is a fair reference for diagnoses and medications, and a weak one for
follow-up, pending tests and red flags. `mimic-bench --ref gold` reruns the comparison once the gold set exists.

| Mode | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b zero-shot | 0.916 | 0.707 | 0.631 | 0.000 | 0.049 | 0.084 | **0.566** (0.530-0.602) | 0.477 | 0.72 | 7.2* |
| qwen3:14b zero-shot | 0.869 | 0.659 | 0.671 | 0.000 | 0.043 | 0.069 | **0.569** (0.533-0.605) | 0.478 | 0.76 | 12.3 |
| qwen3:8b few-shot (3) | 0.924 | 0.847 | 0.689 | 0.000 | 0.079 | 0.104 | **0.638** (0.602-0.672) | 0.554 | 0.77 | 9.1 |
| qwen3:14b few-shot (3) | 0.932 | 0.776 | 0.685 | 0.000 | 0.094 | 0.123 | **0.632** (0.597-0.666) | 0.552 | 0.78 | 13.5 |

All four rows were rescored on 2026-10-28 with repeated list items dropped (schema 1.3, Finding 2). Micro F1
rose by 0.001-0.008, except 8b zero-shot (0.546 → 0.566), which had repeated 189 red flags. The zero-shot
rows are the original predictions scored again against the new labels (`--rescore`, no model calls), so their
timings are unchanged. The few-shot rows are new runs: examples are
picked by their silver labels, so the new labels picked different examples. One schema failure (8b zero-shot,
after the retry); none in the other runs. \*Mean over notes 91-200: the first 90 shared the GPU with a stale run
that was stopped (the full-run mean, 11.7 s, is in the results file).

What this shows:
- **The models were right about diagnoses; the old reference was wrong.** Against billing-code titles, diagnosis
  F1 was 0.07-0.12 and micro F1 0.38-0.44 (in git history). Against the note's own Discharge Diagnosis
  section, principal diagnosis is 0.87-0.93 and secondary 0.66-0.85, and micro F1 rises to 0.57-0.64. 14b
  zero-shot is again the weakest on the principal diagnosis (0.869), as in M4 (0.849).
- **8b ties 14b with three examples and is 1.5x faster.** Few-shot micro F1 is 0.638 vs 0.632 at 9.1 vs 13.5 s
  per note, and 8b has the best secondary-diagnosis score (0.847). This supports fine-tuning the 8b model.
- **Medications remain the hard field with a fair reference:** F1 0.63-0.69 and change-type accuracy 0.72-0.78,
  against 0.95-0.98 and 0.99-1.00 on the synthetic notes. Three examples help 8b (0.631 → 0.689). The new
  examples gave 8b a slightly lower medication score than the old ones (0.689 vs 0.707, before repeats were
  dropped; the CIs overlap), so the choice of examples moves results by a few points. The picker is deterministic, and the
  paper should say which examples it used.
- **Follow-up, pending tests and red flags are still limited by the silver labels:**
  - Follow-up: MIMIC redacts Followup Instructions to `___`, so silver has none. Zero-shot models find about one
    follow-up per note elsewhere in the input, and every one counts as a false positive.
  - Red flags: the rules only take a list after phrases like "call your doctor", and 92.5% of notes get none.
    The models list 2.6-3.6 per note.
  - Pending tests: silver takes any sentence containing "pending" (0.39 per note). The models give 0.36-0.83.
- **Few-shot examples teach the silver conventions as well as the task.** None of the 3 examples has a
  follow-up (they are redacted). With them, the share of notes with no follow-up rises from 10% to 55% (14b) and
  from 47% to 72% (8b). 8b few-shot gives 2.2 secondary diagnoses per note, with 36% of notes having none, almost
  exactly silver's 2.25 and 36%. Part of the few-shot gain against silver comes from matching conventions; the
  gold set will tell how much.

Three bugs were found and fixed along the way (all with tests):
1. Silver follow-ups were `___` placeholders, because MIMIC redacts Followup Instructions (198 of 200 gold-set
   notes). They are now dropped, and `docs/ANNOTATION.md` tells annotators to record follow-ups stated
   elsewhere.
2. After that fix no train note had a silver follow-up. The few-shot picker required one, so the first few-shot
   runs went ahead with 0 examples (14b scored the same as zero-shot). The picker no longer requires a
   follow-up, and a few-shot run with fewer than 3 examples now stops with an error.
3. The first version of the option-B prep re-read the 6-million-row billing-code file once per note, so prep
   stalled. It now reads it once (a test counts the reads), and prep took 11 minutes again.

**Decision before M7: option B (Thiago, 2026-10-26).** Training on billing-code titles would have taught the
model to write billing titles, which would score well against that silver and poorly against gold. Silver
diagnoses now come from the note's Discharge Diagnosis section (headings such as "Chronic issues" and `___`
placeholders are skipped). Medication changes are unchanged. Follow-up, red flags and pending tests stay
rule-based and weak, and the paper will show per-field gold results so this is visible. Option C (14b teacher
labels for the weak fields) was left for later; it started on 2026-10-27 (Option C section).

### M7: QLoRA fine-tune
**Pipeline test.** `scripts/train_qlora.py --source synthetic --base Qwen/Qwen3-0.6B --max-train 40 --max-eval 10`
in `.venv-train` (torch 2.11 + CUDA 12.8, TRL 1.16, PEFT 0.21, bitsandbytes 0.50): 4-bit NF4 base, LoRA rank 16
on all linear layers, loss on the JSON answer only. It ran end to end: 18 s of training, eval loss 0.375, eval
token accuracy 0.93, adapter written outside the repo. The training prompt reuses the zero-shot extractor's
prompt, field guidance included, so the fine-tuned model is served with the same prompt.

**Memory on the 16 GB GPU: attention was running on the slow kernel.** A first 8b smoke run at a
4,096-token limit took 68 s per optimizer step with the GPU full: on Windows the CUDA driver spills into system
RAM instead of failing. A probe on synthetic text (same model, LoRA and trainer settings) showed memory growing
faster than linearly with length, the signature of an attention matrix being built in full. The cause:
Qwen3-8B has 32 query heads and 8 key/value heads, and transformers passes that to PyTorch's attention as
`enable_gqa`. PyTorch for Windows has no flash attention, and its memory-efficient kernel does not accept
`enable_gqa`, so every layer fell back to the math kernel. Measured on one layer at 3,072 tokens, that costs
5.0 GB of extra memory against 0.2 GB for the memory-efficient kernel with the key/value heads repeated.
`train_qlora.py` now repeats the heads (`use_efficient_attention`), which is the same computation.

| Tokens per sequence | Before: s per 4 sequences, peak allocated | After the fix |
|---|---|---|
| ~1,000 | 6.0 s, 8.1 GB | |
| ~2,000 | 13.3 s, 10.0 GB | 7.5 s, 9.2 GB |
| ~3,000 | 23.7 s, 13.3 GB | 11.7 s, 10.6 GB |
| ~4,000 | did not fit | 16.0 s, 12.0 GB |

The first real run (before the fix) was stopped after 5 minutes without finishing a step: the process held
15.2 GB on the GPU plus 1.1 GB spilled into system RAM, about 2 GB above the probe, and the Windows desktop
used another 1.3 GB. With the fix, the second run trained at 16.5 s per step, but PyTorch's cache still grew to
fill the GPU (15.2 GB) and had begun to spill (0.2 GB). It was stopped after 6 steps and restarted with a cap
(`--gpu-fraction`, default 0.85, about 13.6 GB), which makes PyTorch free cached memory instead of spilling.
The GPU then sits at 11.8 GB in total, desktop included.

**Lengths and the limit.** On 3,000 sampled train examples, prompt plus answer is 1,989 tokens at the median,
3,112 at the 90th percentile and 4,382 at the 99th; the JSON answer alone is 416 at the median. A 3,072-token
limit keeps 89.4% (4,096 would keep 98.2%, but with the ~2 GB real-run overhead and the desktop it would leave
no margin on 16 GB, and a spill overnight would stall the run). Longer examples are dropped, not truncated, since truncation would cut the end of the JSON
answer and train on invalid JSON. The fine-tuned model therefore trains on the shorter 89% of notes and is
evaluated on all of them, which the paper should state.

**The real run** (2026-10-26 11:39 PM to 2026-10-27 4:06 AM): Qwen3-8B, 4-bit NF4, LoRA rank 16 (alpha 32)
on all linear layers, learning rate 2e-4 with cosine decay and 3% warm-up, 1 epoch, effective batch 8 (batch
1, 8 gradient accumulation steps), gradient checkpointing, bf16, eval batch 1. Data: 10,000 random train
notes, of which 8,851 fit (1,149 dropped), and 264 of 300 validation notes for eval loss every 200 steps.
1,107 optimizer steps at about 13 s each: 4.4 hours (15,940 s), peak GPU memory 10.6 GB allocated, no errors.
Checkpoints were saved every 200 steps (`--resume` continues after an interruption), and the adapter and
checkpoints are in `MIMIC_WORK_DIR/models`, outside the repo.

| Step | 200 | 400 | 600 | 800 | 1,000 | 1,107 (end) |
|---|---|---|---|---|---|---|
| Eval loss | 0.0233 | 0.0175 | 0.0153 | 0.0138 | 0.0136 | 0.0136 |
| Eval token accuracy | 99.4% | 99.6% | 99.6% | 99.7% | 99.7% | 99.7% |

Train loss fell from 0.55 at the first log (step 10) to about 0.015 at the end (mean 0.024). Eval loss was
flat over the last ~300 steps, with no sign of overfitting. The low loss only says the model reproduces silver
tokens: most answer tokens are JSON structure or text copied from the note. M8 measures what it learned.

### M8: export to Ollama, and the fine-tuned model's scores
**Export.** `scripts/export_ollama.py` merges the adapter into the bf16 base on the CPU, converts it to GGUF
with llama.cpp's `convert_hf_to_gguf.py`, quantizes it to Q4_K_M with `llama-quantize` (4.9 bits per weight,
5.0 GB), and imports it into Ollama as `ctagent-extractor` with qwen3:8b's prompt template and stop tokens.
The plan was to let Ollama quantize, but Ollama 0.40 no longer quantizes safetensors to Q4_K_M or loads LoRA
adapters. llama.cpp is a shallow clone plus the official release b11921 CPU binaries, in
a tools folder, outside the repos. Two snags, both fixed in the script:
- transformers 5 re-saves tokenizer files in a format the converter's pinned transformers 4 can't read. The
  fine-tune doesn't change the tokenizer, so the base model's own files are copied in.
- The converter warns "Unknown RoPE type: default". It is harmless: the GGUF's RoPE base was checked
  (1,000,000, as in the base config).

A smoke test on one synthetic note gave valid JSON with no thinking tokens, in 5 s.

**Control.** The untouched base model went through the same conversion and quantization (`qwen3-8b-local`).
The library qwen3:8b file is 5.23 GB against 5.03 GB for ours, so the two quantizers made slightly different
per-tensor choices. With the control, the comparison differs only in the fine-tune. The control scores within
0.02 of the library model on both benchmarks, so quantization explains almost none of the gap.

MIMIC, 200 gold-set notes, silver labels (preliminary until the gold set is annotated):

| Model | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b zero-shot (Ollama library) | 0.916 | 0.707 | 0.631 | 0.000 | 0.049 | 0.084 | **0.566** (0.530-0.602) | 0.477 | 0.72 | 7.2* |
| qwen3-8b-local zero-shot (control) | 0.922 | 0.723 | 0.628 | 0.000 | 0.059 | 0.083 | **0.567** (0.532-0.602) | 0.476 | 0.72 | 7.2 |
| qwen3:8b few-shot (3) | 0.924 | 0.847 | 0.689 | 0.000 | 0.079 | 0.104 | **0.638** (0.602-0.672) | 0.554 | 0.77 | 9.1 |
| qwen3:14b few-shot (3) | 0.932 | 0.776 | 0.685 | 0.000 | 0.094 | 0.123 | **0.632** (0.597-0.666) | 0.552 | 0.78 | 13.5 |
| **ctagent-extractor (fine-tuned 8b)** | 0.995 | 0.997 | 0.980 | 0.000 | 0.518 | 0.669 | **0.959** (0.942-0.973) | 0.956 | 0.99 | 4.5 |

\*As in M6: mean over notes 91-200. Both tables use schema 1.3 (repeated list items dropped, Finding 2).

Synthetic benchmark, 107 test notes with known answers:

| Model | Principal dx | Secondary dx | Med changes | Follow-up | Pending tests | Red flags | **Micro F1 (95% CI)** | Exact micro F1 | Med change type acc. | s / note (GPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b zero-shot (Ollama library) | 0.953 | 0.986 | 0.954 | 0.776 | 0.944 | 0.994 | **0.947** (0.933-0.959) | 0.881 | 0.99 | 5.4 |
| qwen3-8b-local zero-shot (control) | 0.939 | 0.982 | 0.961 | 0.771 | 0.958 | 0.994 | **0.948** (0.935-0.960) | 0.882 | 1.00 | 5.4 |
| qwen3:8b few-shot (3) | 0.925 | 0.979 | 0.983 | 0.856 | 0.972 | 0.994 | **0.966** (0.956-0.974) | 0.941 | 1.00 | 5.8 |
| qwen3:14b few-shot (3) | 0.935 | 0.981 | 0.981 | 0.981 | 0.928 | 0.994 | **0.979** (0.970-0.987) | 0.969 | 0.99 | 9.4 |
| **ctagent-extractor (fine-tuned 8b)** | 1.000 | 0.964 | 0.917 | 0.057 | 0.604 | 0.679 | **0.789** (0.766-0.812) | 0.758 | 0.96 | 3.8 |

What this shows:
- **Against silver, the fine-tuned model is close to a copy of the rule-based labeler.** Its per-note item
  counts on the 200 notes match silver's: 2.25 secondary diagnoses per note with 36% of notes empty (silver:
  2.25 and 36%), 11.3 medication changes (silver: 11.3), and no follow-ups (silver: none). The zero-shot model
  gives 2.7 secondary diagnoses with 22% empty, and about one follow-up per note. 0.959 measures imitation of
  the rules, not clinical correctness. The gold set measures that.
- **It learned silver's blind spots.** It never outputs a follow-up, and it lists fewer pending tests than the
  base model (0.17 per note vs 0.83; silver 0.39). On the synthetic notes, which state every field
  explicitly, it lost most of the base model's skill on follow-up, pending tests and red flags. It still
  lists red flags on MIMIC more often than silver (1.06 per note vs 0.60), so it did not copy that rule
  exactly.
- **It is faster:** 4.5 s per note on MIMIC against 7.2-9.1 s, since its answers are shorter and the prompt
  carries no examples.
- **On diagnoses and medications it matches silver almost perfectly** (0.98-1.00, against 0.63-0.92 for the
  base model). Silver is a fair reference there, so this is probably a real gain, but only the gold set can
  confirm it.

**Next step for training: better labels for the weak fields** (option C). Train on silver diagnoses and
medications plus 14b few-shot labels for follow-up, pending tests and red flags, then compare three arms on
gold: few-shot, silver-only fine-tune (this run), and silver + teacher fine-tune. Labeling 10,000 train notes
with 14b few-shot takes about 37 GPU hours at 13.5 s per note (or about 9 hours for 2,500 notes). Done on
2,500 notes on 2026-10-27 (Finding 2).

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
   `docs/ANNOTATION.md`. Run `uv run python -m ctagent gold-check` to see progress and agreement. With no
   co-author, the 50-note overlap needs either a second annotator with their own PhysioNet credentialing (the
   MIMIC DUA forbids showing notes to anyone else), or your own second pass on those 50 notes at least two
   weeks later, reported as intra-annotator agreement.
3. **Publishing:** done on 2026-10-28, after an audit, on Thiago's go. The audit checked every tracked file and
   the full history. It found no MIMIC data, nothing from MMC and no secrets, and no file was ever committed and
   then deleted. Personal context was removed from the history, with commit dates kept. Local paths were
   replaced with placeholders. The repo has aggregate metrics only (`evals/results/`), and the CI guard
   checks this. Publishing also meets the MIMIC DUA's code-sharing clause once results are published.
4. **Silver labels for M7:** decided (B) and done.
5. **Labels for the next fine-tune:** option C is done (Finding 2). The gold set (item 2) now decides two open
   questions: whether the fine-tune's gains on diagnoses and medications are real, and whether the teacher or
   silver is right on follow-up, pending tests and red flags.
6. **Next, agreed 2026-10-27:** the two papers. Paper 1 is MMC readmission, in the work session. Paper 2 is this
   extraction comparison, built on gold.

## How to run
```
uv sync --group dev
uv run pytest
uv run python -m ctagent cohort
uv run python -m ctagent bench --model qwen3:14b --n 30          # GPU; add --cpu for CPU-only
uv run python -m ctagent make-benchmark --n 150
uv run python -m ctagent extract-bench --mode few --model qwen3:8b
uv run python -m ctagent extract-bench --mode few --model qwen3:8b --rescore   # rescore saved predictions
uv run python -m ctagent demo --n 5                               # morning batch into data/demo
uv run streamlit run app/review_app.py -- --workspace data/demo   # review screen
uv run python -m ctagent mimic-prep                               # after the MIMIC download
uv run python -m ctagent gold-export --n 200 --overlap 50
uv run python -m ctagent mimic-bench --mode few --model qwen3:8b   # add --ref gold once annotated
uv run python -m ctagent mimic-bench --mode few --model qwen3:8b --rescore   # rescore saved predictions
.venv-train/Scripts/python scripts/train_qlora.py --source mimic --base <models>/Qwen3-8B
.venv-train/Scripts/python scripts/export_ollama.py --base <models>/Qwen3-8B --adapter <MIMIC_WORK_DIR>/models/Qwen3-8B-extractor-mimic/adapter
.venv-train/Scripts/python scripts/export_ollama.py --base <models>/Qwen3-8B --name qwen3-8b-local --out <models>/qwen3-8b-gguf   # control
uv run python -m ctagent mimic-bench --mode zero --model ctagent-extractor
uv run python -m ctagent extract-bench --mode zero --model ctagent-extractor
.venv-train/Scripts/python scripts/teacher_label.py --base <models>/Qwen3-8B --n-train 2500 --n-val 300   # option C, resumable
.venv-train/Scripts/python scripts/train_qlora.py --source mimic --base <models>/Qwen3-8B --teacher-dir <MIMIC_WORK_DIR>/teacher --labels teacher --max-train 2500   # --labels silver: control
```
