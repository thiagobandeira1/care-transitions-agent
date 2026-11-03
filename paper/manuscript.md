# From discharge alerts to follow-up work: design and evaluation of a local, human-in-the-loop AI agent for primary care

**Thiago Bandeira**

General Manager and Head of AI, Mendez Medical Center, Lauderhill, Florida, USA. Correspondence:
tnunez@mendezmedicalcenter.com

---

## Abstract

**Objective.** Primary care practices learn about their patients' hospital discharges from payer alerts, but
turning each alert into timely follow-up is manual work, and the hospital's discharge summary rarely reaches the
practice. We designed an AI agent that runs entirely on a practice's own hardware and evaluated each of its
steps on public hospital data.

**Materials and Methods.** For each discharge, the agent runs a fixed workflow. Deterministic code scores
readmission risk (LACE) and sets the follow-up tier, transitional care management deadlines and check-in dates.
A local language model extracts structured facts from the discharge summary when one exists, then drafts a
coordinator call script and a provider brief, which a grounding check compares with the record. A person
approves every output. We evaluated the risk tiers against 30-day readmissions in 64,036 MIMIC-IV hospital
stays. We evaluated extraction on 200 MIMIC-IV-Note discharge summaries, comparing prompted Qwen3 8B and 14B
models with QLoRA fine-tunes trained on rule-based or teacher labels, against hand and model-made references.
We evaluated the drafts on 195 MIMIC discharges and 30 synthetic patients.

**Results.** The high-risk tier flagged 38.2% of stays, whose 30-day readmission rate was 28.3%, against 14.3%
for the rest (LACE AUROC 0.638, 95% CI 0.631-0.646). Against hand labels, fine-tuned 8B models extracted
diagnoses best (micro F1 0.93-0.94, against 0.69-0.72 for prompted models). Across all fields, prompted models
were best (0.70-0.72), while the fine-tunes performed at the level of their rule-based training labels
(0.58-0.60, against 0.58) and reproduced a rule error in medication changes. On 195 discharges, drafting took
16.8 s per discharge on one consumer GPU. There were no validation failures, no medication absent from the
record, and 93.6% of extracted medication changes reached the provider brief.

**Discussion.** Deterministic rules can carry the time-critical decisions while small local models handle text.
Fine-tuning on cheap labels transfers the labels' errors. In practice, the scarce input is the discharge
summary, not the model.

**Conclusion.** A local, human-in-the-loop agent can turn discharge alerts into reviewable follow-up work on
commodity hardware. A prompted base model is the safer default for extraction; fine-tuning should be limited to
fields whose labels have been checked against an independent reference.

---

## 1. Introduction

After a hospital stay, the primary care team has to act on what happened. It contacts the patient, reconciles
medications, schedules a visit, follows up on results still pending, and watches for warning signs. Discharge
information often reaches primary care late or incomplete,¹ and adverse events after discharge are common, many
of them related to medications.² In the United States, transitional care management (TCM) adds deadlines: an
interactive contact within two business days of discharge, and a face-to-face visit within 7 or 14 days
depending on the complexity of the patient's problems.³

Many practices learn about discharges from payer encounter notifications, daily alerts that list which of their
patients were admitted or discharged. An alert is not follow-up work. Someone still has to decide who is at high
risk, what is due by when, what the hospital changed, and what to say on the phone. In a small practice that is
usually one coordinator working through the list by hand.

Large language models (LLMs) can extract clinical information and draft text,⁴ and agent systems chain such steps
together. Three obstacles keep them out of small practices. Sending patient notes to a cloud service raises
privacy and contractual questions. Open-weight models small enough to run on a single consumer GPU are weaker
than the largest cloud models. And autonomous agents that choose their own actions are hard to make safe and
auditable in clinical work.

We designed a care-transitions agent around these constraints: a fixed workflow, deterministic code for every
rule-based decision, small local models only for reading and writing text, and a person who approves every
output. The work aims to help primary care practices adopt such tools on hardware they already own, so the code,
the benchmark and every aggregate result are open. This paper asks what the agent does and how well each step
performs on real hospital data. It also asks a question that arose while building it: what a small local model
learns when it is fine-tuned on cheap, rule-based labels, compared with prompting the same model.

## 2. Materials and methods

### 2.1 The agent

The agent runs one short, fixed-order workflow per discharge (Figure 1), built with LangGraph. Deterministic code
makes every rule-based decision. The language model has narrow jobs inside specific steps and never chooses
actions or tools.

![Figure 1](figure1.png)

**Figure 1.** One run per discharge alert, in a fixed order. Blue steps are deterministic code, orange steps use
a local language model, and the green step is a person. Extraction is skipped when no discharge summary is
available. A daily job starts a short new run for each patient with a check-in due, so no run stays open for the
30 days.

The workflow has eight steps. **Ingest** normalizes the day's payer discharge file to one event per stay, with
dates, age, disposition, acuity and prior use. **EHR context** retrieves active problems and medications from the
practice's record. **Score** computes the LACE index⁵ (length of stay, acuity of admission, Charlson
comorbidity, and emergency visits in the prior six months) and flags Medicare beneficiaries under 65, who
qualify through disability and are readmitted more often. We chose a published score over a trained model
because, in the author's earlier work on payer notifications, machine-learning models did not outperform LACE.
**Extract**, when a discharge summary is available, has a local model turn it into structured JSON: principal and
secondary diagnoses, medication changes, follow-up appointments, pending tests and red-flag symptoms. **Plan**
applies rules (Table 1) and has the model write a two-to-three-sentence explanation for the coordinator. High
risk means LACE ≥ 10, or LACE ≥ 7 for a Medicare beneficiary under 65; these thresholds are provisional, to be
calibrated on each practice's outcomes. **Draft** has the model write a coordinator call script (opening,
questions, medication review, warning signs, scheduling) and a one-page provider brief (reason for admission,
key conditions, medication changes, what to watch for, risk summary). A deterministic grounding check then flags
any medication or condition a draft names that is not in the patient's record. **Human review** pauses the run,
with its state saved in a local SQLite checkpointer, until a coordinator or provider approves, edits or rejects
the drafts. Nothing reaches a patient or the record without that decision. **Track** stores decisions and
outcomes locally; a daily job starts a short run for each patient with a check-in due, instead of keeping a run
open for 30 days.

**Table 1.** The follow-up plan's rules.

| | High risk | Standard |
|---|---|---|
| Interactive contact | within 2 business days | within 2 business days |
| Visit target | within 3 days | within 7 days |
| TCM visit deadline | 7 days (99496) | 14 days (99495) |
| Check-ins | days 10 and 20 | none |

Every model output is validated against a schema and retried once. A second failure falls back to the rule
text, or to an empty value, and flags the case for a person. Drafts have length limits, and a list that repeats
an item is rejected as a runaway generation. The code refuses cloud model tags, remote endpoints and hosted
tracing; models are served locally by Ollama. The review screen times each review and records how much of each
draft was edited and whether flags were real problems, so the agent can measure its own use.

### 2.2 Data

**MIMIC.** We used the discharge summaries of MIMIC-IV-Note v2.2⁶,⁷ (331,793 notes from 145,914 patients) and
the hospital module of MIMIC-IV v3.1⁶,⁸ on PhysioNet.⁹ Notes were split by patient into training, validation and
test sets (70/10/20), so no patient appears in two splits. The model reads seven sections of each summary in a
fixed order: discharge diagnosis, admission medications, discharge medications, discharge instructions,
follow-up instructions, pending results and hospital course. Inputs are capped at 14,000 characters, which 0.3%
reach; the median input is about 1,050 tokens. MIMIC replaces the Follow-up Instructions section with a
placeholder, so follow-ups have to come from other sections.

**Synthetic benchmark.** Qwen3-14B wrote 150 discharge summaries from Synthea patient records.¹⁰ A note was kept
only if it mentioned every reference fact, so every field has a known answer. The notes are split 43 for
few-shot examples and 107 for testing, and are public with the code.

### 2.3 Evaluating the risk tiers

The cohort was every admission with a discharge summary in the test split, excluding deaths in hospital and
discharges to hospice, for which the agent plans no follow-up. We computed LACE as the agent does at discharge.
Length of stay came from admission and discharge times. An admission counted as acute if it was emergency or
urgent, or if the patient was registered in the emergency department. Charlson comorbidity came from the
admission's billing diagnoses, using Quan's enhanced ICD-9-CM and ICD-10 coding algorithms¹¹ matched by code
prefix (ICD-10-CM codes are matched with the ICD-10 prefixes). Prior emergency visits were ED-registered
admissions 2-182 days before; MIMIC's hospital module has no ED-only visits, so this undercounts them. The
Medicare flag came from the admission's insurance. The primary outcome was an unplanned admission (not
elective, not same-day surgery) 1-30 days after discharge; MIMIC records only its own hospital, so readmissions
elsewhere are missed. The secondary outcome was death or readmission within 30 days, LACE's original outcome. We
report the AUROC of the score, the share of stays in the high tier, the readmission rate in each tier, and the
share of readmissions caught by the high tier (sensitivity), with the same measures for LACE ≥ 10 alone. 95%
confidence intervals come from 1,000 bootstrap resamples of patients.

### 2.4 Evaluating extraction

**Task.** Every key of the six-field JSON is required, and the prompt states each field's definition and
conventions; without this guidance, zero-shot models skipped whole fields in early experiments. A note whose
answer still fails validation after the retry counts as an empty prediction. Repeated list items are dropped
before scoring, in predictions and references alike.

**Training labels.** Rule-based "silver" labels come from each note's own structure. Diagnoses come from the
Discharge Diagnosis section: the first line is the principal diagnosis and the rest are secondary (a principal
diagnosis is found in 95.9% of notes). Medication changes compare the numbered or bulleted admission and
discharge lists by drug name and dose: `started` if only on the discharge list, `stopped` if only on the
admission list, and otherwise `dose_changed` or `continued`. Follow-ups come from the redacted section, so silver
has none; pending tests are sentences containing "pending"; red flags are the list after phrases such as "call
your doctor". As an alternative, Qwen3-14B with three training examples labeled follow-up, pending tests and red
flags for 2,500 training and 300 validation notes (11.2 s per note, about 9 GPU hours), while diagnoses and
medication changes stayed silver. This teacher listed follow-ups in 40% of notes, pending tests in 19% and red
flags in 74%.

**Models.** We prompted Qwen3-8B and Qwen3-14B¹² zero-shot and with three fixed training examples, served by
Ollama in 4-bit (Q4_K_M) with greedy decoding, a 16,384-token context and a 3,000-token answer cap. We fine-tuned
Qwen3-8B with QLoRA:¹³,¹⁴ a 4-bit NF4 base, LoRA rank 16 (alpha 32) on all linear layers, one epoch at learning
rate 2e-4 with cosine decay, an effective batch of 8, and loss on the answer only, using the zero-shot prompt.
Examples longer than 3,072 tokens were dropped rather than truncated, keeping about 89% of notes. We trained four
models (Table 2). Each adapter was merged, converted to GGUF with llama.cpp, quantized to Q4_K_M (5.0 GB) and
served like the base models. The untouched base model went through the same conversion as a quantization control
(Base-8B-local); it scored within 0.02 of the library model on both benchmarks.

**Table 2.** The fine-tuned models.

| Model | Training labels | Training notes |
|---|---|---|
| Silver-8851 | silver | 8,851 (of 10,000 sampled; 4.4 h) |
| Silver-2500 | silver | 2,500 (71 min) |
| Teacher-2500 | teacher for follow-up, pending tests and red flags; silver for the rest | the same 2,500 (74 min) |
| Teacher-1250 | as Teacher-2500 | the first 1,250 of them (37 min) |

**References.** Training labels cannot serve as the reference, because a model trained on them is rewarded for
imitation. We built two references on the 200 test notes, both in the same local annotation tool, which shows
the note beside the silver labels as editable tables and saves only answers that pass the schema. The author
hand-labeled 51 notes (median 0.3 minutes per note). This reference covers the principal and secondary diagnoses
only: follow-up, pending tests and red flags were left out as too ambiguous for a quick single pass, and
medication changes were excluded after a check found a systematic error in them (Section 3.2). As a second
reference, Claude Opus 5.5 (Anthropic), a cloud service, filled in all six fields on all 200 notes by operating
the same tool through its browser extension (median 0.5 minutes per note). This was the only step in which note
text left the local machine;¹⁵ every evaluated model, and every other step, ran locally. We treat these labels
as model-made, not as hand annotation. Both references started from the silver labels.

**Metrics.** List fields are scored as sets, with greedy pairing. A match is exact (normalized equality) or
relaxed (token Jaccard overlap of at least 0.5); relaxed is the primary metric. A medication change matches only
when the drug name and the change type both match, and change-type accuracy among matched drug names is reported
separately. Precision, recall and F1 are micro-averaged over notes, and micro F1 pools the fields each reference
covers. 95% confidence intervals are note-level percentile bootstraps (1,000 resamples on MIMIC, 2,000 on the
synthetic notes). Predictions were generated once and rescored against each reference.

### 2.5 Evaluating the drafts

On synthetic patients, the full workflow ran on 30 Synthea discharges with Qwen3-14B and Qwen3-8B on the GPU, and
on 5 discharges without it. Drafting used the Qwen3 model card's recommended non-thinking sampling with a fixed
seed, because greedy decoding caused repetition loops in drafts.

On MIMIC, the full workflow ran up to the review pause on the 200 test notes. Five patients who died or went to
hospice were skipped, leaving 195 discharges. Each discharge event came from the hospital module (times, age,
disposition, acuity, prior emergency visits and Medicare), and the record stand-in held the admission's billing
diagnoses plus the diagnoses and medications from the note's extraction. Extraction used the saved Qwen3-14B
zero-shot predictions, the best overall in Table 6, and the same model drafted, so one model stays loaded. A
first run showed long condition lists and occasional validation failures; the final prompt asks the brief for at
most six conditions, most important first, and forbids placeholders such as "[Your Name]". We report the final
run, with the first for comparison.

We measured by automated counts only; no draft was read. Record grounding is the agent's own check against the
record. Note grounding counts the medication names in the drafts that appear in the note text (the whole name,
or every word of four or more letters in it), and the conditions in the brief that share at least one content
word with the note. Completeness, measured against the extraction, counts the changed medications (started,
stopped or dose changed) named in the brief, with the same change type, and in the call script, and the red
flags covered by a warning sign in the call script (one sign holding at least half of the flag's content
words). We also counted validation failures and seconds per discharge.

### 2.6 Hardware and software

All work ran on one NVIDIA RTX 4080 (16 GB), an Intel i9-14900K CPU and 64 GB of RAM, under Windows. The agent
uses Python 3.12.10, LangGraph 1.2.17 with langgraph-checkpoint-sqlite 3.1.2, LangChain Core 1.7.0,
langchain-ollama 1.1.0, Pydantic 2.14.1 and Streamlit 1.66.1, with models served by Ollama 0.40.13. Fine-tuning
used PyTorch 2.11.0 (CUDA 12.8), Transformers 5.20.2, TRL 1.16.0, PEFT 0.21.2, bitsandbytes 0.50.2 and Accelerate
1.15.0; conversion used llama.cpp release b11921. Analyses used pandas 3.1.0 and NumPy 2.5.4, and the figure
matplotlib 3.11.2.

### 2.7 Ethics

MIMIC-IV and MIMIC-IV-Note are de-identified. Their collection was reviewed by the Institutional Review Board at
the Beth Israel Deaconess Medical Center, which granted a waiver of informed consent and approved the data
sharing, and we used them under the PhysioNet credentialed data use agreement. The synthetic benchmark and the
synthetic patients involve no real person. Section 4.5 describes Mendez Medical Center's follow-up system, with
the practice's written approval; no data from the practice were used. No MIMIC data, labels, predictions or drafts are in the public
repository, which holds only code, the synthetic benchmark and aggregate metrics.

## 3. Results

### 3.1 Risk tiers

The cohort held 64,036 stays from 28,220 patients, after excluding 1,692 deaths in hospital and 671 hospice
discharges. 12,601 stays (19.7%) were followed by an unplanned readmission within 30 days, and 1,332 patients died
within 30 days. 71.1% of stays had at least one Charlson condition, 29.2% a prior ED-registered admission and
80.9% an acute admission; the median LACE was 8 (interquartile range 6-11).

LACE discriminated modestly, with an AUROC of 0.638 for readmission and 0.654 for death or readmission (Table 3).
The agent's high tier flagged 38.2% of stays; their readmission rate was 28.3%, twice the standard tier's 14.3%,
and the tier caught 55.1% of all readmissions. LACE ≥ 10 alone flagged 34.9% at a similar rate (28.9%) and caught
51.3%, so the Medicare under-65 rule added 3.3 points of stays and 3.8 points of sensitivity.

**Table 3.** The agent's risk tiers on MIMIC-IV, 64,036 stays (95% CI from a patient bootstrap).

| Measure | Estimate (95% CI) |
|---|---|
| LACE AUROC, 30-day readmission | 0.638 (0.631-0.646) |
| LACE AUROC, 30-day death or readmission | 0.654 (0.647-0.660) |
| Stays in the high tier | 38.2% (37.5-39.0) |
| Readmission rate, high tier | 28.3% (27.5-29.2) |
| Readmission rate, standard tier | 14.3% (13.9-14.7) |
| Readmissions caught by the high tier | 55.1% (53.6-56.6) |
| Stays with LACE ≥ 10 | 34.9% (34.3-35.6) |
| Readmission rate, LACE ≥ 10 | 28.9% (28.0-29.7) |
| Readmissions caught by LACE ≥ 10 | 51.3% (49.9-52.7) |

### 3.2 Extraction

**Synthetic notes: the label source decides the prose fields.** On notes with known answers (Table 4), the
prompted models scored 0.948-0.979 micro F1 and the rules 0.696. The silver fine-tunes lost most of the base
model's skill on the prose fields: Silver-8851 scored 0.057 on follow-up, against 0.771 for the base model
through the same conversion. On the same 2,500 notes, replacing silver with teacher labels raised micro F1 from
0.842 to 0.948 and follow-up from 0.190 to 0.797. Teacher-2500 matched the base model overall (0.948) and stayed
below its teacher (0.979), and doubling the teacher-labeled notes from 1,250 to 2,500 left micro F1 unchanged
(0.949 and 0.948).

**Table 4.** Synthetic benchmark, 107 test notes, relaxed F1 (95% CI for micro F1).

| Model | Principal | Secondary | Medications | Follow-up | Pending | Red flags | Micro F1 |
|---|---|---|---|---|---|---|---|
| Rules | 1.000 | 0.963 | 0.907 | 0.159 | 0.452 | 0.037 | 0.696 (0.675-0.716) |
| Base-8B-local, zero-shot | 0.939 | 0.982 | 0.961 | 0.771 | 0.958 | 0.994 | 0.948 (0.935-0.960) |
| Qwen3-8B, few-shot | 0.925 | 0.979 | 0.983 | 0.856 | 0.972 | 0.994 | 0.966 (0.956-0.974) |
| Qwen3-14B, few-shot (teacher) | 0.935 | 0.981 | 0.981 | 0.981 | 0.928 | 0.994 | 0.979 (0.970-0.987) |
| Silver-8851 | 1.000 | 0.964 | 0.917 | 0.057 | 0.604 | 0.679 | 0.789 (0.766-0.812) |
| Silver-2500 | 1.000 | 0.963 | 0.942 | 0.190 | 0.609 | 0.902 | 0.842 (0.822-0.861) |
| Teacher-1250 | 1.000 | 0.995 | 0.975 | 0.752 | 0.892 | 0.960 | 0.949 (0.938-0.960) |
| Teacher-2500 | 1.000 | 0.995 | 0.949 | 0.797 | 0.905 | 0.978 | 0.948 (0.934-0.961) |

**MIMIC diagnoses: the fine-tunes' gain holds against hand labels.** Against the hand labels (Table 5), the
fine-tuned models scored 0.927-0.943 micro F1 on the diagnoses, against 0.695-0.724 for the prompted models
zero-shot, with no overlap between the two groups' confidence intervals, and 0.808-0.849 few-shot. The principal
diagnosis was easy for every model (0.91-0.98); the gap was in secondary diagnoses. The prompted models found
most of them (recall 0.80-0.88 zero-shot) but listed many that the reference did not (precision 0.48-0.53),
while the fine-tunes kept to the Discharge Diagnosis list (precision 0.87-0.91, recall 0.95-0.99). The rules
agreed with the hand labels at 0.933 and the model-made reference at 0.845, and the model-made reference ranked
the models the same way.

**Table 5.** MIMIC-IV-Note, 51 hand-labeled notes, diagnoses only, relaxed F1 (95% CI for micro F1).

| Model | Principal | Secondary | Micro F1 |
|---|---|---|---|
| Rules (silver) | 0.931 | 0.933 | 0.933 |
| Silver-8851 | 0.950 | 0.937 | 0.942 (0.902-0.980) |
| Silver-2500 | 0.931 | 0.925 | 0.927 (0.877-0.970) |
| Teacher-1250 | 0.970 | 0.931 | 0.943 (0.894-0.984) |
| Teacher-2500 | 0.940 | 0.933 | 0.935 (0.891-0.979) |
| Base-8B-local, zero-shot | 0.941 | 0.642 | 0.724 (0.619-0.821) |
| Qwen3-8B, zero-shot | 0.911 | 0.625 | 0.702 (0.583-0.815) |
| Qwen3-8B, few-shot | 0.961 | 0.800 | 0.849 (0.746-0.938) |
| Qwen3-14B, zero-shot | 0.922 | 0.618 | 0.695 (0.580-0.793) |
| Qwen3-14B, few-shot | 0.980 | 0.737 | 0.808 (0.700-0.909) |

**MIMIC, all fields: the fine-tunes score at the level of their rules.** Against the model-made reference on all
six fields (Table 6), the prompted models zero-shot scored highest (0.703-0.723), while every fine-tune scored
0.583-0.600, against 0.584 for the rules themselves. On medication changes the prompted models reached 0.73-0.78
F1 with 0.85-0.89 change-type accuracy, and the fine-tunes 0.61-0.63 and 0.73-0.75, the rules' level (0.640 and
0.74). The silver fine-tunes never listed a follow-up; the teacher fine-tunes recovered part of it (0.30) but
stayed below the prompted models (0.40-0.49), and red flags showed the same order. Pending tests were hard for
every model (0.10-0.30) and for the rules (0.08). At most one of 200 notes per model failed validation.

**Table 6.** MIMIC-IV-Note, 200 notes, LLM reference, relaxed F1 (95% CI for micro F1).

| Model | Principal | Secondary | Medications | Follow-up | Pending | Red flags | Micro F1 | Change type acc. |
|---|---|---|---|---|---|---|---|---|
| Rules (silver) | 0.838 | 0.833 | 0.640 | 0.000 | 0.083 | 0.056 | 0.584 (0.548-0.621) | 0.74 |
| Silver-8851 | 0.843 | 0.834 | 0.628 | 0.000 | 0.101 | 0.155 | 0.583 (0.546-0.622) | 0.74 |
| Silver-2500 | 0.833 | 0.827 | 0.634 | 0.000 | 0.102 | 0.318 | 0.593 (0.554-0.630) | 0.75 |
| Teacher-1250 | 0.841 | 0.842 | 0.625 | 0.303 | 0.304 | 0.357 | 0.600 (0.567-0.633) | 0.75 |
| Teacher-2500 | 0.835 | 0.831 | 0.610 | 0.300 | 0.203 | 0.366 | 0.588 (0.550-0.625) | 0.73 |
| Base-8B-local, zero-shot | 0.852 | 0.749 | 0.725 | 0.416 | 0.225 | 0.754 | 0.704 (0.676-0.729) | 0.85 |
| Qwen3-8B, zero-shot | 0.852 | 0.729 | 0.735 | 0.404 | 0.210 | 0.725 | 0.703 (0.672-0.727) | 0.86 |
| Qwen3-8B, few-shot | 0.865 | 0.873 | 0.739 | 0.260 | 0.250 | 0.333 | 0.668 (0.640-0.694) | 0.88 |
| Qwen3-14B, zero-shot | 0.845 | 0.704 | 0.776 | 0.492 | 0.234 | 0.683 | 0.723 (0.698-0.744) | 0.89 |
| Qwen3-14B, few-shot | 0.877 | 0.808 | 0.729 | 0.357 | 0.293 | 0.412 | 0.675 (0.647-0.700) | 0.86 |

**A rule error the fine-tunes learned.** The medication rule marks a drug `started` when it is not on a numbered
admission list. In 22 of the 51 hand-labeled notes, the admission section had no numbered items (18) or was
absent (4), so the rule saw no admission drugs. Checking each `started` drug for its name in the admission
section text, 304 of the rules' 1,211 `started` rows (25%) across the 200 notes named a drug found there, against
18 of 688 (3%) in the model-made reference. The hand labels still had 84 of 348 (24%), against the rules' 90 on
the same notes, which is why medication changes were left out of the hand reference. The fine-tunes' medication
scores sit at the rules' level, consistent with having learned this rule.

**Scored against their own training labels, the fine-tunes looked best.** Against the silver labels of the same
200 notes, Silver-8851 reached 0.959 micro F1, against 0.566-0.638 for the prompted models. Its output matched
silver's statistics closely: 2.25 secondary diagnoses per note with 36% of notes empty (silver: 2.25 and 36%),
11.3 medication entries per note (silver: 11.3), and no follow-ups.

### 3.3 Drafts and speed

On 30 synthetic discharges (Table 7), Qwen3-14B drafted the plan explanation, call script and provider brief in
15.8 s per discharge on the GPU, with no validation failures and no grounding flags. Qwen3-8B was 2.5 times faster;
its single flag was a real error, a misspelled condition. Without a GPU, a morning batch of 15 discharges would
take about 47 minutes with 14B or 17 minutes with 8B, workable overnight.

**Table 7.** Drafting on Synthea discharges (three model calls per discharge).

| Model | Device | n | Mean s / discharge | Median | Max | Schema failures | Discharges with grounding flags |
|---|---|---|---|---|---|---|---|
| Qwen3-14B | GPU | 30 | 15.8 | 15.5 | 23.1 | 0 | 0 |
| Qwen3-8B | GPU | 30 | 6.2 | 5.5 | 10.9 | 0 | 1 |
| Qwen3-14B | CPU | 5 | 186.3 | 187.4 | 279.0 | 0 | 0 |
| Qwen3-8B | CPU | 5 | 69.2 | 67.7 | 96.7 | 0 | 0 |

On 195 MIMIC discharges (Table 8), the final run drafted each discharge in 16.8 s on average (median 16.1, maximum
31.3) with no validation failures. The record check flagged no medication; 98.4% of the medication names in the
drafts appeared in the note text, and the other 34 came from the extraction without appearing verbatim. Briefs
listed 5.9 conditions on average; the record check flagged 3 of 1,144 (0.3%), in 2 discharges, and 65.2% shared a
content word with the note, most of the rest coming from the admission's billing diagnoses. The provider brief
named 93.6% of the extraction's changed medications (92.7% with the same change type), the call script 86.5%,
and the call script covered 86.9% of the extraction's red flags. With comorbidity scored, 67 of the 195
discharges (34%) were high risk.

The first run, before the two prompt changes, listed 12.2 conditions per brief and had 16 of 2,371 flagged, 4
provider briefs that failed validation, and 18.2 s per discharge, with completeness within 1.2 points of the
final run. The MIMIC runs had no practice name to give, and despite the instruction every call script still
contained a practice-name placeholder; with the name in the prompt, as in 30 further synthetic cases, none did.

**Table 8.** The agent's drafts on 195 MIMIC-IV-Note discharges (Qwen3-14B extraction and drafting, final
prompt; first run in brackets).

| Measure | Count | Share |
|---|---|---|
| Discharges with a failed document after the retry | 0 / 195 [4] | 0.0% [2.1%] |
| Medications in drafts not in the record | 0 / 2,098 | 0.0% [0.0%] |
| Medications in drafts found in the note text | 2,064 / 2,098 | 98.4% [98.3%] |
| Conditions in briefs not in the record | 3 / 1,144 | 0.3% [0.7%] |
| Conditions in briefs sharing a word with the note | 746 / 1,144 | 65.2% [61.8%] |
| Extracted changed medications named in the brief | 1,075 / 1,149 | 93.6% [93.9%] |
| ... with the same change type | 1,065 / 1,149 | 92.7% [93.0%] |
| Extracted changed medications named in the call script | 994 / 1,149 | 86.5% [86.4%] |
| Extracted red flags covered by the call script | 615 / 708 | 86.9% [85.7%] |

Extraction took 4.5-5.3 s per note for the fine-tuned models, 7.2-12.3 s for the prompted models zero-shot and
9.1-13.5 s few-shot. With 14B extraction (12.3 s) and drafting (16.8 s), a discharge with a summary takes about 30
seconds on one consumer GPU, or about 7 minutes for a morning batch of 15.

## 4. Discussion

### 4.1 Principal findings

Each step of the agent can be tested on real hospital data, and each behaved differently. The rule-based risk
tier separated patients modestly but usefully: the high tier's readmission rate was twice the standard tier's,
although LACE's AUROC was only 0.64, so the tier is for ordering the coordinator's work, not a prediction to act
on alone. Extraction was the hard step: the best prompted models reached about 0.70-0.72 micro F1 against a
model-made reference. Drafting was the reliable step: on 195 real discharges the drafts named no medication
absent from the record, flagged the few unsupported conditions, and carried 94% of the extracted medication
changes into the provider brief. Because the drafts are faithful to their input, extraction errors pass into
grounded-looking documents, which is why human review is necessary, not a formality.

Small prompt changes fixed what the first run showed. Asking for at most six conditions, most important first,
halved the brief's condition list, removed the validation failures and cut flagged conditions from 16 to 3,
without losing completeness. Placeholders, however, disappeared only when the practice's name was in the prompt;
an instruction alone did not remove them, so a deployment has to configure the practice's details.

### 4.2 Fine-tuning transferred the rules

The evidence is consistent across both benchmarks and both references (Table 9). Where the rules were right, as
for secondary diagnoses, fine-tuning gave a clear gain over prompted models, which over-listed. Where the rules
were wrong, as for medication change types when the admission list was not numbered, the fine-tunes learned the
error and fell below the prompted models. Where the rules were silent, as for follow-up redacted in MIMIC, the
fine-tunes learned to say nothing. The fine-tune did not learn the task beyond its labels; it learned the
labeler.

**Table 9.** What fine-tuning on rule-based labels transferred.

| Where the rules were | Example | Effect of fine-tuning |
|---|---|---|
| right | secondary diagnoses | a clear gain over prompted models |
| wrong | medication change types with unnumbered admission lists | the error learned; below prompted models |
| silent | follow-up, redacted in MIMIC | nothing learned |

The matched control isolates the cause: on the same notes, with the same training budget, changing only the label
source moved follow-up from 0.19 to 0.80 on the synthetic notes. Rule-based labels are a form of weak
supervision,¹⁶ and labels from a larger model a form of distillation;¹⁷ both are cheap, and both put a ceiling
on the student. On MIMIC, the teacher-label models recovered part of follow-up but stayed below the prompted
models and below their own teacher. Scoring against the training labels hid all of this: against silver, the
silver fine-tune looked far better than the prompted models (0.959 against 0.57-0.64), but against independent
references it was no better than the rules. Any project that trains on weak labels and tests on held-out labels
of the same kind will see the first picture.

For the agent, a prompted base model is therefore the default extractor. A fine-tuned model is worth using only
for fields whose labels have been checked against an independent reference, here the diagnoses, where it was
also about twice as fast.

### 4.3 Building references from pre-filled labels

The hand reference was made by correcting the silver labels, and on medication changes it still had 84 likely
errors of the kind the rules made, against the rules' 90 on the same notes. This is consistent with anchoring on
pre-annotation, which speeds up labeling but pulls annotators toward the suggestion.¹⁸ A targeted check found
it: the two references disagreed on medication change types, and a simple test of the disagreeing rows located
the error. Such checks are worth running whenever a reference is built from pre-filled labels.

### 4.4 Why a fixed workflow rather than an autonomous agent

Small local models are unreliable at choosing actions, while the tasks with deadlines and billing consequences
(TCM contact within two business days, visit windows, check-in dates) have exact rules. Keeping those in
deterministic code, and confining the model to reading and writing text that a person approves, makes every
decision reproducible and auditable. Every model failure becomes a visible flag rather than a silent error.

### 4.5 In practice, the discharge summary is the missing input

Mendez Medical Center, a multi-specialty primary care practice in Lauderhill, Florida, already runs the agent's
deterministic steps in production, on a hospital follow-up board the author built in the practice's own
dashboard. No data from the practice were used in this study; this section describes the system, not its
patients. The payer's admission alert opens a card while
the patient is still in the hospital, so follow-up is ready before discharge. Cards move through columns for new
discharges, call again, visit scheduled and unreachable, and a daily overview counts open cards, contacts due or
overdue, visits past the 14-day deadline and high-risk patients on the readmission protocol. Each card shows the
LACE score stated as a frequency ("about N in 100 similar past stays were readmitted within 30 days", from the
practice's own payer history), the high-risk rule with a coordinator override, the follow-up steps, and the call
results; an appointment check looks for a booked visit in the practice's schedule.

The board has no language model, because the practice receives the payer's alert but not the hospital's discharge
summary. This matches the long-standing gap in discharge communication to primary care,¹ and it decides where the
agent can help. The rule-based half works with the alert alone, which is why the practice could deploy it. The
language-model half needs the summary: without it, there is nothing to extract, and the call script and the brief
lose the medication review and the hospital's warning signs. The agent runs either way, skipping extraction when
no summary exists, but its full value depends on getting the summary to the practice, for example through a
health information exchange or the hospital's patient portal. The board also suggests one change to the agent:
risk stated as a frequency is easier for a coordinator to read than a score, so the plan explanation should do
the same, from each practice's own history.

### 4.6 Limitations

The hand reference is small and narrow: 51 notes and two fields, labeled quickly by one annotator from a
pre-filled form, with no agreement measure. Medication changes and the prose fields rest on a model-made
reference that also started from the pre-filled form; its accuracy on the prose fields is unmeasured, and the
low scores of every model on pending tests may partly reflect it. MIMIC comes from one academic center and redacts
its follow-up section, and the synthetic benchmark was written by Qwen3-14B, which may favor that model. We tested
one model family in 4-bit only, and relaxed matching treats brand and generic names as different. The drafts on
MIMIC were evaluated by automated counts, not read by a clinician; grounding checks catch invented names, not
every wrong statement, completeness is measured against the extraction rather than the note, and condition
grounding needs only one shared word. On MIMIC, the alert and record were stand-ins built from hospital tables,
prior emergency visits were undercounted, readmissions to other hospitals were missed, and billing diagnoses,
coded after discharge, stood in for the problem list a practice would have at discharge. The risk thresholds are
provisional and not calibrated on any practice's outcomes. Finally, this version reports no evaluation with
users.

### 4.7 Next steps

The next evaluation is of use. The review screen already records decisions, edits, review time and the reviewer's
verdict on flags, and a protocol is ready in which a practice's follow-up coordinator reviews 30 synthetic
discharges and completes a short questionnaire; aggregate counts from the practice's live board would add how the
rule-based steps are used day to day. Beyond that, the agent needs live outcomes (contact, visits and
readmissions), a frequency statement of risk calibrated on each practice's history, and above all a route for the
discharge summary to reach the practice.

## 5. Conclusion

A local, human-in-the-loop agent can turn hospital discharge alerts into reviewable follow-up work on a single
consumer GPU. Deterministic rules can carry the time-critical decisions, and its drafts stay faithful to the
record. Its weakest step is extraction, and fine-tuning a small model on rule-based labels transfers the rules,
errors included; a prompted base model is the safer default, and fine-tuning should be limited to fields whose
labels have been checked against an independent reference. In practice, what most limits such an agent in primary
care is not the model but access to the discharge summary.

---

## Declarations

**Ethics.** See Section 2.7.

**Data availability.** The code, the synthetic benchmark, the annotation and review tools, the script for Figure 1
and every aggregate metric reported here are public at https://github.com/thiagobandeira1/care-transitions-agent.
MIMIC-IV and MIMIC-IV-Note are available to credentialed users from PhysioNet. No MIMIC-derived data, labels,
predictions, drafts or model weights are distributed, as the data use agreement requires. No data from Mendez
Medical Center were used for training or evaluation.

**Code availability.** The author owns the code and releases it as open source. The repository's commands
reproduce every result from the data above. Fine-tuned weights are not shared, because they were trained on
MIMIC.

**Author contributions.** T.B. conceived the study, built the software, ran the analyses and wrote the
manuscript.

**Competing interests.** The author is General Manager and Head of AI at Mendez Medical Center, where the
follow-up board described in Section 4.5 runs. The software is open source. The author declares no other
competing interests.

**Funding.** This work received no specific funding.

**Use of generative AI.** Claude Opus 5.5 (Anthropic) produced the model-made reference labels (Section 2.4).
Claude Code (Anthropic) assisted with software development, analysis scripts and the drafting of this
manuscript. The author reviewed all code, analyses and text and takes responsibility for them.

**Acknowledgments.** The author thanks Oscar Mendez, MD, and Mendez Medical Center for supporting this work and
for testing the prototype.

## References

1. Kripalani S, LeFevre F, Phillips CO, Williams MV, Basaviah P, Baker DW. Deficits in communication and
   information transfer between hospital-based and primary care physicians: implications for patient safety and
   continuity of care. *JAMA*. 2007;297(8):831-841.
2. Forster AJ, Murff HJ, Peterson JF, Gandhi TK, Bates DW. The incidence and severity of adverse events affecting
   patients after discharge from the hospital. *Ann Intern Med*. 2003;138(3):161-167.
3. Centers for Medicare & Medicaid Services. Transitional care management services. MLN Booklet MLN908628.
   Baltimore, MD: CMS; August 2025.
4. Agrawal M, Hegselmann S, Lang H, Kim Y, Sontag D. Large language models are few-shot clinical information
   extractors. In: *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*. Abu
   Dhabi: Association for Computational Linguistics; 2022:1998-2022. doi:10.18653/v1/2022.emnlp-main.130
5. van Walraven C, Dhalla IA, Bell C, et al. Derivation and validation of an index to predict early death or
   unplanned readmission after discharge from hospital to the community. *CMAJ*. 2010;182(6):551-557.
6. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset.
   *Sci Data*. 2023;10:1. doi:10.1038/s41597-022-01899-x
7. Johnson A, Pollard T, Horng S, Celi LA, Mark R. MIMIC-IV-Note: deidentified free-text clinical notes (version
   2.2). PhysioNet. 2023. doi:10.13026/1n74-ne17
8. Johnson A, Bulgarelli L, Pollard T, Gow B, Moody B, Horng S, Celi LA, Mark R. MIMIC-IV (version 3.1).
   PhysioNet. 2024. doi:10.13026/kpb9-mt58
9. Pollard T, Moody BE, et al. PhysioNet as a global platform for biomedical research. *Nature Health*. 2026.
   doi:10.1038/s44360-026-00096-z
10. Walonoski J, Kramer M, Nichols J, et al. Synthea: an approach, method, and software mechanism for generating
    synthetic patients and the synthetic electronic health care record. *J Am Med Inform Assoc*.
    2018;25(3):230-238.
11. Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10
    administrative data. *Med Care*. 2005;43(11):1130-1139.
12. Yang A, et al. Qwen3 technical report. arXiv:2505.09388. 2025.
13. Hu EJ, Shen Y, Wallis P, et al. LoRA: low-rank adaptation of large language models. In: *ICLR*. 2022.
14. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: efficient finetuning of quantized LLMs. In:
    *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. 2023.
15. PhysioNet. Responsible use of MIMIC data with online services like GPT. April 18, 2023.
    https://physionet.org/news/post/gpt-responsible-use
16. Ratner A, Bach SH, Ehrenberg H, Fries J, Wu S, Ré C. Snorkel: rapid training data creation with weak
    supervision. *Proc VLDB Endowment*. 2017;11(3):269-282.
17. Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531. 2015.
18. Fort K, Sagot B. Influence of pre-annotation on POS-tagged corpus development. In: *Proceedings of the Fourth
    Linguistic Annotation Workshop*. Uppsala: Association for Computational Linguistics; 2010:56-63.
