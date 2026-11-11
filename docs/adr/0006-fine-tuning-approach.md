# ADR 0006: QLoRA with TRL + PEFT, silver-label training, gold-only evaluation

**Status:** accepted (2026-10-26)

## Context
Paper 2's core result compares a base 8B model (zero-shot and few-shot) with the same model fine-tuned to extract
structured facts from MIMIC-IV-Note discharge summaries. Hardware is one RTX 4080 (16 GB) on Windows. Hand
labels are expensive. A ~200-note gold set is feasible; tens of thousands of hand labels are not.

## Decision
- **Library:** TRL `SFTTrainer` + PEFT LoRA + bitsandbytes 4-bit (QLoRA), which installs cleanly on Windows with
  CUDA 12.8 PyTorch. Unsloth is an optional speed-up, not a dependency.
- **Format:** prompt-completion pairs, so the loss covers only the JSON answer. The prompt is byte-identical to
  the zero-shot extractor's, so the fine-tuned model is served with the same prompt and compared fairly.
- **Training labels:** rule-based silver labels (diagnoses from coded data, medication changes from the
  admission vs discharge lists, follow-up, pending tests and red flags from sections) on the train split only,
  split by patient. Silver label quality is measured against gold and reported.
- **Evaluation:** the hand-annotated gold set (test split, 50 notes double-annotated for agreement) is the
  primary reference. Silver-label metrics are secondary, and the synthetic benchmark is public but easier.
- **Model selection:** hyperparameters and checkpoint are chosen on the validation split only. The test split
  is touched once. (Not followed as written; see the 2026-11-09 update below.)
- **Weights stay local:** adapters, merged and GGUF models are written outside the repo and never uploaded.

## Consequences
The fine-tuned model may partly learn the silver rules' biases. Comparing against gold, and against the rules
themselves as a baseline, shows whether it learned more than the rules. That comparison is part of paper 2.

## Update (2026-10-27): what the first fine-tune learned
- **Silver diagnoses now come from the note** (the Discharge Diagnosis section), not from billing-code titles
  (option B, decided by Thiago on 2026-10-26). Billing titles would have taught the model to write billing
  titles.
- **The expected bias is confirmed, and it is large.** The first run (silver labels only, 8,851 train notes)
  reproduces silver almost exactly: 0.959 micro F1 against silver on the 200 gold-set notes, with per-note
  item counts matching silver's. It also learned silver's blind spots. Silver has no follow-ups (MIMIC redacts
  that section), so the model never outputs one. On the public synthetic notes, where every field has a
  known answer, micro F1 fell from 0.947 (base 8b, zero-shot) to 0.789: follow-up 0.776 → 0.057, pending
  tests 0.944 → 0.604, red flags 0.994 → 0.679.
- **Consequence:** labels for the weak fields (follow-up, pending tests, red flags) must improve before a
  fine-tuned model replaces the few-shot extractor in the agent. Option C (a 14b few-shot teacher for those
  fields, silver for diagnoses and medications) is the leading candidate. This run stays in paper 2 as the
  silver-only arm.
- **Option C started (2026-10-27).** qwen3:14b few-shot (the M6 configuration) labels follow-up, pending
  tests and red flags on 2,500 train notes and 300 validation notes; silver keeps diagnoses and medications.
  The notes are a nested, resumable sample, so the set can grow if a learning curve (1,250 vs 2,500) says more
  notes help. A silver-only model on the same 2,500 notes is the control, so only the label source differs.
  2,500 gives about 600 notes with pending tests, the rarest weak field, and the first run's validation loss
  had mostly flattened by 5,000 examples. The student cannot beat its teacher on these fields, so teacher
  quality matters more than note count.
- **Option C result (2026-10-27):**
  - **Synthetic notes:** teacher labels brought the weak fields back. On the same 2,500 notes, micro F1 is
    0.948 with teacher labels and 0.842 with silver. Follow-up rose from 0.19 to 0.80, and pending tests from
    0.61 to 0.91. c2500 matches the base 8b zero-shot (0.948) and trails its 14b teacher (0.979).
  - **Learning curve:** halving the notes (c1250) changes micro F1 by 0.001, so more notes from the same
    teacher are not the bottleneck.
  - **MIMIC:** the 2,500-note models sometimes repeated one list item hundreds of times, and those loops
    drove their first MIMIC scores down. Extraction schema 1.3 (2026-10-28) drops repeated list items on
    validation, for every model, and all saved predictions were rescored. c2500 then scores 0.808 against
    silver (it was 0.778), with silver-level diagnoses (0.987 / 0.967) and medications (0.912). On the weak
    fields it agrees with silver as little as its teacher does.
  - **Consequence:** c2500 is the candidate fine-tuned arm for paper 2. Gold decides teacher vs rules on the
    weak fields, and until then the few-shot extractor stays the agent's default.
- **Gold agreement without a second annotator:** there is no co-author to annotate. The 50-note overlap needs a second
  PhysioNet-credentialed annotator, or else a second pass by the same annotator at least two weeks later,
  reported as intra-annotator agreement.
- **Export:** Ollama 0.40 no longer quantizes safetensors to Q4_K_M or loads LoRA adapters. The merged model is
  converted and quantized with llama.cpp, and a control (the base model through the same steps) isolates the
  fine-tune from quantization differences.

## Update (2026-11-09): outcome
The comparison is reported in the paper ([paper/manuscript.md](../../paper/manuscript.md)), published as a
preprint on Zenodo: doi:10.5281/zenodo.24090449, which always resolves to the latest version (the figures here
are from version 1.3). Older entries here call it "paper 2". The references are in its Section 2.4, the results
in Section 3.2, and what fine-tuning learned in Section 4.2.
- **The references actually used.** Both were made in the local annotation tool by correcting the silver
  pre-fill, so they differ from the training labels but are not independent of the rules.
  - **Author-corrected diagnosis reference:** the first 51 of the 200 test notes, principal and secondary
    diagnoses only. The weak fields were left out before labeling, and medication changes after a check found a
    systematic error in them, before any model was scored against the hand labels. There is no second
    annotator; the same annotator's second pass, for intra-annotator agreement, is pending
    ([docs/ANNOTATION.md](../ANNOTATION.md)).
  - **Model-made reference for all six fields:** all 200 notes, corrected by Claude Opus 5.5 (a cloud service)
    through Anthropic's Claude browser extension. It is the exception in [ADR 0001](0001-two-track-data-policy.md),
    the only step in which note text left this machine, and it is never reported as hand annotation.
- **Outcome: a prompted base model is the agent's preferred default extractor.** On the diagnoses, the
  fine-tunes agreed best with the author-corrected reference (micro F1 0.93-0.94, against 0.69-0.72 for the
  prompted models zero-shot and 0.81-0.85 few-shot; the rules themselves 0.933). On all fields they scored at the
  level of their training labels (0.58-0.60, rules 0.584), below the prompted models (0.70-0.72 zero-shot,
  0.67-0.68 few-shot). The weak fields were scored against the model-made reference only. There the
  teacher-label fine-tunes beat the silver ones on follow-up (0.30 F1, where the silver ones list none) and red
  flags, but stayed below the prompted models zero-shot on both (follow-up 0.40-0.49), and pending tests were
  hard for every model. A fine-tuned model is worth using only for fields whose labels were checked against a
  reference other than the training labels, here the section-based diagnoses.
- **Two base-model builds.** `qwen3:8b` is the Ollama library build. Base-8B-local (`qwen3-8b-local`) is the
  untouched base model through the same GGUF conversion as the fine-tunes: the control under Export above. Their
  micro F1 was within 0.001 on the synthetic notes and the 200-note reference, and 0.022 higher for
  Base-8B-local on the 51-note reference. "Base 8b" in older entries can be either: on the synthetic notes, 0.947
  is `qwen3:8b` and 0.948 is Base-8B-local. [docs/MODEL_CARD.md](../MODEL_CARD.md) maps the fine-tunes' earlier
  names (M7, s2500, c1250, c2500) to the paper's.
- **Model selection and the test split, as done** (added 2026-11-10): each fine-tune used its final checkpoint,
  with no early stopping or checkpoint selection (paper Appendix A4). The test split was used more than once:
  repeated list items were dropped (schema 1.3) after loops were seen on the 200 test notes, all saved
  predictions were rescored under that rule and against each reference, the agent's extractor was chosen as
  the best overall on these notes (paper Table 6), and the drafting prompt was revised after a first run on
  195 of them. The paper therefore reports a development evaluation, not an untouched test (Sections 2.5 and
  4.6, Appendix C).
