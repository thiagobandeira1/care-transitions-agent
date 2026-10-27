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
  is touched once.
- **Weights stay local:** adapters, merged and GGUF models are written outside the repo and never uploaded.

## Consequences
The fine-tuned model may partly learn the silver rules' biases. Comparing against gold, and against the rules
themselves as a baseline, shows whether it learned more than the rules. That comparison is part of paper 2.

## Update (2026-10-27): what the first fine-tune learned
- **Silver diagnoses now come from the note** (the Discharge Diagnosis section), not from billing-code titles
  (option B, decided by Thiago on 2026-10-26). Billing titles would have taught the model to write billing
  titles.
- **The expected bias is confirmed, and it is large.** The first run (silver labels only, 8,851 train notes)
  reproduces silver almost exactly: 0.958 micro F1 against silver on the 200 gold-set notes, with per-note
  item counts matching silver's. It also learned silver's blind spots. Silver has no follow-ups (MIMIC redacts
  that section), so the model never outputs one. On the public synthetic notes, where every field has a
  known answer, micro F1 fell from 0.947 (base 8b, zero-shot) to 0.788: follow-up 0.776 → 0.057, pending
  tests 0.944 → 0.592, red flags 0.994 → 0.679.
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
- **Gold agreement without a second annotator:** this is not an FIU project. The 50-note overlap needs a second
  PhysioNet-credentialed annotator, or else a second pass by the same annotator at least two weeks later,
  reported as intra-annotator agreement.
- **Export:** Ollama 0.40 no longer quantizes safetensors to Q4_K_M or loads LoRA adapters. The merged model is
  converted and quantized with llama.cpp, and a control (the base model through the same steps) isolates the
  fine-tune from quantization differences.
