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
