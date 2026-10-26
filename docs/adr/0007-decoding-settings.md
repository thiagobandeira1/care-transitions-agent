# ADR 0007: Sampled decoding for drafting, greedy for extraction, and a runaway guard

**Status:** accepted (2026-10-26)

## Context
Every LLM call first ran at temperature 0 (greedy decoding). The review screen then showed a plan explanation,
meant to be 2-3 sentences, that repeated the same few sentences until the 1,500-token cap. 4 of the 6 saved
sample drafts (14b and 8b) had explanations of about 8,000 characters. The JSON still parsed, so nothing flagged
it, and the looping inflated the drafting latency numbers.

The Qwen3 model card says not to use greedy decoding because it can lead to endless repetitions. For
non-thinking mode it recommends temperature 0.7, top-p 0.8, top-k 20.

Greedy decoding also made the retry useless: the same prompt gives the same failed answer twice.

## Decision
- **Drafting** (plan explanation, call script, provider brief) uses `drafting_llm()`: the Qwen3 non-thinking
  settings, with a seed. Call k of a run uses seed + k, so a run is repeatable and a retry samples a new answer.
- **Extraction** stays greedy. Its output is short, constrained JSON, it had no schema failures in 428 benchmark
  calls, and deterministic output keeps the paper's scoring repeatable. Revisit if extraction on MIMIC (longer
  notes) shows looping.
- **Runaway guard** on every free-text draft field: longer than a field-specific limit (400 or 1,000 characters),
  a repeated sentence, or a repeated list item fails validation. The call is retried once and then flagged for a
  human (ADR 0002). The check runs after decoding, so text is never cut off silently.

## Consequences
Drafts differ from run to run unless the seed and call order match. Drafting latency and flag counts were
re-measured after this change; the earlier numbers included the looping.
