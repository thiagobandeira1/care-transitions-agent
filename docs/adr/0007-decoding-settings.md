# ADR 0007: Field guidance in the prompt, sampled decoding for drafting, and a runaway guard

**Status:** accepted (2026-10-26)

## Context
The review screen showed a plan explanation, meant to be 2-3 sentences, that repeated the same few
sentences until the 1,500-token cap. 4 of the 6 saved sample drafts (14b and 8b) had explanations of about
8,000 characters. The JSON still parsed, so nothing flagged it, and the looping inflated the drafting latency
numbers.

The first fix assumed greedy decoding was the cause. The Qwen3 model card says not to use greedy decoding
because it can lead to endless repetitions, and recommends temperature 0.7, top-p 0.8, top-k 20 for
non-thinking mode. Sampling alone did not fix it: with the runaway guard below in place, qwen3:8b still
failed 26 of 30 plan explanations (14b: 7 of 30). On 8 of those discharges, 4 still looped to the cap and
2 ran to 9 sentences.

The real cause: **the model never saw the instructions.** "2-3 plain sentences" lived only in the JSON
schema's field description. Ollama uses the schema to constrain decoding; it does not show it to the model.
The prompt said only "Explain the follow-up tier and next steps for the coordinator." With the field
descriptions added to the prompt, qwen3:8b wrote 12 of 12 valid explanations of about 250 characters, under
sampled and under greedy decoding.

The same gap applied to extraction: "include continued medications" was not in the zero-shot prompt, and
8b zero-shot left continued medications out.

## Decision
- **Field guidance in every prompt.** `field_guide(schema)` renders each field's name, type, allowed values
  and description as prompt text. Drafting prompts and the extraction prompt (and so the fine-tuning
  prompt, which reuses it) include it. Every field has a description (extraction schema 1.2).
- **Drafting** (plan explanation, call script, provider brief) uses `drafting_llm()`: the Qwen3 non-thinking
  settings, with a seed. Call k of a run uses seed + k, so a run is repeatable and a retry samples a new
  answer instead of repeating the failed one.
- **Extraction** stays greedy: deterministic output keeps the paper's scoring repeatable. Revisit if
  extraction on MIMIC (longer notes) shows looping.
- **Runaway guard** on every free-text draft field: longer than a field-specific limit (400 or 1,000
  characters), a repeated sentence, or a repeated list item fails validation. The call is retried once and
  then flagged for a human (ADR 0002). The check runs after decoding, so text is never cut off silently.
- Retry errors keep the end of the message, where the validator states its reason.

## Update (2026-10-28): looping in extraction
Extraction on MIMIC did loop: the smaller option C fine-tunes sometimes repeated one list item hundreds of
times in otherwise valid JSON (PROGRESS.md, Finding 2). Extraction stays greedy, so scores stay repeatable.
Instead, extraction schema 1.3 drops repeated list items (ignoring case and spacing) when an extraction is
validated. A repeat is dropped rather than failing validation, as it does in drafting, because the rest of
the answer is still usable, and a retry under greedy decoding would repeat the same answer.

## Consequences
Drafts differ from run to run unless the seed and call order match. Drafting latency and flag counts, and the
LLM extraction results, were re-measured after this change; PROGRESS.md has the numbers. Earlier drafting
numbers included the looping, and earlier zero-shot extraction numbers came from a prompt without the field
guidance.
