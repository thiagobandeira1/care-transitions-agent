# ADR 0002: Fixed LangGraph pipeline, not an autonomous tool-choosing agent

**Status:** accepted (2026-10-25)

## Context
The agent runs on local 7-14B models. Small models are unreliable at choosing tools and planning multi-step
actions, and a clinical workflow needs predictable, auditable behavior.

## Decision
- The order of steps is fixed in code: ingest → ehr_context → score → extract → plan → draft → human_review →
  track. `ehr_context` runs before `score` because LACE's comorbidity part needs the problem list when the
  payer feed has no diagnosis codes.
- State holds JSON-ready dicts only, so checkpoints never need custom deserialization.
- The LLM does only narrow jobs inside specific nodes: extraction, drafting, and explaining a decision that the
  rules already made.
- Scoring, tiers, TCM deadlines and check-in scheduling are deterministic code.
- Each LLM output is validated against a Pydantic schema and retried once. If it still fails, the case is flagged
  for a human.
- A deterministic grounding check flags any medication or diagnosis in a draft that isn't in the patient's
  context.

## Consequences
Behavior is testable with fakes, and failures are explicit flags instead of silent errors. The trade-off is that
new behavior needs code changes, not prompt changes.
