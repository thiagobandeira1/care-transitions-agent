# Gold-set annotation guideline (MIMIC-IV-Note, local only)

The gold set is the reference for paper 2's main comparison (base model zero-shot and few-shot vs the fine-tuned
model). Annotate **only what the model can see**: the `input.txt` file, which holds the extraction sections of one
discharge summary. Everything stays on this PC under `MIMIC_WORK_DIR\gold\`. Never paste note text into any cloud
tool, Claude included.

## Workflow
1. `uv run python -m ctagent gold-export --n 200 --overlap 50` creates one folder per note with `input.txt` and
   `prefill.json` (rule-based silver labels as a starting point).
2. Copy `prefill.json` to `gold.json` and correct it while reading `input.txt`. Delete wrong items, add missed
   ones, fix change types. Save as valid JSON.
3. The second annotator (a co-author) does the same for the 50 `overlap` notes and writes `gold_b.json`.
4. `uv run python -m ctagent gold-check` validates every file against the schema and prints aggregate
   agreement: silver vs gold, and annotator A vs B.

## Field definitions
| Field | Include | Exclude |
|---|---|---|
| `principal_diagnosis` | The main reason for the hospitalization, as written in Discharge Diagnosis (first or "primary") | Symptoms when a diagnosis is given |
| `secondary_diagnoses` | Other diagnoses listed under Discharge Diagnosis ("secondary") | Past history that isn't listed at discharge |
| `medication_changes` | Each drug on either medication list: `started` (only on discharge list), `stopped` (only on admission list, or stated stopped), `dose_changed` (same drug, different dose or frequency), `continued` (same on both) | Inpatient-only drugs not on either list. Use the lowercase generic name (e.g. `furosemide`, not `Lasix`) |
| `follow_up` | Each appointment or follow-up instruction: who, when, why | General advice ("rest") |
| `pending_tests` | Results explicitly pending at discharge | Tests already resulted |
| `red_flags` | Symptoms the patient is told should prompt a call or ER visit | General education |

## Rules
- Copy wording from the note where possible; don't infer what isn't written.
- The prefill's diagnoses are the hospital's billing-code titles (ICD-9 or ICD-10), not the note's wording.
  Replace them with what Discharge Diagnosis says.
- MIMIC-IV-Note redacts the Followup Instructions section to `___`, so the prefill has no follow-ups. Record
  follow-ups stated elsewhere in the input (discharge instructions, hospital course); leave the list empty if
  there are none.
- If a field has nothing, leave the list empty (or `principal_diagnosis` null).
- Record annotation time per note (optional, `minutes` key) to report the effort.
- Disagreements between A and B are resolved by discussion after agreement is measured. The resolved file
  becomes `gold.json`.
