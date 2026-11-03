# Gold-set annotation guideline (MIMIC-IV-Note, local only)

The gold set is the reference for paper 2's main comparison (base model zero-shot and few-shot vs the fine-tuned
model). Annotate **only what the model can see**: the `input.txt` file, which holds the extraction sections of one
discharge summary. Everything stays on this PC under `MIMIC_WORK_DIR\gold\`. Label by hand: no AI tool or browser
extension may read the note or fill in the form.

## Reference sets
| File | Notes | How it was made | Use |
|---|---|---|---|
| `gold.json` | the 50 overlap notes (plus 1), done 2026-11-02 | by hand in the screen | the gold standard for principal and secondary diagnoses |
| `claude.json` | all 200 | filled in with Claude's browser extension on 2026-11-01 (2 hours, median 0.5 min per note) | a second, model-made reference (`--ref claude`); never reported as hand annotation |
| `gold_b.json` | the 50 overlap notes | by hand, 14 days later (optional) | intra-annotator agreement |

`gold-check` reports how closely Claude's labels agree with the hand labels (`claude_vs_gold`). On those terms
the 200-note Claude results can back up the 50-note hand results.

**Hand labels cover the 2 diagnosis fields** (Thiago's decisions, 2026-11-02). Each `gold.json` lists its
fields under `fields`, and every score against it (`gold-check`, `mimic-bench --ref gold`) uses only those. All
other fields are scored against `claude.json` only.
- Follow-up, pending tests and red flags were left out before the hand pass, as too ambiguous to label quickly.
  The synthetic benchmark has known answers for them.
- Medication changes were hand-checked (median 0.3 min per note) and then dropped. The prefill marks a drug
  `started` when it is not on a numbered admission list, but in 22 of the 51 notes that list has no numbered
  items or is missing. In the hand labels, 84 of 348 `started` rows (24%) were drugs named in the admission
  list, against 3 of 181 (2%) in Claude's and 90 of 354 in the prefill. The rows stay in the files, unscored.
  The screen now shows only the diagnoses.

## Workflow
1. `uv run python -m ctagent gold-export --n 200 --overlap 50` creates one folder per note with `input.txt` and
   `prefill.json` (rule-based silver labels as a starting point). Already done: the 200 folders exist.
2. Open the annotation screen, which runs only on this PC:
   ```
   uv run streamlit run app/annotate_app.py
   ```
   - The note is on the left. "Find in note" lists the matching lines.
   - The prefill is on the right, as editable tables. Delete wrong rows, add missed ones (the empty last row of
     each table), and fix change types.
   - **Save and next** checks the answer against the schema and writes `gold.json`. It records the minutes
     spent and the save time, then opens the next note to do, overlap notes first. The sidebar counts the
     overlap notes done.
   - The screen never opens `claude.json`, so the hand labels start from the prefill.
   - You can stop at any time; saved notes reopen with your answers.
3. **Second pass, for agreement:**
   - There is no second annotator, so the same annotator redoes the 50 overlap notes (folders with an
     `OVERLAP` file) and writes `gold_b.json`. In the screen, choose **Second pass**.
   - Each overlap note opens 14 days after its first pass was saved. It starts again from the prefill, and the
     screen never shows `gold.json`. Don't open it yourself either.
   - Report the result as intra-annotator agreement. A second PhysioNet-credentialed annotator can use the same
     screen instead and report inter-annotator agreement.
4. `uv run python -m ctagent gold-check` validates every file against the schema and prints aggregate
   agreement: silver vs gold, Claude vs gold, and first vs second pass.

To try the screen first, build a practice set from the synthetic notes:
```
uv run python -m ctagent gold-demo --n 10 --overlap 3
uv run streamlit run app/annotate_app.py -- --gold-dir data/gold-demo
```

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
- The prefill's diagnoses come from the note's Discharge Diagnosis section by simple rules: the first line is
  the principal and the rest are secondary. Check the split and the wording against the note.
- MIMIC-IV-Note redacts the Followup Instructions section to `___`, so the prefill has no follow-ups. Record
  follow-ups stated elsewhere in the input (discharge instructions, hospital course); leave the list empty if
  there are none.
- If a field has nothing, leave the list empty (or `principal_diagnosis` null).
- The screen records the time per note (`minutes`), so the effort can be reported.
- `gold.json` (the first pass) is the reference for the models. The second pass only measures agreement. If
  two annotators are used, disagreements are resolved by discussion after agreement is measured, and the
  resolved file becomes `gold.json`.
