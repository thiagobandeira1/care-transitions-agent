# Simulated-use study: reviewing the agent's drafts

The paper's "how it is used" question, answered without patient data: a person who does hospital follow-up at a
primary care practice reviews 30 discharges drafted by the agent, exactly as they would at work. The patients
are synthetic (Synthea), so no patient data are involved. No MIMIC access and no practice records are needed.

## Setup (Thiago, once)
0. Before running the session with practice staff, obtain independent ethical review; an exemption counts
   only if that independent reviewer grants it in writing. The paper's Section 4.7 says any evaluation with
   staff or patients needs independent ethical review first. Participation is voluntary.
1. Build the 30 cases: 30 Synthea discharges, drafted with the practice's name (about 10 minutes on the GPU).
   ```
   uv run python -m ctagent demo --n 30 --seed 31 --workspace data/simulated-use --practice "Your Practice Name"
   ```
2. Open the review screen on this PC (it serves to this PC only):
   ```
   uv run streamlit run app/review_app.py -- --workspace data/simulated-use
   ```
3. After the session, get the measures (counts only):
   ```
   uv run python -m ctagent pilot-report --workspace data/simulated-use
   ```

## For the reviewer (about 60-90 minutes; it can be split)
These are made-up patients. Treat each one as a real discharge you have to act on today.

1. Type your name in the sidebar. It is saved with each decision.
2. In **Review queue**, pick the next discharge (high risk comes first).
3. Read the plan (risk tier, call-by date, visit-by date), the **call script** and the **provider brief**, as if
   you were about to call the patient.
4. If the screen shows **Check:** warnings, answer **Were these flags real problems?**
5. Decide:
   - **Approve** if you would use both documents as they are.
   - **Approve with edits** if you would fix something first. Open **Edit drafts**, fix it (one item per line;
     medications as `name | started, stopped, dose_changed or continued | note`), then click the button.
   - **Reject** if you would not use them. Write a short note on why.
6. Work at your normal pace. Fix what you would fix at work, nothing more. The screen times each review.

Skip the **Follow-up** and **Check-ins due** tabs: they are for real patients.

## Afterwards: questionnaire (5 minutes)
Rate each statement from 1 (strongly disagree) to 5 (strongly agree).

| # | Statement | 1-5 |
|---|---|---|
| 1 | The call scripts matched the patient's record. | |
| 2 | The call scripts were ready to use with little or no editing. | |
| 3 | The provider briefs gave the provider what they need for the follow-up visit. | |
| 4 | The risk tier and the deadlines were clear. | |
| 5 | When the screen flagged something, it was worth checking. | |
| 6 | This would save time compared with how we prepare follow-up calls today. | |
| 7 | I would use this in my daily work. | |

- About how many minutes does preparing one follow-up call take you today, and how many with this?
- What would you change first?
- What made you edit or reject a draft most often?

Optional: the System Usability Scale (Brooke, 1996), a standard 10-item usability questionnaire.

## What the study measures
- **Recorded by the screen:**
  - decisions: approved as drafted, edited or rejected
  - how much of each document was changed, and which fields
  - minutes per review
  - whether flags were real problems
- **From the questionnaire:** perceived accuracy, readiness, time saved and willingness to use.

What it does not measure: patient outcomes, or how the agent behaves on real records. The MIMIC evaluations in
PROGRESS.md cover performance on real notes.
