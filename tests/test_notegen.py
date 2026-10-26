import random
from dataclasses import dataclass, field
from datetime import date

from ctagent.models import Condition, MedChange, MedChangeType, PatientContext
from ctagent.notegen import generate_note, missing_facts, note_prompt, reference
from test_graph import EVENT

CTX = PatientContext(
    conditions=[
        Condition(
            code="1",
            description="Chronic congestive heart failure (disorder)",
            onset=date(2020, 1, 1),
        ),
        Condition(
            code="2", description="Diabetes mellitus type 2 (disorder)", onset=date(2019, 1, 1)
        ),
    ],
    med_changes=[
        MedChange(name="furosemide", change=MedChangeType.DOSE_CHANGED, detail="20 -> 40 mg"),
        MedChange(name="metformin", change=MedChangeType.CONTINUED, detail="500 mg"),
        MedChange(name="spironolactone", change=MedChangeType.STARTED, detail="25 mg"),
    ],
)


def test_reference_is_built_from_the_record() -> None:
    ref = reference(EVENT, CTX, high_risk=True, rng=random.Random(1))
    assert (
        ref.principal_diagnosis
        and ref.principal_diagnosis.text == "Chronic congestive heart failure"
    )
    assert [d.text for d in ref.secondary_diagnoses] == ["Diabetes mellitus type 2"]
    assert [m.name for m in ref.medication_changes][-1] == "metformin"  # continued meds last
    assert [f.with_whom for f in ref.follow_up] == ["Primary care", "Cardiology"]
    assert ref.follow_up[0].timeframe == "within 3 days"
    assert "weight gain of more than 3 pounds in a day" in ref.red_flags
    prompt = note_prompt(EVENT, ref)
    assert "furosemide: dose_changed" in prompt and "Followup Instructions" in prompt


@dataclass
class ScriptedLLM:
    notes: list[str]
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.notes.pop(0)


def test_note_is_regenerated_once_then_dropped() -> None:
    ref = reference(EVENT, CTX, high_risk=False, rng=random.Random(1))
    full = "Heart failure. Furosemide, metformin and spironolactone."
    llm = ScriptedLLM(["Heart failure. Furosemide only.", full])
    assert generate_note(llm, EVENT, ref) == full
    assert "left out: spironolactone, metformin" in llm.prompts[1]
    assert generate_note(ScriptedLLM(["nothing", "still nothing"]), EVENT, ref) is None
    assert missing_facts(full, ref) == []
