import pytest

from ctagent.extract_eval import evaluate, exact, match, normalize, relaxed, token_overlap
from ctagent.extraction import Diagnosis, DischargeExtraction, FollowUp, MedicationChange
from ctagent.extractor import Example, Extractor
from ctagent.llm import FakeLLM, LLMOutputError
from ctagent.models import MedChangeType as T
from test_notes import NOTE

REF = DischargeExtraction(
    principal_diagnosis=Diagnosis(text="Acute on chronic systolic heart failure"),
    secondary_diagnoses=[Diagnosis(text="Chronic kidney disease")],
    medication_changes=[
        MedicationChange(name="furosemide", change=T.DOSE_CHANGED),
        MedicationChange(name="spironolactone", change=T.STARTED),
        MedicationChange(name="aspirin", change=T.STOPPED),
    ],
    follow_up=[FollowUp(with_whom="Primary care within 7 days")],
    pending_tests=["Blood cultures"],
    red_flags=["chest pain", "shortness of breath at rest"],
)


def test_string_matching() -> None:
    assert normalize(" Heart-Failure, acute ") == "heart failure acute"
    assert exact("Chest pain.", "chest pain") and not exact("chest pain", "pain")
    assert relaxed("systolic heart failure", "heart failure systolic acute")
    assert token_overlap("", "") == 1.0 and not relaxed("cough", "fever")
    c = match(["a", "b", "x"], ["a", "b", "c", "d"], exact)
    assert (c.tp, c.fp, c.fn) == (2, 1, 2)


def test_perfect_prediction_scores_one() -> None:
    s = evaluate([(REF, REF)]).summary()
    assert s["exact"]["micro"]["f1"] == 1.0 and s["med_change_type_accuracy"] == 1.0  # type: ignore[index]


def test_partial_prediction_and_wrong_change_type() -> None:
    pred = REF.model_copy(
        update={
            "medication_changes": [
                MedicationChange(name="furosemide", change=T.STARTED),
                MedicationChange(name="spironolactone", change=T.STARTED),
            ],
            "red_flags": ["Chest pain!", "fever"],
            "principal_diagnosis": Diagnosis(text="Systolic heart failure, acute on chronic"),
        }
    )
    s = evaluate([(pred, REF)]).summary()
    meds = s["exact"]["medication_changes"]  # type: ignore[index]
    assert meds["precision"] == 0.5 and meds["recall"] == pytest.approx(1 / 3)
    assert s["med_change_type_accuracy"] == 0.5
    assert s["exact"]["principal_diagnosis"]["f1"] == 0.0  # type: ignore[index]
    assert s["relaxed"]["principal_diagnosis"]["f1"] == 1.0  # type: ignore[index]
    assert s["exact"]["red_flags"]["precision"] == 0.5  # type: ignore[index]


def test_schema_failure_counts_as_all_missed() -> None:
    s = evaluate([(None, REF)]).summary()
    assert s["relaxed"]["micro"]["recall"] == 0.0 and s["n"] == 1  # type: ignore[index]


def test_extractor_zero_and_few_shot_prompts() -> None:
    llm = FakeLLM({"DischargeExtraction": lambda p: REF})
    zero = Extractor(llm)
    assert zero(NOTE) == REF
    assert "EXAMPLE" not in llm.calls[-1][1]
    assert llm.calls[-1][1].startswith("Return JSON with these fields:")
    assert "\n\nSUMMARY:\n" in llm.calls[-1][1]
    few = Extractor(llm, [Example(note=NOTE, label=REF)])
    few(NOTE)
    assert "EXAMPLE 1" in llm.calls[-1][1] and "NOW THE REAL ONE" in llm.calls[-1][1]
    bad = Extractor(FakeLLM({"DischargeExtraction": lambda p: LLMOutputError("x")}))
    assert bad.run(NOTE).value is None
    with pytest.raises(ValueError):
        bad(NOTE)


def test_schema_makes_the_model_emit_every_field() -> None:
    schema = DischargeExtraction.model_json_schema()
    assert schema["required"] == list(DischargeExtraction.model_fields)
    assert DischargeExtraction(principal_diagnosis=None).follow_up == []  # defaults still work


def test_repeated_list_items_are_dropped() -> None:
    looped = REF.model_dump(mode="json") | {
        "red_flags": ["chest pain", "Chest  pain", "fever", *["chest pain"] * 500],
        "secondary_diagnoses": [
            {"text": "CKD"},
            {"text": "ckd"},
            {"text": "CKD", "icd10": "N18.9"},
        ],
        "medication_changes": [{"name": "aspirin", "change": "stopped"}] * 2
        + [{"name": "aspirin", "change": "started"}],
    }
    x = DischargeExtraction.model_validate(looped)
    assert x.red_flags == ["chest pain", "fever"]
    assert [d.icd10 for d in x.secondary_diagnoses] == [None, "N18.9"]  # another code: kept
    assert [m.change for m in x.medication_changes] == [T.STOPPED, T.STARTED]
    assert evaluate([(x, REF)]).summary()["relaxed"]["red_flags"]["precision"] == 0.5  # type: ignore[index]
