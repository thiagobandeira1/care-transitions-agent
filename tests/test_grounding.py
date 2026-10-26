"""Grounding check: real record names in different spacing/punctuation are not flagged."""

from datetime import date

from ctagent.drafts import MedLine, ProviderBrief, grounding_violations
from ctagent.models import MedChange, MedChangeType, Medication, PatientContext
from ctagent.synthea import base_name

DESC = "insulin isophane  human 70 UNT/ML / insulin  regular  human 30 UNT/ML Injectable Suspension"
CTX = PatientContext(
    active_meds=[Medication(code="1", description=DESC, start=date(2024, 1, 1))],
    med_changes=[MedChange(name=base_name(DESC), change=MedChangeType.CONTINUED)],
)


def brief(*names: str) -> ProviderBrief:
    return ProviderBrief(
        reason_for_admission="x",
        key_conditions=[],
        watch_for=[],
        risk_summary="",
        medication_changes=[MedLine(name=n, change=MedChangeType.CONTINUED) for n in names],
    )


def test_spacing_and_punctuation_variants_are_grounded() -> None:
    assert base_name(DESC) == "insulin isophane human"
    for variant in ["insulin isophane human", "Insulin  Isophane, Human", "insulin isophane"]:
        assert grounding_violations(CTX, "", None, brief(variant)) == [], variant


def test_invented_medication_still_flagged() -> None:
    assert grounding_violations(CTX, "", None, brief("warfarin")) == [
        "medication not in record: warfarin"
    ]
