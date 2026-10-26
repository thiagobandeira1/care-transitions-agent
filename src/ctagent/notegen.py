"""Synthetic discharge summaries with known answers (the public extraction benchmark, M4).

Facts come from a Synthea episode plus deterministic rules (follow-up, pending tests, red flags).
A local model writes the summary in the common US template; a note is kept only if it mentions
every medication and the principal diagnosis it was given (one regeneration, then dropped).
Synthetic notes are easier than real ones: results on them are never presented as clinical
validity (ADR 0001).
"""

import random
import re
from typing import Protocol

from .drafts import _clean
from .extraction import Diagnosis, DischargeExtraction, FollowUp, MedicationChange
from .models import DischargeEvent, MedChangeType, PatientContext

SPECIALTY = [
    (r"heart|cardiac|myocardial|coronary|aortic|mitral|atrial", "Cardiology"),
    (r"kidney|renal", "Nephrology"),
    (r"pulmonary|lung|bronch|emphysema|asthma|pneumonia|respiratory", "Pulmonology"),
    (r"neoplasm|carcinoma|cancer|leukemia|myeloma", "Oncology"),
    (r"diabetes|hyperglycemia", "Endocrinology"),
]
RED_FLAGS = [
    (
        r"heart failure",
        [
            "weight gain of more than 3 pounds in a day",
            "shortness of breath at rest",
            "new swelling in the legs",
        ],
    ),
    (
        r"myocardial|coronary|angina",
        ["chest pain or pressure", "pain spreading to the arm or jaw", "fainting"],
    ),
    (
        r"pneumonia|respiratory|covid|coronavirus",
        ["fever above 101 F", "trouble breathing", "coughing up blood"],
    ),
    (r"kidney|renal", ["much less urine than usual", "confusion", "new swelling"]),
]
DEFAULT_RED_FLAGS = ["fever above 101 F", "chest pain", "new confusion or trouble speaking"]
PENDING = ["blood cultures", "thyroid function tests", "final pathology report", "urine culture"]

SYSTEM = (
    "You write realistic, de-identified hospital discharge summaries for a software test set. "
    "Use ___ for any name, date or place. Include EVERY fact you are given and do not add new "
    "medications, diagnoses, tests or follow-up appointments."
)


class TextLLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


def reference(
    event: DischargeEvent,
    ctx: PatientContext,
    high_risk: bool,
    rng: random.Random,
    max_meds: int = 8,
) -> DischargeExtraction:
    principal = _clean(event.reason) or "Acute illness"
    secondary = [
        _clean(c.description) for c in ctx.conditions if _clean(c.description) != principal
    ]
    meds = sorted(ctx.med_changes, key=lambda m: m.change is MedChangeType.CONTINUED)[:max_meds]
    specialty = next((s for pat, s in SPECIALTY if re.search(pat, principal, re.I)), None)
    follow = [
        FollowUp(
            with_whom="Primary care",
            timeframe="within 3 days" if high_risk else "within 7 days",
            reason="post-hospital follow-up",
        )
    ]
    if specialty:
        follow.append(FollowUp(with_whom=specialty, timeframe="within 2 weeks", reason=principal))
    flags = next((f for pat, f in RED_FLAGS if re.search(pat, principal, re.I)), DEFAULT_RED_FLAGS)
    return DischargeExtraction(
        principal_diagnosis=Diagnosis(text=principal),
        secondary_diagnoses=[Diagnosis(text=s) for s in secondary[:4]],
        medication_changes=[
            MedicationChange(name=m.name, change=m.change, detail=m.detail) for m in meds
        ],
        follow_up=follow,
        pending_tests=rng.sample(PENDING, k=rng.choice([0, 0, 1, 2])),
        red_flags=list(flags),
    )


def note_prompt(event: DischargeEvent, ref: DischargeExtraction) -> str:
    meds = "\n".join(f"- {m.name}: {m.change.value} ({m.detail})" for m in ref.medication_changes)
    principal = ref.principal_diagnosis.text if ref.principal_diagnosis else ""
    return (
        f"Write a discharge summary for a {event.age}-year-old discharged after {event.los_days} "
        f"days to {event.disposition.value}. Use these headers: Chief Complaint, "
        "History of Present Illness, Brief Hospital Course, Medications on Admission, "
        "Discharge Medications, "
        "Discharge Diagnosis, Discharge Instructions, Followup Instructions.\n"
        f"Principal diagnosis: {principal}\n"
        "Secondary diagnoses: " + "; ".join(d.text for d in ref.secondary_diagnoses) + "\n"
        "Medications (admission vs discharge lists must reflect each change; dose_changed means "
        f"same drug, different dose):\n{meds or '- none'}\n"
        "Follow-up: " + "; ".join(f"{f.with_whom} {f.timeframe}" for f in ref.follow_up) + "\n"
        "Pending at discharge: " + ("; ".join(ref.pending_tests) or "nothing") + "\n"
        "Warning signs to list after 'call your doctor or return to the emergency department if "
        "you have:': " + "; ".join(ref.red_flags)
    )


def missing_facts(note: str, ref: DischargeExtraction) -> list[str]:
    text = note.lower()
    missing = [m.name for m in ref.medication_changes if m.name.split()[0].lower() not in text]
    if ref.principal_diagnosis:
        words = re.findall(r"[a-z]{5,}", ref.principal_diagnosis.text.lower())
        if words and not any(w in text for w in words):
            missing.append(ref.principal_diagnosis.text)
    return missing


def generate_note(llm: TextLLM, event: DischargeEvent, ref: DischargeExtraction) -> str | None:
    prompt = note_prompt(event, ref)
    for _ in range(2):
        note = llm.complete(SYSTEM, prompt)
        missing = missing_facts(note, ref)
        if not missing:
            return note
        prompt += f"\nYour previous draft left out: {', '.join(missing)}. Include them."
    return None
