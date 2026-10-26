"""Draft schemas, prompts, and the deterministic grounding check for LLM-written text.

Structured fields (medication names, condition names) make the grounding check mechanical: every
medication or condition a draft names must exist in the patient's record as of discharge.
"""

import re

from pydantic import BaseModel, Field

from .models import DischargeEvent, FollowUpPlan, MedChangeType, PatientContext, RiskScore
from .synthea import base_name

SYSTEM = (
    "You draft documents for a primary care practice's hospital follow-up coordinator. "
    "Use ONLY facts in the provided record. Never invent medications, diagnoses, results or dates. "
    "If something is unknown, leave it out. Every draft is reviewed by a human before use."
)


class PlanExplanation(BaseModel):
    summary: str = Field(description="2-3 plain sentences: why this tier and what happens next")


class MedLine(BaseModel):
    name: str = Field(description="Medication name exactly as in the record (generic name)")
    change: MedChangeType
    note: str = Field(default="", description="What the patient or provider should know")


class CallScript(BaseModel):
    opening: str = Field(description="Greeting and purpose of the call, plain language")
    questions: list[str] = Field(description="3-6 questions about symptoms, understanding, needs")
    medication_review: list[MedLine] = Field(description="Only medications that changed")
    warning_signs: list[str] = Field(description="Symptoms that should prompt a call or the ER")
    scheduling: str = Field(description="The visit to offer, with the target date")


class ProviderBrief(BaseModel):
    reason_for_admission: str
    key_conditions: list[str] = Field(description="Relevant chronic conditions, names from record")
    medication_changes: list[MedLine]
    watch_for: list[str] = Field(description="What to check at the follow-up visit")
    risk_summary: str = Field(description="One sentence on readmission risk and its drivers")


def record_text(
    event: DischargeEvent, ctx: PatientContext, score: RiskScore, plan: FollowUpPlan
) -> str:
    changes = [m for m in ctx.med_changes if m.change is not MedChangeType.CONTINUED]
    lines = [
        f"Age: {event.age}. Admitted {event.admit:%Y-%m-%d}, discharged {event.discharge:%Y-%m-%d} "
        f"({event.los_days} days) to {event.disposition.value}.",
        f"Reason for admission: {event.reason or 'not recorded'}.",
        "Conditions: " + ("; ".join(_clean(c.description) for c in ctx.conditions) or "none"),
        "Medication changes during the stay: "
        + ("; ".join(f"{m.name} ({m.change.value})" for m in changes) or "none recorded"),
        "Active medications at discharge: "
        + ("; ".join(base_name(m.description) for m in ctx.active_meds) or "none"),
        f"Risk: LACE {score.lace} (length of stay {score.lace_l}, acute {score.lace_a}, "
        f"comorbidity {score.lace_c}, ER visits {score.lace_e}); under 65: {score.under_65}.",
        f"Plan: {plan.tier.value} tier; visit by {plan.visit_by}; "
        f"TCM contact by {plan.contact_by}; "
        f"TCM {plan.tcm_code} face-to-face by {plan.tcm_visit_by}; check-ins: "
        + (", ".join(str(d) for d in plan.check_ins) or "none")
        + ".",
    ]
    return "\n".join(lines)


def _clean(description: str) -> str:
    return re.sub(r"\s*\((disorder|finding|situation)\)\s*$", "", description).strip()


_STOP = {
    "disease",
    "disorder",
    "chronic",
    "acute",
    "syndrome",
    "history",
    "primary",
    "type",
    "stage",
    "with",
    "without",
    "due",
    "of",
    "the",
    "and",
    "mellitus",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in _STOP}


def grounding_violations(
    ctx: PatientContext, reason: str, script: CallScript | None, brief: ProviderBrief | None
) -> list[str]:
    """Medications or conditions named in drafts that are not in the record."""
    meds = {base_name(m.description) for m in ctx.active_meds} | {m.name for m in ctx.med_changes}
    known_conditions = set().union(
        *(_tokens(c.description) for c in ctx.conditions), _tokens(reason)
    )
    lines = (script.medication_review if script else []) + (
        brief.medication_changes if brief else []
    )
    out = []
    for line in lines:
        name = line.name.strip().lower()
        if not any(name in m or m in name for m in meds if m):
            out.append(f"medication not in record: {line.name}")
    for cond in brief.key_conditions if brief else []:
        if not _tokens(cond) & known_conditions:
            out.append(f"condition not in record: {cond}")
    return out
