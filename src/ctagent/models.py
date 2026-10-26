"""Typed records passed between pipeline steps and graph nodes."""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Disposition(StrEnum):
    HOME = "Home"
    HOME_HEALTH = "Home Health"
    SNF = "SNF"
    HOSPICE = "Hospice"
    EXPIRED = "Expired"


class Tier(StrEnum):
    HIGH = "high"
    STANDARD = "standard"


class Condition(BaseModel):
    code: str
    description: str
    onset: date


class Medication(BaseModel):
    code: str
    description: str
    start: date
    stop: date | None = None


class MedChangeType(StrEnum):
    STARTED = "started"
    STOPPED = "stopped"
    DOSE_CHANGED = "dose_changed"
    CONTINUED = "continued"


class MedChange(BaseModel):
    name: str
    change: MedChangeType
    detail: str = ""


class DischargeEvent(BaseModel):
    """One index hospitalization (transfers already merged), with what is known by discharge."""

    patient_id: str
    episode_id: str
    admit: datetime
    discharge: datetime
    age: int
    disposition: Disposition
    reason: str = ""
    acute: bool = Field(description="ER contact on the admission day or the day before")
    er_visits_6m: int = Field(ge=0, description="Distinct ER days 2-182 days before admission")
    n_stays: int = 1

    @property
    def los_days(self) -> int:
        return (self.discharge.date() - self.admit.date()).days

    @property
    def under_65(self) -> bool:
        return self.age < 65


class PatientContext(BaseModel):
    """The outpatient-record view of a patient as of the discharge (the agent's EHR stand-in)."""

    conditions: list[Condition] = []
    active_meds: list[Medication] = []
    med_changes: list[MedChange] = []


class RiskScore(BaseModel):
    lace: int
    lace_l: int
    lace_a: int
    lace_c: int
    lace_e: int
    charlson: int
    charlson_conditions: list[str]
    under_65: bool


class FollowUpPlan(BaseModel):
    tier: Tier
    visit_by: date = Field(description="Practice target for the follow-up visit")
    contact_by: date = Field(description="TCM interactive contact deadline (2 business days)")
    tcm_code: str = Field(
        description="Candidate TCM code; final code depends on the provider's MDM"
    )
    tcm_visit_by: date = Field(description="TCM face-to-face deadline for the candidate code")
    check_ins: list[date]
    reasons: list[str]
