"""The structured output of the discharge-summary extractor (Part A), versioned."""

from pydantic import BaseModel, Field

from .models import MedChangeType

SCHEMA_VERSION = "1.0"


class Diagnosis(BaseModel):
    text: str
    icd10: str | None = Field(default=None, description="ICD-10 code when stated or obvious")


class MedicationChange(BaseModel):
    name: str = Field(description="Generic drug name, lowercase")
    change: MedChangeType
    detail: str = Field(default="", description="Dose or reason, as written in the summary")


class FollowUp(BaseModel):
    with_whom: str
    timeframe: str = ""
    reason: str = ""


class DischargeExtraction(BaseModel):
    principal_diagnosis: Diagnosis | None
    secondary_diagnoses: list[Diagnosis] = []
    medication_changes: list[MedicationChange] = []
    follow_up: list[FollowUp] = []
    pending_tests: list[str] = []
    red_flags: list[str] = Field(default=[], description="Symptoms that should prompt a call or ER")
