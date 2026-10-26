"""The structured output of the discharge-summary extractor (Part A), versioned."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import MedChangeType

# 1.1: every top-level key is required in the JSON schema (same instances as 1.0)
# 1.2: every field has a description, and the prompt shows them (same instances)
SCHEMA_VERSION = "1.2"


class Diagnosis(BaseModel):
    text: str = Field(description="As written in the summary")
    icd10: str | None = Field(default=None, description="ICD-10 code when stated or obvious")


class MedicationChange(BaseModel):
    name: str = Field(description="Generic drug name, lowercase")
    change: MedChangeType
    detail: str = Field(default="", description="Dose or reason, as written in the summary")


class FollowUp(BaseModel):
    with_whom: str = Field(description="Clinician, clinic or specialty")
    timeframe: str = Field(default="", description="When, as written")
    reason: str = Field(default="", description="Why, if stated")


def _require_every_key(schema: dict[str, Any]) -> None:
    schema["required"] = list(schema.get("properties", {}))


class DischargeExtraction(BaseModel):
    """Every key is required in the JSON schema the model decodes against.

    With optional keys, constrained decoding lets a model leave a whole field out; zero-shot
    models did exactly that for follow-up and pending tests. Python defaults stay for callers.
    """

    model_config = ConfigDict(json_schema_extra=_require_every_key)

    principal_diagnosis: Diagnosis | None = Field(
        description="The main reason for this hospitalization"
    )
    secondary_diagnoses: list[Diagnosis] = Field(
        default=[], description="Other diagnoses for this stay"
    )
    medication_changes: list[MedicationChange] = Field(
        default=[],
        description="Every medication on the admission or discharge list, continued ones too",
    )
    follow_up: list[FollowUp] = Field(default=[], description="Follow-up appointments to make")
    pending_tests: list[str] = Field(
        default=[], description="Tests or results still pending at discharge"
    )
    red_flags: list[str] = Field(default=[], description="Symptoms that should prompt a call or ER")
