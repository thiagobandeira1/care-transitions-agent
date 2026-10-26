"""The structured output of the discharge-summary extractor (Part A), versioned."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import MedChangeType

# 1.1: every top-level key is required in the JSON schema (same instances as 1.0)
SCHEMA_VERSION = "1.1"


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


def _require_every_key(schema: dict[str, Any]) -> None:
    schema["required"] = list(schema.get("properties", {}))


class DischargeExtraction(BaseModel):
    """Every key is required in the JSON schema the model decodes against.

    With optional keys, constrained decoding lets a model leave a whole field out; zero-shot
    models did exactly that for follow-up and pending tests. Python defaults stay for callers.
    """

    model_config = ConfigDict(json_schema_extra=_require_every_key)

    principal_diagnosis: Diagnosis | None
    secondary_diagnoses: list[Diagnosis] = []
    medication_changes: list[MedicationChange] = []
    follow_up: list[FollowUp] = []
    pending_tests: list[str] = []
    red_flags: list[str] = Field(default=[], description="Symptoms that should prompt a call or ER")
