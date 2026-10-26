"""Discharge-summary extractor: zero-shot or few-shot prompting of a local model.

The same interface serves the fine-tuned model later (its own Ollama tag, zero-shot prompt).
"""

from dataclasses import dataclass, field

from .extraction import DischargeExtraction
from .llm import CallResult, StructuredLLM, call_with_retry, field_guide
from .notes import extraction_input

SYSTEM = (
    "You extract structured facts from a hospital discharge summary. Copy facts only from the "
    "summary; never add anything not stated. Medication changes compare the admission and "
    "discharge medication lists (started, stopped, dose_changed, continued); use lowercase "
    "generic drug names. Red flags are symptoms the patient is told should prompt a call or an "
    "ER visit. Pending tests are results still pending at discharge. Return JSON only."
)


@dataclass
class Example:
    note: str
    label: DischargeExtraction


@dataclass
class Extractor:
    llm: StructuredLLM
    examples: list[Example] = field(default_factory=list)  # empty = zero-shot

    def prompt(self, note: str) -> str:
        shots = []
        for i, ex in enumerate(self.examples, 1):
            text, _ = extraction_input(ex.note)
            shots.append(f"EXAMPLE {i}\nSUMMARY:\n{text}\nJSON:\n{ex.label.model_dump_json()}")
        text, _ = extraction_input(note)
        head = ("\n\n".join(shots) + "\n\nNOW THE REAL ONE\n") if shots else ""
        return f"{field_guide(DischargeExtraction)}\n\n{head}SUMMARY:\n{text}\nJSON:"

    def __call__(self, note: str) -> DischargeExtraction:
        r = self.run(note)
        if r.value is None:
            raise ValueError(r.error)
        return r.value

    def run(self, note: str) -> CallResult[DischargeExtraction]:
        return call_with_retry(self.llm, DischargeExtraction, SYSTEM, self.prompt(note))
