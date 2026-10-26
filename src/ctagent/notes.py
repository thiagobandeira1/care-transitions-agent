"""Discharge-summary sectioning and rule-based silver labels.

Works on any discharge summary in the common US template (MIMIC-IV-Note, and the synthetic
summaries generated for the public benchmark). Callers print aggregates only: counts of known
sections, never header text or content, because a colon-terminated line could hold patient text.
"""

import re

from .extraction import Diagnosis, DischargeExtraction, FollowUp, MedicationChange
from .models import MedChangeType

# canonical name -> header spellings seen in the template (matched case-insensitively at line start)
SECTIONS: dict[str, tuple[str, ...]] = {
    "chief_complaint": ("chief complaint",),
    "hpi": ("history of present illness",),
    "pmh": ("past medical history",),
    "results": ("pertinent results",),
    "hospital_course": ("brief hospital course", "hospital course"),
    "meds_admission": ("medications on admission", "admission medications"),
    "meds_discharge": ("discharge medications",),
    "disposition": ("discharge disposition",),
    "diagnosis": ("discharge diagnosis", "discharge diagnoses"),
    "condition": ("discharge condition",),
    "instructions": ("discharge instructions",),
    "followup": ("followup instructions", "follow-up instructions", "follow up instructions"),
}
# the sections the extractor reads (keeps inputs inside a ~4k-token budget)
EXTRACTION_SECTIONS = (
    "diagnosis",
    "meds_admission",
    "meds_discharge",
    "instructions",
    "followup",
    "hospital_course",
)

_HEADER = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(h) for hs in SECTIONS.values() for h in hs) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_CANON = {h: name for name, hs in SECTIONS.items() for h in hs}


def split_sections(text: str) -> dict[str, str]:
    """Known sections -> body text. Unknown headers stay inside the preceding section."""
    out: dict[str, str] = {}
    matches = list(_HEADER.finditer(text))
    if not matches:
        return out
    ends = [m.start() for m in matches[1:]] + [len(text)]
    for m, end in zip(matches, ends, strict=True):
        name = _CANON[m.group(1).lower()]
        out.setdefault(name, text[m.end() : end].strip())  # first occurrence wins
    return out


def extraction_input(text: str, max_chars: int = 14000) -> tuple[str, bool]:
    """The sections the extractor reads, in a fixed order, trimmed to a budget (~4k tokens)."""
    sec = split_sections(text)
    parts = [f"{name.upper()}:\n{sec[name]}" for name in EXTRACTION_SECTIONS if sec.get(name)]
    joined = "\n\n".join(parts) if parts else text
    return joined[:max_chars], len(joined) > max_chars


# ---- silver labels (rules; their agreement with the gold set is reported, never assumed) ----

_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(.+)$", re.MULTILINE)
_DOSE = re.compile(r"\d")
_NAME_STOP = {
    "tablet",
    "tab",
    "capsule",
    "cap",
    "po",
    "iv",
    "sc",
    "subq",
    "oral",
    "solution",
    "extended",
    "release",
    "er",
    "sr",
    "xl",
    "dr",
    "hcl",
    "inhaler",
}


def med_entries(section: str) -> dict[str, str]:
    """Numbered or bulleted medication lines -> {normalized name: full line}."""
    out: dict[str, str] = {}
    for line in _LIST_ITEM.findall(section):
        line = line.strip()
        cut = _DOSE.search(line)
        head = line[: cut.start()] if cut else line
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z-]+", head) if w.lower() not in _NAME_STOP]
        if words:
            out.setdefault(" ".join(words[:2]).lower(), line)
    return out


def _dose(line: str) -> str:
    return " ".join(re.findall(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml|meq)\b", line.lower()))


def silver_med_changes(sections: dict[str, str]) -> list[MedicationChange]:
    before = med_entries(sections.get("meds_admission", ""))
    after = med_entries(sections.get("meds_discharge", ""))
    out = []
    for name in sorted(before.keys() | after.keys()):
        if name in after and name not in before:
            out.append(
                MedicationChange(name=name, change=MedChangeType.STARTED, detail=after[name])
            )
        elif name in before and name not in after:
            out.append(
                MedicationChange(name=name, change=MedChangeType.STOPPED, detail=before[name])
            )
        elif _dose(before[name]) != _dose(after[name]):
            out.append(
                MedicationChange(
                    name=name,
                    change=MedChangeType.DOSE_CHANGED,
                    detail=f"{before[name]} -> {after[name]}",
                )
            )
        else:
            out.append(
                MedicationChange(name=name, change=MedChangeType.CONTINUED, detail=after[name])
            )
    return out


_PENDING = re.compile(r"[^.\n]*\bpending\b[^.\n]*", re.IGNORECASE)
_RED_FLAG_LEAD = re.compile(
    r"(call your doctor|return to the (?:emergency|ed|er)|seek (?:immediate )?medical attention|"
    r"come back to the (?:emergency|ed|er)|go to the (?:emergency|ed|er))",
    re.IGNORECASE,
)


def silver_labels(
    text: str, principal: Diagnosis | None = None, secondary: list[Diagnosis] | None = None
) -> DischargeExtraction:
    """Rule-based labels from one summary; diagnoses come from structured codes when available."""
    sec = split_sections(text)
    follow = [
        FollowUp(with_whom=line.strip()) for line in _LIST_ITEM.findall(sec.get("followup", ""))
    ]
    if not follow and sec.get("followup"):
        follow = [FollowUp(with_whom=s.strip()) for s in sec["followup"].split("\n") if s.strip()]
    pending = [
        m.group(0).strip()
        for m in _PENDING.finditer(sec.get("hospital_course", "") + "\n" + sec.get("results", ""))
    ]
    flags: list[str] = []
    instr = sec.get("instructions", "")
    lead = _RED_FLAG_LEAD.search(instr)
    if lead:
        flags = [line.strip() for line in _LIST_ITEM.findall(instr[lead.end() :])][:12]
    if principal is None and sec.get("diagnosis"):
        first = next((x for x in sec["diagnosis"].split("\n") if x.strip()), "").strip(" -*:")
        principal = Diagnosis(text=first) if first else None
    return DischargeExtraction(
        principal_diagnosis=principal,
        secondary_diagnoses=secondary or [],
        medication_changes=silver_med_changes(sec),
        follow_up=follow,
        pending_tests=pending,
        red_flags=flags,
    )
