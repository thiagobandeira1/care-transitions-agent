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
    "pending": ("pending at discharge", "pending results", "pending tests"),
}
# the sections the extractor reads (keeps inputs inside a ~4k-token budget)
EXTRACTION_SECTIONS = (
    "diagnosis",
    "meds_admission",
    "meds_discharge",
    "instructions",
    "followup",
    "pending",
    "hospital_course",
)

# A header is a known name at line start, as plain text ("Discharge Diagnosis:", MIMIC) or
# Markdown ("**Discharge Diagnosis**", "**Discharge Diagnosis:**", "## Discharge Diagnosis").
# Without a colon it must stand alone on its line, so a sentence that starts with a header
# name ("Hospital course was uneventful.") is not a header.
_HEADER = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*("
    + "|".join(re.escape(h) for hs in SECTIONS.values() for h in hs)
    + r")(?:[ \t]*:[ \t]*(?:\*\*)?|[ \t]*\*\*[ \t]*:|[ \t]*(?:\*\*)?(?=[ \t]*$))",
    re.IGNORECASE | re.MULTILINE,
)
_CANON = {h: name for name, hs in SECTIONS.items() for h in hs}
_RULE_LINE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,})[ \t]*$", re.MULTILINE)


def _clean_body(body: str) -> str:
    """Drop Markdown rules and bold markers (MIMIC's ___ placeholders are left alone)."""
    return _RULE_LINE.sub("", body).replace("**", "").strip()


def split_sections(text: str) -> dict[str, str]:
    """Known sections -> body text. Unknown headers stay inside the preceding section."""
    out: dict[str, str] = {}
    matches = list(_HEADER.finditer(text))
    if not matches:
        return out
    ends = [m.start() for m in matches[1:]] + [len(text)]
    for m, end in zip(matches, ends, strict=True):
        name = _CANON[m.group(1).lower()]
        out.setdefault(name, _clean_body(text[m.end() : end]))  # first occurrence wins
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
_DX_LABEL = re.compile(r"^(?:primary|principal|secondary)(?:\s+diagnos[ie]s)?\s*:\s*", re.I)
# a heading line without a colon ("PRIMARY DIAGNOSIS", "Chronic issues"), never a diagnosis
_DX_HEADING = re.compile(
    r"^(?:(?:primary|principal|secondary|final|discharge|additional|other|active|chronic|acute|"
    r"resolved|inactive|medical|stable|hospital)\s*)*(?:diagnos[ie]s|dx|issues?|problems?|"
    r"conditions?)?$",
    re.I,
)
_WITH = re.compile(
    r"\bwith\s+(?:the\s+|your\s+|an?\s+|dr\.?\s+)?(.+?)"
    r"(?=\s+(?:within|in|on|for|by|at|after|before)\b|[,.;(]|$)",
    re.IGNORECASE,
)


def _dx_lines(section: str) -> list[str]:
    """Diagnosis lines without bullets, numbering or Primary:/Secondary: labels.

    Drops headings ("Chronic issues", "Active:") and lines that are only ___ placeholders.
    """
    out = []
    for raw in section.split("\n"):
        line = re.sub(r"^\d+[.)]\s*", "", raw.strip(" -*•\t"))
        heading = line.endswith(":")
        line = _DX_LABEL.sub("", line).strip(" -*:")
        if line and not heading and not _DX_HEADING.match(line) and not _deidentified(line):
            out.append(line)
    return out


def _whom(item: str) -> str:
    """'See/follow up with the X within N days' -> 'X'; other lines are kept whole."""
    m = _WITH.search(item)
    return m.group(1).strip() if m else item.strip()


def _deidentified(item: str) -> bool:
    """Only ___ placeholders. MIMIC-IV-Note redacts its Followup Instructions section this way."""
    return not re.sub(r"[_\s]", "", item)


def silver_labels(
    text: str, principal: Diagnosis | None = None, secondary: list[Diagnosis] | None = None
) -> DischargeExtraction:
    """Rule-based labels from one summary; diagnoses come from structured codes when available.

    Without codes, the first diagnosis line is the principal and the rest are secondary.
    """
    sec = split_sections(text)
    follow = [
        FollowUp(with_whom=_whom(line)) for line in _LIST_ITEM.findall(sec.get("followup", ""))
    ]
    if not follow and sec.get("followup"):
        follow = [FollowUp(with_whom=_whom(s)) for s in sec["followup"].split("\n") if s.strip()]
    follow = [f for f in follow if not _deidentified(f.with_whom)]
    pending = [
        m.group(0).strip()
        for m in _PENDING.finditer(sec.get("hospital_course", "") + "\n" + sec.get("results", ""))
    ]
    pending += [line.strip() for line in _LIST_ITEM.findall(sec.get("pending", ""))]
    flags: list[str] = []
    instr = sec.get("instructions", "")
    lead = _RED_FLAG_LEAD.search(instr)
    if lead:
        flags = [line.strip() for line in _LIST_ITEM.findall(instr[lead.end() :])][:12]
    dx_lines = _dx_lines(sec.get("diagnosis", ""))
    if principal is None and dx_lines:
        principal = Diagnosis(text=dx_lines[0])
        if secondary is None:
            secondary = [Diagnosis(text=t) for t in dx_lines[1:]]
    return DischargeExtraction(
        principal_diagnosis=principal,
        secondary_diagnoses=secondary or [],
        medication_changes=silver_med_changes(sec),
        follow_up=follow,
        pending_tests=pending,
        red_flags=flags,
    )
