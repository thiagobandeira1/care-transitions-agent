"""Gold-set annotation: the service layer behind the annotation screen (app/annotate_app.py).

The screen stays thin; opening a note, turning the form into a DischargeExtraction, saving and the
second-pass rules live here and are tested without a browser. Nothing here prints note text.

- First pass: correct `prefill.json` into `gold.json` by hand, the 50 OVERLAP notes first. Only
  HAND_FIELDS are labeled; the file lists them under `fields`, and scoring uses only those.
- Second pass (the OVERLAP notes): the same annotator starts again from the prefill, without
  seeing `gold.json`, and writes `gold_b.json` at least SECOND_PASS_DAYS after the first pass.
- `claude.json` (labels filled in with Claude, kept as a second reference) is never opened here.
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .extract_eval import FIELDS
from .extraction import Diagnosis, DischargeExtraction, FollowUp, MedicationChange

FIRST, SECOND = "gold.json", "gold_b.json"
SECOND_PASS_DAYS = 14
# The fields labeled by hand; every other field is scored against claude.json only (Thiago's
# decisions, 2026-11-02). Medication changes were dropped after the hand pass: 24% of its
# "started" rows were drugs on the admission list, a prefill error the pass kept.
HAND_FIELDS = ("principal_diagnosis", "secondary_diagnoses")
# editable table per list field: column names, and which pydantic model a row becomes
TABLES: dict[str, tuple[str, ...]] = {
    "secondary_diagnoses": ("text", "icd10"),
    "medication_changes": ("name", "change", "detail"),
    "follow_up": ("with_whom", "timeframe", "reason"),
    "pending_tests": ("text",),
    "red_flags": ("text",),
}
_ROW_MODELS = {
    "secondary_diagnoses": Diagnosis,
    "medication_changes": MedicationChange,
    "follow_up": FollowUp,
}


@dataclass(frozen=True)
class Note:
    folder: Path

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def overlap(self) -> bool:
        return (self.folder / "OVERLAP").exists()

    def text(self) -> str:
        return (self.folder / "input.txt").read_text(encoding="utf-8")

    def target(self, second: bool) -> Path:
        return self.folder / (SECOND if second else FIRST)


def notes(gold_dir: Path) -> list[Note]:
    return [Note(p) for p in sorted(gold_dir.iterdir()) if (p / "input.txt").exists()]


def read_saved(path: Path) -> tuple[DischargeExtraction, dict[str, Any]] | None:
    """A saved annotation and its bookkeeping keys (minutes, saved_at); None if missing/invalid."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DischargeExtraction.model_validate(raw), raw
    except (OSError, ValueError):
        return None


def checked_fields(path: Path) -> tuple[str, ...]:
    """The fields a saved annotation covers: its `fields` key, or every field if it has none."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return FIELDS
    listed = raw.get("fields", FIELDS) if isinstance(raw, dict) else FIELDS
    return tuple(f for f in FIELDS if f in listed)


def common_fields(paths: Iterable[Path]) -> tuple[str, ...]:
    """The fields every one of these annotations covers, which is what scoring may use."""
    checked = [set(checked_fields(p)) for p in paths]
    return tuple(f for f in FIELDS if all(f in c for c in checked))


def start_labels(note: Note, second: bool) -> tuple[DischargeExtraction, float, bool]:
    """What the form opens with, the minutes already spent, and whether a saved file was invalid.

    A saved annotation for this pass resumes. Otherwise the prefill: the second pass never opens
    gold.json, so the two passes are independent.
    """
    target = note.target(second)
    saved = read_saved(target)
    if saved is not None:
        return saved[0], float(saved[1].get("minutes", 0.0)), False
    prefill = DischargeExtraction.model_validate_json(
        (note.folder / "prefill.json").read_text(encoding="utf-8")
    )
    return prefill, 0.0, target.exists()


def to_rows(ext: DischargeExtraction) -> dict[str, list[dict[str, str]]]:
    """One list of table rows per list field (strings only, for an editable table)."""
    out: dict[str, list[dict[str, str]]] = {}
    for field, cols in TABLES.items():
        items = getattr(ext, field)
        if field in _ROW_MODELS:
            dumped = [item.model_dump(mode="json") for item in items]
            out[field] = [{c: d.get(c) or "" for c in cols} for d in dumped]
        else:
            out[field] = [{"text": s} for s in items]
    return out


def _blank(value: Any) -> bool:
    # an empty cell in an edited table arrives as None or NaN
    return value is None or (isinstance(value, float) and math.isnan(value))


def _clean(row: dict[str, Any], cols: tuple[str, ...]) -> dict[str, str]:
    return {c: "" if _blank(row.get(c)) else str(row[c]).strip() for c in cols}


def from_form(
    principal_text: str, principal_icd10: str, rows: dict[str, list[dict[str, Any]]]
) -> DischargeExtraction:
    """The annotation the form describes. Blank rows are dropped; raises ValueError if invalid."""
    data: dict[str, Any] = {
        "principal_diagnosis": (
            {"text": principal_text.strip(), "icd10": principal_icd10.strip() or None}
            if principal_text.strip()
            else None
        )
    }
    for field, cols in TABLES.items():
        kept = [r for r in (_clean(r, cols) for r in rows.get(field, [])) if any(r.values())]
        if field == "secondary_diagnoses":
            data[field] = [{**r, "icd10": r["icd10"] or None} for r in kept]
        elif field in _ROW_MODELS:
            data[field] = kept
        else:
            data[field] = [r["text"] for r in kept]
    return DischargeExtraction.model_validate(data)


def save(
    note: Note,
    ext: DischargeExtraction,
    minutes: float,
    second: bool,
    now: datetime | None = None,
    fields: tuple[str, ...] = HAND_FIELDS,
) -> Path:
    """Write gold.json (or gold_b.json) atomically: the checked fields (the rest left empty),
    which fields those are, the minutes spent and the save time."""
    now = now or datetime.now()
    blank: dict[str, Any] = {
        f: None if f == "principal_diagnosis" else [] for f in FIELDS if f not in fields
    }
    data = ext.model_copy(update=blank).model_dump(mode="json") | {
        "fields": list(fields),
        "minutes": round(minutes, 1),
        "saved_at": now.isoformat(timespec="seconds"),
    }
    path = note.target(second)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def second_pass_opens(note: Note) -> date | None:
    """The first day the second pass may start: SECOND_PASS_DAYS after the first pass was saved.

    None if the note is not an overlap note or has no valid first pass yet.
    """
    saved = read_saved(note.target(second=False)) if note.overlap else None
    if saved is None or "saved_at" not in saved[1]:
        return None
    first = datetime.fromisoformat(saved[1]["saved_at"]).date()
    return first + timedelta(days=SECOND_PASS_DAYS)


def status(note: Note, second: bool, today: date) -> str:
    """done, invalid, todo, or (second pass only) waiting / not_started."""
    target = note.target(second)
    if target.exists():
        return "done" if read_saved(target) is not None else "invalid"
    if not second:
        return "todo"
    opens = second_pass_opens(note)
    if opens is None:
        return "not_started"
    return "todo" if today >= opens else "waiting"


def queue(all_notes: list[Note], second: bool) -> list[Note]:
    """The notes a pass works on: every note, or only the overlap notes for the second pass."""
    return [n for n in all_notes if n.overlap] if second else all_notes


def next_todo(all_notes: list[Note], second: bool, today: date) -> Note | None:
    """The first note left to do, overlap notes first (the hand-labeled set)."""
    for n in sorted(queue(all_notes, second), key=lambda n: not n.overlap):
        if status(n, second, today) in ("todo", "invalid"):
            return n
    return None


def progress(all_notes: list[Note], today: date) -> dict[str, int]:
    overlap = queue(all_notes, second=True)
    second = [status(n, True, today) for n in overlap]
    return {
        "notes": len(all_notes),
        "first_pass_done": sum(status(n, False, today) == "done" for n in all_notes),
        "overlap_notes": len(overlap),
        "first_pass_overlap_done": sum(status(n, False, today) == "done" for n in overlap),
        "second_pass_done": second.count("done"),
        "second_pass_open": second.count("todo"),
    }


def synthetic_gold(dest: Path, n: int, overlap: int) -> dict[str, int]:
    """A practice gold folder from the synthetic benchmark notes (never MIMIC) to try the screen."""
    from .benchmark import load
    from .notes import extraction_input, silver_labels

    items = load("test")[:n]
    for i, x in enumerate(items):
        d = dest / f"{i:04d}_{x['id']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.txt").write_text(extraction_input(x["note"])[0], encoding="utf-8")
        (d / "prefill.json").write_text(silver_labels(x["note"]).model_dump_json(indent=2))
        if i < overlap:
            (d / "OVERLAP").write_text("second pass: write gold_b.json\n")
    return {"exported": len(items), "overlap": min(overlap, len(items))}
