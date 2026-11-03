"""Service layer for the coordinator review screen: what's pending, what it says, and decisions.

The screen (app/review_app.py) stays thin; everything here is testable without a browser or model.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from .graph import Deps, ReviewDecision, build_graph, resume_review, start_discharge, thread
from .llm import StructuredLLM
from .models import DischargeEvent, PatientContext
from .tracking import Tracker


@dataclass
class Workspace:
    """Local files for one deployment: tracking DB and graph checkpoints (both gitignored)."""

    root: Path

    @property
    def tracking(self) -> Path:
        return self.root / "tracking.sqlite"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints.sqlite"


def open_graph(
    ws: Workspace,
    llm: StructuredLLM,
    contexts: Callable[[DischargeEvent], PatientContext],
    practice: str = "",
    coordinator: str = "",
) -> tuple[CompiledStateGraph[Any], Tracker]:
    ws.root.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(ws.tracking)
    saver = SqliteSaver(sqlite3.connect(str(ws.checkpoints), check_same_thread=False))
    deps = Deps(
        llm=llm, contexts=contexts, tracker=tracker, practice=practice, coordinator=coordinator
    )
    return build_graph(deps, saver), tracker


def pending(tracker: Tracker) -> list[dict[str, Any]]:
    rows = tracker.con.execute(
        """select episode_id, discharge_date, tier, lace, contact_by, flags from discharges
           where status = 'pending_review' order by tier = 'high' desc, discharge_date"""
    ).fetchall()
    keys = ("episode_id", "discharge_date", "tier", "lace", "contact_by", "flags")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def review_payload(graph: CompiledStateGraph[Any], episode_id: str) -> dict[str, Any] | None:
    """The interrupt payload a reviewer sees (None if the discharge isn't waiting for review)."""
    snapshot = graph.get_state(thread(episode_id))
    for task in snapshot.tasks:
        for intr in task.interrupts:
            value: dict[str, Any] = intr.value
            return value
    return None


def decide(graph: CompiledStateGraph[Any], episode_id: str, decision: ReviewDecision) -> str:
    out = resume_review(graph, episode_id, decision)
    return str(out.value["status"])


def process(graph: CompiledStateGraph[Any], events: list[DischargeEvent]) -> int:
    """Run new discharges up to review (the morning batch). Returns how many now await review."""
    return sum(bool(start_discharge(graph, ev).interrupts) for ev in events)


MED_FIELDS = {"medication_review", "medication_changes"}  # lines of "name | change | note"
LIST_FIELDS = {"questions", "warning_signs", "key_conditions", "watch_for"}  # one item per line


def draft_to_form(doc: dict[str, Any] | None) -> dict[str, str]:
    """A draft as plain text boxes for editing, field by field (no JSON for the reviewer)."""
    out: dict[str, str] = {}
    for k, v in (doc or {}).items():
        if k in MED_FIELDS:
            out[k] = "\n".join(f"{m['name']} | {m['change']} | {m.get('note', '')}" for m in v)
        elif k in LIST_FIELDS:
            out[k] = "\n".join(v)
        else:
            out[k] = str(v)
    return out


def draft_from_form[D: BaseModel](schema: type[D], fields: dict[str, str]) -> D:
    """The edited text boxes as a draft; raises ValidationError on a bad line (e.g. change type)."""
    data: dict[str, Any] = {}
    for k, text in fields.items():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if k in MED_FIELDS:
            parts = [[p.strip() for p in line.split("|")] + ["", ""] for line in lines]
            data[k] = [{"name": p[0], "change": p[1], "note": p[2]} for p in parts]
        elif k in LIST_FIELDS:
            data[k] = lines
        else:
            data[k] = text.strip()
    return schema.model_validate(data)


def follow_ups(tracker: Tracker) -> list[dict[str, Any]]:
    """Approved discharges with their deadlines and the outcomes recorded so far, newest first."""
    cur = tracker.con.execute(
        """select d.episode_id, d.discharge_date, d.tier, d.contact_by, d.visit_by,
                  d.tcm_visit_by, o.contacted_at, o.visit_attended
           from discharges d left join outcomes o using(episode_id)
           where d.status = 'approved' order by d.discharge_date desc, d.episode_id"""
    )
    keys = [c[0] for c in cur.description]
    return [dict(zip(keys, r, strict=True)) for r in cur.fetchall()]


def outcome_fields(
    contacted: date | None,
    attempts: int | None,
    minutes: int | None,
    scheduled: date | None,
    attended: date | None,
    readmitted: str,
) -> dict[str, str | int | None]:
    """The follow-up form's values as tracker outcome fields ('not known' readmission -> None)."""

    def iso(d: date | None) -> str | None:
        return d.isoformat() if d else None

    return {
        "contacted_at": iso(contacted),
        "contact_attempts": attempts,
        "contact_minutes": minutes,
        "visit_scheduled": iso(scheduled),
        "visit_attended": iso(attended),
        "readmitted": {"yes": 1, "no": 0}.get(readmitted),
    }
