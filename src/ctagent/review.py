"""Service layer for the coordinator review screen: what's pending, what it says, and decisions.

The screen (app/review_app.py) stays thin; everything here is testable without a browser or model.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

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
    ws: Workspace, llm: StructuredLLM, contexts: Callable[[DischargeEvent], PatientContext]
) -> tuple[CompiledStateGraph[Any], Tracker]:
    ws.root.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(ws.tracking)
    saver = SqliteSaver(sqlite3.connect(str(ws.checkpoints), check_same_thread=False))
    return build_graph(Deps(llm=llm, contexts=contexts, tracker=tracker), saver), tracker


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
