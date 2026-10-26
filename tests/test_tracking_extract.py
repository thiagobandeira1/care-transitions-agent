from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from ctagent.graph import Deps, build_graph, start_discharge
from ctagent.llm import FakeLLM
from ctagent.tracking import Tracker
from test_graph import CTX, EVENT, good

PLAN = {
    "tier": "high",
    "contact_by": "2024-03-14",
    "visit_by": "2024-03-15",
    "tcm_code": "99496",
    "tcm_visit_by": "2024-03-19",
    "check_ins": ["2024-03-22", "2024-04-01"],
}


def test_check_in_lifecycle_and_outcomes(tmp_path: Path) -> None:
    t = Tracker(tmp_path / "nested" / "t.sqlite")
    t.record_plan("ep1", "p1", date(2024, 3, 12), PLAN, 13, ["flag a"])
    assert t.status("ep1") == "pending_review" and t.due_check_ins(date(2025, 1, 1)) == []
    t.record_review("ep1", "approve", "coordinator", PLAN["check_ins"])  # type: ignore[arg-type]
    t.complete_check_in("ep1", "2024-03-22", "doing well")
    assert t.due_check_ins(date(2025, 1, 1)) == [("ep1", "2024-04-01")]
    t.record_outcome("ep1", contacted_at="2024-03-13", visit_attended="2024-03-15", readmitted=0)
    t.record_outcome("ep1", readmitted=1)
    row = t.con.execute("select contacted_at, readmitted from outcomes").fetchone()
    assert row == ("2024-03-13", 1)
    assert t.summary() == {"approved": 1}
    assert t.status("missing") is None
    with pytest.raises(ValueError):
        t.record_outcome("ep1", patient_name="x")


class Extracted(BaseModel):
    principal_diagnosis: str


def test_extract_node_uses_note_and_flags_failures(tmp_path: Path) -> None:
    def extractor(note: str) -> BaseModel:
        if "boom" in note:
            raise RuntimeError("parse")
        return Extracted(principal_diagnosis="CHF")

    llm = FakeLLM(good())
    deps = Deps(
        llm=llm, contexts=lambda e: CTX, tracker=Tracker(tmp_path / "t.sqlite"), extractor=extractor
    )
    graph = build_graph(deps, InMemorySaver())
    start_discharge(graph, EVENT, note_text="Discharge summary text")
    assert "From the discharge summary" in llm.calls[0][1]

    graph2 = build_graph(deps, InMemorySaver())
    pause = start_discharge(graph2, EVENT, note_text="boom").interrupts[0].value
    assert "extraction failed: RuntimeError" in pause["flags"]
