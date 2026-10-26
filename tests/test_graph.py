import sqlite3
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from ctagent.drafts import CallScript, MedLine, PlanExplanation, ProviderBrief
from ctagent.graph import Deps, ReviewDecision, build_graph, resume_review, start_discharge
from ctagent.llm import FakeLLM, LLMOutputError
from ctagent.models import (
    Condition,
    DischargeEvent,
    Disposition,
    MedChange,
    MedChangeType,
    Medication,
    PatientContext,
)
from ctagent.tracking import Tracker

EVENT = DischargeEvent(
    patient_id="p1",
    episode_id="ep1",
    admit=datetime(2024, 3, 4, 8),
    discharge=datetime(2024, 3, 12, 14),
    age=78,
    disposition=Disposition.HOME,
    reason="Chronic congestive heart failure (disorder)",
    acute=True,
    er_visits_6m=2,
)
CTX = PatientContext(
    conditions=[
        Condition(
            code="88805009",
            description="Chronic congestive heart failure (disorder)",
            onset=date(2020, 1, 1),
        ),
        Condition(
            code="431857002",
            description="Chronic kidney disease stage 4 (disorder)",
            onset=date(2021, 1, 1),
        ),
    ],
    active_meds=[
        Medication(code="2", description="furosemide 40 MG Oral Tablet", start=date(2024, 3, 5))
    ],
    med_changes=[MedChange(name="furosemide", change=MedChangeType.DOSE_CHANGED)],
)


def good() -> dict[str, Callable[[str], BaseModel | Exception]]:
    line = MedLine(name="furosemide", change=MedChangeType.DOSE_CHANGED, note="higher dose")
    return {
        "PlanExplanation": lambda p: PlanExplanation(summary="High risk: early visit."),
        "CallScript": lambda p: CallScript(
            opening="Hi",
            questions=["How are you?"],
            medication_review=[line],
            warning_signs=["Weight gain"],
            scheduling="Visit by Friday",
        ),
        "ProviderBrief": lambda p: ProviderBrief(
            reason_for_admission="CHF",
            key_conditions=["Congestive heart failure", "Kidney disease stage 4"],
            medication_changes=[line],
            watch_for=["Weight"],
            risk_summary="LACE 14",
        ),
    }


def make(
    tmp_path: Path,
    responses: dict[str, Callable[[str], BaseModel | Exception]],
    saver: object | None = None,
):  # type: ignore[no-untyped-def]
    tracker = Tracker(tmp_path / "track.sqlite")
    llm = FakeLLM(responses)
    graph = build_graph(
        Deps(llm=llm, contexts=lambda e: CTX, tracker=tracker), saver or InMemorySaver()
    )  # type: ignore[arg-type]
    return graph, tracker, llm


def test_runs_to_human_review_then_approves(tmp_path: Path) -> None:
    graph, tracker, _ = make(tmp_path, good())
    out = start_discharge(graph, EVENT)
    [pause] = out.interrupts
    assert (
        pause.value["tier"] == "high" and pause.value["lace"] == 13
    )  # LOS 5 + acute 3 + Charlson 3 (CHF 1 + CKD 2) + ER 2
    assert pause.value["flags"] == [] and pause.value["call_script"]["opening"] == "Hi"
    assert tracker.status("ep1") == "pending_review"

    done = resume_review(graph, "ep1", ReviewDecision(decision="approve", reviewer="coordinator"))
    assert done.value["status"] == "approved" and not done.interrupts
    assert tracker.status("ep1") == "approved"
    assert tracker.due_check_ins(date(2024, 3, 22)) == [("ep1", "2024-03-22")]
    assert tracker.due_check_ins(date(2024, 4, 30)) == [
        ("ep1", "2024-03-22"),
        ("ep1", "2024-04-01"),
    ]


def test_reject_schedules_nothing(tmp_path: Path) -> None:
    graph, tracker, _ = make(tmp_path, good())
    start_discharge(graph, EVENT)
    resume_review(graph, "ep1", ReviewDecision(decision="reject", reviewer="dr", note="wrong pt"))
    assert tracker.status("ep1") == "rejected" and tracker.due_check_ins(date(2025, 1, 1)) == []


def test_edit_replaces_the_draft(tmp_path: Path) -> None:
    graph, _, _ = make(tmp_path, good())
    start_discharge(graph, EVENT)
    edited = CallScript(
        opening="Edited",
        questions=[],
        medication_review=[],
        warning_signs=[],
        scheduling="Tuesday 10am",
    )
    out = resume_review(
        graph, "ep1", ReviewDecision(decision="edit", reviewer="c", call_script=edited)
    )
    assert out.value["call_script"]["opening"] == "Edited" and out.value["status"] == "approved"


def test_one_retry_then_flag_with_fallback(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def flaky(prompt: str) -> BaseModel | Exception:
        attempts["n"] += 1
        return LLMOutputError("bad json") if attempts["n"] == 1 else PlanExplanation(summary="ok")

    graph, _, _ = make(tmp_path, good() | {"PlanExplanation": flaky})
    assert start_discharge(graph, EVENT).interrupts[0].value["flags"] == []  # recovered on retry

    graph2, _, _ = make(tmp_path / "b", good() | {"PlanExplanation": lambda p: LLMOutputError("x")})
    pause = start_discharge(graph2, EVENT).interrupts[0].value
    assert any("plan explanation failed after 2 tries" in f for f in pause["flags"])
    assert "LACE 13" in pause["explanation"]  # deterministic fallback text


def test_grounding_flags_invented_medication_and_condition(tmp_path: Path) -> None:
    bad = MedLine(name="warfarin", change=MedChangeType.STARTED)
    responses = good() | {
        "ProviderBrief": lambda p: ProviderBrief(
            reason_for_admission="CHF",
            key_conditions=["Atrial fibrillation"],
            medication_changes=[bad],
            watch_for=[],
            risk_summary="",
        )
    }
    graph, _, _ = make(tmp_path, responses)
    flags = start_discharge(graph, EVENT).interrupts[0].value["flags"]
    assert "medication not in record: warfarin" in flags
    assert "condition not in record: Atrial fibrillation" in flags


def test_prompt_contains_only_record_facts(tmp_path: Path) -> None:
    graph, _, llm = make(tmp_path, good())
    start_discharge(graph, EVENT)
    prompt = llm.calls[0][1]
    assert "Chronic kidney disease stage 4" in prompt and "furosemide (dose_changed)" in prompt
    assert "TCM 99496" in prompt
    assert (
        "- summary (text): 2-3 plain sentences" in prompt
    )  # the schema's guidance reaches the model


def test_review_survives_restart_with_sqlite_checkpointer(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    with sqlite3.connect(db, check_same_thread=False) as con1:
        graph, tracker, _ = make(tmp_path, good(), SqliteSaver(con1))
        start_discharge(graph, EVENT)
    with sqlite3.connect(db, check_same_thread=False) as con2:  # a new process after a restart
        graph2 = build_graph(
            Deps(llm=FakeLLM(good()), contexts=lambda e: CTX, tracker=tracker), SqliteSaver(con2)
        )
        out = resume_review(graph2, "ep1", ReviewDecision(decision="approve", reviewer="c"))
    assert out.value["status"] == "approved"


def test_invalid_review_decision_rejected_before_resume() -> None:
    with pytest.raises(ValueError):
        ReviewDecision.model_validate({"decision": "maybe", "reviewer": "c"})
