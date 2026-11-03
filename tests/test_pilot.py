"""Pilot measures: review edits through the graph, outcomes, and the counts-only report."""

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from ctagent.__main__ import main
from ctagent.drafts import ProviderBrief
from ctagent.graph import ReviewDecision, resume_review, start_discharge
from ctagent.pilot import edit_share, fields_changed, report, review_edits
from ctagent.review import Workspace, follow_ups, outcome_fields
from ctagent.tracking import Tracker
from test_graph import EVENT, good, make

TODAY = date(2026, 12, 1)  # a Tuesday


def plan(tier: str, contact: str, visit: str, tcm: str) -> dict[str, object]:
    code = "99496" if tier == "high" else "99495"
    return {
        "tier": tier,
        "contact_by": contact,
        "visit_by": visit,
        "tcm_code": code,
        "tcm_visit_by": tcm,
    }


def test_edit_measures() -> None:
    a = {"opening": "Hi", "questions": ["How are you?"]}
    assert edit_share(a, a) == 0.0 and edit_share(None, None) == 0.0
    assert 0 < edit_share(a, {**a, "opening": "Hello there"}) < 1
    assert fields_changed("call_script", a, {**a, "opening": "Hello"}) == ["call_script.opening"]
    assert review_edits("reject", (a, a), (a, a)) == {
        "script_edit": None,
        "brief_edit": None,
        "fields_edited": [],
    }


def test_review_measures_flow_through_the_graph(tmp_path: Path) -> None:
    graph, tracker, _ = make(tmp_path, good())
    pause = start_discharge(graph, EVENT).interrupts[0].value
    brief = ProviderBrief.model_validate(pause["brief"])
    edited = brief.model_copy(update={"watch_for": ["Weight", "Ankle swelling"]})
    decision = ReviewDecision(
        decision="edit", reviewer="coord", brief=edited, seconds=42.5, flags_real="no"
    )
    resume_review(graph, EVENT.episode_id, decision)
    row = tracker.con.execute(
        """select decision, review_seconds, flags, flags_real, script_edit, brief_edit,
                  fields_edited from reviews"""
    ).fetchone()
    assert row[:4] == ("edit", 42.5, 0, "no")
    assert row[4] == 0.0 and 0 < row[5] < 1  # only the brief changed
    assert json.loads(row[6]) == ["brief.watch_for"]
    assert tracker.status(EVENT.episode_id) == "approved"


def build(tmp_path: Path) -> Tracker:
    """Five discharges: an edited high-risk one, an approved and a rejected standard one, one
    still pending, and one approved yesterday whose deadlines haven't passed."""
    t = Tracker(tmp_path / "pilot" / "tracking.sqlite")
    rows = [
        ("A", "2026-11-02", plan("high", "2026-11-04", "2026-11-05", "2026-11-09")),
        ("B", "2026-11-06", plan("standard", "2026-11-10", "2026-11-13", "2026-11-20")),
        ("C", "2026-11-09", plan("standard", "2026-11-11", "2026-11-16", "2026-11-23")),
        ("D", "2026-11-27", plan("standard", "2026-12-01", "2026-12-04", "2026-12-11")),
        ("E", "2026-11-30", plan("high", "2026-12-02", "2026-12-03", "2026-12-07")),
    ]
    for ep, day, p in rows:
        t.record_plan(ep, f"p{ep}", date.fromisoformat(day), p, 12, [])
    t.record_review(
        "A",
        "edit",
        "coord",
        ["2026-11-12", "2026-11-22"],
        review_seconds=60,
        flags=1,
        flags_real="no",
        script_edit=0.0,
        brief_edit=0.2,
        fields_edited=["brief.watch_for"],
    )
    t.record_review("B", "approve", "coord", [], review_seconds=30, script_edit=0, brief_edit=0)
    t.record_review("C", "reject", "coord", [], review_seconds=90, flags=2, flags_real="yes")
    t.record_review("E", "approve", "coord", ["2026-12-10"], script_edit=0, brief_edit=0)
    t.record_outcome(
        "A",
        **outcome_fields(date(2026, 11, 3), 1, 12, date(2026, 11, 6), date(2026, 11, 6), "no"),
    )
    t.record_outcome("B", contacted_at="2026-11-11", readmitted=1)  # contact a day late
    with t.con:  # one check-in done on its due date (completion time is "now" otherwise)
        t.con.execute(
            "update check_ins set done_at='2026-11-12T10:00:00', outcome='ok' "
            "where episode_id='A' and due_date='2026-11-12'"
        )
    return t


def test_report_counts(tmp_path: Path) -> None:
    r = report(build(tmp_path), TODAY)
    assert (r["discharges"], r["pending_review"], r["by_tier"]) == (
        5,
        1,
        {"high": 2, "standard": 3},
    )
    rv = r["review"]
    assert rv["decisions"] == {"edit": 1, "approve": 2, "reject": 1}
    assert (rv["call_scripts_edited"], rv["briefs_edited"]) == (0, 1)
    assert rv["median_edit_share_when_edited"] == {"call_script": None, "brief": 0.2}
    assert rv["fields_edited"] == {"brief.watch_for": 1}
    assert rv["review_seconds_median"] == 60.0
    assert (rv["discharges_with_flags"], rv["flags_real"]) == (2, {"no": 1, "yes": 1})

    # E's deadlines haven't passed and nothing is recorded, so it isn't counted yet
    t = r["timeliness"]["all"]
    assert t["contact_by_2_business_days"] == {"n": 2, "met": 1, "share": 0.5, "not_recorded": 0}
    assert t["visit_by_practice_target"] == {"n": 2, "met": 0, "share": 0.0, "not_recorded": 1}
    assert t["visit_by_tcm_deadline"] == {"n": 2, "met": 1, "share": 0.5, "not_recorded": 1}
    assert r["timeliness"]["high"]["contact_by_2_business_days"]["share"] == 1.0
    assert r["timeliness"]["standard"]["contact_by_2_business_days"]["share"] == 0.0
    assert r["business_days_to_contact_median"] == 2.0  # 1 (Mon -> Tue) and 3 (Fri -> Wed)
    assert r["days_to_visit_median"] == 4.0
    assert r["contact_effort"] == {"attempts_median": 1.0, "minutes_median": 12.0}
    assert r["check_ins"] == {"due": 2, "done": 1, "done_by_due_date": 1}
    assert r["readmissions_descriptive"] == {"recorded": 2, "readmitted": 1}
    assert "pA" not in json.dumps(r) and '"A"' not in json.dumps(r)  # no identifiers


def test_follow_up_list_and_form_values(tmp_path: Path) -> None:
    t = build(tmp_path)
    assert [r["episode_id"] for r in follow_ups(t)] == ["E", "B", "A"]  # approved, newest first
    assert t.outcome("A")["contact_minutes"] == 12 and t.outcome("C") == {}
    assert outcome_fields(None, None, None, None, None, "not known") == dict.fromkeys(
        (
            "contacted_at",
            "contact_attempts",
            "contact_minutes",
            "visit_scheduled",
            "visit_attended",
            "readmitted",
        )
    )
    with pytest.raises(ValueError, match="unknown outcome fields"):
        t.record_outcome("A", phone="555")


def test_drafts_edit_as_plain_text() -> None:
    from pydantic import ValidationError

    from ctagent.drafts import CallScript
    from ctagent.review import draft_from_form, draft_to_form
    from test_graph import good

    script = good()["CallScript"]("")
    form = draft_to_form(script.model_dump(mode="json"))
    assert form["medication_review"] == "furosemide | dose_changed | higher dose"
    assert draft_from_form(CallScript, form) == script  # unchanged text gives the same draft
    form["questions"] += "\n  Do you have a scale at home?  \n\n"
    form["medication_review"] += "\nlisinopril | stopped"
    edited = draft_from_form(CallScript, form)
    assert edited.questions[-1] == "Do you have a scale at home?"
    assert edited.medication_review[-1].note == ""
    with pytest.raises(ValidationError):
        draft_from_form(CallScript, form | {"medication_review": "aspirin | paused"})
    assert draft_to_form(None) == {}


def test_older_database_gets_the_new_columns(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "create table outcomes (episode_id text primary key, contacted_at text, "
        "visit_scheduled text, visit_attended text, readmitted integer)"
    )
    con.commit()
    con.close()
    t = Tracker(path)
    t.record_outcome("X", contact_attempts=2, contact_minutes=5)
    assert t.outcome("X")["contact_attempts"] == 2


def test_pilot_report_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build(tmp_path)
    main(["pilot-report", "--workspace", str(tmp_path / "pilot"), "--date", "2026-12-01"])
    assert json.loads(capsys.readouterr().out)["discharges"] == 5
    with pytest.raises(SystemExit, match="no tracking database"):
        main(["pilot-report", "--workspace", str(tmp_path / "nowhere")])
    assert Workspace(tmp_path / "pilot").tracking.exists()
