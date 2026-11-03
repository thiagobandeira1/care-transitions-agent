"""Annotation screen service layer, on a practice set from the synthetic benchmark (no MIMIC)."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from ctagent.__main__ import main
from ctagent.annotate import (
    HAND_FIELDS,
    checked_fields,
    from_form,
    next_todo,
    notes,
    progress,
    save,
    second_pass_opens,
    start_labels,
    status,
    synthetic_gold,
    to_rows,
)
from ctagent.extract_eval import FIELDS
from ctagent.extraction import Diagnosis, DischargeExtraction
from ctagent.gold import check

DAY1 = datetime(2026, 11, 1, 9, 30)


def practice(tmp_path: Path, n: int = 3, overlap: int = 1) -> Path:
    assert synthetic_gold(tmp_path / "gold", n, overlap) == {"exported": n, "overlap": overlap}
    return tmp_path


def test_first_pass_opens_prefill_saves_and_resumes(tmp_path: Path) -> None:
    ns = notes(practice(tmp_path) / "gold")
    assert len(ns) == 3 and ns[0].overlap and not ns[1].overlap
    assert ns[0].text().strip()
    ext, minutes, invalid = start_labels(ns[0], second=False)
    assert (minutes, invalid) == (0.0, False)

    rows = to_rows(ext)
    rows["red_flags"].append({"text": "  new chest pain "})
    rows["pending_tests"].append({"text": ""})  # a blank row the annotator added and left empty
    edited = from_form("Pneumonia", "", rows)
    assert edited.red_flags[-1] == "new chest pain"
    assert edited.pending_tests == ext.pending_tests

    path = save(ns[0], edited, 3.24, second=False, now=DAY1)
    saved = json.loads(path.read_text())
    assert path.name == "gold.json"
    assert (saved["minutes"], saved["saved_at"]) == (3.2, "2026-11-01T09:30:00")
    assert not list(ns[0].folder.glob("*.tmp"))
    # only the hand-labeled fields are kept; the rest are saved empty and not listed
    assert saved["fields"] == list(HAND_FIELDS) and saved["red_flags"] == []
    kept = edited.model_copy(
        update={
            "medication_changes": [],
            "follow_up": [],
            "pending_tests": [],
            "red_flags": [],
        }
    )
    assert start_labels(ns[0], second=False) == (kept, 3.2, False)
    assert checked_fields(path) == HAND_FIELDS
    assert checked_fields(ns[0].folder / "prefill.json") == FIELDS  # no `fields` key: all
    assert status(ns[0], False, DAY1.date()) == "done"
    assert next_todo(ns, False, DAY1.date()) == ns[1]

    out = check(tmp_path)  # the existing gold-check reads what the screen writes
    assert (out["annotated"], out["invalid"]) == (1, 0)


def test_form_rows_round_trip_and_blanks() -> None:
    empty = from_form("  ", "", {})
    assert empty.principal_diagnosis is None and empty.medication_changes == []

    ext = from_form(
        "Pneumonia",
        " J18.9 ",
        {
            "secondary_diagnoses": [{"text": "Heart failure", "icd10": ""}],
            "medication_changes": [{"name": "furosemide", "change": "started", "detail": None}],
            "follow_up": [{"with_whom": "Cardiology", "timeframe": "2 weeks", "reason": ""}],
            "pending_tests": [{"text": " blood culture "}, {"text": None}, {"text": float("nan")}],
        },
    )
    assert ext.principal_diagnosis == Diagnosis(text="Pneumonia", icd10="J18.9")
    assert ext.secondary_diagnoses[0].icd10 is None
    assert ext.medication_changes[0].detail == ""
    assert ext.pending_tests == ["blood culture"]
    pd_ = ext.principal_diagnosis
    assert from_form(pd_.text, pd_.icd10 or "", to_rows(ext)) == ext

    with pytest.raises(ValueError):
        from_form("x", "", {"medication_changes": [{"name": "aspirin", "change": "paused"}]})


def test_second_pass_waits_two_weeks_and_never_sees_the_first(tmp_path: Path) -> None:
    ns = notes(practice(tmp_path, n=2, overlap=1) / "gold")
    assert status(ns[0], True, DAY1.date()) == "not_started"
    prefill, _, _ = start_labels(ns[0], second=False)
    first = prefill.model_copy(update={"principal_diagnosis": Diagnosis(text="Changed")})
    save(ns[0], first, 5.0, second=False, now=DAY1)

    assert second_pass_opens(ns[0]) == date(2026, 11, 15)
    assert second_pass_opens(ns[1]) is None  # not an overlap note
    assert status(ns[0], True, date(2026, 11, 14)) == "waiting"
    assert next_todo(ns, True, date(2026, 11, 14)) is None
    assert next_todo(ns, True, date(2026, 11, 15)) == ns[0]
    assert start_labels(ns[0], second=True)[0] == prefill  # starts again from the prefill
    assert progress(ns, date(2026, 11, 15)) == {
        "notes": 2,
        "first_pass_done": 1,
        "overlap_notes": 1,
        "first_pass_overlap_done": 1,
        "second_pass_done": 0,
        "second_pass_open": 1,
    }

    save(ns[0], prefill, 4.0, second=True, now=datetime(2026, 11, 15, 10))
    assert status(ns[0], True, date(2026, 11, 15)) == "done"
    assert check(tmp_path)["double_annotated"] == 1


def test_claude_labels_stay_out_of_the_hand_pass(tmp_path: Path) -> None:
    ns = notes(practice(tmp_path, n=3, overlap=1) / "gold")
    prefill, _, _ = start_labels(ns[0], second=False)
    claude = prefill.model_copy(
        update={"principal_diagnosis": Diagnosis(text="zz not in any note"), "red_flags": ["zz"]}
    )
    for n in ns:
        (n.folder / "claude.json").write_text(claude.model_dump_json())
    assert start_labels(ns[0], second=False) == (prefill, 0.0, False)  # never opens claude.json
    assert status(ns[0], False, DAY1.date()) == "todo"

    save(ns[1], prefill, 6.0, second=False, now=DAY1)  # a non-overlap note done first
    assert next_todo(ns, False, DAY1.date()) == ns[0]  # the overlap note still comes first
    save(ns[0], prefill, 6.0, second=False, now=DAY1)
    out = check(tmp_path)
    assert (out["annotated"], out["claude_labeled"]) == (2, 3)
    assert out["gold_fields"] == list(HAND_FIELDS)
    agree = out["claude_vs_gold"]
    assert agree["n"] == 2 and agree["fields"] == list(HAND_FIELDS)
    assert "red_flags" not in agree["exact"]  # not hand-labeled, so not scored
    assert agree["exact"]["principal_diagnosis"]["precision"] < 1.0
    rules = out["silver_vs_claude"]  # the rules against the model-made reference, every field
    assert rules["n"] == 3 and "red_flags" in rules["exact"]


def test_broken_saved_file_reopens_from_prefill(tmp_path: Path) -> None:
    ns = notes(practice(tmp_path) / "gold")
    ns[0].target(second=False).write_text("{not json")
    ext, minutes, invalid = start_labels(ns[0], second=False)
    prefill = DischargeExtraction.model_validate_json((ns[0].folder / "prefill.json").read_text())
    assert (ext, minutes, invalid) == (prefill, 0.0, True)
    assert status(ns[0], False, DAY1.date()) == "invalid"
    assert next_todo(ns, False, DAY1.date()) == ns[0]
    assert second_pass_opens(ns[0]) is None


def test_gold_demo_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["gold-demo", "--gold-dir", str(tmp_path / "practice"), "--n", "2", "--overlap", "1"])
    assert json.loads(capsys.readouterr().out) == {"exported": 2, "overlap": 1}
    assert len(notes(tmp_path / "practice")) == 2
