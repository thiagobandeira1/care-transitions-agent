"""Option C teacher labels on fictional notes (no MIMIC data, no model)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ctagent import teacher
from ctagent.extraction import DischargeExtraction
from ctagent.extractor import Extractor
from ctagent.llm import FakeLLM
from test_train_examples import load_script


def _label(red_flag: str = "", follow_up: bool = False) -> DischargeExtraction:
    return DischargeExtraction.model_validate(
        {
            "principal_diagnosis": {"text": "heart failure"},
            "medication_changes": [{"name": "furosemide", "change": "started"}],
            "follow_up": [{"with_whom": "cardiology"}] if follow_up else [],
            "red_flags": [red_flag] if red_flag else [],
        }
    )


def _teacher(prompt: str) -> DischargeExtraction | Exception:
    if "fails" in prompt:
        return ValueError("quoted note text")
    # the teacher disagrees with silver on diagnoses; only its weak fields are kept
    out = _label("chest pain", follow_up=True)
    return out.model_copy(update={"principal_diagnosis": None})


def test_merge_keeps_silver_diagnoses_and_meds_and_teacher_weak_fields() -> None:
    weak = _label("chest pain", follow_up=True).model_dump(mode="json")
    merged = teacher.merge(_label("fever"), weak)
    assert merged.principal_diagnosis and merged.principal_diagnosis.text == "heart failure"
    assert merged.red_flags == ["chest pain"] and merged.follow_up[0].with_whom == "cardiology"


def test_label_stops_at_n_and_resumes_without_relabeling(tmp_path: Path) -> None:
    notes = [(f"n{i}", "fails" if i == 1 else f"note {i}") for i in range(6)]
    llm = FakeLLM({"DischargeExtraction": _teacher})
    path = tmp_path / "teacher" / "train.jsonl"
    first = teacher.label(notes, Extractor(llm), path, n=2)
    assert (first["labeled"], first["failures"], first["notes"]) == (2, 1, 3)
    assert first["share_with_items"] == {"follow_up": 1.0, "pending_tests": 0.0, "red_flags": 1.0}
    assert "quoted note text" not in json.dumps(first)
    calls = len(llm.calls)
    again = teacher.label(notes, Extractor(llm), path, n=4)
    assert again["labeled"] == 4 and len(llm.calls) == calls + 2  # n0-n2 are not relabeled
    assert list(teacher.load(path)) == ["n0", "n2", "n3", "n4"]
    # a stop in mid-write leaves a cut-off line; resuming drops it and relabels that note
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"note_id": "n5", "wea')
    assert teacher.summary(path)["notes"] == 5
    last = teacher.label(notes, Extractor(llm), path, n=5)
    assert last["labeled"] == 5 and list(teacher.load(path))[-1] == "n5"


def test_empty_file_summary() -> None:
    assert teacher.summary(Path("missing.jsonl"))["labeled"] == 0


def test_training_uses_teacher_notes_in_labeling_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = [_label("fever").model_dump_json() for _ in range(3)]
    pd.DataFrame(
        {"note_id": ["a", "b", "c"], "input": ["note a", "note b", "note c"], "silver": silver}
    ).to_parquet(tmp_path / "train.parquet")
    monkeypatch.setenv("MIMIC_WORK_DIR", str(tmp_path))
    llm = FakeLLM({"DischargeExtraction": _teacher})
    teacher.label([("c", "note c"), ("a", "note a")], Extractor(llm), tmp_path / "train.jsonl", 2)
    script = load_script()

    merged = script.examples("mimic", "train", 10, 0, tmp_path, "teacher")
    assert ["note c" in ex["prompt"][1]["content"] for ex in merged] == [True, False]
    targets = [json.loads(ex["completion"][0]["content"]) for ex in merged]
    assert len(targets) == 2 and all(t["red_flags"] == ["chest pain"] for t in targets)
    assert targets[0]["principal_diagnosis"]["text"] == "heart failure"
    control = script.examples("mimic", "train", 1, 0, tmp_path, "silver")
    assert len(control) == 1
    assert json.loads(control[0]["completion"][0]["content"])["red_flags"] == ["fever"]
    assert "note c" in control[0]["prompt"][1]["content"]  # labeling order, nested subsets

    with pytest.raises(ValueError, match="teacher-dir"):
        script.examples("mimic", "train", 10, 0, None, "teacher")
