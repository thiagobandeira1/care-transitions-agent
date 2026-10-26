"""MIMIC baseline runner on a fictional work dir (no MIMIC data, no model)."""

import json
from pathlib import Path

import pandas as pd
import pytest

import ctagent.mimic_bench as mb
from ctagent.extraction import DischargeExtraction
from ctagent.gold import export
from ctagent.llm import FakeLLM
from ctagent.notes import extraction_input
from test_notes import NOTE


def _label(pending: bool = False, follow_up: bool = True) -> DischargeExtraction:
    return DischargeExtraction.model_validate(
        {
            "principal_diagnosis": {"text": "heart failure"},
            "secondary_diagnoses": [{"text": "atrial fibrillation"}],
            "medication_changes": [{"name": "furosemide", "change": "started"}],
            "follow_up": [{"with_whom": "cardiology"}] if follow_up else [],
            "pending_tests": ["blood culture"] if pending else [],
            "red_flags": ["chest pain"],
        }
    )


def _answer(_: str) -> DischargeExtraction:
    return _label()


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    text, _ = extraction_input(NOTE)
    empty = DischargeExtraction(principal_diagnosis=None).model_dump_json()
    # like MIMIC train silver: follow-ups are redacted, so none is labeled
    train = [_label(pending=i == 4, follow_up=False).model_dump_json() for i in range(6)]
    pd.DataFrame(
        {"input": [text] * 6 + ["a note with nothing labeled"], "silver": [*train, empty]}
    ).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame(
        {
            "note_id": [f"n{i}" for i in range(4)],
            "input": text,
            "silver": _label().model_dump_json(),
        }
    ).to_parquet(tmp_path / "test.parquet")
    export(tmp_path, n=4, overlap=1)
    monkeypatch.setattr(mb, "SHOT_CHARS", (0, 10**6))
    monkeypatch.setattr(mb, "REPO", tmp_path / "repo")
    return tmp_path


def test_zero_shot_scores_silver_and_records_full_runs(work: Path) -> None:
    llm = FakeLLM({"DischargeExtraction": _answer})
    res = mb.run(work, "zero", "qwen3:8b", llm=llm)
    assert (res["n"], res["schema_failures"], res["few_shot_examples"]) == (4, 0, 0)
    assert res["relaxed"]["micro"]["f1"] == 1.0
    assert "EXAMPLE" not in llm.calls[0][1]
    saved = work / "repo" / "evals" / "results" / "extraction_mimic_silver_qwen3-8b_zero.json"
    assert json.loads(saved.read_text())["n"] == 4
    preds = (work / "preds" / "silver_qwen3-8b_zero.jsonl").read_text().splitlines()
    assert len(preds) == 4 and json.loads(preds[0])["pred"] is not None


def test_few_shot_uses_labeled_train_notes_with_pending_tests_first(work: Path) -> None:
    llm = FakeLLM({"DischargeExtraction": _answer})
    res = mb.run(work, "few", "qwen3:8b", limit=2, llm=llm)
    prompt = llm.calls[0][1]
    assert res["n"] == 2 and res["few_shot_examples"] == 3 and prompt.count("\nJSON:\n") == 3
    assert "blood culture" in prompt.split("EXAMPLE 2")[0]
    assert not (work / "repo").exists()  # a smoke test with --limit is not recorded


def test_few_shot_without_enough_examples_fails(work: Path) -> None:
    with pytest.raises(ValueError, match="found 6 few-shot examples, need 7"):
        mb.run(work, "few", "qwen3:8b", k_shots=7, llm=FakeLLM())


def test_failures_keep_only_the_error_type(work: Path) -> None:
    llm = FakeLLM({"DischargeExtraction": lambda _: ValueError("quoted note text SECRET")})
    res = mb.run(work, "zero", "qwen3:8b", llm=llm)
    assert res["schema_failures"] == 4 and res["failure_types"] == {"ValueError": 4}
    assert res["relaxed"]["micro"]["f1"] == 0.0
    assert "SECRET" not in json.dumps(res)
    assert "SECRET" in (work / "preds" / "silver_qwen3-8b_zero.jsonl").read_text()  # local only


def test_gold_reference_uses_valid_annotations_only(work: Path) -> None:
    folders = sorted((work / "gold").iterdir())
    (folders[0] / "gold.json").write_text(_label().model_dump_json())
    (folders[1] / "gold.json").write_text("{not json")
    res = mb.run(
        work, "zero", "qwen3:8b", ref="gold", llm=FakeLLM({"DischargeExtraction": _answer})
    )
    assert res["n"] == 1 and res["benchmark"].endswith("gold labels")


def test_rules_mode_points_to_gold_check(work: Path) -> None:
    with pytest.raises(ValueError, match="gold-check"):
        mb.run(work, "rules", "qwen3:8b", llm=FakeLLM())
