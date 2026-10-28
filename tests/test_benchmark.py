import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import ctagent.benchmark as bm
from ctagent.extraction import DischargeExtraction
from fixtures import SyntheaFixture
from test_notes import NOTE


class StubLLM:
    """Writes the fixed test note and returns the reference answer for every extraction call."""

    def __init__(self, model: str, **_: Any) -> None:
        self.name = model

    def complete(self, system: str, prompt: str) -> str:
        # mention every medication from the prompt so the note passes the completeness check
        meds = [
            line.split(":")[0].strip("- ") for line in prompt.splitlines() if line.startswith("- ")
        ]
        return NOTE + "\n" + " ".join(meds) + "\nheart failure acute illness encounter"

    def generate(self, schema: type[BaseModel], system: str, prompt: str) -> Any:
        return schema.model_validate({"principal_diagnosis": {"text": "Acute illness"}})


@pytest.fixture
def bench_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fx = SyntheaFixture()
    for i in range(6):
        fx.patient(f"P{i}")
        fx.encounter(f"P{i}", "inpatient", 3000 + i, 3003 + i)
    fx.patient("ZZ")
    fx.encounter("ZZ", "ambulatory", 3400, 3400.1)
    monkeypatch.setattr(bm, "OllamaLLM", StubLLM)
    monkeypatch.setattr(bm, "BENCH_DIR", tmp_path / "bench")
    monkeypatch.setattr(bm, "REPO", tmp_path)
    return fx.write(tmp_path / "csv")


def test_build_and_score_all_modes(bench_env: Path, tmp_path: Path) -> None:
    meta = bm.build(bench_env, n=6, model="qwen3:14b", train_frac=0.5)
    assert meta["kept"] == 6 and meta["train"] == 3 and meta["test"] == 3  # type: ignore[comparison-overlap]
    items = json.loads((tmp_path / "bench" / "notes.json").read_text())
    assert {x["split"] for x in items} == {"train", "test"}
    DischargeExtraction.model_validate(items[0]["reference"])

    rules = bm.score("rules")
    assert rules["n_test"] == 3 and rules["schema_failures"] == 0 and "relaxed_f1_ci95" in rules
    few = bm.score("few", "qwen3:8b", k_shots=2)
    zero = bm.score("zero", "qwen3:8b")
    assert few["relaxed"]["principal_diagnosis"]["f1"] == 1.0  # stub answers "Acute illness"
    assert zero["model"] == "qwen3:8b"
    assert (tmp_path / "evals" / "results" / "extraction_synthetic_qwen3-8b_few.json").exists()
    assert (tmp_path / "evals" / "results" / "extraction_synthetic_rules.json").exists()


def test_rescore_scores_saved_predictions_without_the_model(
    bench_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bm.build(bench_env, n=6, model="qwen3:14b", train_frac=0.5)
    with pytest.raises(ValueError, match="run without --rescore"):
        bm.score("zero", "qwen3:8b", rescore=True)
    first = bm.score("zero", "qwen3:8b")
    assert (tmp_path / "outputs" / "preds" / "synthetic_qwen3-8b_zero.jsonl").exists()
    monkeypatch.setattr(bm, "OllamaLLM", None)  # a model call would now fail
    again = bm.score("zero", "qwen3:8b", rescore=True)
    assert again["rescored"] and not first["rescored"]
    assert again["relaxed"] == first["relaxed"] and again["schema_failures"] == 0
