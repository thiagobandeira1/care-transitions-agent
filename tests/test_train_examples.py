"""The training script's data path (no torch needed): prompts must match the zero-shot extractor."""

import importlib.util
import json
from pathlib import Path

import pytest

import ctagent.benchmark as bm
from ctagent.extractor import SYSTEM, Extractor
from ctagent.llm import FakeLLM
from ctagent.notes import silver_labels
from test_notes import NOTE

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_qlora.py"


def load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("train_qlora", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_prompt_is_identical_to_zero_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label = silver_labels(NOTE).model_dump(mode="json")
    items = [
        {
            "id": f"s{i}",
            "patient": f"p{i}",
            "split": "train" if i < 3 else "test",
            "note": NOTE,
            "reference": label,
        }
        for i in range(4)
    ]
    (tmp_path / "notes.json").write_text(json.dumps(items))
    monkeypatch.setattr(bm, "BENCH_DIR", tmp_path)
    train = load_script().examples("synthetic", "train", limit=10, seed=1)
    assert len(train) == 3
    prompt, completion = train[0]["prompt"], train[0]["completion"]
    assert prompt[0] == {"role": "system", "content": SYSTEM}
    assert prompt[1]["content"] == Extractor(FakeLLM()).prompt(NOTE)
    assert json.loads(completion[0]["content"]) == label
