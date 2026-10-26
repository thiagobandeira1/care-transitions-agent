"""Gold export/check on a fictional work dir (no MIMIC data)."""

import json
from pathlib import Path

import pandas as pd

from ctagent.gold import check, export
from ctagent.notes import silver_labels
from test_notes import NOTE


def test_export_then_check_agreement(tmp_path: Path) -> None:
    silver = silver_labels(NOTE).model_dump_json()
    pd.DataFrame(
        {"note_id": [f"n{i}" for i in range(5)], "input": NOTE, "silver": silver}
    ).to_parquet(tmp_path / "test.parquet")
    assert export(tmp_path, n=4, overlap=2) == {"exported": 4, "overlap": 2}
    folders = sorted((tmp_path / "gold").iterdir())
    assert (
        len(folders) == 4
        and (folders[0] / "OVERLAP").exists()
        and not (folders[3] / "OVERLAP").exists()
    )

    gold = json.loads((folders[0] / "prefill.json").read_text())
    gold["red_flags"] = gold["red_flags"][:1]  # annotator removes two items
    (folders[0] / "gold.json").write_text(json.dumps(gold))
    (folders[0] / "gold_b.json").write_text(json.dumps(gold))
    (folders[1] / "gold.json").write_text("{not json")
    out = check(tmp_path)
    assert (out["folders"], out["annotated"], out["invalid"], out["double_annotated"]) == (
        4,
        2,
        1,
        1,
    )
    assert out["annotator_b_vs_a"]["exact"]["micro"]["f1"] == 1.0
    assert out["silver_vs_gold"]["exact"]["red_flags"]["precision"] < 1.0
