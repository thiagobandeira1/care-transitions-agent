"""Gold-set export and checking for MIMIC-IV-Note (local only; prints aggregates only).

Export writes one folder per sampled test note under MIMIC_WORK_DIR/gold/ with the model input and
silver labels as a pre-fill. Check validates annotations and reports agreement (silver vs gold,
annotator A vs B) as aggregate F1, never note text. See docs/ANNOTATION.md.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from .extract_eval import evaluate
from .extraction import DischargeExtraction


def export(work_dir: Path, n: int, overlap: int, seed: int = 21) -> dict[str, int]:
    test = pd.read_parquet(work_dir / "test.parquet")
    sample = test.sample(n=min(n, len(test)), random_state=seed).reset_index(drop=True)
    gold = work_dir / "gold"
    for i, row in sample.iterrows():
        d = gold / f"{int(str(i)):04d}_{row['note_id']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.txt").write_text(str(row["input"]), encoding="utf-8")
        (d / "prefill.json").write_text(
            json.dumps(json.loads(str(row["silver"])), indent=2), encoding="utf-8"
        )
        if int(str(i)) < overlap:
            (d / "OVERLAP").write_text("second annotator: write gold_b.json\n")
    return {"exported": len(sample), "overlap": min(overlap, len(sample))}


def _load(path: Path) -> DischargeExtraction | None:
    try:
        return DischargeExtraction.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValidationError, ValueError):
        return None


def check(work_dir: Path) -> dict[str, Any]:
    folders = sorted(p for p in (work_dir / "gold").iterdir() if p.is_dir())
    done = [f for f in folders if (f / "gold.json").exists()]
    invalid = [f for f in done if _load(f / "gold.json") is None]
    valid = [f for f in done if f not in invalid]
    silver_vs_gold = [
        (_load(f / "prefill.json"), g) for f in valid if (g := _load(f / "gold.json"))
    ]
    pairs_ab = [
        (b, a)
        for f in valid
        if (f / "gold_b.json").exists()
        and (a := _load(f / "gold.json"))
        and (b := _load(f / "gold_b.json"))
    ]
    out: dict[str, Any] = {
        "folders": len(folders),
        "annotated": len(done),
        "invalid": len(invalid),
        "double_annotated": len(pairs_ab),
    }
    if silver_vs_gold:
        out["silver_vs_gold"] = evaluate(silver_vs_gold).summary()
    if pairs_ab:
        out["annotator_b_vs_a"] = evaluate(pairs_ab).summary()
    return out
