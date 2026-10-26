"""MIMIC-IV-Note extraction baselines (M6). Runs locally only and prints aggregates only.

Runs zero-shot or few-shot extraction on the gold-set notes (MIMIC_WORK_DIR/gold/, written by
`gold-export`) and scores it against their silver labels (prefill.json) now, or against the gold
labels (gold.json) once annotated. Few-shot examples come from the train split only. Predictions
go to MIMIC_WORK_DIR/preds/, outside the repository; only aggregate metrics go to evals/results/.
Error messages can quote model output, so they stay in the local predictions file; results keep
only the error type. The rules baseline against gold is `gold-check`'s silver_vs_gold.
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .bench import REPO
from .benchmark import bootstrap
from .extract_eval import evaluate
from .extraction import DischargeExtraction
from .extractor import Example, Extractor
from .llm import OllamaLLM, StructuredLLM
from .privacy import assert_outside_repo

# few-shot examples: short train notes (input characters) whose silver labels fill every field
SHOT_CHARS = (1500, 4000)
SHOT_POOL = 5000
# MIMIC answers list ~11 medications; 1,500 tokens (the drafting cap) can cut them off
NUM_PREDICT = 3000


def gold_notes(work_dir: Path, ref: str) -> list[tuple[str, str, DischargeExtraction]]:
    """(folder, model input, reference) per gold-set note; `ref` is 'silver' or 'gold'."""
    out = []
    for d in sorted(p for p in (work_dir / "gold").iterdir() if p.is_dir()):
        path = d / ("prefill.json" if ref == "silver" else "gold.json")
        if not path.exists():
            continue
        try:
            label = DischargeExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # an invalid annotation; gold-check reports it
        out.append((d.name, (d / "input.txt").read_text(encoding="utf-8"), label))
    return out


def pick_shots(work_dir: Path, k: int, seed: int) -> list[Example]:
    """k short train notes with every core field labeled, one with pending tests if possible.

    Follow-up is not required: MIMIC redacts it, so no silver label has one.
    """
    df = pd.read_parquet(work_dir / "train.parquet", columns=["input", "silver"])
    df = df.sample(n=min(SHOT_POOL, len(df)), random_state=seed)
    df = df[df["input"].str.len().between(*SHOT_CHARS)]
    cands = []
    for text, silver in zip(df["input"], df["silver"], strict=True):
        label = DischargeExtraction.model_validate_json(str(silver))
        core = (label.secondary_diagnoses, label.medication_changes, label.red_flags)
        if label.principal_diagnosis and all(core):
            cands.append(Example(str(text), label))
    with_pending = [ex for ex in cands if ex.label.pending_tests]
    return (with_pending[:1] + [ex for ex in cands if not ex.label.pending_tests])[:k]


def run(
    work_dir: Path,
    mode: str,
    model: str,
    ref: str = "silver",
    limit: int | None = None,
    k_shots: int = 3,
    seed: int = 5,
    llm: StructuredLLM | None = None,
) -> dict[str, Any]:
    """mode: 'zero' or 'few'. A run with `limit` is a smoke test and is not recorded."""
    if mode not in ("zero", "few"):
        raise ValueError("MIMIC modes are zero and few; the rules baseline is gold-check")
    notes = gold_notes(work_dir, ref)[:limit]
    shots = pick_shots(work_dir, k_shots, seed) if mode == "few" else []
    if mode == "few" and len(shots) < k_shots:
        raise ValueError(f"found {len(shots)} few-shot examples, need {k_shots}")
    ext = Extractor(llm or OllamaLLM(model, num_ctx=16384, num_predict=NUM_PREDICT), shots)
    preds_dir = assert_outside_repo(work_dir / "preds")
    preds_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model.replace(':', '-')}_{mode}"

    t0 = time.perf_counter()
    failures: dict[str, int] = {}
    pairs: list[tuple[DischargeExtraction | None, DischargeExtraction]] = []
    with (preds_dir / f"{ref}_{tag}.jsonl").open("w", encoding="utf-8") as fh:
        for i, (folder, text, label) in enumerate(notes, 1):
            r = ext.run(text)
            if r.value is None:
                kind = r.error.split(":", 1)[0]
                failures[kind] = failures.get(kind, 0) + 1
            row = {"folder": folder, "attempts": r.attempts, "seconds": r.seconds}
            row |= {"pred": r.value.model_dump() if r.value else None, "error": r.error}
            fh.write(json.dumps(row) + "\n")
            pairs.append((r.value, label))
            print(f"  {mode} {i}/{len(notes)}", flush=True)

    result = {
        "benchmark": f"MIMIC-IV-Note gold-set notes, {ref} labels",
        "mode": mode,
        "model": model,
        "n": len(notes),
        "few_shot_examples": len(shots),
        "schema_failures": sum(failures.values()),
        "failure_types": failures,
        "seconds_per_note": (time.perf_counter() - t0) / max(len(notes), 1),
        **evaluate(pairs).summary(),
        "relaxed_f1_ci95": bootstrap(pairs) if pairs else {},
    }
    if limit is None:
        out = REPO / "evals" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"extraction_mimic_{ref}_{tag}.json").write_text(json.dumps(result, indent=2))
    return result
