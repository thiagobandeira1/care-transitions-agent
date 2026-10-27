"""MIMIC-IV-Note extraction baselines (M6). Runs locally only and prints aggregates only.

Runs zero-shot or few-shot extraction on the gold-set notes (MIMIC_WORK_DIR/gold/, written by
`gold-export`) and scores it against their silver labels (prefill.json) now, or against the gold
labels (gold.json) once annotated. Few-shot examples come from the train split only. Predictions
go to MIMIC_WORK_DIR/preds/, outside the repository; only aggregate metrics go to evals/results/.
Error messages can quote model output, so they stay in the local predictions file; results keep
only the error type. The rules baseline against gold is `gold-check`'s silver_vs_gold.
"""

import json
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


def saved_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """folder -> saved prediction row from a full run."""
    if not path.exists():
        raise ValueError(f"no saved predictions at {path.name}; run without --rescore first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {r["folder"]: r for r in rows}


def run(
    work_dir: Path,
    mode: str,
    model: str,
    ref: str = "silver",
    limit: int | None = None,
    k_shots: int = 3,
    seed: int = 5,
    llm: StructuredLLM | None = None,
    rescore: bool = False,
) -> dict[str, Any]:
    """mode: 'zero' or 'few'. A run with `limit` is a smoke test and is not recorded.

    `rescore` scores the predictions saved by the full silver-label run against the current labels
    without calling the model: after the silver labels change, or on the gold set once annotated.
    """
    if mode not in ("zero", "few"):
        raise ValueError("MIMIC modes are zero and few; the rules baseline is gold-check")
    notes = gold_notes(work_dir, ref)[:limit]
    preds_dir = assert_outside_repo(work_dir / "preds")
    tag = f"{model.replace(':', '-')}_{mode}"

    rows: list[dict[str, Any]] = []
    if rescore:
        saved = saved_predictions(preds_dir / f"silver_{tag}.jsonl")
        missing = [folder for folder, _, _ in notes if folder not in saved]
        if missing:
            raise ValueError(f"{len(missing)} notes have no saved prediction")
        rows = [saved[folder] for folder, _, _ in notes]
        n_shots = k_shots if mode == "few" else 0  # a full few-shot run checks it had k
    else:
        shots = pick_shots(work_dir, k_shots, seed) if mode == "few" else []
        if mode == "few" and len(shots) < k_shots:
            raise ValueError(f"found {len(shots)} few-shot examples, need {k_shots}")
        ext = Extractor(llm or OllamaLLM(model, num_ctx=16384, num_predict=NUM_PREDICT), shots)
        preds_dir.mkdir(parents=True, exist_ok=True)
        with (preds_dir / f"{ref}_{tag}.jsonl").open("w", encoding="utf-8") as fh:
            for i, (folder, text, _) in enumerate(notes, 1):
                r = ext.run(text)
                row = {"folder": folder, "attempts": r.attempts, "seconds": r.seconds}
                row |= {"pred": r.value.model_dump() if r.value else None, "error": r.error}
                fh.write(json.dumps(row) + "\n")
                rows.append(row)
                print(f"  {mode} {i}/{len(notes)}", flush=True)
        n_shots = len(shots)

    failures: dict[str, int] = {}
    pairs: list[tuple[DischargeExtraction | None, DischargeExtraction]] = []
    for row, (_, _, label) in zip(rows, notes, strict=True):
        if row["pred"] is None:
            kind = str(row["error"]).split(":", 1)[0]
            failures[kind] = failures.get(kind, 0) + 1
        pred = DischargeExtraction.model_validate(row["pred"]) if row["pred"] else None
        pairs.append((pred, label))

    result = {
        "benchmark": f"MIMIC-IV-Note gold-set notes, {ref} labels",
        "mode": mode,
        "model": model,
        "n": len(notes),
        "few_shot_examples": n_shots,
        "rescored": rescore,
        "schema_failures": sum(failures.values()),
        "failure_types": failures,
        "seconds_per_note": sum(r["seconds"] for r in rows) / max(len(rows), 1),
        **evaluate(pairs).summary(),
        "relaxed_f1_ci95": bootstrap(pairs) if pairs else {},
    }
    if limit is None:
        out = REPO / "evals" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"extraction_mimic_{ref}_{tag}.json").write_text(json.dumps(result, indent=2))
    return result
