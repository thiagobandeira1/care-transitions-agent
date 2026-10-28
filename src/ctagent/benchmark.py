"""Public synthetic extraction benchmark: build it (local model writes notes) and score extractors.

Outputs are synthetic and committable: notes + references in synthetic/samples/benchmark/, metrics
in evals/results/. Splits are by Synthea patient; few-shot examples come only from `train`.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bench import REPO, sample_discharges
from .extract_eval import FIELDS, Report, evaluate
from .extraction import DischargeExtraction
from .extractor import Example, Extractor
from .llm import OllamaLLM
from .notegen import generate_note, reference
from .notes import silver_labels
from .rules import plan_follow_up
from .scoring import score_discharge

BENCH_DIR = REPO / "synthetic" / "samples" / "benchmark"


def build(
    csv_dir: Path, n: int, model: str, seed: int = 11, train_frac: float = 0.3
) -> dict[str, object]:
    events, ctx = sample_discharges(csv_dir, n, seed)
    rng = random.Random(seed)
    llm = OllamaLLM(model, temperature=0.7, num_ctx=8192)
    patients = sorted({e.patient_id for e in events})
    rng.shuffle(patients)
    train_pts = set(patients[: int(len(patients) * train_frac)])
    items, dropped = [], 0
    for i, ev in enumerate(events):
        c = ctx[ev.episode_id]
        risk = score_discharge(ev, [x.code for x in c.conditions])
        high = plan_follow_up(ev.discharge.date(), risk, ev.disposition).tier.value == "high"
        ref = reference(ev, c, high, rng)
        note = generate_note(llm, ev, ref)
        if note is None:
            dropped += 1
            continue
        items.append(
            {
                "id": f"syn-{i:04d}",
                "patient": f"pt-{patients.index(ev.patient_id):04d}",
                "split": "train" if ev.patient_id in train_pts else "test",
                "note": note,
                "reference": ref.model_dump(mode="json"),
            }
        )
        print(f"  note {i + 1}/{n}", flush=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "notes.json").write_text(json.dumps(items, indent=1))
    meta = {
        "generator": model,
        "seed": seed,
        "requested": n,
        "kept": len(items),
        "dropped": dropped,
        "train": sum(x["split"] == "train" for x in items),
        "test": sum(x["split"] == "test" for x in items),
    }
    (BENCH_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(split: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = json.loads((BENCH_DIR / "notes.json").read_text())
    return [x for x in items if x["split"] == split]


def bootstrap(
    pairs: list[tuple[DischargeExtraction | None, DischargeExtraction]],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs for relaxed micro-F1 and per-field relaxed F1 (resampling notes)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        rep = evaluate([pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]).summary()
        relaxed: dict[str, dict[str, float]] = rep["relaxed"]  # type: ignore[assignment]
        for k in (*FIELDS, "micro"):
            draws.setdefault(k, []).append(relaxed[k]["f1"])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()
    }


def predict(
    mode: str, model: str | None, test: list[dict[str, Any]], k_shots: int, seed: int
) -> list[dict[str, Any]]:
    """One row per test note: id, seconds and the prediction (None on a schema failure)."""
    ext: Extractor | None = None
    if mode != "rules":
        assert model is not None
        shots: list[Example] = []
        if mode == "few":
            pool = load("train")
            pick = random.Random(seed).sample(pool, k=min(k_shots, len(pool)))
            shots = [
                Example(x["note"], DischargeExtraction.model_validate(x["reference"])) for x in pick
            ]
        ext = Extractor(OllamaLLM(model, num_ctx=16384), shots)
    rows = []
    for i, x in enumerate(test):
        t0 = time.perf_counter()
        pred = silver_labels(x["note"]) if ext is None else ext.run(x["note"]).value
        seconds = time.perf_counter() - t0
        dump = pred.model_dump(mode="json") if pred else None
        rows.append({"id": x["id"], "seconds": seconds, "pred": dump})
        if ext is not None:
            print(f"  {mode} {i + 1}/{len(test)}", flush=True)
    return rows


def score(
    mode: str, model: str | None = None, k_shots: int = 3, seed: int = 5, rescore: bool = False
) -> dict[str, Any]:
    """mode: 'rules' (silver-label rules), 'zero' or 'few' (local model).

    Predictions are saved under outputs/preds/ (not committed). `rescore` scores them again
    without calling the model, e.g. after a change to the schema's validation.
    """
    test = load("test")
    refs = [DischargeExtraction.model_validate(x["reference"]) for x in test]
    tag = mode if model is None else f"{model.replace(':', '-')}_{mode}"
    saved = REPO / "outputs" / "preds" / f"synthetic_{tag}.jsonl"
    if rescore:
        if not saved.exists():
            raise ValueError(f"no saved predictions at {saved.name}; run without --rescore first")
        by_id = {r["id"]: r for r in map(json.loads, saved.read_text().splitlines())}
        rows = [by_id[x["id"]] for x in test]
    else:
        rows = predict(mode, model, test, k_shots, seed)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text("".join(json.dumps(r) + "\n" for r in rows))
    preds = [DischargeExtraction.model_validate(r["pred"]) if r["pred"] else None for r in rows]
    pairs = list(zip(preds, refs, strict=True))
    rep: Report = evaluate(pairs)
    result = {
        "benchmark": "synthetic (Synthea-derived)",
        "mode": mode,
        "model": model,
        "n_test": len(test),
        "rescored": rescore,
        "schema_failures": sum(p is None for p in preds),
        "seconds_per_note": sum(r["seconds"] for r in rows) / max(len(rows), 1),
        **rep.summary(),
        "relaxed_f1_ci95": bootstrap(pairs),
    }
    out = REPO / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"extraction_synthetic_{tag}.json").write_text(json.dumps(result, indent=2))
    return result
