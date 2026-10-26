"""Drafting benchmark on Synthea discharges: latency (GPU vs CPU), schema failures, grounding.

Runs the real graph up to human review with a local Ollama model. Synthea only: outputs are
synthetic and may be committed as aggregates (evals/results/) and a few samples
(synthetic/samples/).
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver

from .graph import Deps, build_graph, start_discharge
from .llm import drafting_llm
from .models import DischargeEvent, Disposition, PatientContext
from .synthea import SyntheaSource, build_cohort, patient_contexts
from .tracking import Tracker

REPO = Path(__file__).resolve().parents[2]


def to_event(row: "pd.Series[Any]") -> DischargeEvent:
    return DischargeEvent(
        patient_id=str(row["patient_id"]),
        episode_id=str(row["episode_id"]),
        admit=row["admit"],
        discharge=row["discharge"],
        age=int(row["age"]),
        disposition=Disposition(row["disposition"]),
        reason=str(row["reason"] or ""),
        acute=bool(row["acute"]),
        er_visits_6m=int(row["er_visits_6m"]),
        n_stays=int(row["n_stays"]),
    )


def sample_discharges(
    csv_dir: Path, n: int, seed: int
) -> tuple[list[DischargeEvent], dict[str, PatientContext]]:
    src = SyntheaSource(csv_dir)
    eps = build_cohort(src)
    el = eps[eps["eligible"]].sample(n=n, random_state=seed)
    return [to_event(r) for _, r in el.iterrows()], patient_contexts(src, el)


def _drive(
    graph: Any, events: list[DischargeEvent], ctx: dict[str, PatientContext], samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run each discharge to the review pause; return timing rows, kept samples, details."""
    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        t0 = time.perf_counter()
        pause = start_discharge(graph, ev).interrupts[0].value
        wall = time.perf_counter() - t0
        flags = pause["flags"]
        details.append(
            {
                "episode": i,
                "flags": flags,
                "record_meds": sorted({m.name for m in ctx[ev.episode_id].med_changes}),
                "call_script": pause["call_script"],
                "brief": pause["brief"],
            }
        )
        rows.append(
            {
                "seconds": wall,
                "tier": pause["tier"],
                "schema_fail": sum("failed after" in f for f in flags),
                "grounding": sum("not in record" in f for f in flags),
            }
        )
        if i < samples:
            kept.append(
                {
                    "episode": i,
                    **{
                        k: pause[k]
                        for k in (
                            "tier",
                            "lace",
                            "explanation",
                            "flags",
                            "call_script",
                            "brief",
                            "plan",
                        )
                    },
                }
            )
        print(f"  {i + 1}/{len(events)}: {wall:.1f}s, flags={len(flags)}", flush=True)
    return rows, kept, details


def run_bench(
    csv_dir: Path, model: str, n: int, cpu_only: bool, seed: int = 7, samples: int = 0
) -> dict[str, Any]:
    events, ctx = sample_discharges(csv_dir, n, seed)
    llm = drafting_llm(model, cpu_only=cpu_only)
    tracker = Tracker(":memory:")
    graph = build_graph(
        Deps(llm=llm, contexts=lambda e: ctx[e.episode_id], tracker=tracker), InMemorySaver()
    )
    try:
        rows, kept, details = _drive(graph, events, ctx, samples)
    finally:
        if cpu_only:
            llm.unload()
    secs = [r["seconds"] for r in rows]
    result = {
        "model": llm.name,
        "n": n,
        "seed": seed,
        "device": "cpu" if cpu_only else "gpu",
        "seconds_per_discharge": {
            "mean": statistics.mean(secs),
            "median": statistics.median(secs),
            "max": max(secs),
        },
        "llm_calls_per_discharge": 3,
        "discharges_with_schema_failure": sum(r["schema_fail"] > 0 for r in rows),
        "discharges_with_grounding_violation": sum(r["grounding"] > 0 for r in rows),
        "grounding_violations_total": sum(r["grounding"] for r in rows),
        "tiers": pd.Series([r["tier"] for r in rows]).value_counts().to_dict(),
        "morning_batch_minutes_for_15_discharges": statistics.mean(secs) * 15 / 60,
    }
    out = REPO / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{model.replace(':', '-')}_{result['device']}"
    (out / f"drafting_{tag}.json").write_text(json.dumps(result, indent=2))
    # every draft with its record's medications, for reviewing each flag (synthetic; gitignored)
    (REPO / "outputs").mkdir(exist_ok=True)
    (REPO / "outputs" / f"bench_details_{tag}.json").write_text(
        json.dumps(details, indent=1, default=str)
    )
    if kept:
        sample_dir = REPO / "synthetic" / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / f"drafts_{tag}.json").write_text(json.dumps(kept, indent=2, default=str))
    return result
