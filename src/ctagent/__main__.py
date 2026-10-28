"""Command line. Prints aggregates only.

python -m ctagent cohort                        # Synthea cohort, LACE distribution, tiers
python -m ctagent bench --model qwen3:14b [--cpu]  # drafting latency + grounding (Synthea)
python -m ctagent daily --date 2024-03-22          # check-ins due today (tracking DB)
python -m ctagent make-benchmark --n 150           # synthetic notes with known answers
python -m ctagent extract-bench --mode zero|few|rules [--model qwen3:8b] [--rescore]
python -m ctagent mimic-prep                       # local only; prints aggregates
python -m ctagent gold-export --n 200 --overlap 50 / gold-check   # local annotation
python -m ctagent mimic-bench --mode zero|few --model qwen3:8b [--ref gold] [--limit 5] [--rescore]
python -m ctagent demo --n 10 [--workspace data/demo]   # morning batch for the review screen
"""

import argparse
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .metrics import auroc
from .models import DischargeEvent, Disposition
from .rules import plan_follow_up
from .scoring import score_discharge
from .synthea import SyntheaSource, build_cohort, patient_contexts
from .tracking import Tracker

DEFAULT_SYNTHEA = Path(__file__).resolve().parents[3] / "_synthea" / "output" / "csv"


def to_event(row: pd.Series) -> DischargeEvent:
    return DischargeEvent(
        patient_id=row["patient_id"],
        episode_id=row["episode_id"],
        admit=row["admit"],
        discharge=row["discharge"],
        age=int(row["age"]),
        disposition=Disposition(row["disposition"]),
        reason=str(row["reason"] or ""),
        acute=bool(row["acute"]),
        er_visits_6m=int(row["er_visits_6m"]),
        n_stays=int(row["n_stays"]),
    )


def cmd_cohort(csv_dir: Path) -> None:
    src = SyntheaSource(csv_dir)
    eps = build_cohort(src)
    el = eps[eps["eligible"]]
    ctx = patient_contexts(src, el)
    scores: list[int] = []
    tiers: Counter[str] = Counter()
    for _, row in el.iterrows():
        ev = to_event(row)
        s = score_discharge(ev, [c.code for c in ctx[ev.episode_id].conditions])
        scores.append(s.lace)
        tiers[plan_follow_up(ev.discharge.date(), s, ev.disposition).tier.value] += 1
    lace = pd.Series(scores, index=el.index)
    n_pat = el["patient_id"].nunique()
    print(f"episodes {len(eps)} | eligible index stays {len(el)} | patients {n_pat}")
    print("exclusions:", eps.loc[~eps["eligible"], "exclusion"].value_counts().to_dict())
    print(f"30-day readmissions {int(el['readmit30'].sum())} ({el['readmit30'].mean():.1%})")
    print("LACE quantiles:", lace.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
    print("tiers:", dict(tiers))
    auc = auroc(el["readmit30"].to_numpy(dtype=int), lace.to_numpy(dtype=float))
    print(f"LACE AUROC (synthetic sanity check, not clinical evidence): {auc:.3f}")


def cmd_daily(db: Path, today: date) -> None:
    due = Tracker(db).due_check_ins(today)
    print(f"check-ins due on or before {today}: {len(due)}")
    for episode_id, due_date in due:
        print(f"  {episode_id} (due {due_date})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ctagent")
    p.add_argument(
        "command",
        choices=[
            "cohort",
            "bench",
            "daily",
            "make-benchmark",
            "extract-bench",
            "mimic-prep",
            "gold-export",
            "gold-check",
            "mimic-bench",
            "demo",
        ],
    )
    p.add_argument(
        "--synthea", type=Path, default=Path(os.environ.get("SYNTHEA_CSV_DIR", DEFAULT_SYNTHEA))
    )
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--cpu", action="store_true", help="force CPU-only inference (num_gpu=0)")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--samples", type=int, default=0, help="save this many drafts as samples")
    p.add_argument("--db", type=Path, default=Path("data/tracking.sqlite"))
    p.add_argument("--mode", choices=["zero", "few", "rules"], default="zero")
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--ref", choices=["silver", "gold"], default="silver")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N notes, not saved")
    p.add_argument(
        "--rescore", action="store_true", help="score saved predictions again (no model calls)"
    )
    p.add_argument("--workspace", type=Path, default=Path("data/demo"))
    p.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = p.parse_args(argv)
    load_dotenv()
    if args.command == "cohort":
        cmd_cohort(args.synthea)
    elif args.command == "bench":
        from .bench import run_bench

        print(
            json.dumps(
                run_bench(args.synthea, args.model, args.n, args.cpu, samples=args.samples),
                indent=2,
            )
        )
    elif args.command == "make-benchmark":
        from .benchmark import build

        print(json.dumps(build(args.synthea, args.n, args.model), indent=2))
    elif args.command == "mimic-prep":
        from .mimic import MimicPaths, prepare

        print(json.dumps(prepare(MimicPaths.from_env()), indent=2, default=str))
    elif args.command == "demo":
        from .bench import sample_discharges
        from .llm import drafting_llm
        from .review import Workspace, open_graph, pending, process

        events, ctx = sample_discharges(args.synthea, args.n, seed=29)
        graph, tracker = open_graph(
            Workspace(args.workspace), drafting_llm(args.model), lambda e: ctx[e.episode_id]
        )
        done = process(graph, events)
        print(f"processed {done} discharges; pending review: {len(pending(tracker))}")
    elif args.command in ("gold-export", "gold-check"):
        from .gold import check, export
        from .mimic import MimicPaths

        work = MimicPaths.from_env().work_dir
        res = export(work, args.n, args.overlap) if args.command == "gold-export" else check(work)
        print(json.dumps(res, indent=2, default=str))
    elif args.command == "mimic-bench":
        from .mimic import MimicPaths
        from .mimic_bench import run

        work = MimicPaths.from_env().work_dir
        res = run(work, args.mode, args.model, args.ref, args.limit, rescore=args.rescore)
        print(json.dumps(res, indent=2))
    elif args.command == "extract-bench":
        from .benchmark import score

        model = None if args.mode == "rules" else args.model
        print(json.dumps(score(args.mode, model, rescore=args.rescore), indent=2))
    else:
        cmd_daily(args.db, args.date)


if __name__ == "__main__":
    main()
