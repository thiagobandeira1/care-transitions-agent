"""Teacher labels for the fields the silver rules miss (option C, ADR 0006). Runs locally only.

The first fine-tune copied silver's gaps: MIMIC redacts the Followup Instructions section, and the
pending-test and red-flag rules are narrow. Here a larger model with the M6 few-shot prompt
labels follow-up, pending tests and red flags; diagnoses and medication changes stay silver.
Labels go outside the repository, one JSON line per note, appended as each note finishes, so a
stopped run resumes where it left off. Error messages can quote model output, so they stay in
that local file. Only counts and timings are printed or returned.
"""

import json
import statistics as st
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .extraction import DischargeExtraction
from .extractor import Extractor

WEAK_FIELDS = ("follow_up", "pending_tests", "red_flags")


def merge(silver: DischargeExtraction, weak: dict[str, Any]) -> DischargeExtraction:
    """Silver diagnoses and medication changes, teacher follow-up, pending tests and red flags."""
    return DischargeExtraction.model_validate(
        silver.model_dump(mode="json") | {k: weak[k] for k in WEAK_FIELDS}
    )


def rows(path: Path) -> list[dict[str, Any]]:
    """Complete rows only: a last line without a newline was cut off by a stop."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
    return [json.loads(line) for line in lines]


def load(path: Path) -> dict[str, dict[str, Any]]:
    """note_id -> teacher fields, in labeling order. Notes the teacher failed are left out."""
    return {r["note_id"]: r["weak"] for r in rows(path) if r["weak"] is not None}


def label(notes: Iterable[tuple[str, str]], ext: Extractor, path: Path, n: int) -> dict[str, Any]:
    """Label (note_id, text) pairs in order until n notes have teacher labels.

    Notes already in `path` are skipped, so rerunning the same command resumes a stopped run.
    """
    kept = rows(path)
    if path.exists():  # drop a cut-off last line before appending after it
        path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    done = {r["note_id"] for r in kept}
    ok = sum(r["weak"] is not None for r in kept)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for note_id, text in notes:
            if ok >= n:
                break
            if note_id in done:
                continue
            r = ext.run(text)
            weak = r.value.model_dump(mode="json", include=set(WEAK_FIELDS)) if r.value else None
            row = {"note_id": note_id, "attempts": r.attempts, "seconds": r.seconds}
            fh.write(json.dumps(row | {"weak": weak, "error": r.error}) + "\n")
            fh.flush()
            ok += weak is not None
            print(f"  {path.stem} {ok}/{n}", flush=True)
    return summary(path)


def summary(path: Path) -> dict[str, Any]:
    """Counts and timings only: how many notes, how fast, and how often each field is filled."""
    all_rows = rows(path)
    good = [r["weak"] for r in all_rows if r["weak"] is not None]
    counts = {f: [len(w[f]) for w in good] for f in WEAK_FIELDS}
    return {
        "notes": len(all_rows),
        "labeled": len(good),
        "failures": len(all_rows) - len(good),
        "seconds_per_note": st.mean(r["seconds"] for r in all_rows) if all_rows else 0.0,
        "share_with_items": {f: sum(c > 0 for c in v) / max(len(v), 1) for f, v in counts.items()},
        "mean_items": {f: st.mean(v) if v else 0.0 for f, v in counts.items()},
    }
