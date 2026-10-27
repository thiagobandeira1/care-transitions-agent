"""MIMIC-IV-Note preparation for the extraction model (M5). Runs locally only.

Reads discharge summaries and hospital diagnosis codes from directories OUTSIDE the repository,
splits by patient, builds extraction inputs and silver labels, and writes them to a local work
directory, also outside the repository. It prints aggregates only (counts, rates, length
quantiles) and never prints note text, header text, or identifiers. Claude reads only that output.

Silver diagnoses come from the note's Discharge Diagnosis section, the convention the gold set
follows (docs/ANNOTATION.md). Billing codes are only counted: their titles are not the note's
wording, so a model trained on them would learn to write billing titles.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .notes import SECTIONS, extraction_input, silver_labels, split_sections
from .privacy import assert_outside_repo

SPLITS = (("train", 0.7), ("validation", 0.1), ("test", 0.2))


@dataclass(frozen=True)
class MimicPaths:
    note_dir: Path
    hosp_dir: Path
    work_dir: Path

    @classmethod
    def from_env(cls) -> "MimicPaths":
        note = Path(os.environ["MIMIC_NOTE_DIR"])
        hosp = Path(os.environ["MIMIC_HOSP_DIR"])
        work = Path(os.environ.get("MIMIC_WORK_DIR", note.parent / "ctagent_work"))
        return cls(assert_outside_repo(note), assert_outside_repo(hosp), assert_outside_repo(work))


def split_patients(subject_ids: "pd.Series[Any]", seed: int) -> dict[Any, str]:
    ids = np.array(sorted(subject_ids.unique()))
    np.random.default_rng(seed).shuffle(ids)
    cuts = np.cumsum([round(frac * len(ids)) for _, frac in SPLITS])[:-1]
    out: dict[Any, str] = {}
    for (name, _), part in zip(SPLITS, np.split(ids, cuts), strict=True):
        out.update(dict.fromkeys(part.tolist(), name))
    return out


def coded_principal(hosp_dir: Path) -> dict[int, int]:
    """hadm_id -> ICD version (9 or 10) of the admission's first billing code (for the stats)."""
    dx = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_version"]
    )
    first = dx.sort_values(["hadm_id", "seq_num"]).groupby("hadm_id")["icd_version"].first()
    return {int(str(h)): int(v) for h, v in first.items()}


def prepare(paths: MimicPaths, seed: int = 13, limit: int | None = None) -> dict[str, Any]:
    notes = pd.read_csv(
        paths.note_dir / "discharge.csv.gz",
        usecols=["note_id", "subject_id", "hadm_id", "text"],
        nrows=limit,
    )
    notes["split"] = notes["subject_id"].map(split_patients(notes["subject_id"], seed))
    versions = coded_principal(paths.hosp_dir)
    coded = [versions.get(int(str(h))) for h in notes["hadm_id"]]

    rows, section_hits = [], dict.fromkeys(SECTIONS, 0)
    lengths, trimmed = [], 0
    with_principal, secondary = 0, 0
    for note_id, split, text in zip(notes["note_id"], notes["split"], notes["text"], strict=True):
        sec = split_sections(text)
        for name in sec:
            section_hits[name] += 1
        inp, cut = extraction_input(text)
        trimmed += cut
        lengths.append(len(inp))
        label = silver_labels(text)
        with_principal += label.principal_diagnosis is not None
        secondary += len(label.secondary_diagnoses)
        rows.append(
            {"note_id": note_id, "split": split, "input": inp, "silver": label.model_dump_json()}
        )

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).merge(notes[["note_id", "subject_id", "hadm_id"]], on="note_id")
    for name, _ in SPLITS:
        out[out["split"] == name].to_parquet(paths.work_dir / f"{name}.parquet", index=False)

    n = len(notes)
    lens = pd.Series(lengths) / 4  # ~4 characters per token
    stats = {
        "notes": n,
        "patients": int(notes["subject_id"].nunique()),
        "by_split": notes["split"].value_counts().to_dict(),
        "patients_by_split": notes.groupby("split")["subject_id"].nunique().to_dict(),
        "section_presence": {k: round(v / max(n, 1), 3) for k, v in section_hits.items()},
        "input_tokens_approx": lens.quantile([0.5, 0.9, 0.99]).round().to_dict(),
        "inputs_trimmed": round(trimmed / max(n, 1), 3),
        "notes_with_coded_principal_dx": round(float(np.mean([v is not None for v in coded])), 3),
        "notes_with_icd10_principal": round(float(np.mean([v == 10 for v in coded])), 3),
        "silver_principal_dx_from_note": round(with_principal / max(n, 1), 3),
        "silver_secondary_dx_per_note": round(secondary / max(n, 1), 2),
        "silver_med_changes_per_note": round(
            float(np.mean([len(json.loads(r["silver"])["medication_changes"]) for r in rows])), 2
        )
        if rows
        else 0.0,
    }
    (paths.work_dir / "prep_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats
