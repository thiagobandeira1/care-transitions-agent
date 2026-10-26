"""MIMIC-IV-Note preparation for the extraction model (M5). Runs locally only.

Reads discharge summaries and hospital diagnosis codes from directories OUTSIDE the repository,
splits by patient, builds extraction inputs and silver labels, and writes them to a local work
directory, also outside the repository. It prints aggregates only (counts, rates, length
quantiles) and never prints note text, header text, or identifiers. Claude reads only that output.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extraction import Diagnosis
from .notes import SECTIONS, extraction_input, silver_labels, split_sections
from .privacy import assert_outside_repo

SPLITS = (("train", 0.7), ("validation", 0.1), ("test", 0.2))
MAX_SECONDARY = 8


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


def diagnoses(hosp_dir: Path) -> pd.DataFrame:
    dx = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_code", "icd_version"]
    )
    titles = pd.read_csv(
        hosp_dir / "d_icd_diagnoses.csv.gz", usecols=["icd_code", "icd_version", "long_title"]
    )
    return dx.merge(titles, on=["icd_code", "icd_version"], how="left").sort_values(
        ["hadm_id", "seq_num"]
    )


def _dx_lists(dx: pd.DataFrame) -> dict[int, tuple[Diagnosis | None, list[Diagnosis]]]:
    out: dict[int, tuple[Diagnosis | None, list[Diagnosis]]] = {}
    for hadm, g in dx.groupby("hadm_id"):
        items = [
            Diagnosis(text=str(t), icd10=str(c) if v == 10 else None)
            for t, c, v in zip(
                g["long_title"].fillna(""), g["icd_code"], g["icd_version"], strict=True
            )
        ]
        out[int(str(hadm))] = (items[0] if items else None, items[1 : 1 + MAX_SECONDARY])
    return out


def _principal(
    dx: dict[int, tuple[Diagnosis | None, list[Diagnosis]]], hadm: object
) -> Diagnosis | None:
    return dx.get(int(str(hadm)), (None, []))[0]


def prepare(paths: MimicPaths, seed: int = 13, limit: int | None = None) -> dict[str, Any]:
    notes = pd.read_csv(
        paths.note_dir / "discharge.csv.gz",
        usecols=["note_id", "subject_id", "hadm_id", "text"],
        nrows=limit,
    )
    notes["split"] = notes["subject_id"].map(split_patients(notes["subject_id"], seed))
    dx = _dx_lists(diagnoses(paths.hosp_dir))

    rows, section_hits = [], dict.fromkeys(SECTIONS, 0)
    lengths, trimmed = [], 0
    for note_id, hadm, split, text in zip(
        notes["note_id"], notes["hadm_id"], notes["split"], notes["text"], strict=True
    ):
        sec = split_sections(text)
        for name in sec:
            section_hits[name] += 1
        inp, cut = extraction_input(text)
        trimmed += cut
        lengths.append(len(inp))
        principal, secondary = dx.get(int(hadm), (None, []))
        label = silver_labels(text, principal, secondary)
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
        "notes_with_coded_principal_dx": round(
            float(np.mean([_principal(dx, h) is not None for h in notes["hadm_id"]])), 3
        ),
        "notes_with_icd10_principal": round(
            float(np.mean([bool((p := _principal(dx, h)) and p.icd10) for h in notes["hadm_id"]])),
            3,
        ),
        "silver_med_changes_per_note": round(
            float(np.mean([len(json.loads(r["silver"])["medication_changes"]) for r in rows])), 2
        )
        if rows
        else 0.0,
    }
    (paths.work_dir / "prep_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats
