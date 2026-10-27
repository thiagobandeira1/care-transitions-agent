"""MIMIC prep on tiny MIMIC-shaped files filled with fictional text (no MIMIC data)."""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import ctagent.mimic as mimic
from ctagent.mimic import MimicPaths, prepare, split_patients
from ctagent.privacy import PrivacyError
from test_notes import NOTE

SENTINEL = "FICTIONAL-SENTINEL-TEXT"


def write_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, index=False)


@pytest.fixture
def paths(tmp_path: Path) -> MimicPaths:
    note_dir, hosp_dir = tmp_path / "note", tmp_path / "hosp"
    notes = pd.DataFrame(
        {
            "note_id": [f"n{i}" for i in range(20)],
            "subject_id": [100 + i // 2 for i in range(20)],  # 10 patients, 2 notes each
            "hadm_id": [1000 + i for i in range(20)],
            "note_type": "DS",
            "note_seq": 1,
            "charttime": "",
            "storetime": "",
            "text": [NOTE + f"\n{SENTINEL}" for _ in range(20)],
        }
    )
    write_gz(note_dir / "discharge.csv.gz", notes)
    write_gz(
        hosp_dir / "diagnoses_icd.csv.gz",
        pd.DataFrame(
            {
                "subject_id": [100, 100, 101],
                "hadm_id": [1000, 1000, 1002],
                "seq_num": [1, 2, 1],
                "icd_code": ["I5023", "N183", "4280"],
                "icd_version": [10, 10, 9],
            }
        ),
    )
    return MimicPaths(note_dir, hosp_dir, tmp_path / "work")


def test_split_is_by_patient_and_reproducible() -> None:
    ids = pd.Series(list(range(100)) * 2)
    a, b = split_patients(ids, 13), split_patients(ids, 13)
    assert a == b and len(a) == 100
    assert sorted(set(a.values())) == ["test", "train", "validation"]


def test_prepare_writes_splits_and_reports_aggregates_only(
    paths: MimicPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[Path] = []
    real = mimic.coded_principal
    monkeypatch.setattr(mimic, "coded_principal", lambda d: reads.append(d) or real(d))
    stats = prepare(paths)
    assert len(reads) == 1  # the 6-million-row code file is read once, not once per note
    assert stats["notes"] == 20 and stats["patients"] == 10
    assert stats["section_presence"]["meds_discharge"] == 1.0
    assert stats["notes_with_coded_principal_dx"] == 0.1  # 2 of 20 admissions coded
    assert stats["notes_with_icd10_principal"] == 0.05
    assert stats["silver_principal_dx_from_note"] == 1.0
    assert stats["silver_secondary_dx_per_note"] == 1.0
    assert SENTINEL not in json.dumps(stats)  # no note text in what gets printed
    frames = {
        s: pd.read_parquet(paths.work_dir / f"{s}.parquet") for s in ("train", "validation", "test")
    }
    assert sum(len(f) for f in frames.values()) == 20
    subjects = [set(f["subject_id"]) for f in frames.values()]
    assert not (subjects[0] & subjects[1]) and not (subjects[0] & subjects[2])
    # diagnoses come from the note's wording, even where billing codes exist
    coded = json.loads(pd.concat(frames.values()).set_index("hadm_id").loc[1000, "silver"])
    assert coded["principal_diagnosis"] == {
        "text": "Acute on chronic systolic heart failure",
        "icd10": None,
    }


def test_from_env_refuses_paths_inside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MIMIC_NOTE_DIR", str(repo / "data" / "note"))
    monkeypatch.setenv("MIMIC_HOSP_DIR", str(repo.parent / "elsewhere"))
    with pytest.raises(PrivacyError):
        MimicPaths.from_env()


def test_from_env_defaults_work_dir_next_to_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MIMIC_NOTE_DIR", str(tmp_path / "note"))
    monkeypatch.setenv("MIMIC_HOSP_DIR", str(tmp_path / "hosp"))
    monkeypatch.delenv("MIMIC_WORK_DIR", raising=False)
    assert MimicPaths.from_env().work_dir == (tmp_path / "ctagent_work").resolve()
