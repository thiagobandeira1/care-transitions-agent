"""The agent's MIMIC drafting runner on fictional hosp tables and a fictional note."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import ctagent.mimic_agent as ma
from ctagent.drafts import CallScript, MedLine, PlanExplanation, ProviderBrief
from ctagent.extraction import DischargeExtraction
from ctagent.gold import export
from ctagent.llm import FakeLLM
from ctagent.models import Disposition, MedChangeType
from ctagent.notes import extraction_input
from test_notes import NOTE

EXT = DischargeExtraction.model_validate(
    {
        "principal_diagnosis": {"text": "Acute on chronic systolic heart failure"},
        "secondary_diagnoses": [{"text": "Chronic kidney disease"}],
        "medication_changes": [
            {"name": "furosemide", "change": "dose_changed"},
            {"name": "aspirin", "change": "stopped"},
            {"name": "spironolactone", "change": "started"},
            {"name": "metoprolol succinate xl", "change": "continued"},
        ],
        "red_flags": ["shortness of breath at rest", "chest pain"],
    }
)


def _line(name: str, change: MedChangeType) -> MedLine:
    return MedLine(name=name, change=change)


def drafts() -> FakeLLM:
    return FakeLLM(
        {
            "PlanExplanation": lambda p: PlanExplanation(summary="Standard tier: visit in a week."),
            # furosemide is right; warfarin is in neither the note nor the record
            "CallScript": lambda p: CallScript(
                opening="Hello",
                questions=["How are you feeling?"],
                medication_review=[
                    _line("furosemide", MedChangeType.DOSE_CHANGED),
                    _line("warfarin", MedChangeType.STARTED),
                ],
                warning_signs=["Call us if you have chest pain"],
                scheduling="Visit within 7 days",
            ),
            # aspirin has the wrong change type; spironolactone is missing; zebra fever is invented
            "ProviderBrief": lambda p: ProviderBrief(
                reason_for_admission="Heart failure",
                key_conditions=["Heart failure", "Zebra fever"],
                medication_changes=[
                    _line("furosemide", MedChangeType.DOSE_CHANGED),
                    _line("aspirin", MedChangeType.STARTED),
                ],
                watch_for=["Weight"],
                risk_summary="Moderate risk.",
            ),
        }
    )


def _gz(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression="gzip")


def write_hosp(hosp: Path) -> None:
    """Fictional hosp tables. Index stays 101, 201, 301 and 401; the rest is history or outcome."""
    hosp.mkdir()
    adm = [
        # subject 1: the index stay via the ED, one ED stay 2 months before, one 14 months before,
        # and an unplanned readmission 5 days after discharge
        (1, 101, "2150-03-04 08:00", "2150-03-12 14:00", "EW EMER.", "HOME", "2150-03-04 06:00"),
        (1, 100, "2150-01-10 09:00", "2150-01-12 10:00", "EW EMER.", "HOME", "2150-01-10 07:00"),
        (1, 99, "2149-01-01 09:00", "2149-01-03 10:00", "EW EMER.", "HOME", "2149-01-01 07:00"),
        (1, 102, "2150-03-17 09:00", "2150-03-19 10:00", "EW EMER.", "HOME", "2150-03-17 07:00"),
        (2, 201, "2160-05-01 08:00", "2160-05-03 08:00", "EW EMER.", "DIED", "2160-05-01 06:00"),
        # subject 3: a planned admission 10 days later is not a readmission
        (3, 301, "2170-06-01 08:00", "2170-06-04 08:00", "ELECTIVE", "HOME HEALTH CARE", None),
        (3, 302, "2170-06-14 08:00", "2170-06-15 08:00", "ELECTIVE", "HOME", None),
        # subject 4: back 40 days later (outside the window); died 15 days after discharge
        (4, 401, "2180-07-01 08:00", "2180-07-05 08:00", "URGENT", "REHAB", None),
        (4, 402, "2180-08-14 08:00", "2180-08-15 08:00", "URGENT", "HOME", None),
    ]
    cols = ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]
    frame = pd.DataFrame(adm, columns=[*cols, "discharge_location", "edregtime"])
    frame["insurance"] = frame["subject_id"].map({1: "Medicare", 2: "Medicare", 3: "Medicaid"})
    frame["insurance"] = frame["insurance"].fillna("Medicare")
    frame["hospital_expire_flag"] = (frame["discharge_location"] == "DIED").astype(int)
    _gz(frame, hosp / "admissions.csv.gz")
    _gz(
        pd.DataFrame(
            [
                (1, 70, 2150, None),
                (2, 80, 2160, "2160-05-03"),
                (3, 50, 2168, None),
                (4, 60, 2180, "2180-07-20"),
            ],
            columns=["subject_id", "anchor_age", "anchor_year", "dod"],
        ),
        hosp / "patients.csv.gz",
    )
    _gz(
        pd.DataFrame(
            [(101, 1, "I509", 10), (101, 2, "4019", 9), (301, 1, "I509", 10), (401, 1, "N184", 10)],
            columns=["hadm_id", "seq_num", "icd_code", "icd_version"],
        ),
        hosp / "diagnoses_icd.csv.gz",
    )
    _gz(
        pd.DataFrame(
            [
                ("I509", 10, "Heart failure, unspecified"),
                ("4019", 9, "Unspecified essential hypertension"),
                ("N184", 10, "Chronic kidney disease, stage 4 (severe)"),
            ],
            columns=["icd_code", "icd_version", "long_title"],
        ),
        hosp / "d_icd_diagnoses.csv.gz",
    )


def write_work(work: Path) -> None:
    """A gold-set work dir over the four index stays, with saved extractions (one missing)."""
    work.mkdir()
    text, _ = extraction_input(NOTE)
    pd.DataFrame(
        {
            "note_id": ["n0", "n1", "n2", "n3"],
            "hadm_id": [101, 201, 301, 401],
            "subject_id": [1, 2, 3, 4],
            "input": text,
            "silver": EXT.model_dump_json(),
        }
    ).to_parquet(work / "test.parquet")
    export(work, n=4, overlap=1)
    (work / "preds").mkdir()
    rows = [
        {"folder": d.name, "pred": None if d.name.endswith("_n2") else EXT.model_dump()}
        for d in sorted((work / "gold").iterdir())
    ]
    (work / "preds" / "silver_qwen3-14b_zero.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    work, hosp = tmp_path / "work", tmp_path / "hosp"
    write_hosp(hosp)
    write_work(work)
    monkeypatch.setattr(ma, "REPO", tmp_path / "repo")
    return work, hosp


def test_dispositions_and_events(dirs: tuple[Path, Path]) -> None:
    assert ma.disposition(float("nan")) is Disposition.HOME
    assert ma.disposition("REHAB") is Disposition.SNF
    assert ma.disposition("DIED") is Disposition.EXPIRED
    _, hosp = dirs
    evs = ma.events(hosp, {"0000_a": 101, "0001_b": 301})
    first, elective = evs["0000_a"], evs["0001_b"]
    assert first.admit == datetime(2150, 3, 4, 8) and first.los_days == 8
    assert (first.age, first.acute, first.er_visits_6m) == (
        70,
        True,
        1,
    )  # 14 months back: not counted
    assert (elective.age, elective.acute, elective.disposition) == (
        52,
        False,
        Disposition.HOME_HEALTH,
    )
    assert first.episode_id == "e0000"  # no MIMIC identifier leaves this module


def test_drafting_counts(dirs: tuple[Path, Path]) -> None:
    work, hosp = dirs
    res = ma.run(work, hosp, "qwen3:14b", llm=drafts())
    assert (res["n_notes"], res["n_drafted"], res["with_extraction"]) == (4, 3, 2)
    assert res["skipped"] == {"hospice_or_expired": 1}
    assert res["discharges_with_schema_failure"] == 0
    # the record has the extraction's drugs: warfarin is flagged in both notes with an extraction;
    # with no extraction there are no drugs in the record, so all four lines are flagged
    assert res["record_grounding"] == {
        "discharges_flagged": 3,
        "medications_flagged": 6,
        "conditions_flagged": 3,
    }
    note = res["note_grounding"]
    assert (note["medications"]["draft_meds_in_note"], note["medications"]["draft_meds"]) == (9, 12)
    assert (note["conditions"]["share"]) == 0.5
    done = res["completeness"]
    assert (
        done["changed_meds_in_brief"]["changed_in_brief"],
        done["changed_meds_in_brief"]["changed_meds"],
    ) == (4, 6)
    assert done["changed_meds_in_brief_same_type"]["changed_in_brief_same_type"] == 2
    assert done["changed_meds_in_script"]["changed_in_script"] == 2
    assert done["red_flags_in_script"]["share"] == 0.5
    assert sum(res["tiers"].values()) == 3

    saved = (work / "preds" / "drafts_qwen3-14b.jsonl").read_text().splitlines()
    assert len(saved) == 3 and "warfarin" in saved[0]  # drafts stay in the local work dir
    results = (
        dirs[0].parent / "repo" / "evals" / "results" / "drafting_mimic_qwen3-14b.json"
    ).read_text()
    assert "warfarin" not in results and "Zebra" not in results and "n0" not in results


def test_smoke_run_saves_nothing(dirs: tuple[Path, Path]) -> None:
    work, hosp = dirs
    res = ma.run(work, hosp, "qwen3:14b", limit=1, llm=drafts())
    assert res["n_notes"] == 1
    assert not (work / "preds" / "drafts_qwen3-14b.jsonl").exists()
    assert not (dirs[0].parent / "repo").exists()


def test_matching_helpers() -> None:
    words = {"furosemide", "metoprolol", "succinate", "xl"}
    assert ma.in_note("Metoprolol succinate XL", words, "metoprolol succinate xl 25 mg")
    assert not ma.in_note("warfarin", words, "furosemide 40 mg")
    assert ma.covered("chest pain", ["Call if you have chest pain"])
    assert not ma.covered("shortness of breath at rest", ["Call if you have chest pain"])
    assert not ma.covered("SOB", ["shortness of breath"])  # no content words to match
