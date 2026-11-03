"""The agent's risk step against readmissions, on the fictional hosp tables of test_mimic_agent."""

import json
from pathlib import Path

import numpy as np
import pytest

import ctagent.mimic_risk as mr
from ctagent.__main__ import main
from test_mimic_agent import write_hosp, write_work


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    write_hosp(tmp_path / "hosp")
    write_work(tmp_path / "work")
    monkeypatch.setattr(mr, "REPO", tmp_path / "repo")
    return tmp_path / "work", tmp_path / "hosp"


def test_weighted_auroc_counts_ties_as_half() -> None:
    score, y = np.array([1, 1, 2, 3]), np.array([0, 1, 0, 1])
    assert mr.auroc(score, y, np.ones(4)) == 0.625  # (0.5 + 0 + 1 + 1) / 4 pairs
    assert mr.auroc(score, y, np.array([1, 1, 0, 1])) == 0.75  # the negative at 2 drops out
    assert np.isnan(mr.auroc(score, np.zeros(4), np.ones(4)))


def test_index_stays(dirs: tuple[Path, Path]) -> None:
    _, hosp = dirs
    stays, excluded = mr.index_stays(hosp, {101, 201, 301, 401})
    assert excluded == {"died_in_hospital": 1, "hospice": 0}
    s = stays.set_index("hadm_id")
    # 101: 8 days (5) + ED (3) + heart failure (1) + one prior ED stay (1) = 10, readmitted day 5
    assert s.loc[101, ["lace", "er_visits", "charlson", "readmit30"]].tolist() == [10, 1, 1, True]
    # 301: elective, 3 days, heart failure: 4; the planned admission 10 days later doesn't count
    assert s.loc[301, ["lace", "acute", "readmit30", "high"]].tolist() == [4, False, False, False]
    # 401: 4 days (4) + urgent (3) + CKD stage 4 (2) = 9, high as Medicare under 65; died day 15
    row = s.loc[401]
    assert (row["lace"], row["high"], row["readmit30"], row["death30"]) == (9, True, False, True)


def test_run_reports_aggregates_only(dirs: tuple[Path, Path]) -> None:
    work, hosp = dirs
    res = mr.run(work, hosp, n_boot=50)
    assert (res["index_stays"], res["patients"], res["readmissions_30d"], res["deaths_30d"]) == (
        3,
        3,
        1,
        1,
    )
    e = res["estimates"]
    assert (e["lace_auroc_readmit30"], e["lace_auroc_death_or_readmit30"]) == (1.0, 1.0)
    assert (e["readmit30_rate"], e["high_tier_share"], e["readmit30_rate_high"]) == (
        0.333,
        0.667,
        0.5,
    )
    assert (e["readmit30_rate_standard"], e["high_tier_sensitivity"]) == (0.0, 1.0)
    assert (e["lace10_share"], e["readmit30_rate_lace10"]) == (0.333, 1.0)
    lo, hi = res["ci95"]["readmit30_rate"]
    assert 0.0 <= lo <= hi <= 1.0
    saved = (work.parent / "repo" / "evals" / "results" / "risk_mimic.json").read_text()
    assert json.loads(saved)["index_stays"] == 3 and "subject" not in saved


def test_command(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    work, hosp = dirs
    monkeypatch.setenv("MIMIC_NOTE_DIR", str(hosp))
    monkeypatch.setenv("MIMIC_HOSP_DIR", str(hosp))
    monkeypatch.setenv("MIMIC_WORK_DIR", str(work))
    main(["mimic-risk", "--boot", "20"])
    assert json.loads(capsys.readouterr().out)["index_stays"] == 3
