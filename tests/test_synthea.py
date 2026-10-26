from pathlib import Path

import pandas as pd
import pytest

from ctagent.models import MedChangeType
from ctagent.synthea import SyntheaSource, base_name, build_cohort, is_planned, patient_contexts
from fixtures import SyntheaFixture


@pytest.mark.parametrize(
    ("description", "reason", "planned"),
    [
        ("Drug rehabilitation and detoxification (regime/therapy)", "", True),
        (
            "Encounter for problem (procedure)",
            "Non-small cell carcinoma of lung  TNM stage 1 (disorder)",
            True,
        ),
        ("Encounter for problem (procedure)", "Appendicitis (disorder)", False),
        ("Admission to ward (procedure)", "Sterilization requested (situation)", True),
        ("Hospital admission (procedure)", float("nan"), False),
    ],
)
def test_planned_stays(description: str, reason: object, planned: bool) -> None:
    assert is_planned(description, reason) is planned


def build(fx: SyntheaFixture, tmp_path: Path) -> pd.DataFrame:
    fx.encounter("ZZ", "ambulatory", 400, 400.1)  # pushes the end of data far away
    return build_cohort(SyntheaSource(fx.write(tmp_path / "csv")), since="2020-01-01").set_index(
        "patient_id"
    )


def test_index_stay_with_er_history_and_readmission(tmp_path: Path) -> None:
    fx = SyntheaFixture().patient("A").patient("ZZ")
    fx.encounter("A", "emergency", -50, -49.9)
    fx.encounter("A", "emergency", -50.2, -50.1)  # same ER day counts once
    fx.encounter("A", "emergency", -1, -0.9)  # the day before admission: acute, not a prior visit
    fx.encounter("A", "inpatient", 0, 3)
    fx.encounter(
        "A",
        "inpatient",
        8,
        9,
        "Encounter for problem (procedure)",
        "Malignant neoplasm of colon (disorder)",
    )
    fx.encounter("A", "inpatient", 13, 15)
    a = build(fx, tmp_path).loc[["A"]].sort_values("admit").iloc[0]
    assert a["acute"] and a["er_visits_6m"] == 1
    assert a["readmit30"] and a["days_to_next"] == 10  # the planned chemo stay on day 8 is skipped
    assert a["eligible"] and a["age"] == 74


def test_transfer_chain_merges(tmp_path: Path) -> None:
    fx = SyntheaFixture().patient("B").patient("ZZ")
    fx.encounter("B", "inpatient", 0, 2)
    fx.encounter("B", "inpatient", 2.1, 5)
    b = build(fx, tmp_path).loc["B"]
    assert b["n_stays"] == 2 and b["discharge"] == pd.Timestamp("2024-03-09 08:00")


def test_dispositions_and_exclusions(tmp_path: Path) -> None:
    fx = (
        SyntheaFixture()
        .patient("SNF")
        .patient("DEAD", died="2024-03-07")
        .patient("AET")
        .patient("ZZ")
    )
    fx.encounter("SNF", "inpatient", 0, 3)
    fx.encounter("SNF", "snf", 3.2, 10)
    fx.encounter("DEAD", "inpatient", 0, 3)
    fx.encounter("AET", "inpatient", 0, 3, payer="PA")
    c = build(fx, tmp_path)
    assert c.loc["SNF", "disposition"] == "SNF" and c.loc["SNF", "eligible"]
    assert (
        c.loc["DEAD", "disposition"] == "Expired"
        and c.loc["DEAD", "exclusion"] == "died or hospice"
    )
    assert c.loc["AET", "exclusion"] == "not Medicare"


def test_no_follow_up_window_at_end_of_data(tmp_path: Path) -> None:
    fx = SyntheaFixture().patient("LATE")
    fx.encounter("LATE", "inpatient", 0, 3)
    fx.encounter("LATE", "ambulatory", 20, 20.1)  # data ends 17 days after discharge
    c = build_cohort(SyntheaSource(fx.write(tmp_path / "csv")), since="2020-01-01")
    assert c.iloc[0]["exclusion"] == "no 30-day follow-up"


def test_patient_context_as_of_discharge(tmp_path: Path) -> None:
    fx = SyntheaFixture().patient("M").patient("ZZ")
    fx.encounter("M", "inpatient", 0, 3)
    fx.condition("M", "88805009", "Chronic congestive heart failure (disorder)", -900)
    fx.condition(
        "M", "44054006", "Diabetes mellitus type 2 (disorder)", 10
    )  # diagnosed after discharge
    fx.condition(
        "M", "59621000", "Essential hypertension (disorder)", -4000, -3000
    )  # resolved > 5 years ago
    fx.medication("M", "1", "furosemide 20 MG Oral Tablet", -100, 1)  # stopped ...
    fx.medication("M", "2", "furosemide 40 MG Oral Tablet", 1)  # ... and restarted at a new dose
    fx.medication("M", "3", "metoprolol 25 MG Oral Tablet", 2)  # started
    fx.medication("M", "4", "aspirin 81 MG Oral Tablet", -300, 2)  # stopped
    fx.medication("M", "5", "atorvastatin 20 MG Oral Tablet", -300)  # continued
    src = SyntheaSource(fx.write(tmp_path / "csv"))
    eps = build_cohort(src, since="2020-01-01")
    ctx = patient_contexts(src, eps[eps.patient_id == "M"])
    c = next(iter(ctx.values()))
    assert [x.code for x in c.conditions] == ["88805009"]
    changes = {m.name: m.change for m in c.med_changes}
    assert changes == {
        "furosemide": MedChangeType.DOSE_CHANGED,
        "metoprolol": MedChangeType.STARTED,
        "aspirin": MedChangeType.STOPPED,
        "atorvastatin": MedChangeType.CONTINUED,
    }
    assert {m.code for m in c.active_meds} == {"2", "3", "5"}


def test_base_name() -> None:
    assert base_name("lisinopril 10 MG Oral Tablet") == "lisinopril"
    assert (
        base_name("24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet")
        == "metformin hydrochloride"
    )
    assert (
        base_name("NDA020503 200 ACTUAT Albuterol 0.09 MG/ACTUAT Metered Dose Inhaler")
        == "albuterol"
    )
    assert base_name("1 ML Epoetin Alfa 4000 UNT/ML Injection [Epogen]") == "epoetin alfa"


SYNTHEA = Path(__file__).resolve().parents[2] / "_synthea" / "output" / "csv"


@pytest.mark.synthea
@pytest.mark.skipif(not SYNTHEA.exists(), reason="local Synthea output not present")
def test_real_synthea_cohort_builds() -> None:
    eps = build_cohort(SyntheaSource(SYNTHEA))
    el = eps[eps.eligible]
    assert len(el) > 500 and 0 < el["readmit30"].mean() < 0.5
