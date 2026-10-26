"""Synthea discharge cohort: index hospitalizations, outcomes, and the record as of discharge.

Synthea is synthetic, so rows here may be inspected freely. Definitions mirror the Stage 1 work:
- Planned stays (rehab/detox, chemotherapy cycles, sleep studies, sterilization, transplant
  admissions, follow-up stays) are neither index stays nor readmissions. This approximates the
  CMS planned-readmission rule.
- Same-day or overlapping admissions merge into one episode.
- Disposition comes from the encounter that follows within a day (SNF, hospice, home health) or a
  death on or before the discharge date.
- Readmission: the next unplanned episode starts 1-30 days after discharge.
- Index stays need a Medicare or dual-eligible payer, an adult patient, discharge alive and not to
  hospice, and a full 30-day follow-up window.

SQL strings are built only from file paths and constants in this module, never from user input.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .models import Condition, Disposition, MedChange, MedChangeType, Medication, PatientContext

PLANNED_DESCRIPTIONS = {
    "Drug rehabilitation and detoxification (regime/therapy)",
    "Follow-up visit (procedure)",
    "Admission to surgical transplant department (procedure)",
}
PLANNED_REASON = re.compile(r"sterilization requested|awaiting transplantation", re.I)
TREATMENT_DESCRIPTIONS = {
    "Encounter for problem (procedure)",
    "Admission to clinical oncology department (procedure)",
}
TREATMENT_REASON = re.compile(r"neoplasm|carcinoma|leukemia|myeloma|sleep disorder", re.I)
MEDICARE_PAYERS = ("Medicare", "Dual Eligible")
FOLLOW_UP_DAYS = 30
ER_LOOKBACK_DAYS = 182
CONDITION_LOOKBACK_YEARS = 5


def is_planned(description: str, reason: object) -> bool:
    reason = reason if isinstance(reason, str) else ""  # missing reasons arrive as NaN
    if description in PLANNED_DESCRIPTIONS or PLANNED_REASON.search(reason):
        return True
    return description in TREATMENT_DESCRIPTIONS and bool(TREATMENT_REASON.search(reason))


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


@dataclass
class SyntheaSource:
    csv_dir: Path

    def table(self, name: str) -> str:
        path = (self.csv_dir / f"{name}.csv").as_posix()
        return f"read_csv('{path}', header=true, all_varchar=true)"

    def encounters(self, classes: tuple[str, ...]) -> pd.DataFrame:
        in_list = ", ".join(f"'{c}'" for c in classes)
        con = _connect()
        sql = f"""
            select e.Id as encounter_id, e.PATIENT as patient_id, e.ENCOUNTERCLASS as class,
                   e.START::timestamptz::timestamp as start, e.STOP::timestamptz::timestamp as stop,
                   e.DESCRIPTION as description, e.REASONCODE as reason_code,
                   e.REASONDESCRIPTION as reason, p.NAME as payer
            from {self.table("encounters")} e left join {self.table("payers")} p on e.PAYER = p.Id
            where e.ENCOUNTERCLASS in ({in_list})"""
        return _df(con, sql)

    def patients(self) -> pd.DataFrame:
        sql = f"""select Id as patient_id, BIRTHDATE::date as birthdate,
                         DEATHDATE::date as deathdate
                  from {self.table("patients")}"""
        return _df(_connect(), sql)

    def data_end(self) -> pd.Timestamp:
        sql = f"select max(START::timestamptz::timestamp) from {self.table('encounters')}"
        return pd.Timestamp(_connect().sql(sql).fetchall()[0][0])


def _df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    out: pd.DataFrame = con.sql(sql).df()
    return out


def merge_episodes(inp: pd.DataFrame) -> "pd.Series[int]":
    """Episode number per stay: same-day or overlapping unplanned stays of a patient share one."""
    inp = inp.sort_values(["patient_id", "start"])
    out = np.empty(len(inp), dtype=int)
    n, cur_patient, cur_end = -1, None, pd.Timestamp.min
    for i, (p, s, e) in enumerate(zip(inp["patient_id"], inp["start"], inp["stop"], strict=True)):
        if p == cur_patient and s.normalize() <= cur_end.normalize():
            cur_end = max(cur_end, e)
        else:
            n, cur_patient, cur_end = n + 1, p, e
        out[i] = n
    return pd.Series(out, index=inp.index, dtype=int)


def build_cohort(src: SyntheaSource, since: str = "2016-01-01") -> pd.DataFrame:
    """All unplanned inpatient episodes, with labels and index eligibility."""
    enc = src.encounters(("inpatient", "emergency", "snf", "hospice", "home"))
    pts = src.patients().set_index("patient_id")
    inp = enc[enc["class"] == "inpatient"].copy()
    inp["planned"] = [
        is_planned(d, r) for d, r in zip(inp["description"], inp["reason"], strict=True)
    ]
    unplanned = inp[~inp["planned"]].copy()
    unplanned["episode"] = merge_episodes(unplanned)
    g = unplanned.sort_values("start").groupby("episode")
    eps = (
        pd.DataFrame(
            {
                "episode_id": g["encounter_id"].first(),
                "patient_id": g["patient_id"].first(),
                "admit": g["start"].min(),
                "discharge": g["stop"].max(),
                "n_stays": g.size(),
                "reason": g["reason"].first(),
                "payer": g["payer"].first(),
            }
        )
        .sort_values(["patient_id", "admit"])
        .reset_index(drop=True)
    )

    # outcome: the next unplanned episode
    eps["next_admit"] = eps.groupby("patient_id")["admit"].shift(-1)
    eps["days_to_next"] = (
        eps["next_admit"].dt.normalize() - eps["discharge"].dt.normalize()
    ).dt.days
    eps["readmit30"] = eps["days_to_next"].between(1, FOLLOW_UP_DAYS)

    born = eps["patient_id"].map(pts["birthdate"])
    died = pd.to_datetime(eps["patient_id"].map(pts["deathdate"]))
    eps["age"] = [_age(b, a) for b, a in zip(born, eps["admit"], strict=True)]
    eps["disposition"] = _dispositions(eps, enc, died)
    eps = eps.join(_er_history(eps, enc[enc["class"] == "emergency"]))

    cutoff = src.data_end() - pd.Timedelta(days=FOLLOW_UP_DAYS)
    eps["exclusion"] = np.select(
        [
            eps["admit"] < pd.Timestamp(since),
            ~eps["payer"].isin(MEDICARE_PAYERS),
            eps["age"] < 18,
            eps["disposition"].isin([Disposition.EXPIRED.value, Disposition.HOSPICE.value]),
            eps["discharge"] > cutoff,
        ],
        [
            "before study window",
            "not Medicare",
            "under 18",
            "died or hospice",
            "no 30-day follow-up",
        ],
        default="",
    )
    eps["eligible"] = eps["exclusion"] == ""
    return eps.drop(columns=["next_admit"])


def _age(born: object, when: pd.Timestamp) -> int:
    b = pd.Timestamp(str(born))
    return int(when.year - b.year - ((when.month, when.day) < (b.month, b.day)))


def _dispositions(
    eps: pd.DataFrame, enc: pd.DataFrame, died: "pd.Series[pd.Timestamp]"
) -> list[str]:
    post = {"snf": Disposition.SNF, "hospice": Disposition.HOSPICE, "home": Disposition.HOME_HEALTH}
    nxt = enc[enc["class"].isin(list(post))][["patient_id", "class", "start"]]
    m = eps[["patient_id", "discharge"]].reset_index().merge(nxt, on="patient_id")
    gap = m["start"] - m["discharge"]
    m = m[(gap >= pd.Timedelta(0)) & (gap <= pd.Timedelta(days=1))].sort_values("start")
    first = m.groupby("index")["class"].first()
    out = []
    for i, (dis, death) in enumerate(zip(eps["discharge"], died, strict=True)):
        if pd.notna(death) and death.normalize() <= dis.normalize():
            out.append(Disposition.EXPIRED.value)
        elif i in first.index:
            out.append(post[first[i]].value)
        else:
            out.append(Disposition.HOME.value)
    return out


def _er_history(eps: pd.DataFrame, er: pd.DataFrame) -> pd.DataFrame:
    days = er.assign(day=er["start"].dt.normalize())[["patient_id", "day"]].drop_duplicates()
    m = eps[["patient_id", "admit"]].reset_index().merge(days, on="patient_id")
    before = (m["admit"].dt.normalize() - m["day"]).dt.days
    out = pd.DataFrame(index=eps.index)
    out["acute"] = eps.index.isin(m.loc[before.between(0, 1), "index"])
    counts = m[before.between(2, ER_LOOKBACK_DAYS)].groupby("index")["day"].nunique()
    out["er_visits_6m"] = counts.reindex(eps.index, fill_value=0).astype(int)
    return out


# ---- the record as of discharge (the agent's EHR stand-in) ----

_LEADING_QTY = re.compile(r"^(?:NDA\d+\s+)?(?:[\d.]+\s*(?:HR|ML|ACTUAT|MG|UNT)\s+)+", re.I)
_BASE_NAME = re.compile(r"^([^\d\[]+)")


def base_name(description: str) -> str:
    """Drug name without strength or form ('24 HR Metformin ... 500 MG' -> 'metformin ...')."""
    s = _LEADING_QTY.sub("", description.strip())
    m = _BASE_NAME.match(s)
    # Synthea descriptions contain double spaces ("insulin isophane  human"); collapse them
    return re.sub(r"\s+", " ", (m.group(1) if m else s)).strip().lower()


def patient_contexts(src: SyntheaSource, eps: pd.DataFrame) -> dict[str, PatientContext]:
    """Per episode: conditions (onset before discharge, active within 5 years), active meds and
    medication changes during the stay."""
    ids = eps[["episode_id", "patient_id", "admit", "discharge"]]
    con = _connect()
    con.register("ids", ids)
    lookback = f"interval {CONDITION_LOOKBACK_YEARS} year"
    conds = _df(
        con,
        f"""
        select i.episode_id, c.CODE as code, c.DESCRIPTION as description, c.START::date as onset
        from ids i join {src.table("conditions")} c on c.PATIENT = i.patient_id
        where c.START::date <= i.discharge::date
          and (c.STOP is null or c.STOP::date >= i.discharge::date - {lookback})
          and c.DESCRIPTION like '%(disorder)%'""",
    )
    meds = _df(
        con,
        f"""
        select i.episode_id, i.admit, i.discharge, m.CODE as code, m.DESCRIPTION as description,
               m.START::timestamptz::timestamp as start, m.STOP::timestamptz::timestamp as stop
        from ids i join {src.table("medications")} m on m.PATIENT = i.patient_id
        where m.START::timestamptz::timestamp <= i.discharge
          and (m.STOP is null or m.STOP::timestamptz::timestamp >= i.admit)""",
    )
    out: dict[str, PatientContext] = {e: PatientContext() for e in ids["episode_id"]}
    for e, grp in conds.groupby("episode_id"):
        out[str(e)].conditions = [
            Condition(
                code=str(r.code),
                description=str(r.description),
                onset=pd.Timestamp(str(r.onset)).date(),
            )
            for r in grp.drop_duplicates("code").itertuples()
        ]
    for e, grp in meds.groupby("episode_id"):
        ctx = out[str(e)]
        ctx.active_meds, ctx.med_changes = _med_changes(grp)
    return out


def _med_changes(grp: pd.DataFrame) -> tuple[list[Medication], list[MedChange]]:
    admit, discharge = grp["admit"].iloc[0], grp["discharge"].iloc[0]
    active_at_discharge = grp["stop"].isna() | (grp["stop"] > discharge)
    started = grp["start"] >= admit
    stopped = grp["stop"].notna() & grp["stop"].between(admit, discharge) & ~started
    active = grp[active_at_discharge].drop_duplicates("code")
    meds = [
        Medication(
            code=r.code,
            description=r.description,
            start=r.start.date(),
            stop=None if pd.isna(r.stop) else r.stop.date(),
        )
        for r in active.itertuples()
    ]
    new = {base_name(d): d for d in grp.loc[started & active_at_discharge, "description"]}
    gone = {base_name(d): d for d in grp.loc[stopped, "description"]}
    kept = {base_name(d): d for d in grp.loc[~started & active_at_discharge, "description"]}
    changes = [
        MedChange(name=n, change=MedChangeType.DOSE_CHANGED, detail=f"{gone[n]} -> {new[n]}")
        for n in sorted(new.keys() & gone.keys())
    ]
    changes += [
        MedChange(name=n, change=MedChangeType.STARTED, detail=new[n])
        for n in sorted(new.keys() - gone.keys() - kept.keys())
    ]
    changes += [
        MedChange(name=n, change=MedChangeType.STOPPED, detail=gone[n])
        for n in sorted(gone.keys() - new.keys() - kept.keys())
    ]
    changes += [
        MedChange(name=n, change=MedChangeType.CONTINUED, detail=kept[n])
        for n in sorted(kept.keys() - new.keys() - gone.keys())
    ]
    return meds, changes
