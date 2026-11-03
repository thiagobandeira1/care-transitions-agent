"""The agent's risk step on MIMIC-IV: LACE tiers against real 30-day readmissions. Local only;
prints aggregates only.

Index stays are the admissions of the test-split discharge summaries, minus in-hospital deaths and
discharges to hospice (the agent plans no follow-up for them). LACE is computed as the agent does
at discharge:
- L: length of stay. A: an emergency or urgent admission, or registration in the ED.
- C: Charlson from the admission's billing diagnoses (Quan 2005, ICD-9-CM and ICD-10).
- E: ED-registered admissions 2-182 days before. MIMIC-IV hosp has no ED-only visits, so this
  undercounts.
The tier is the agent's rule: high when LACE >= 10, or LACE >= 7 for a Medicare beneficiary
under 65. The outcome is an unplanned admission (not elective, not same-day surgery) 1-30 days
after discharge; MIMIC only sees its own hospital. Death or readmission within 30 days is the
secondary outcome (LACE's original one). CIs come from a bootstrap that resamples patients.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bench import REPO
from .charlson import charlson_score
from .mimic_agent import ACUTE_TYPES
from .scoring import charlson_points, er_points, los_points

PLANNED = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}


def index_stays(hosp_dir: Path, hadm_ids: set[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per eligible index stay: LACE parts, tier inputs and the outcomes."""
    adm = pd.read_csv(
        hosp_dir / "admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "discharge_location",
            "edregtime",
            "insurance",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "edregtime"],
    )
    pts = pd.read_csv(
        hosp_dir / "patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year", "dod"]
    )
    idx = adm[adm["hadm_id"].isin(hadm_ids)].merge(pts, on="subject_id")
    died = (idx["hospital_expire_flag"] == 1) | (idx["discharge_location"] == "DIED")
    hospice = idx["discharge_location"] == "HOSPICE"
    excluded = {"died_in_hospital": int(died.sum()), "hospice": int((hospice & ~died).sum())}
    idx = idx[~died & ~hospice].copy()

    # outcome: the next unplanned admission of the same patient, 1-30 days after discharge
    later = adm.loc[~adm["admission_type"].isin(PLANNED), ["subject_id", "admittime"]]
    pairs = idx[["hadm_id", "subject_id", "dischtime"]].merge(later, on="subject_id")
    gap = pairs["admittime"] - pairs["dischtime"]
    hit = pairs[(gap > pd.Timedelta(0)) & (gap <= pd.Timedelta(days=30))]
    idx["readmit30"] = idx["hadm_id"].isin(set(hit["hadm_id"]))
    dod = pd.to_datetime(idx["dod"])
    left = idx["dischtime"].dt.normalize()
    idx["death30"] = dod.notna() & (dod >= left) & (dod <= left + pd.Timedelta(days=30))
    idx["death_or_readmit30"] = idx["readmit30"] | idx["death30"]

    # E: distinct ED registration days 2-182 days before this admission
    ed = adm.loc[adm["edregtime"].notna(), ["subject_id", "edregtime"]]
    ed = ed.assign(day=ed["edregtime"].dt.normalize()).drop_duplicates(["subject_id", "day"])
    prior = idx[["hadm_id", "subject_id", "admittime"]].merge(ed, on="subject_id")
    days = (prior["admittime"].dt.normalize() - prior["day"]).dt.days
    er = prior[(days >= 2) & (days <= 182)].groupby("hadm_id").size()
    idx["er_visits"] = idx["hadm_id"].map(er).fillna(0).astype(int)

    # C: Charlson from the admission's billing diagnoses
    dx = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    dx = dx[dx["hadm_id"].isin(set(idx["hadm_id"]))]
    codes = ("ICD" + dx["icd_version"].astype(str) + ":" + dx["icd_code"]).groupby(dx["hadm_id"])
    charlson = codes.apply(lambda c: charlson_score(c)[0])
    idx["charlson"] = idx["hadm_id"].map(charlson).fillna(0).astype(int)

    los = (idx["dischtime"].dt.normalize() - idx["admittime"].dt.normalize()).dt.days
    idx["acute"] = idx["admission_type"].isin(ACUTE_TYPES) | idx["edregtime"].notna()
    idx["lace"] = (
        los.map(los_points)
        + 3 * idx["acute"].astype(int)
        + idx["charlson"].map(charlson_points)
        + idx["er_visits"].map(er_points)
    )
    age = idx["anchor_age"] + idx["admittime"].dt.year - idx["anchor_year"]
    medicare_under_65 = (idx["insurance"] == "Medicare") & (age < 65)
    idx["high"] = (idx["lace"] >= 10) | (medicare_under_65 & (idx["lace"] >= 7))
    return idx.reset_index(drop=True), excluded


def auroc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUROC for an integer score (ties count half), so a bootstrap is a reweighting."""
    values = np.unique(score)
    at = np.searchsorted(values, score)
    pos = np.bincount(at, weights=w * y, minlength=len(values))
    neg = np.bincount(at, weights=w * (1 - y), minlength=len(values))
    below = np.cumsum(neg) - neg
    total = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / total) if total else float("nan")


def _rate(y: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    den = (w * mask).sum()
    return float((w * mask * y).sum() / den) if den else float("nan")


def stats(stays: pd.DataFrame, w: np.ndarray) -> dict[str, float]:
    y = stays["readmit30"].to_numpy(dtype=float)
    yd = stays["death_or_readmit30"].to_numpy(dtype=float)
    lace = stays["lace"].to_numpy()
    high = stays["high"].to_numpy(dtype=bool)
    ten = lace >= 10
    return {
        "lace_auroc_readmit30": auroc(lace, y, w),
        "lace_auroc_death_or_readmit30": auroc(lace, yd, w),
        "readmit30_rate": _rate(y, w, np.ones_like(high)),
        "high_tier_share": float((w * high).sum() / w.sum()),
        "readmit30_rate_high": _rate(y, w, high),
        "readmit30_rate_standard": _rate(y, w, ~high),
        "high_tier_sensitivity": _rate(high.astype(float), w, y.astype(bool)),
        "lace10_share": float((w * ten).sum() / w.sum()),
        "readmit30_rate_lace10": _rate(y, w, ten),
        "lace10_sensitivity": _rate(ten.astype(float), w, y.astype(bool)),
    }


def bootstrap(stays: pd.DataFrame, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% percentile CIs, resampling patients (all of a patient's stays move together)."""
    codes, uniq = pd.factorize(stays["subject_id"])
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        for k, v in stats(stays, counts[codes].astype(float)).items():
            draws.setdefault(k, []).append(v)
    return {
        k: (round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3))
        for k, v in draws.items()
    }


def run(work_dir: Path, hosp_dir: Path, n_boot: int = 1000, seed: int = 11) -> dict[str, Any]:
    test = pd.read_parquet(work_dir / "test.parquet", columns=["hadm_id"])
    stays, excluded = index_stays(hosp_dir, set(test["hadm_id"].astype(int)))
    point = stats(stays, np.ones(len(stays)))
    lace = stays["lace"]
    result: dict[str, Any] = {
        "benchmark": "the agent's LACE tiers vs 30-day readmission, MIMIC-IV test split",
        "index_stays": len(stays),
        "patients": int(stays["subject_id"].nunique()),
        "excluded": excluded,
        "readmissions_30d": int(stays["readmit30"].sum()),
        "deaths_30d": int(stays["death30"].sum()),
        "estimates": {k: round(v, 3) for k, v in point.items()},
        "ci95": bootstrap(stays, n_boot, seed),
        "lace_quantiles": {str(q): float(lace.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "share_with_charlson": round(float((stays["charlson"] > 0).mean()), 3),
        "share_with_prior_ed": round(float((stays["er_visits"] > 0).mean()), 3),
        "share_acute": round(float(stays["acute"].mean()), 3),
        "bootstrap": {"resamples": n_boot, "unit": "patient", "seed": seed},
    }
    out = REPO / "evals" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "risk_mimic.json").write_text(json.dumps(result, indent=2))
    return result
