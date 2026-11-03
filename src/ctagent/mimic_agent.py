"""The agent's drafting step on the MIMIC-IV gold-set notes. Local only; prints aggregates only.

Each gold-set note runs through the real graph up to the human-review pause, with:
- the discharge event from MIMIC-IV hosp (admissions, patients): times, age, disposition, acuity,
  Medicare. Prior ER visits count only ED-registered admissions 2-182 days before, since hosp has
  no ED-only visits.
- an EHR stand-in: the admission's billing diagnoses (with their ICD codes, so LACE gets Charlson
  points) plus the note extraction's diagnoses and medications.
- the extraction `mimic-bench` saved for the chosen extractor, so no extraction call is made.

The tiers themselves are evaluated against readmissions in `mimic_risk`. Drafts and flags go to
MIMIC_WORK_DIR/preds/, outside the repository, and are never printed. evals/results/ gets counts
only:
- grounding: medications and conditions named in the drafts that are missing from the record (the
  graph's own check) or from the note text itself
- completeness: the extraction's changed medications and red flags that reach the drafts
"""

import json
import statistics
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver

from .bench import REPO
from .drafts import CallScript, ProviderBrief, _norm, _tokens  # the grounding check's matching
from .extraction import DischargeExtraction
from .graph import Deps, build_graph, start_discharge
from .llm import StructuredLLM, drafting_llm
from .mimic_bench import gold_notes, saved_predictions
from .models import (
    Condition,
    DischargeEvent,
    Disposition,
    MedChange,
    MedChangeType,
    Medication,
    PatientContext,
)
from .privacy import assert_outside_repo
from .tracking import Tracker

# MIMIC discharge_location -> the agent's dispositions (facilities follow the SNF path)
DISCHARGE_TO = {
    "HOME": Disposition.HOME,
    "AGAINST ADVICE": Disposition.HOME,
    "ASSISTED LIVING": Disposition.HOME,
    "HOME HEALTH CARE": Disposition.HOME_HEALTH,
    "SKILLED NURSING FACILITY": Disposition.SNF,
    "REHAB": Disposition.SNF,
    "CHRONIC/LONG TERM ACUTE CARE": Disposition.SNF,
    "PSYCH FACILITY": Disposition.SNF,
    "ACUTE HOSPITAL": Disposition.SNF,
    "OTHER FACILITY": Disposition.SNF,
    "HEALTHCARE FACILITY": Disposition.SNF,
    "HOSPICE": Disposition.HOSPICE,
    "DIED": Disposition.EXPIRED,
}
ACUTE_TYPES = {"EW EMER.", "DIRECT EMER.", "URGENT", "EU OBSERVATION"}
NO_FOLLOW_UP = {Disposition.HOSPICE, Disposition.EXPIRED}
KEPT = ("tier", "explanation", "call_script", "brief", "plan")  # saved locally, never printed


def disposition(location: object) -> Disposition:
    """Unrecorded or unknown locations count as home."""
    return DISCHARGE_TO.get(str(location).strip().upper(), Disposition.HOME)


def events(hosp_dir: Path, stays: dict[str, int]) -> dict[str, DischargeEvent]:
    """folder -> discharge event, for the folders' hospital admissions (hadm_id)."""
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
        ],
        parse_dates=["admittime", "dischtime", "edregtime"],
    )
    pts = pd.read_csv(
        hosp_dir / "patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year"]
    )
    mine = adm[adm["hadm_id"].isin(stays.values())]
    rows: dict[Any, dict[Any, Any]] = mine.set_index("hadm_id").to_dict("index")
    subjects = set(mine["subject_id"])
    ages: dict[Any, dict[Any, Any]] = (
        pts[pts["subject_id"].isin(subjects)].set_index("subject_id").to_dict("index")
    )
    ed = adm[adm["subject_id"].isin(subjects) & adm["edregtime"].notna()]
    ed_days: dict[int, set[date]] = {}
    for subject, t in zip(ed["subject_id"], ed["edregtime"], strict=True):
        ed_days.setdefault(int(subject), set()).add(t.date())
    out = {}
    for folder, hadm in stays.items():
        r = rows[hadm]
        admit: datetime = r["admittime"].to_pydatetime()
        p = ages[r["subject_id"]]
        prior = ed_days.get(int(r["subject_id"]), set())
        out[folder] = DischargeEvent(
            patient_id=f"p{folder[:4]}",
            episode_id=f"e{folder[:4]}",
            admit=admit,
            discharge=r["dischtime"].to_pydatetime(),
            age=int(p["anchor_age"]) + admit.year - int(p["anchor_year"]),
            disposition=disposition(r["discharge_location"]),
            acute=bool(r["admission_type"] in ACUTE_TYPES or pd.notna(r["edregtime"])),
            er_visits_6m=sum(2 <= (admit.date() - d).days <= 182 for d in prior),
            medicare=r["insurance"] == "Medicare",
        )
    return out


def diagnoses(hosp_dir: Path, hadm_ids: set[int]) -> dict[int, list[tuple[str, str]]]:
    """hadm_id -> the admission's billing diagnoses as (code, title), in sequence order."""
    dx = pd.read_csv(
        hosp_dir / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    dx = dx[dx["hadm_id"].isin(hadm_ids)]
    titles = pd.read_csv(hosp_dir / "d_icd_diagnoses.csv.gz", dtype={"icd_code": str})
    dx = dx.merge(titles, on=["icd_code", "icd_version"], how="left").sort_values(
        ["hadm_id", "seq_num"]
    )
    out: dict[int, list[tuple[str, str]]] = {}
    for h, code, ver, title in zip(
        dx["hadm_id"], dx["icd_code"], dx["icd_version"], dx["long_title"], strict=True
    ):
        if isinstance(title, str):
            out.setdefault(int(h), []).append((f"ICD{ver}:{code}", title))
    return out


def context(
    event: DischargeEvent, billing: list[tuple[str, str]], ext: DischargeExtraction | None
) -> PatientContext:
    onset = event.admit.date()
    conditions = [Condition(code=c, description=t, onset=onset) for c, t in billing]
    if ext is None:
        return PatientContext(conditions=conditions)
    noted = ([ext.principal_diagnosis] if ext.principal_diagnosis else []) + list(
        ext.secondary_diagnoses
    )
    conditions += [Condition(code="", description=d.text, onset=onset) for d in noted]
    meds = ext.medication_changes
    return PatientContext(
        conditions=conditions,
        active_meds=[
            Medication(code="", description=m.name, start=event.discharge.date())
            for m in meds
            if m.change is not MedChangeType.STOPPED
        ],
        med_changes=[MedChange(name=m.name, change=m.change, detail=m.detail) for m in meds],
    )


def _same_drug(a: str, b: str) -> bool:
    x, y = _norm(a), _norm(b)
    return bool(x and y) and (x in y or y in x)


def in_note(name: str, note_words: set[str], note_norm: str) -> bool:
    """A drug name appears in the note: the whole name, or every word of 4+ letters in it."""
    n = _norm(name)
    words = [w for w in n.split() if len(w) >= 4]
    return bool(n) and (f" {n} " in f" {note_norm} " or (bool(words) and set(words) <= note_words))


def covered(flag: str, signs: list[str]) -> bool:
    """A red flag reaches the call script if one warning sign holds half its content words."""
    need = _tokens(flag)
    return bool(need) and any(len(need & _tokens(s)) * 2 >= len(need) for s in signs)


def score_drafts(
    note: str,
    ext: DischargeExtraction | None,
    script: CallScript | None,
    brief: ProviderBrief | None,
) -> dict[str, int]:
    """Counts for one discharge: names in the drafts found in the note, and extraction items
    that reach the drafts. Nothing here keeps text."""
    note_norm = _norm(note)
    note_words, note_tokens = set(note_norm.split()), _tokens(note)
    lines = (script.medication_review if script else []) + (
        brief.medication_changes if brief else []
    )
    conds = brief.key_conditions if brief else []
    c = {
        "draft_meds": len(lines),
        "draft_meds_in_note": sum(in_note(m.name, note_words, note_norm) for m in lines),
        "draft_conditions": len(conds),
        "draft_conditions_in_note": sum(bool(_tokens(x) & note_tokens) for x in conds),
    }
    changed = [
        m for m in (ext.medication_changes if ext else []) if m.change != MedChangeType.CONTINUED
    ]
    brief_meds = brief.medication_changes if brief else []
    script_meds = script.medication_review if script else []
    hits = [next((b for b in brief_meds if _same_drug(m.name, b.name)), None) for m in changed]
    flags = [f for f in (ext.red_flags if ext else []) if _tokens(f)]
    c |= {
        "changed_meds": len(changed),
        "changed_in_brief": sum(h is not None for h in hits),
        "changed_in_brief_same_type": sum(
            h is not None and h.change == m.change for h, m in zip(hits, changed, strict=True)
        ),
        "changed_in_script": sum(
            any(_same_drug(m.name, s.name) for s in script_meds) for m in changed
        ),
        "red_flags": len(flags),
        "red_flags_in_script": sum(
            covered(f, script.warning_signs if script else []) for f in flags
        ),
    }
    return c


def _share(rows: list[dict[str, Any]], part: str, whole: str) -> dict[str, float | int]:
    num, den = sum(r[part] for r in rows), sum(r[whole] for r in rows)
    return {part: num, whole: den, "share": round(num / den, 3) if den else 0.0}


def run(
    work_dir: Path,
    hosp_dir: Path,
    model: str,
    extractor: str = "qwen3-14b_zero",
    limit: int | None = None,
    llm: StructuredLLM | None = None,
) -> dict[str, Any]:
    """Draft for every gold-set note with `model`, using the saved `extractor` predictions.

    A run with `limit` is a smoke test: its drafts and results are not saved.
    """
    preds_dir = assert_outside_repo(work_dir / "preds")
    saved = saved_predictions(preds_dir / f"silver_{extractor}.jsonl")
    notes = [(f, text) for f, text, _ in gold_notes(work_dir, "silver")][:limit]
    test = pd.read_parquet(work_dir / "test.parquet", columns=["note_id", "hadm_id"])
    hadm = dict(zip(test["note_id"], test["hadm_id"].astype(int), strict=True))
    stays = {f: hadm[f.split("_", 1)[1]] for f, _ in notes}
    evs = events(hosp_dir, stays)
    billing = diagnoses(hosp_dir, set(stays.values()))

    exts: dict[str, DischargeExtraction | None] = {}
    for f, _ in notes:
        pred = saved.get(f, {}).get("pred")
        exts[f] = ext = DischargeExtraction.model_validate(pred) if pred else None
        if ext and ext.principal_diagnosis:  # the record's reason for admission
            evs[f] = evs[f].model_copy(update={"reason": ext.principal_diagnosis.text})
    ctx = {evs[f].episode_id: context(evs[f], billing.get(stays[f], []), exts[f]) for f, _ in notes}
    by_text: dict[str, DischargeExtraction] = {t: e for (f, t) in notes if (e := exts[f])}
    lookup: Callable[[str], DischargeExtraction] = by_text.__getitem__
    graph = build_graph(
        Deps(
            llm=llm or drafting_llm(model),
            contexts=lambda e: ctx[e.episode_id],
            tracker=Tracker(":memory:"),
            extractor=lookup,
        ),
        InMemorySaver(),
    )

    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    skipped = {"hospice_or_expired": 0}
    for i, (f, text) in enumerate(notes, 1):
        ev = evs[f]
        if ev.disposition in NO_FOLLOW_UP:
            skipped["hospice_or_expired"] += 1
            continue
        t0 = time.perf_counter()
        pause = start_discharge(graph, ev, text if exts[f] else None).interrupts[0].value
        wall = time.perf_counter() - t0
        script = CallScript.model_validate(pause["call_script"]) if pause["call_script"] else None
        brief = ProviderBrief.model_validate(pause["brief"]) if pause["brief"] else None
        flags = pause["flags"]
        rows.append(
            {
                "seconds": wall,
                "tier": pause["tier"],
                "with_extraction": exts[f] is not None,
                "schema_failures": sum("failed after" in x for x in flags),
                "record_meds_flagged": sum(x.startswith("medication not in record") for x in flags),
                "record_conditions_flagged": sum(
                    x.startswith("condition not in record") for x in flags
                ),
                **score_drafts(text, exts[f], script, brief),
            }
        )
        kept.append({"folder": f, "flags": flags} | {k: pause[k] for k in KEPT})
        print(f"  draft {i}/{len(notes)}: {wall:.1f}s", flush=True)

    secs = [r["seconds"] for r in rows]
    result: dict[str, Any] = {
        "benchmark": "agent drafting on MIMIC-IV-Note gold-set notes",
        "model": model,
        "extractor": extractor,
        "n_notes": len(notes),
        "n_drafted": len(rows),
        "skipped": skipped,
        "with_extraction": sum(r["with_extraction"] for r in rows),
        "seconds_per_discharge": {
            "mean": round(statistics.mean(secs), 1),
            "median": round(statistics.median(secs), 1),
            "max": round(max(secs), 1),
        }
        if secs
        else {},
        "discharges_with_schema_failure": sum(r["schema_failures"] > 0 for r in rows),
        "record_grounding": {
            "discharges_flagged": sum(
                r["record_meds_flagged"] + r["record_conditions_flagged"] > 0 for r in rows
            ),
            "medications_flagged": sum(r["record_meds_flagged"] for r in rows),
            "conditions_flagged": sum(r["record_conditions_flagged"] for r in rows),
        },
        "note_grounding": {
            "medications": _share(rows, "draft_meds_in_note", "draft_meds"),
            "conditions": _share(rows, "draft_conditions_in_note", "draft_conditions"),
        },
        "completeness": {
            "changed_meds_in_brief": _share(rows, "changed_in_brief", "changed_meds"),
            "changed_meds_in_brief_same_type": _share(
                rows, "changed_in_brief_same_type", "changed_meds"
            ),
            "changed_meds_in_script": _share(rows, "changed_in_script", "changed_meds"),
            "red_flags_in_script": _share(rows, "red_flags_in_script", "red_flags"),
        },
        "tiers": pd.Series([r["tier"] for r in rows]).value_counts().to_dict(),
    }
    if limit is None:
        with (preds_dir / f"drafts_{model.replace(':', '-')}.jsonl").open(
            "w", encoding="utf-8"
        ) as fh:
            for row in kept:
                fh.write(json.dumps(row) + "\n")
        out = REPO / "evals" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"drafting_mimic_{model.replace(':', '-')}.json").write_text(
            json.dumps(result, indent=2)
        )
    return result
