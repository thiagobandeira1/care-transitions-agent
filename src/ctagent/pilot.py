"""Pilot measures: how coordinators use the agent's output, from the local tracking database.

`review_edits` measures each review against the drafts as written (the graph calls it). `report`
turns the tracking database into the paper's "how it is used" measures. It returns counts, shares
and medians only, never identifiers, names or text, so its output can leave the practice.
- review: approved as drafted, edited or rejected; how much and which fields were edited; review
  time; whether the flags were real problems
- timeliness: interactive contact by the TCM deadline (2 business days), and visits by the
  practice's target and by the TCM deadline (7 or 14 days), overall and by tier
- check-ins done by their due date, contact effort, and readmissions (descriptive only)
A deadline counts only once it has passed or the event is recorded, so a running pilot can report.
"""

import difflib
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

import numpy as np

from .tracking import Tracker

Doc = dict[str, Any] | None


def edit_share(before: Doc, after: Doc) -> float:
    """Share of a document the reviewer changed: 0 = as drafted, 1 = entirely rewritten."""
    a = json.dumps(before, sort_keys=True) if before else ""
    b = json.dumps(after, sort_keys=True) if after else ""
    if a == b:
        return 0.0
    return round(1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)


def fields_changed(name: str, before: Doc, after: Doc) -> list[str]:
    """Which fields of a document changed, as 'document.field' (names only, never content)."""
    old, new = before or {}, after or {}
    return [f"{name}.{k}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def review_edits(decision: str, script: tuple[Doc, Doc], brief: tuple[Doc, Doc]) -> dict[str, Any]:
    """Edit measures for one review; None for a rejected discharge (nothing was used)."""
    if decision == "reject":
        return {"script_edit": None, "brief_edit": None, "fields_edited": []}
    return {
        "script_edit": edit_share(*script),
        "brief_edit": edit_share(*brief),
        "fields_edited": fields_changed("call_script", *script) + fields_changed("brief", *brief),
    }


def _day(value: object) -> date | None:
    return date.fromisoformat(str(value)[:10]) if value else None


def _median(values: Iterable[float]) -> float | None:
    v = list(values)
    return round(statistics.median(v), 1) if v else None


def _deadline(rows: list[dict[str, Any]], due: str, done: str, today: date) -> dict[str, Any]:
    """Of the discharges whose deadline has passed or whose event is recorded, how many met it."""
    met = late = missing = 0
    for r in rows:
        deadline, when = _day(r[due]), _day(r[done])
        if deadline is None:
            continue
        if when is None:
            missing += deadline < today
        elif when <= deadline:
            met += 1
        else:
            late += 1
    n = met + late + missing
    return {"n": n, "met": met, "share": round(met / n, 3) if n else 0.0, "not_recorded": missing}


def report(tracker: Tracker, today: date) -> dict[str, Any]:
    con = tracker.con
    cur = con.execute(
        """select d.tier, d.discharge_date, d.contact_by, d.visit_by, d.tcm_visit_by, d.status,
                  o.contacted_at, o.contact_attempts, o.contact_minutes, o.visit_attended,
                  o.readmitted
           from discharges d left join outcomes o using(episode_id)"""
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    approved = [r for r in rows if r["status"] == "approved"]

    keys = ("decision", "seconds", "flags", "real", "script", "brief", "fields")
    rv = [
        dict(zip(keys, r, strict=True))
        for r in con.execute(
            """select decision, review_seconds, flags, flags_real, script_edit, brief_edit,
                      fields_edited from reviews"""
        )
    ]
    edited = [r for r in rv if r["decision"] == "edit"]
    flagged = [r for r in rv if r["flags"]]

    timeliness: dict[str, Any] = {}
    groups = [("all", approved)] + [
        (t, [r for r in approved if r["tier"] == t]) for t in ("high", "standard")
    ]
    for label, group in groups:
        timeliness[label] = {
            "contact_by_2_business_days": _deadline(group, "contact_by", "contacted_at", today),
            "visit_by_practice_target": _deadline(group, "visit_by", "visit_attended", today),
            "visit_by_tcm_deadline": _deadline(group, "tcm_visit_by", "visit_attended", today),
        }
    contact_days, visit_days = [], []
    for r in approved:
        left, contacted, seen = (
            _day(r["discharge_date"]),
            _day(r["contacted_at"]),
            _day(r["visit_attended"]),
        )
        if left and contacted:  # business days after discharge, the TCM count
            contact_days.append(int(np.busday_count(left + timedelta(1), contacted + timedelta(1))))
        if left and seen:
            visit_days.append((seen - left).days)

    check_ins = [
        (_day(due), _day(at))
        for due, at in con.execute(
            """select c.due_date, c.done_at from check_ins c join discharges d using(episode_id)
               where d.status = 'approved' and c.due_date <= ?""",
            (today.isoformat(),),
        )
    ]
    done = [(due, at) for due, at in check_ins if due and at]

    recorded = [r for r in approved if r["readmitted"] is not None]
    return {
        "as_of": today.isoformat(),
        "discharges": len(rows),
        "by_tier": dict(Counter(r["tier"] for r in rows)),
        "pending_review": sum(r["status"] == "pending_review" for r in rows),
        "review": {
            "reviewed": len(rv),
            "decisions": dict(Counter(r["decision"] for r in rv)),
            "call_scripts_edited": sum(1 for r in edited if r["script"]),
            "briefs_edited": sum(1 for r in edited if r["brief"]),
            "median_edit_share_when_edited": {
                "call_script": _median(r["script"] for r in edited if r["script"]),
                "brief": _median(r["brief"] for r in edited if r["brief"]),
            },
            "fields_edited": dict(
                Counter(f for r in edited for f in json.loads(r["fields"])).most_common()
            ),
            "review_seconds_median": _median(r["seconds"] for r in rv if r["seconds"] is not None),
            "discharges_with_flags": len(flagged),
            "flags_real": dict(Counter(r["real"] or "not_answered" for r in flagged)),
        },
        "timeliness": timeliness,
        "business_days_to_contact_median": _median(contact_days),
        "days_to_visit_median": _median(visit_days),
        "contact_effort": {
            "attempts_median": _median(
                r["contact_attempts"] for r in approved if r["contact_attempts"] is not None
            ),
            "minutes_median": _median(
                r["contact_minutes"] for r in approved if r["contact_minutes"] is not None
            ),
        },
        "check_ins": {
            "due": len(check_ins),
            "done": len(done),
            "done_by_due_date": sum(1 for due, at in done if at <= due),
        },
        "readmissions_descriptive": {
            "recorded": len(recorded),
            "readmitted": sum(1 for r in recorded if r["readmitted"]),
        },
    }
