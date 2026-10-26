"""Local SQLite tracking of plans, reviews, check-ins and outcomes (no graph stays open 30 days)."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

SCHEMA = """
create table if not exists discharges (
    episode_id text primary key, patient_id text not null, discharge_date text not null,
    tier text not null, lace integer not null, contact_by text not null, visit_by text not null,
    tcm_code text not null, tcm_visit_by text not null, status text not null,
    reviewer text, decided_at text, flags text not null default '[]'
);
create table if not exists check_ins (
    episode_id text not null references discharges(episode_id), due_date text not null,
    done_at text, outcome text, primary key (episode_id, due_date)
);
create table if not exists outcomes (
    episode_id text primary key references discharges(episode_id), contacted_at text,
    visit_scheduled text, visit_attended text, readmitted integer
);
"""


class Tracker:
    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.executescript(SCHEMA)

    def record_plan(
        self,
        episode_id: str,
        patient_id: str,
        discharge: date,
        plan: dict[str, object],
        lace: int,
        flags: list[str],
    ) -> None:
        with self.con:
            self.con.execute(
                """insert into discharges
                   values (?,?,?,?,?,?,?,?,?, 'pending_review', null, null, ?)
                   on conflict(episode_id) do update set tier=excluded.tier, lace=excluded.lace,
                   flags=excluded.flags""",
                (
                    episode_id,
                    patient_id,
                    discharge.isoformat(),
                    plan["tier"],
                    lace,
                    str(plan["contact_by"]),
                    str(plan["visit_by"]),
                    plan["tcm_code"],
                    str(plan["tcm_visit_by"]),
                    json.dumps(flags),
                ),
            )

    def record_review(
        self, episode_id: str, decision: str, reviewer: str, check_ins: list[str]
    ) -> None:
        status = {"approve": "approved", "edit": "approved", "reject": "rejected"}[decision]
        with self.con:
            self.con.execute(
                "update discharges set status=?, reviewer=?, decided_at=? where episode_id=?",
                (status, reviewer, datetime.now().isoformat(timespec="seconds"), episode_id),
            )
            if status == "approved":
                self.con.executemany(
                    "insert or ignore into check_ins(episode_id, due_date) values (?, ?)",
                    [(episode_id, d) for d in check_ins],
                )

    def status(self, episode_id: str) -> str | None:
        row = self.con.execute(
            "select status from discharges where episode_id=?", (episode_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def due_check_ins(self, today: date) -> list[tuple[str, str]]:
        rows = self.con.execute(
            """select c.episode_id, c.due_date from check_ins c join discharges d using(episode_id)
               where c.done_at is null and c.due_date <= ? and d.status = 'approved'
               order by c.due_date""",
            (today.isoformat(),),
        ).fetchall()
        return [(str(e), str(d)) for e, d in rows]

    def complete_check_in(self, episode_id: str, due_date: str, outcome: str) -> None:
        with self.con:
            self.con.execute(
                "update check_ins set done_at=?, outcome=? where episode_id=? and due_date=?",
                (datetime.now().isoformat(timespec="seconds"), outcome, episode_id, due_date),
            )

    def record_outcome(self, episode_id: str, **fields: str | int | None) -> None:
        allowed = {"contacted_at", "visit_scheduled", "visit_attended", "readmitted"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown outcome fields: {sorted(bad)}")
        cols = ", ".join(fields)
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields)
        with self.con:
            self.con.execute(
                f"insert into outcomes(episode_id, {cols}) values (?{', ?' * len(fields)}) "  # noqa: S608
                f"on conflict(episode_id) do update set {updates}",
                (episode_id, *fields.values()),
            )

    def summary(self) -> dict[str, int]:
        rows = self.con.execute(
            "select status, count(*) from discharges group by status"
        ).fetchall()
        return {str(s): int(n) for s, n in rows}
