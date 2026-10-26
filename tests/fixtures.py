"""Tiny handmade Synthea-shaped CSVs for tests (written to a temp dir, never committed)."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

T0 = datetime(2024, 3, 4, 8, 0)  # a Monday


def ts(day: float) -> str:
    return (T0 + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")


def d(day: float) -> str:
    return (T0 + timedelta(days=day)).strftime("%Y-%m-%d")


class SyntheaFixture:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, str]]] = {
            "patients": [],
            "payers": [{"Id": "PM", "NAME": "Medicare"}, {"Id": "PA", "NAME": "Aetna"}],
            "encounters": [],
            "conditions": [],
            "medications": [],
        }
        self._n = 0

    def patient(self, pid: str, born: str = "1950-01-01", died: str = "") -> "SyntheaFixture":
        self.rows["patients"].append({"Id": pid, "BIRTHDATE": born, "DEATHDATE": died})
        return self

    def encounter(
        self,
        pid: str,
        cls: str,
        start: float,
        stop: float,
        description: str = "Hospital admission (procedure)",
        reason: str = "",
        payer: str = "PM",
    ) -> str:
        self._n += 1
        eid = f"E{self._n}"
        self.rows["encounters"].append(
            {
                "Id": eid,
                "START": ts(start),
                "STOP": ts(stop),
                "PATIENT": pid,
                "PAYER": payer,
                "ENCOUNTERCLASS": cls,
                "CODE": "1",
                "DESCRIPTION": description,
                "REASONCODE": "",
                "REASONDESCRIPTION": reason,
            }
        )
        return eid

    def condition(
        self, pid: str, code: str, description: str, start: float, stop: float | None = None
    ) -> None:
        self.rows["conditions"].append(
            {
                "START": d(start),
                "STOP": "" if stop is None else d(stop),
                "PATIENT": pid,
                "ENCOUNTER": "",
                "SYSTEM": "http://snomed.info/sct",
                "CODE": code,
                "DESCRIPTION": description,
            }
        )

    def medication(
        self, pid: str, code: str, description: str, start: float, stop: float | None = None
    ) -> None:
        self.rows["medications"].append(
            {
                "START": ts(start),
                "STOP": "" if stop is None else ts(stop),
                "PATIENT": pid,
                "ENCOUNTER": "",
                "CODE": code,
                "DESCRIPTION": description,
            }
        )

    HEADERS: ClassVar[dict[str, list[str]]] = {
        "patients": ["Id", "BIRTHDATE", "DEATHDATE"],
        "payers": ["Id", "NAME"],
        "encounters": [
            "Id",
            "START",
            "STOP",
            "PATIENT",
            "PAYER",
            "ENCOUNTERCLASS",
            "CODE",
            "DESCRIPTION",
            "REASONCODE",
            "REASONDESCRIPTION",
        ],
        "conditions": ["START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION"],
        "medications": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
    }

    def write(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in self.rows.items():
            fields = self.HEADERS[name]
            with (folder / f"{name}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        return folder
