"""Field-level evaluation of discharge extractions against reference labels.

Each list field is scored as a set: predictions and references are paired greedily by
similarity, a pair counts when it meets the threshold (exact = normalized equality; relaxed =
token overlap >= 0.5), and precision/recall/F1 are micro-averaged over the corpus. Medication
changes count only when the drug AND the change type both match; change-type accuracy among
matched drugs is reported separately.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .extraction import DischargeExtraction

FIELDS = (
    "principal_diagnosis",
    "secondary_diagnoses",
    "medication_changes",
    "follow_up",
    "pending_tests",
    "red_flags",
)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    return len(ta & tb) / len(ta | tb) if ta and tb else float(normalize(a) == normalize(b))


def exact(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def relaxed(a: str, b: str) -> bool:
    return exact(a, b) or token_overlap(a, b) >= 0.5


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp, self.fp, self.fn = self.tp + other.tp, self.fp + other.fp, self.fn + other.fn

    def prf(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f}


def match(pred: list[str], ref: list[str], same: Callable[[str, str], bool]) -> Counts:
    left, tp = list(ref), 0
    for p in pred:
        hit = next((r for r in left if same(p, r)), None)
        if hit is not None:
            left.remove(hit)
            tp += 1
    return Counts(tp, len(pred) - tp, len(ref) - tp)


def field_strings(x: DischargeExtraction, name: str) -> list[str]:
    if name == "principal_diagnosis":
        return [x.principal_diagnosis.text] if x.principal_diagnosis else []
    if name == "secondary_diagnoses":
        return [d.text for d in x.secondary_diagnoses]
    if name == "medication_changes":
        return [f"{m.name}|{m.change.value}" for m in x.medication_changes]
    if name == "follow_up":
        return [f.with_whom for f in x.follow_up]
    values: list[str] = getattr(x, name)
    return values


def _med_same(strict: bool) -> Callable[[str, str], bool]:
    def same(a: str, b: str) -> bool:
        (na, ca), (nb, cb) = a.split("|"), b.split("|")
        names = exact(na, nb) if strict else relaxed(na, nb)
        return names and ca == cb

    return same


@dataclass
class Report:
    exact: dict[str, Counts] = field(default_factory=lambda: {f: Counts() for f in FIELDS})
    relaxed: dict[str, Counts] = field(default_factory=lambda: {f: Counts() for f in FIELDS})
    med_names_matched: int = 0
    med_type_correct: int = 0
    n: int = 0

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {"n": self.n}
        for mode in ("exact", "relaxed"):
            counts: dict[str, Counts] = getattr(self, mode)
            total = Counts()
            for f in FIELDS:
                total.add(counts[f])
            out[mode] = {**{f: counts[f].prf() for f in FIELDS}, "micro": total.prf()}
        out["med_change_type_accuracy"] = (
            self.med_type_correct / self.med_names_matched if self.med_names_matched else 0.0
        )
        return out


def evaluate(pairs: Iterable[tuple[DischargeExtraction | None, DischargeExtraction]]) -> Report:
    """`pred` may be None (schema failure): everything in the reference counts as missed."""
    rep = Report()
    for pred, ref in pairs:
        rep.n += 1
        p = pred or DischargeExtraction(principal_diagnosis=None)
        for f in FIELDS:
            pf, rf = field_strings(p, f), field_strings(ref, f)
            strict_fn = _med_same(True) if f == "medication_changes" else exact
            loose_fn = _med_same(False) if f == "medication_changes" else relaxed
            rep.exact[f].add(match(pf, rf, strict_fn))
            rep.relaxed[f].add(match(pf, rf, loose_fn))
        left = list(ref.medication_changes)
        for m in p.medication_changes:
            hit = next((r for r in left if relaxed(m.name, r.name)), None)
            if hit is not None:
                left.remove(hit)
                rep.med_names_matched += 1
                rep.med_type_correct += m.change == hit.change
    return rep
