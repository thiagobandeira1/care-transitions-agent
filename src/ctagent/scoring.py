"""LACE index (van Walraven et al., CMAJ 2010) plus an under-65 flag. Deterministic (ADR 0003)."""

from collections.abc import Iterable

from .charlson import charlson_score
from .models import DischargeEvent, RiskScore


def los_points(days: int) -> int:
    if days < 1:
        return 0
    if days <= 3:
        return days
    if days <= 6:
        return 4
    if days <= 13:
        return 5
    return 7


def charlson_points(score: int) -> int:
    return 5 if score >= 4 else score


def er_points(visits: int) -> int:
    return min(visits, 4)


def score_discharge(event: DischargeEvent, condition_codes: Iterable[str]) -> RiskScore:
    charlson, cats = charlson_score(condition_codes)
    parts = {
        "lace_l": los_points(event.los_days),
        "lace_a": 3 if event.acute else 0,
        "lace_c": charlson_points(charlson),
        "lace_e": er_points(event.er_visits_6m),
    }
    return RiskScore(
        lace=sum(parts.values()),
        charlson=charlson,
        charlson_conditions=cats,
        under_65=event.under_65,
        **parts,
    )
