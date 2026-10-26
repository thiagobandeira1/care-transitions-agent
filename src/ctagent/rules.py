"""Deterministic follow-up planning: risk tier, TCM clocks and check-ins (ADR 0002, ADR 0005).

TCM (CMS): 99495 and 99496 both need an interactive contact within 2 business days of discharge.
The face-to-face visit is due within 14 days for 99495 (moderate MDM) or 7 days for 99496 (high
MDM). The code here is only a candidate; the provider's medical decision making sets the final code.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

from .models import Disposition, FollowUpPlan, RiskScore, Tier


@dataclass(frozen=True)
class PlanRules:
    """Provisional thresholds, to be calibrated on a practice's own outcomes (ADR 0005)."""

    high_lace: int = 10
    high_lace_under_65: int = 7
    high_visit_days: int = 3
    standard_visit_days: int = 7
    check_in_days: tuple[int, ...] = (10, 20)
    holidays: frozenset[date] = field(default_factory=frozenset)


def add_business_days(start: date, n: int, holidays: frozenset[date] = frozenset()) -> date:
    """The n-th business day after `start` (weekends and listed holidays don't count)."""
    d, left = start, n
    while left:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in holidays:
            left -= 1
    return d


DEFAULT_RULES = PlanRules()


def plan_follow_up(
    discharge: date, score: RiskScore, disposition: Disposition, rules: PlanRules = DEFAULT_RULES
) -> FollowUpPlan:
    reasons = []
    if score.lace >= rules.high_lace:
        reasons.append(f"LACE {score.lace} >= {rules.high_lace}")
    if score.under_65 and score.lace >= rules.high_lace_under_65:
        reasons.append(f"Medicare under 65 with LACE {score.lace} >= {rules.high_lace_under_65}")
    tier = Tier.HIGH if reasons else Tier.STANDARD
    if disposition is Disposition.SNF:
        reasons.append("Discharged to SNF: coordinate with the facility before the TCM contact")

    high = tier is Tier.HIGH
    tcm_days = 7 if high else 14
    return FollowUpPlan(
        tier=tier,
        visit_by=discharge
        + timedelta(days=rules.high_visit_days if high else rules.standard_visit_days),
        contact_by=add_business_days(discharge, 2, rules.holidays),
        tcm_code="99496" if high else "99495",
        tcm_visit_by=discharge + timedelta(days=tcm_days),
        check_ins=[discharge + timedelta(days=d) for d in rules.check_in_days] if high else [],
        reasons=reasons or [f"LACE {score.lace} below the high-risk thresholds"],
    )
