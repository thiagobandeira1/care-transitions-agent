from datetime import date, datetime

import pytest

from ctagent.charlson import charlson_categories, charlson_score
from ctagent.models import DischargeEvent, Disposition, RiskScore, Tier
from ctagent.rules import PlanRules, add_business_days, plan_follow_up
from ctagent.scoring import charlson_points, er_points, los_points, score_discharge


def event(los: int = 4, age: int = 75, acute: bool = True, er: int = 2) -> DischargeEvent:
    return DischargeEvent(
        patient_id="p",
        episode_id="e",
        admit=datetime(2024, 3, 4, 8),
        discharge=datetime(2024, 3, 4 + los, 14),
        age=age,
        disposition=Disposition.HOME,
        acute=acute,
        er_visits_6m=er,
    )


def test_lace_point_tables() -> None:
    assert [los_points(d) for d in [0, 1, 2, 3, 4, 6, 7, 13, 14, 30]] == [
        0,
        1,
        2,
        3,
        4,
        4,
        5,
        5,
        7,
        7,
    ]
    assert [charlson_points(s) for s in [0, 1, 3, 4, 9]] == [0, 1, 3, 5, 5]
    assert [er_points(n) for n in [0, 2, 4, 7]] == [0, 2, 4, 4]


def test_charlson_hierarchy_and_weights() -> None:
    # CHF (1) + diabetic kidney disease (2, replaces diabetes) + metastatic (6, replaces cancer)
    codes = ["88805009", "44054006", "127013003", "254837009", "94503003", "59621000"]
    assert charlson_categories(codes) == {"chf", "diab_cc", "metastatic"}
    assert charlson_score(codes) == (9, ["chf", "diab_cc", "metastatic"])
    assert charlson_score([]) == (0, [])


def test_score_discharge() -> None:
    s = score_discharge(event(), ["88805009", "431857002"])  # CHF + CKD 4 -> Charlson 3
    assert (s.lace_l, s.lace_a, s.lace_c, s.lace_e, s.lace) == (4, 3, 3, 2, 12)
    assert not s.under_65


def test_business_days_skip_weekends_and_holidays() -> None:
    fri = date(2024, 3, 8)
    assert add_business_days(fri, 2) == date(2024, 3, 12)  # Mon, Tue
    assert add_business_days(fri, 2, frozenset({date(2024, 3, 11)})) == date(2024, 3, 13)


def score(lace: int, under_65: bool = False) -> RiskScore:
    return RiskScore(
        lace=lace,
        lace_l=0,
        lace_a=0,
        lace_c=0,
        lace_e=0,
        charlson=0,
        charlson_conditions=[],
        under_65=under_65,
    )


@pytest.mark.parametrize(
    ("lace", "under_65", "tier"),
    [
        (10, False, Tier.HIGH),
        (9, False, Tier.STANDARD),
        (7, True, Tier.HIGH),
        (6, True, Tier.STANDARD),
    ],
)
def test_tiers(lace: int, under_65: bool, tier: Tier) -> None:
    assert plan_follow_up(date(2024, 3, 8), score(lace, under_65), Disposition.HOME).tier is tier


def test_high_risk_plan() -> None:
    p = plan_follow_up(date(2024, 3, 8), score(12), Disposition.SNF)
    assert p.visit_by == date(2024, 3, 11) and p.contact_by == date(2024, 3, 12)
    assert (p.tcm_code, p.tcm_visit_by) == ("99496", date(2024, 3, 15))
    assert p.check_ins == [date(2024, 3, 18), date(2024, 3, 28)]
    assert any("SNF" in r for r in p.reasons)


def test_standard_plan_uses_practice_rules() -> None:
    p = plan_follow_up(
        date(2024, 3, 8), score(5), Disposition.HOME, PlanRules(standard_visit_days=5)
    )
    assert (p.tier, p.visit_by, p.tcm_code, p.tcm_visit_by) == (
        Tier.STANDARD,
        date(2024, 3, 13),
        "99495",
        date(2024, 3, 22),
    )
    assert p.check_ins == [] and "below" in p.reasons[0]
