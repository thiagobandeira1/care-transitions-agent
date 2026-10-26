from pathlib import Path

import numpy as np
import pytest

from ctagent.__main__ import main
from ctagent.metrics import auroc
from fixtures import SyntheaFixture


def test_auroc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 6, 200).astype(float)  # many ties
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    expected = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert auroc(y, s) == pytest.approx(expected)
    assert auroc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    with pytest.raises(ValueError):
        auroc([1, 1], [0.2, 0.3])


def test_cohort_command_prints_aggregates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fx = SyntheaFixture().patient("A").patient("B").patient("ZZ")
    fx.encounter("A", "emergency", -1, -0.9)
    fx.encounter("A", "inpatient", 3000, 3010)  # ~2032: inside the default study window
    fx.encounter("A", "inpatient", 3015, 3017)
    fx.encounter("B", "inpatient", 3000, 3002)
    fx.condition("A", "88805009", "Chronic congestive heart failure (disorder)", 2000)
    fx.encounter("ZZ", "ambulatory", 3400, 3400.1)
    main(["cohort", "--synthea", str(fx.write(tmp_path / "csv"))])
    out = capsys.readouterr().out
    assert "eligible index stays 3" in out and "tiers:" in out and "AUROC" in out
    assert "TEST" not in out
