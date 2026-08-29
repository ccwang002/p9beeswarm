"""Compare the Python position helpers with ggbeeswarm's R helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from plotnine.scales import scale_x_continuous, scale_x_discrete

from p9beeswarm.beeswarm import determine_pos
from p9beeswarm.positions import get_range


@pytest.fixture(scope="module")
def r_ggbeeswarm_helpers() -> Any:
    try:
        from rpy2 import robjects
        from rpy2.robjects import packages
        from rpy2.robjects.packages import PackageNotInstalledError
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"R and rpy2 are unavailable: {error}")
    try:
        packages.importr("ggbeeswarm")
        packages.importr("ggplot2")
    except (PackageNotInstalledError, RuntimeError) as error:
        pytest.skip(f"R ggbeeswarm dependencies are unavailable: {error}")
    return robjects


@pytest.mark.parametrize(
    ("scale_kind", "limits"),
    [("discrete", ["a", "b", "a"]), ("continuous", [2.0, 8.0]), ("continuous", [5.0, 5.0])],
)
def test_get_range_matches_ggbeeswarm(
    r_ggbeeswarm_helpers: Any, scale_kind: str, limits: list[Any]
) -> None:
    robjects = r_ggbeeswarm_helpers
    if scale_kind == "discrete":
        r_limits = robjects.StrVector(limits)
        r_scale = robjects.r["ggplot2::scale_x_discrete"](limits=r_limits)
        python_scale = scale_x_discrete(limits=limits)
    else:
        r_limits = robjects.FloatVector(limits)
        r_scale = robjects.r["ggplot2::scale_x_continuous"](limits=r_limits)
        python_scale = scale_x_continuous(limits=limits)

    expected = float(robjects.r["ggbeeswarm:::get_range"](r_scale)[0])
    assert get_range(python_scale) == expected


@pytest.mark.parametrize("method", ["center", "square", "hex"])
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_determine_pos_matches_ggbeeswarm(
    r_ggbeeswarm_helpers: Any, method: str, side: int
) -> None:
    values = np.array([1, 1, 1, 2, 2, 2], dtype=float)
    robjects = r_ggbeeswarm_helpers
    expected = np.asarray(
        robjects.r["ggbeeswarm:::determine_pos"](
            robjects.FloatVector(values.tolist()),
            method=robjects.StrVector([method]),
            side=side,
        ),
        dtype=float,
    )

    np.testing.assert_allclose(
        determine_pos(values, method, side), expected, rtol=0, atol=1e-12
    )
