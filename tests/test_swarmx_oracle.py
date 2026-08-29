"""Optional parity coverage against the R beeswarm ``swarmx`` oracle."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def r_beeswarm():
    try:
        from rpy2 import robjects  # type: ignore[import-not-found]
        from rpy2.robjects import packages  # type: ignore[import-not-found]
        from rpy2.robjects.packages import (  # type: ignore[import-not-found]
            PackageNotInstalledError,
        )
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"R and rpy2 are unavailable: {error}")
    try:
        return packages.importr("beeswarm"), robjects
    except (PackageNotInstalledError, RuntimeError) as error:
        pytest.skip(f"R beeswarm package is unavailable: {error}")


def test_swarmx_matches_r_oracle(r_beeswarm):
    r_package, robjects = r_beeswarm
    from p9beeswarm import swarmx

    values = np.array([-1.5, -0.8, -0.2, 0.0, 0.1, 0.7, 1.4])
    y_size = np.ptp(values) / 100
    expected = r_package.swarmx(
        x=robjects.FloatVector(np.zeros(values.size).tolist()),
        y=robjects.FloatVector(values.tolist()),
        xsize=0.4,
        ysize=float(y_size),
        cex=1,
        side=0,
        priority=robjects.StrVector(["ascending"]),
        fast=True,
        compact=True,
    ).rx2("x")
    np.testing.assert_allclose(
        swarmx(
            np.zeros(values.size),
            values,
            x_size=0.4,
            y_size=y_size,
            cex=1,
            priority="ascending",
            compact=True,
        ).x,
        np.asarray(expected, dtype=float),
    )
