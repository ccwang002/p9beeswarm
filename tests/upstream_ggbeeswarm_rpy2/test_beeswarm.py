"""Coordinate parity for the ggbeeswarm README's beeswarm examples.

These tests use ``beeswarm::swarmx`` directly instead of rendering figures.
ggbeeswarm delegates its ``swarm`` and ``compactswarm`` layouts to that
function, so its returned x values are the upstream point locations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
from plotnine.data import mpg

from p9beeswarm import beeswarm


@pytest.fixture(scope="module")
def r_beeswarm() -> tuple[Any, Any]:
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


def _assert_matches_swarmx(
    r_beeswarm: tuple[Any, Any],
    values: Iterable[float],
    *,
    cex: float = 1,
    method: str = "swarm",
    priority: str = "ascending",
    side: int = 0,
) -> None:
    r_package, robjects = r_beeswarm
    values_array = np.asarray(list(values), dtype=float)
    y_size = max(float(np.ptp(values_array)), 1.0) / 100
    expected = r_package.swarmx(
        x=robjects.FloatVector(np.zeros(values_array.size).tolist()),
        y=robjects.FloatVector(values_array.tolist()),
        xsize=0.4,
        ysize=y_size,
        cex=cex,
        side=side,
        priority=robjects.StrVector([priority]),
        fast=True,
        compact=method == "compactswarm",
    ).rx2("x")
    np.testing.assert_allclose(
        beeswarm(
            values_array,
            width=0.4,
            cex=cex,
            method=method,
            priority=priority,
            side=side,
        ),
        np.asarray(expected, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_readme_iris_beeswarm_default_and_one_sided(r_beeswarm: tuple[Any, Any]) -> None:
    from sklearn.datasets import load_iris

    dataset: Any = load_iris()
    for values in np.split(dataset.data[:, 0], 3):
        _assert_matches_swarmx(r_beeswarm, values)
        _assert_matches_swarmx(r_beeswarm, values, side=1)


@pytest.mark.parametrize("cex", [0.5, 1.1])
def test_readme_mpg_beeswarm_sizes(r_beeswarm: tuple[Any, Any], cex: float) -> None:
    for _, group in mpg.groupby("class", observed=True):
        _assert_matches_swarmx(r_beeswarm, group["hwy"], cex=cex)


def test_readme_categorical_y_beeswarm(r_beeswarm: tuple[Any, Any]) -> None:
    for _, group in mpg.groupby("class", observed=True):
        _assert_matches_swarmx(r_beeswarm, group["hwy"], cex=0.5)


@pytest.mark.parametrize("method", ["swarm", "compactswarm"])
def test_readme_beeswarm_swarm_methods(
    r_beeswarm: tuple[Any, Any], method: str
) -> None:
    values = np.random.default_rng(12345).choice(np.arange(1, 101), 200)
    _assert_matches_swarmx(r_beeswarm, values, cex=2.5, method=method)


@pytest.mark.parametrize("priority", ["ascending", "descending", "density"])
def test_readme_beeswarm_priorities(
    r_beeswarm: tuple[Any, Any], priority: str
) -> None:
    rng = np.random.default_rng(12345)
    values = np.concatenate(
        [rng.normal(location, 1, size) for location, size in zip((1, 2, 3), (20, 40, 80))]
    )
    _assert_matches_swarmx(r_beeswarm, values, cex=2, priority=priority)
