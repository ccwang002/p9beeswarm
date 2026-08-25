"""Coordinate parity for the ggbeeswarm README's quasi-random examples."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
from plotnine.data import mpg

from p9beeswarm import quasirandom


@pytest.fixture(scope="module")
def r_vipor() -> tuple[Any, Any]:
    try:
        from rpy2 import robjects
        from rpy2.robjects import packages
        from rpy2.robjects.packages import PackageNotInstalledError
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"R and rpy2 are unavailable: {error}")
    try:
        return packages.importr("vipor"), robjects
    except (PackageNotInstalledError, RuntimeError) as error:
        pytest.skip(f"R vipor package is unavailable: {error}")


def _assert_matches_vipor(
    r_vipor: tuple[Any, Any],
    values: Iterable[float],
    *,
    method: str = "quasirandom",
    max_length: float | None = None,
    varwidth: bool = False,
) -> None:
    r_package, robjects = r_vipor
    values_array = np.asarray(list(values), dtype=float)
    arguments: dict[str, Any] = {
        "method": robjects.StrVector([method]),
        "adjust": 0.5,
    }
    if max_length is not None:
        arguments["maxLength"] = max_length
    expected = 0.4 * np.asarray(
        r_package.offsetSingleGroup(robjects.FloatVector(values_array.tolist()), **arguments),
        dtype=float,
    )
    np.testing.assert_allclose(
        quasirandom(
            values_array,
            width=0.4,
            method=method,
            max_length=max_length,
            varwidth=varwidth,
        ),
        expected,
        rtol=1e-3,
        atol=1e-3,
    )


def test_readme_quasirandom_default_and_categorical_y(
    r_vipor: tuple[Any, Any],
) -> None:
    for _, group in mpg.groupby("class", observed=True):
        _assert_matches_vipor(r_vipor, group["hwy"])


@pytest.mark.parametrize("method", ["smiley", "frowney"])
def test_readme_quasirandom_deterministic_methods(
    r_vipor: tuple[Any, Any], method: str
) -> None:
    from sklearn.datasets import load_iris

    values = load_iris().data[:, 0]
    _assert_matches_vipor(r_vipor, values, method=method)


def test_readme_quasirandom_varwidth(r_vipor: tuple[Any, Any]) -> None:
    class_sizes = mpg.groupby("class", observed=True).size()
    max_length = float(class_sizes.max())
    for _, group in mpg.groupby("class", observed=True):
        _assert_matches_vipor(
            r_vipor, group["hwy"], max_length=max_length, varwidth=True
        )


@pytest.mark.parametrize("method", ["tukey", "tukeyDense", "pseudorandom"])
def test_readme_quasirandom_stochastic_methods_produce_point_locations(method: str) -> None:
    from sklearn.datasets import load_iris

    values = load_iris().data[:, 0]
    offsets = quasirandom(values, width=0.4, method=method, random_state=12345)
    assert offsets.shape == values.shape
    assert np.all(np.isfinite(offsets))
    assert np.abs(offsets).max() <= 0.4


def test_readme_quasirandom_dodge_groups() -> None:
    sub_mpg = mpg[mpg["class"].isin(["midsize", "pickup", "suv"])]
    for _, group in sub_mpg.groupby(["class", "cyl"], observed=True):
        offsets = quasirandom(group["displ"], width=0.4, random_state=12345)
        assert offsets.shape == (len(group),)
        assert np.all(np.isfinite(offsets))
