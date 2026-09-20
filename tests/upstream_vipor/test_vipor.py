"""Tests corresponding to vipor's upstream ``tests.R``."""

from typing import Any

import numpy as np
import pytest

from vipor import (
    aveWithArgs,
    digits2number,
    number2digits,
    offsetSingleGroup,
    offsetX,
    topBottomDistribute,
    vanDerCorput,
    vpPlot,
)
from vipor.core import _density


def test_offseting() -> None:
    assert np.array_equal(offsetX([1], [1]), [0])
    assert np.array_equal(offsetX([1] * 10, list(range(1, 11))), [0] * 10)
    with pytest.raises(ValueError, match="not the same length"):
        offsetX([1], [1, 2])
    assert len(np.unique(offsetX([1] * 100))) == 100
    assert np.array_equal(offsetX([1] * 100), offsetX([1] * 100))
    assert np.allclose(offsetX([1] * 100, width=0.2) * 2, offsetX([1] * 100))
    small_y = [1] * 10 + [2] * 1000
    small_x = [1] * 10 + [2] * 1000
    large_y = [1] * 1000 + [2] * 1000
    large_x = [1] * 1000 + [2] * 1000
    assert np.allclose(
        offsetX(small_y, small_x, varwidth=True)[:10] * 10,
        offsetX(large_y, large_x, varwidth=True)[:10],
    )


def test_single_group_offseting() -> None:
    assert np.array_equal(offsetSingleGroup([1]), [0])
    assert offsetSingleGroup([]).size == 0
    values = np.random.default_rng(1).normal(size=1000)
    assert np.array_equal(
        offsetSingleGroup(values), offsetSingleGroup(values, maxLength=1000)
    )
    assert np.allclose(
        offsetSingleGroup(values) * 10, offsetSingleGroup(values, maxLength=10)
    )
    assert np.allclose(
        offsetSingleGroup(values) / 10, offsetSingleGroup(values, maxLength=100000)
    )
    for method in (
        "quasirandom",
        "pseudorandom",
        "smiley",
        "frowney",
        "tukey",
        "tukeyDense",
    ):
        offsets = offsetSingleGroup(values[:100], method=method, random_state=1)
        assert np.max(offsets) <= 1
        assert np.min(offsets) >= -1
        assert len(offsetSingleGroup(values[:2], method=method, random_state=1)) == 2


def test_tukey_dense_is_jittered() -> None:
    """Regression test: ``method="tukeyDense"`` must jitter its texture, like
    R's ``vipor::tukeyTexture(y)`` (default ``jitter=TRUE``), instead of only
    recycling the 50 discrete Tukey texture levels verbatim.

    Previously the Python implementation passed ``jitter=method == "tukey"``
    to ``tukeyTexture``, which evaluates to ``False`` for ``"tukeyDense"``,
    collapsing what should be near-unique offsets into at most 50 discrete
    bands for groups larger than 50 observations. Using identical ``y``
    values pins the point-density scaling factor to a single constant so the
    banding is not masked by density-driven variation.
    """
    values = np.full(200, 5.0)
    offsets = offsetSingleGroup(values, method="tukeyDense", random_state=1)
    assert len(np.unique(offsets)) == len(values)




def test_ave_with_args() -> None:
    values = np.arange(1, 11)
    groups = [1, 2, 3, 4, 5] * 2
    assert np.array_equal(
        aveWithArgs(values, groups), [3.5, 4.5, 5.5, 6.5, 7.5] * 2
    )
    assert np.array_equal(
        aveWithArgs(values, groups, FUN=np.median), [3.5, 4.5, 5.5, 6.5, 7.5] * 2
    )
    assert np.array_equal(aveWithArgs(values), np.repeat(5.5, 10))


def test_top_bottom_distribution() -> None:
    values = np.arange(1, 11)
    assert np.array_equal(topBottomDistribute(values), topBottomDistribute(values + 100))
    assert np.array_equal(topBottomDistribute(-values), topBottomDistribute(values, True))
    assert np.array_equal(topBottomDistribute([1000]), [0.5])
    assert np.array_equal(np.sort(topBottomDistribute(values)[-2:]), [0, 1])
    assert np.array_equal(np.sort(topBottomDistribute(values, True)[:2]), [0, 1])
    assert np.array_equal(np.sort(topBottomDistribute(values, prop=False)[-2:]), [1, 10])
    assert np.array_equal(
        np.sort(topBottomDistribute(values, prop=False)[:5]), [3, 4, 5, 6, 7]
    )


def test_van_der_corput() -> None:
    assert np.allclose(vanDerCorput(3), [1 / 2, 1 / 4, 3 / 4])
    assert np.allclose(vanDerCorput(8), [1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 6 / 16, 14 / 16, 1 / 16])
    assert np.allclose(vanDerCorput(5, start=4), [1 / 8, 5 / 8, 6 / 16, 14 / 16, 1 / 16])
    assert np.allclose(vanDerCorput(3, 3), [1 / 3, 2 / 3, 1 / 9])
    assert len(np.unique(vanDerCorput(10000))) == 10000
    with pytest.raises(ValueError, match="base"):
        vanDerCorput(3, 1)
    with pytest.raises(ValueError, match="n "):
        vanDerCorput(-10)


def test_number_splitting_and_combining() -> None:
    assert number2digits(0).size == 0
    assert np.array_equal(number2digits(5, 4), [1, 1])
    assert np.array_equal(number2digits(255), np.ones(8))
    assert np.array_equal(number2digits(65535, 16), np.repeat(15, 4))
    assert digits2number([1, 1]) == 3
    assert digits2number([1, 1, 1]) == 7
    assert digits2number([1], base=2, fractional=True) == 0.5
    assert digits2number(20, base=2) == 20
    assert digits2number(number2digits(123456, 23), 23) == 123456
    with pytest.raises(ValueError, match="negative"):
        number2digits(-1)
    with pytest.raises(ValueError, match="base"):
        number2digits(10, 1)
    with pytest.raises(ValueError, match="digit"):
        digits2number([-1], 10)


def test_vp_plot_returns_x_positions() -> None:
    values = np.random.default_rng(1).normal(size=100)
    assert len(vpPlot(y=values)) == 100
    assert np.allclose(vpPlot(y=values), 1 + offsetX(values))


@pytest.mark.parametrize(
    "method",
    ["quasirandom", "pseudorandom", "maxout", "minout", "tukey", "tukeyDense"],
)
def test_offsets_match_upstream_r(
    r_vipor: Any, method: str
) -> None:
    values = np.array([-2, -1.5, -1, -0.2, 0, 0.1, 0.2, 1, 2, 3, 4], dtype=float)
    r_package, robjects = r_vipor
    r_values = r_package.offsetSingleGroup(
        robjects.FloatVector(values.tolist()), method=robjects.StrVector([method])
    )
    expected = np.asarray(r_values, dtype=float)
    actual = offsetSingleGroup(values, method=method, random_state=1)
    if method in {"pseudorandom", "tukey", "tukeyDense"}:
        # These upstream methods consume R's process-global random stream.
        # Compare their shape and finite point locations here; deterministic
        # methods below are compared point-for-point.
        assert actual.shape == expected.shape
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        return
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-4
    )


def test_pseudorandom_offsets_match_upstream_r_distribution(r_vipor: Any) -> None:
    """``method="pseudorandom"`` draws from R's ``stats::runif``/NumPy's
    ``Generator.random`` respectively, whose underlying algorithms differ, so
    individual offsets can't be compared point-for-point across many draws
    (see ``test_offsets_match_upstream_r`` above). Instead, check that both
    implementations draw offsets from the same bounded distribution: each
    offset must stay within `[-point_density, point_density]` (the same
    density-based scaling used by every method), and, aggregated over many
    random draws, the mean of all offsets should be close to zero as
    expected for symmetric uniform noise.
    """
    values = np.array([-2, -1.5, -1, -0.2, 0, 0.1, 0.2, 1, 2, 3, 4], dtype=float)
    r_package, robjects = r_vipor
    r_y = robjects.FloatVector(values.tolist())
    num_trials = 300

    r_offset_batches = []
    for seed in range(num_trials):
        robjects.r(f"set.seed({seed})")
        # Force a copy immediately: rpy2's zero-copy array view into R's
        # memory can be invalidated by the next R call in this loop.
        r_offset_batches.append(
            np.array(
                r_package.offsetSingleGroup(
                    r_y, method=robjects.StrVector(["pseudorandom"])
                ),
                dtype=float,
                copy=True,
            )
        )
    r_offsets = np.concatenate(r_offset_batches)

    python_offsets = np.concatenate(
        [
            offsetSingleGroup(values, method="pseudorandom", random_state=seed)
            for seed in range(num_trials)
        ]
    )

    max_magnitude = 1.0  # (offset - 0.5) * 2 in [-1, 1], point_density <= 1
    assert np.all(np.abs(r_offsets) <= max_magnitude + 1e-9)
    assert np.all(np.abs(python_offsets) <= max_magnitude + 1e-9)

    # Mean of many symmetric Uniform(-p, p) draws should be close to zero;
    # with num_trials * len(values) = 3300 draws the standard error of the
    # mean is small, so a generous tolerance still catches gross scaling or
    # sign bugs without being sensitive to which RNG produced the draws.
    assert r_offsets.mean() == pytest.approx(0.0, abs=0.1)
    assert python_offsets.mean() == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize(
    ("values", "nbins", "adjust"),
    [
        ([1.2, 1.4, 1.4, 2.1, 3.7, 5.2], 2, 0.5),
        ([1.2, 1.4, 1.4, 2.1, 3.7, 5.2], 32, 1),
        ([1.2, 1.4, 1.4, 2.1, 3.7, 5.2], 513, 2),
        ([0, 0, 0, 0], 1024, 1),
        ([5, 5, 5, 5], 1024, 0.5),
    ],
)
def test_density_matches_r_fft_kde(
    r_vipor: Any, values: list[float], nbins: int, adjust: float
) -> None:
    _, robjects = r_vipor
    expected = robjects.r["density"](
        robjects.FloatVector(values), n=nbins, adjust=adjust, kernel="gaussian"
    )
    expected_x = np.asarray(expected.rx2("x"), dtype=float)
    expected_y = np.asarray(expected.rx2("y"), dtype=float)
    expected_y /= expected_y.max()
    actual_x, actual_y = _density(np.asarray(values, dtype=float), nbins, adjust)
    np.testing.assert_allclose(actual_x, expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-12, atol=1e-12)
