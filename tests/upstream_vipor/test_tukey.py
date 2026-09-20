"""Tests corresponding to vipor's upstream ``test_tukey.R``."""

from math import factorial
from typing import Any

import numpy as np
import pytest

from vipor import (
    generatePermuteString,
    permute,
    tukeyPermutes,
    tukeyT,
    tukeyTexture,
)


def test_permuting() -> None:
    assert permute(()) is None
    assert permute((1,)) == [(1,)]
    assert permute((1, 2)) == [(1, 2), (2, 1)]
    assert permute(("a", "b")) == [("a", "b"), ("b", "a")]
    assert len(permute(tuple(range(1, 6))) or []) == factorial(5)


def test_tukey_permutations() -> None:
    assert len(tukeyPermutes()) == 32
    assert len(tukeyPermutes(5, 2)) == 32
    assert len(tukeyPermutes(3, 2)) == 4
    assert len(tukeyPermutes(3, 3)) == 6
    for candidate in tukeyPermutes(6):
        assert not (candidate[0] < candidate[1] < candidate[2])
        assert not (candidate[1] < candidate[2] < candidate[3])
        assert not (candidate[4] > candidate[3] > candidate[2])
        assert not (candidate[5] > candidate[4] > candidate[3])


def test_tukey_permutation_string() -> None:
    generated = generatePermuteString(20, random_state=1)
    assert np.array_equal(np.sort(generated), np.repeat(np.arange(1, 6), 20))
    generated = generatePermuteString(10, 7, random_state=1)
    assert np.array_equal(np.sort(generated), np.repeat(np.arange(1, 8), 10))
    for generated in (
        generatePermuteString(20, 7, random_state=1),
        generatePermuteString(100, 5, random_state=1),
    ):
        directions = np.diff(generated) > 0
        boundaries = np.r_[0, np.flatnonzero(directions[1:] != directions[:-1]) + 1, len(directions)]
        assert np.diff(boundaries).max() < 3


def test_tukey_offset_positions() -> None:
    expected = np.repeat(np.arange(1, 98, 4), 2)
    assert np.array_equal(np.sort(tukeyT(random_state=1)), expected)
    assert np.array_equal(np.sort(tukeyT(10, 5, random_state=1)), expected)
    assert np.array_equal(
        np.sort(tukeyT(20, 5, random_state=1)), np.repeat(np.arange(1, 98, 4), 4)
    )
    assert len(tukeyT(10, 6, random_state=1)) == 60


def test_tukey_texture() -> None:
    assert len(tukeyTexture(np.arange(200), random_state=1)) == 200
    assert len(tukeyTexture(np.arange(1234), random_state=1)) == 1234
    texture = tukeyTexture(np.arange(1234), random_state=1)
    assert np.max(texture) <= 100
    assert np.min(texture) >= 0
    assert len(np.unique(tukeyTexture(np.arange(100), random_state=1))) == 100
    assert len(np.unique(tukeyTexture(np.arange(100), jitter=False, random_state=1))) == 50
    thin_values = tukeyTexture(
        np.array([-100, *range(1, 101), 101.1]),
        delta=1,
        thin=True,
        random_state=1,
    )
    assert np.array_equal(thin_values[[0, 101]], [50, 50])
    assert np.array_equal(
        tukeyTexture(np.arange(1, 101), delta=0.9, thin=True, random_state=1),
        np.repeat(50.0, 100),
    )
    hollow = tukeyTexture(
        np.array([1, 2, 101, 102]), delta=10, hollow=True, random_state=1
    )
    assert np.array_equal(np.min(hollow), 0)
    assert np.array_equal(np.max(hollow), 100)


@pytest.mark.parametrize("n", [51, 100, 150, 234])
def test_tukey_texture_recycles_base_pattern_with_boost(n: int) -> None:
    """Regression test for the base 50-value texture (with its "+2" boost on
    the 26th-50th values) being recycled as a whole for inputs longer than
    50, matching R's ``vipor::tukeyTexture`` (``offset[26:50] <- offset[26:50] + 2``
    applied *before* ``rep(offset, length.out = n)``).

    Previously, ``+2`` was applied only once, to the first 50 output values,
    so subsequent 50-value cycles were missing the boost.
    """
    texture = tukeyTexture(np.arange(n), jitter=False, thin=False, random_state=1)
    full_cycles = n // 50
    for cycle in range(1, full_cycles):
        assert np.array_equal(texture[:50], texture[cycle * 50 : (cycle + 1) * 50])


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8])
def test_tukey_texture_accepts_small_inputs(n: int) -> None:
    assert len(tukeyTexture(np.arange(n), random_state=1)) == n


# ``tukeyTexture``'s random permutation choices (``tukeyT`` -> ``generatePermuteString``
# -> R's ``sample()``) and jitter (``stats::runif``) draw from R's own RNG
# algorithms, which differ from NumPy's, so per-seed outputs are not expected
# to match value-for-value across the two implementations. The tests below
# instead compare structural invariants that must hold regardless of which
# random draws were made, calling the real R `vipor` package via rpy2.


@pytest.mark.parametrize("n", [50, 100, 250])
def test_tukey_texture_recycled_sum_matches_upstream_r(r_vipor: Any, n: int) -> None:
    """The un-jittered, un-thinned texture is always the same 50-value
    pattern (25 values from ``tukeyT()`` doubled, with a "+2" boost applied
    to exactly 25 of them) recycled to length ``n``, so its sum is a fixed
    multiple of 2500 regardless of which permutations the RNG selects. Check
    this invariant holds identically for both R and this package's output.
    """
    r_package, robjects = r_vipor
    expected_sum = 2500.0 * (n / 50)

    robjects.r("set.seed(1)")
    r_values = r_package.tukeyTexture(
        robjects.FloatVector(range(1, n + 1)), jitter=False, thin=False
    )
    r_texture = np.asarray(r_values, dtype=float)
    assert r_texture.sum() == pytest.approx(expected_sum)

    python_texture = tukeyTexture(
        np.arange(1, n + 1), jitter=False, thin=False, random_state=1
    )
    assert python_texture.sum() == pytest.approx(expected_sum)


def test_tukey_texture_thin_forces_isolated_points_to_match_upstream_r(
    r_vipor: Any,
) -> None:
    """Points sufficiently isolated from their neighbours (per ``delta``) are
    forced to the texture's middle value (50) by ``thin=True``, regardless of
    which random Tukey permutations were otherwise drawn. Verify this against
    a real call into R's ``vipor::tukeyTexture``.
    """
    values = [-100.0, *(float(v) for v in range(1, 101)), 101.1]
    r_package, robjects = r_vipor

    robjects.r("set.seed(1)")
    r_values = r_package.tukeyTexture(
        robjects.FloatVector(values), jitter=False, thin=True, delta=1.0
    )
    r_texture = np.asarray(r_values, dtype=float)
    assert np.array_equal(r_texture[[0, -1]], [50.0, 50.0])

    python_texture = tukeyTexture(
        np.asarray(values), jitter=False, thin=True, delta=1.0, random_state=1
    )
    assert np.array_equal(python_texture[[0, -1]], [50.0, 50.0])


@pytest.mark.parametrize("n", [10, 100, 1234])
def test_tukey_texture_shape_and_range_match_upstream_r(r_vipor: Any, n: int) -> None:
    """Regardless of RNG draws, both implementations must return one texture
    value per input point, bounded to the documented ``[0, 100]`` range.
    """
    r_package, robjects = r_vipor
    r_input = robjects.FloatVector(range(n))

    robjects.r("set.seed(1)")
    r_texture = np.asarray(r_package.tukeyTexture(r_input), dtype=float)
    assert r_texture.shape == (n,)
    assert np.all((r_texture >= 0) & (r_texture <= 100))

    python_texture = tukeyTexture(np.arange(n), random_state=1)
    assert python_texture.shape == (n,)
    assert np.all((python_texture >= 0) & (python_texture <= 100))
