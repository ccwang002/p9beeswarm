"""Tests corresponding to vipor's upstream ``test_tukey.R``."""

from math import factorial

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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8])
def test_tukey_texture_accepts_small_inputs(n: int) -> None:
    assert len(tukeyTexture(np.arange(n), random_state=1)) == n
