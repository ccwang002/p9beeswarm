import numpy as np

from plotnine_beeswarm import beeswarm, quasirandom, sina
from vipor import offsetSingleGroup


def test_beeswarm_is_reproducible_and_preserves_missing_values():
    values = np.array([1, 1, 1, np.nan])
    offsets = beeswarm(values)
    assert np.isnan(offsets[-1])
    assert np.all(np.isfinite(offsets[:-1]))
    assert np.array_equal(offsets, beeswarm(values), equal_nan=True)


def test_quasirandom_is_bounded_and_reproducible():
    values = np.arange(10)
    offsets = quasirandom(values)
    assert np.max(np.abs(offsets)) <= 0.4
    assert np.array_equal(offsets, quasirandom(values))


def test_sina_scales_offsets():
    offsets = sina(np.arange(20))
    assert np.max(np.abs(offsets)) <= 0.4
    assert len(offsets) == 20


def test_quasirandom_distribution_methods_are_distinct():
    values = np.array([0, 0.1, 0.2, 0.3, 1, 1.1, 1.2, 2, 3, 4])
    frowney = quasirandom(values, method="frowney")
    smiley = quasirandom(values, method="smiley")
    assert not np.array_equal(frowney, smiley)


def test_quasirandom_delegates_density_distribution_to_vipor():
    values = np.array([-2, -1.5, -1, -0.2, 0, 0.1, 0.2, 1, 2, 3, 4])
    for method in ("quasirandom", "maxout", "minout", "smiley", "frowney", "tukey"):
        expected = offsetSingleGroup(values, method=method, adjust=0.5) * 0.4
        actual = quasirandom(
            values, method=method, bandwidth=0.5, random_state=123
        )
        np.testing.assert_allclose(actual, expected)


def test_quasirandom_forwards_nbins_and_varwidth_to_vipor():
    values = np.arange(10, dtype=float)
    offsets = quasirandom(
        values,
        nbins=32,
        varwidth=True,
        max_length=40,
        random_state=123,
    )
    expected = (
        offsetSingleGroup(
            values,
            nbins=32,
            adjust=0.5,
            maxLength=40,
            random_state=123,
        )
        * 0.4
    )
    np.testing.assert_allclose(offsets, expected)


def test_beeswarm_gutter_uses_upstream_half_width():
    offsets = beeswarm(np.zeros(200), corral="gutter", corral_width=0.9)
    assert np.nanmax(np.abs(offsets)) <= 0.45
    assert np.isclose(np.nanmax(np.abs(offsets)), 0.45)


def test_beeswarm_side_and_corral_follow_upstream_ranges():
    values = np.zeros(200)
    left = beeswarm(values, side=-1, corral="gutter", corral_width=0.9)
    right = beeswarm(values, side=1, corral="gutter", corral_width=0.9)
    assert np.nanmin(left) >= -0.9
    assert np.nanmax(left) <= 0
    assert np.nanmin(right) >= 0
    assert np.nanmax(right) <= 0.9
