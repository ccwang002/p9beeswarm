import numpy as np

from plotnine_beeswarm import beeswarm, quasirandom, sina


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


def test_beeswarm_gutter_uses_upstream_half_width():
    offsets = beeswarm(np.zeros(200), corral="gutter", corral_width=0.9)
    assert np.nanmax(np.abs(offsets)) <= 0.45
    assert np.isclose(np.nanmax(np.abs(offsets)), 0.45)
