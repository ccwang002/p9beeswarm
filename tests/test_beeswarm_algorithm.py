import numpy as np
import pytest

from p9beeswarm.beeswarm import beeswarm, quasirandom, sina, swarmx


def test_swarmx_returns_upstream_style_columns_and_checks_collisions():
    result = swarmx(np.zeros(3), np.zeros(3), x_size=1, y_size=1, fast=False)
    np.testing.assert_allclose(result.x, [0, 1, -1])
    np.testing.assert_allclose(result["y"], np.zeros(3))


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("side, expected", [(1, [0, 1, 2]), (-1, [0, -1, -2])])
def test_swarmx_one_sided_layout(compact, side, expected):
    result = swarmx(np.zeros(3), np.zeros(3), x_size=1, y_size=1, side=side, compact=compact)
    np.testing.assert_allclose(result.x, expected)


@pytest.mark.parametrize("priority", ["ascending", "descending", "density", "random", "none"])
def test_priority_modes_are_deterministic(priority):
    values = np.array([0.0, 0.01, 0.02, 1.0])
    first = swarmx(0, values, x_size=0.5, y_size=1, priority=priority, random_state=42)
    second = swarmx(0, values, x_size=0.5, y_size=1, priority=priority, random_state=42)
    np.testing.assert_array_equal(first.x, second.x)


@pytest.mark.parametrize("corral", ["gutter", "wrap", "random", "omit"])
def test_corral_modes_control_runaway_points(corral):
    result = swarmx(
        0,
        np.zeros(20),
        x_size=1,
        y_size=1,
        corral=corral,
        corral_width=0.9,
        random_state=3,
    )
    if corral == "omit":
        assert np.isnan(result.x).sum() > 0
    else:
        assert np.nanmin(result.x) >= -0.45
        assert np.nanmax(result.x) <= 0.45


def test_swarmx_preserves_missing_values_and_supports_log_data():
    result = swarmx(10, [1, np.nan, 10], x_size=1, y_size=1, log="y")
    assert np.isnan(result.x[1])
    np.testing.assert_array_equal(result.y, [1, np.nan, 10])
    np.testing.assert_allclose(
        swarmx(10, [1, 1], x_size=1, y_size=1, log="x").x, [10, 100]
    )


def test_beeswarm_methods_and_vipor_helpers():
    values = np.array([0.0, 0.0, 1.0, np.nan])
    centered = beeswarm(values, method="center")
    np.testing.assert_allclose(centered[:2], [-0.2, 0.2])
    assert np.isnan(centered[-1])
    assert np.isnan(quasirandom(values)[-1])
    assert np.isnan(sina(values)[-1])


def test_invalid_swarm_arguments_fail_loudly():
    with pytest.raises(ValueError, match="priority"):
        swarmx(0, [1, 2], priority="invalid")
    with pytest.raises(ValueError, match="side"):
        swarmx(0, [1], side=2)
