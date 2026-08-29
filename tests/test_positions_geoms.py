import numpy as np
import pandas as pd
from plotnine import aes, ggplot
from plotnine.scales import scale_x_continuous, scale_x_discrete, scale_y_continuous
from sklearn.datasets import load_iris

from p9beeswarm.beeswarm import determine_pos, swarmx
from p9beeswarm.geoms import geom_beeswarm, geom_quasirandom, geom_sina
from p9beeswarm.positions import get_range, position_beeswarm, position_quasirandom
from vipor import offsetSingleGroup


def _data():
    return pd.DataFrame(
        {
            "x": [1, 1, 1, 1, 2, 2],
            "y": [0.0, 0.1, 0.2, 0.3, 0.0, 0.4],
            "group": [1, 1, 2, 2, 1, 2],
        }
    )


def test_get_range_matches_upstream_scale_rules():
    discrete = scale_x_discrete(limits=["a", "b", "a"])
    continuous = scale_y_continuous(limits=(2, 8))
    zero = scale_x_continuous(limits=(5, 5))

    assert get_range(discrete) == 2
    assert get_range(continuous) == 6
    assert get_range(zero) == 1


def test_determine_pos_matches_upstream_layout_rules():
    rows = np.array([1, 1, 1, 2, 2, 2])
    np.testing.assert_allclose(determine_pos(rows, "center", 0), [-1, 0, 1, -1, 0, 1])
    np.testing.assert_allclose(determine_pos(rows, "square", 0), [-1, 0, 1, -1, 0, 1])
    np.testing.assert_allclose(determine_pos(rows, "hex", 0), [-1.25, -0.25, 0.75, -0.75, 0.25, 1.25])
    np.testing.assert_allclose(determine_pos(rows, "square", -1), [-2, -1, 0, -2, -1, 0])
    np.testing.assert_allclose(determine_pos(rows, "hex", 1), [0, 1, 2, 0.5, 1.5, 2.5])


def test_positions_share_orientation_and_preserve_missing_values():
    data = _data()
    data.loc[1, "y"] = np.nan
    params = position_beeswarm(orientation="x").setup_params(data)
    result = position_beeswarm(orientation="x").compute_panel(data, None, params)
    assert np.isnan(result.loc[1, "x"])
    assert result["x"].dropna().shape == (len(data) - 1,)

    horizontal = position_beeswarm(orientation="y").compute_panel(
        data.fillna({"y": 0.1}), None, position_beeswarm(orientation="y").setup_params(data)
    )
    assert np.allclose(horizontal["y"], data.fillna({"y": 0.1})["y"])


def test_dodge_and_corral_are_applied_by_the_position():
    data = _data()
    dodged = position_beeswarm(
        width=0.2, dodge_width=0.8, corral="gutter", corral_width=0.4
    )
    result = dodged.compute_panel(data, None, dodged.setup_params(data))
    category_one = data["x"] == 1
    assert result.loc[category_one & (data["group"] == 1), "x"].mean() < 1
    assert result.loc[category_one & (data["group"] == 2), "x"].mean() > 1
    assert np.abs(result["x"] - data["x"]).le(0.200001).all()


def test_beeswarm_forwards_algorithm_method():
    position = position_beeswarm(method="swarm", width=0.2)
    data = pd.DataFrame({"x": 1, "y": [0.0, 0.0, 0.0]})
    result = position.compute_panel(data, None, position.setup_params(data))
    assert result["x"].nunique() > 1


def test_beeswarm_default_uses_panel_scale_sizes_for_documented_iris_example():
    dataset = load_iris()
    iris = pd.DataFrame(
        dataset.data,
        columns=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    ).assign(Species=[dataset.target_names[index] for index in dataset.target])

    plot = ggplot(iris, aes("Species", "Sepal.Length")) + geom_beeswarm()
    plot._build()
    result = plot.layers[0].data

    expected = np.empty(len(iris))
    for center, (_, group) in enumerate(iris.groupby("Species", sort=True), start=1):
        expected[group.index] = center + swarmx(
            0,
            group["Sepal.Length"],
            x_size=get_range(plot.layout.panel_params[0].x.scale) / 100,
            y_size=get_range(plot.layout.panel_params[0].y.scale) / 100,
        ).x

    assert plot.layers[0].position.params["method"] == "swarm"
    np.testing.assert_allclose(result["x"], expected)


def test_quasirandom_forwards_vipor_parameters_and_varwidth():
    values = np.arange(8, dtype=float)
    position = position_quasirandom(
        width=0.4, varwidth=True, bandwidth=0.5, nbins=32, method="tukey"
    )
    data = pd.DataFrame({"x": 1, "y": values})
    actual = position.compute_panel(data, None, position.setup_params(data))
    expected = offsetSingleGroup(
        values, maxLength=8, method="tukey", nbins=32, adjust=0.5
    ) * 0.4
    np.testing.assert_allclose(actual["x"].to_numpy() - 1, expected)


def test_quasirandom_default_width_uses_dodged_group_spacing():
    data = pd.DataFrame(
        {
            "x": [1.0] * 6,
            "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "group": [1, 1, 2, 2, 3, 3],
        }
    )
    position = position_quasirandom(dodge_width=1)
    result = position.compute_panel(data, None, position.setup_params(data))

    # Three groups are dodged to 2/3, 1, and 4/3. Like ggbeeswarm, infer the
    # default width from their 1/3 spacing, not the original unit spacing.
    centers = {1: 2 / 3, 2: 1.0, 3: 4 / 3}
    for group, center in centers.items():
        rows = data["group"] == group
        expected = offsetSingleGroup(data.loc[rows, "y"], adjust=0.5) * (0.4 / 3)
        np.testing.assert_allclose(result.loc[rows, "x"] - center, expected)


def test_geoms_use_matching_position_objects_and_build():
    for geom, position_name in (
        (geom_beeswarm(), "position_beeswarm"),
        (geom_quasirandom(varwidth=True), "position_quasirandom"),
        (geom_sina(), "_SinaPosition"),
    ):
        assert type(geom._position).__name__ == position_name
        plot = ggplot(_data(), aes("x", "y")) + geom
        plot._build()
