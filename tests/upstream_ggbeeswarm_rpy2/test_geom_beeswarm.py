"""Built-coordinate parity for ggbeeswarm's documented Iris example."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, ggplot
from sklearn.datasets import load_iris

from p9beeswarm import geom_beeswarm


@pytest.fixture(scope="module")
def r_ggbeeswarm() -> Any:
    try:
        from rpy2 import robjects  # type: ignore[import-not-found]
        from rpy2.robjects import packages  # type: ignore[import-not-found]
        from rpy2.robjects.packages import (  # type: ignore[import-not-found]
            PackageNotInstalledError,
        )
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"R and rpy2 are unavailable: {error}")
    try:
        packages.importr("ggbeeswarm")
        packages.importr("ggplot2")
    except (PackageNotInstalledError, RuntimeError) as error:
        pytest.skip(f"R ggbeeswarm dependencies are unavailable: {error}")
    return robjects


def test_documented_iris_geom_beeswarm_coordinates_match_upstream(
    r_ggbeeswarm: Any,
) -> None:
    dataset: Any = load_iris()
    iris = pd.DataFrame(
        dataset.data,
        columns=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    ).assign(Species=[dataset.target_names[index] for index in dataset.target])

    expected_x = np.asarray(
        r_ggbeeswarm.r(
            """
            ggplot2::ggplot_build(
              ggplot2::ggplot(datasets::iris, ggplot2::aes(Species, Sepal.Length)) +
                ggbeeswarm::geom_beeswarm()
            )$data[[1]]$x
            """
        ),
        dtype=float,
    )
    expected_y = np.asarray(
        r_ggbeeswarm.r(
            """
            ggplot2::ggplot_build(
              ggplot2::ggplot(datasets::iris, ggplot2::aes(Species, Sepal.Length)) +
                ggbeeswarm::geom_beeswarm()
            )$data[[1]]$y
            """
        ),
        dtype=float,
    )

    plot = ggplot(iris, aes("Species", "Sepal.Length")) + geom_beeswarm()
    plot._build()
    actual = plot.layers[0].data

    np.testing.assert_allclose(actual["x"], expected_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual["y"], expected_y, rtol=0, atol=1e-12)
