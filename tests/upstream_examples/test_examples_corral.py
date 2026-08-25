"""README examples for corralling runaway beeswarm points."""

from __future__ import annotations

import pytest
from plotnine import aes, ggplot, ggtitle

from p9beeswarm import geom_beeswarm


@pytest.fixture(scope="module")
def corral_data():
    try:
        from rpy2 import robjects
        from rpy2.rinterface_lib.embedded import RRuntimeError
        from rpy2.robjects import conversion, default_converter, pandas2ri
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"R and rpy2 are unavailable: {error}")

    try:
        with conversion.localconverter(default_converter + pandas2ri.converter):
            return conversion.rpy2py(
                robjects.r(
                    """
                    set.seed(1995)
                    data.frame(
                        y = rnorm(1000),
                        id = sample(c("G1", "G2", "G3"), size = 1000, replace = TRUE)
                    )
                    """
                )
            )
    except (RRuntimeError, RuntimeError, TypeError, ValueError) as error:
        pytest.skip(f"R corral data generation is unavailable: {error}")


@pytest.mark.parametrize("corral", ["none", "gutter", "wrap", "random", "omit"])
def test_readme_beeswarm_corral(corral, corral_data, assert_plot):
    plot = (
        ggplot(corral_data, aes("id", "y", color="id"))
        + geom_beeswarm(
            cex=2.5, corral=corral, corral_width=0.9, random_state=1995
        )
        + ggtitle(f'corral = "{corral}"')
    )
    assert_plot(plot, f"beeswarm_corral_{corral}")
