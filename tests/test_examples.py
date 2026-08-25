"""README examples for the introductory and basic quasirandom sections."""

import pytest
from plotnine import aes
from plotnine.data import mpg

from p9beeswarm import geom_quasirandom

from .example_helpers import base_plot, iris_data, sub_mpg


@pytest.mark.parametrize(
    ("name", "plot"),
    [
        (
            "compare_jitter",
            base_plot(iris_data(), "Species", "Sepal.Length")
            + geom_quasirandom(),
        ),
        (
            "quasirandom_default",
            base_plot(mpg, "class", "hwy") + geom_quasirandom(),
        ),
        (
            "quasirandom_categorical_y",
            base_plot(mpg, "hwy", "class") + geom_quasirandom(group_on_x=False),
        ),
        (
            "quasirandom_varwidth",
            base_plot(mpg, "class", "hwy") + geom_quasirandom(varwidth=True),
        ),
        (
            "quasirandom_dodge",
            base_plot(sub_mpg(), "class", "displ")
            + geom_quasirandom(aes(color="factor(cyl)"), dodge_width=1),
        ),
    ],
    ids=[
        "compare_jitter",
        "quasirandom_default",
        "quasirandom_categorical_y",
        "quasirandom_varwidth",
        "quasirandom_dodge",
    ],
)
def test_readme_quasirandom_example(name, plot, assert_plot):
    assert_plot(plot, name)
