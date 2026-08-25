"""README examples for the basic, alternative, and priority beeswarm sections."""

import pytest
from tests.example_helpers import (
    alternative_data,
    base_plot,
    iris_data,
    priority_data,
    sub_mpg,
)
from plotnine import aes, ggtitle, scale_x_continuous, scale_y_discrete
from plotnine.data import mpg

from p9beeswarm import geom_beeswarm


@pytest.mark.parametrize(
    ("name", "plot"),
    [
        (
            "beeswarm_default",
            base_plot(iris_data(), "Species", "Sepal.Length")
            + geom_beeswarm()
            + ggtitle("Beeswarm"),
        ),
        (
            "beeswarm_one_sided",
            base_plot(iris_data(), "Species", "Sepal.Length") + geom_beeswarm(side=1),
        ),
        ("beeswarm_mpg", base_plot(mpg, "class", "hwy") + geom_beeswarm(size=0.5)),
        (
            "beeswarm_categorical_y",
            base_plot(mpg, "hwy", "class") + geom_beeswarm(size=0.5),
        ),
        (
            "beeswarm_categorical_y_expanded",
            base_plot(mpg, "hwy", "class")
            + geom_beeswarm(size=0.5)
            + scale_y_discrete(expand=(0, 0, 0.5, 1)),
        ),
        (
            "beeswarm_large_points",
            base_plot(mpg, "class", "hwy") + geom_beeswarm(size=1.1),
        ),
        (
            "beeswarm_dodge",
            base_plot(sub_mpg(), "class", "displ")
            + geom_beeswarm(aes(color="factor(cyl)"), dodge_width=0.5),
        ),
    ],
    ids=[
        "beeswarm_default",
        "beeswarm_one_sided",
        "beeswarm_mpg",
        "beeswarm_categorical_y",
        "beeswarm_categorical_y_expanded",
        "beeswarm_large_points",
        "beeswarm_dodge",
    ],
)
def test_readme_beeswarm_example(name, plot, assert_plot):
    assert_plot(plot, name)


@pytest.mark.parametrize(
    "method",
    ["swarm", "compactswarm", "hex", "square", "center"],
)
def test_readme_beeswarm_method(method, assert_plot):
    plot = (
        base_plot(alternative_data(), "x", "y")
        + geom_beeswarm(cex=2.5, method=method)
        + ggtitle(f'method = "{method}"')
    )
    assert_plot(plot, f"beeswarm_method_{method}")


@pytest.mark.parametrize("priority", ["ascending", "descending", "density", "random"])
def test_readme_beeswarm_priority(priority, assert_plot):
    plot = (
        base_plot(priority_data(), "x", "y")
        + geom_beeswarm(cex=2, priority=priority, random_state=12345)
        + ggtitle(priority.title())
        + scale_x_continuous(expand=(0, 0, 0.5, 0.5))
    )
    assert_plot(plot, f"beeswarm_priority_{priority}")
