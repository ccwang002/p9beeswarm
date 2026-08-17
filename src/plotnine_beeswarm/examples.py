"""Plotnine translations of the examples in ggbeeswarm's README."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd
from plotnine import aes, ggplot, ggtitle, scale_x_continuous, scale_y_discrete
from plotnine.data import mpg

from plotnine_beeswarm import geom_beeswarm, geom_quasirandom


def _iris_data():
    from sklearn.datasets import load_iris

    dataset = load_iris()
    return pd.DataFrame(
        dataset.data,
        columns=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    ).assign(Species=[dataset.target_names[index] for index in dataset.target])


def _base(data, x, y):
    return ggplot(data, aes(x, y))


def _sub_mpg():
    return mpg[mpg["class"].isin(["midsize", "pickup", "suv"])]


def _priority_data():
    rng = np.random.default_rng(12345)
    x = np.repeat([1, 2, 3], [20, 40, 80])
    return pd.DataFrame({"x": x, "y": rng.normal(x, 1)})


def _alternative_data():
    rng = np.random.default_rng(12345)
    return pd.DataFrame({"x": "A", "y": rng.choice(np.arange(1, 101), 200)})


def _corral_data():
    return pd.read_csv(files("plotnine_beeswarm.data").joinpath("corral.csv"))


def readme_examples():
    """Return every README plot as ``(name, plot)`` pairs."""
    iris = _iris_data()
    examples = [
        ("compare_jitter", _base(iris, "Species", "Sepal.Length") + geom_quasirandom()),
        ("quasirandom_default", _base(mpg, "class", "hwy") + geom_quasirandom()),
        (
            "quasirandom_categorical_y",
            _base(mpg, "hwy", "class") + geom_quasirandom(group_on_x=False),
        ),
        (
            "quasirandom_varwidth",
            _base(mpg, "class", "hwy") + geom_quasirandom(varwidth=True),
        ),
        (
            "quasirandom_dodge",
            _base(_sub_mpg(), "class", "displ")
            + geom_quasirandom(aes(color="factor(cyl)"), dodge_width=1),
        ),
        (
            "quasirandom_tukey",
            _base(iris, "Species", "Sepal.Length")
            + geom_quasirandom(method="tukey", random_state=12345)
            + ggtitle("Tukey texture"),
        ),
        (
            "quasirandom_tukey_dense",
            _base(iris, "Species", "Sepal.Length")
            + geom_quasirandom(method="tukeyDense", random_state=12345)
            + ggtitle("Tukey + density"),
        ),
        (
            "quasirandom_frowney",
            _base(iris, "Species", "Sepal.Length")
            + geom_quasirandom(method="frowney")
            + ggtitle("Banded frowns"),
        ),
        (
            "quasirandom_smiley",
            _base(iris, "Species", "Sepal.Length")
            + geom_quasirandom(method="smiley")
            + ggtitle("Banded smiles"),
        ),
        (
            "quasirandom_pseudorandom",
            _base(iris, "Species", "Sepal.Length")
            + geom_quasirandom(method="pseudorandom", random_state=12345)
            + ggtitle("Jittered density"),
        ),
        (
            "beeswarm_default",
            _base(iris, "Species", "Sepal.Length")
            + geom_beeswarm()
            + ggtitle("Beeswarm"),
        ),
        (
            "beeswarm_one_sided",
            _base(iris, "Species", "Sepal.Length") + geom_beeswarm(side=1),
        ),
        ("beeswarm_mpg", _base(mpg, "class", "hwy") + geom_beeswarm(size=0.5)),
        (
            "beeswarm_categorical_y",
            _base(mpg, "hwy", "class") + geom_beeswarm(size=0.5),
        ),
        (
            "beeswarm_categorical_y_expanded",
            _base(mpg, "hwy", "class")
            + geom_beeswarm(size=0.5)
            + scale_y_discrete(expand=(0, 0, 0.5, 1)),
        ),
        ("beeswarm_large_points", _base(mpg, "class", "hwy") + geom_beeswarm(size=1.1)),
        (
            "beeswarm_dodge",
            _base(_sub_mpg(), "class", "displ")
            + geom_beeswarm(aes(color="factor(cyl)"), dodge_width=0.5),
        ),
    ]
    for method in ("swarm", "compactswarm", "hex", "square", "center"):
        examples.append(
            (
                f"beeswarm_method_{method}",
                _base(_alternative_data(), "x", "y")
                + geom_beeswarm(cex=2.5, method=method)
                + ggtitle(f'method = "{method}"'),
            )
        )
    priority_data = _priority_data()
    for priority in ("ascending", "descending", "density", "random"):
        examples.append(
            (
                f"beeswarm_priority_{priority}",
                _base(priority_data, "x", "y")
                + geom_beeswarm(cex=2, priority=priority, random_state=12345)
                + ggtitle(priority.title())
                + scale_x_continuous(expand=(0, 0, 0.5, 0.5)),
            )
        )
    corral_data = _corral_data()
    for corral in ("none", "gutter", "wrap", "random", "omit"):
        examples.append(
            (
                f"beeswarm_corral_{corral}",
                ggplot(corral_data, aes("id", "y", color="id"))
                + geom_beeswarm(
                    cex=2.5, corral=corral, corral_width=0.9, random_state=1995
                )
                + ggtitle(f'corral = "{corral}"'),
            )
        )
    return examples
