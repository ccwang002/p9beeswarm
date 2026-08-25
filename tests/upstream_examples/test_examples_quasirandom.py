"""README examples for the alternative quasirandom methods."""

import pytest
from plotnine import ggtitle

from p9beeswarm import geom_quasirandom
from tests.example_helpers import base_plot, iris_data

pytestmark = pytest.mark.image


@pytest.mark.parametrize(
    ("name", "method", "title"),
    [
        ("quasirandom_tukey", "tukey", "Tukey texture"),
        ("quasirandom_tukey_dense", "tukeyDense", "Tukey + density"),
        ("quasirandom_frowney", "frowney", "Banded frowns"),
        ("quasirandom_smiley", "smiley", "Banded smiles"),
        ("quasirandom_pseudorandom", "pseudorandom", "Jittered density"),
    ],
    ids=[
        "quasirandom_tukey",
        "quasirandom_tukey_dense",
        "quasirandom_frowney",
        "quasirandom_smiley",
        "quasirandom_pseudorandom",
    ],
)
def test_readme_quasirandom_method(name, method, title, assert_plot):
    random_state = (
        12345 if method in {"tukey", "tukeyDense", "pseudorandom"} else None
    )
    plot = (
        base_plot(iris_data(), "Species", "Sepal.Length")
        + geom_quasirandom(method=method, random_state=random_state)
        + ggtitle(title)
    )
    assert_plot(plot, name)
