import pytest

from plotnine_beeswarm.examples import readme_examples


@pytest.mark.parametrize("name, plot", readme_examples(), ids=lambda item: item)
def test_readme_example(name, plot, assert_plot):
    assert_plot(plot, name)
