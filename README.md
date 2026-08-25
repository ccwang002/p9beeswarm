# p9beeswarm

`p9beeswarm` provides beeswarm-style geoms for [plotnine](https://plotnine.org/), modelled after the R [ggbeeswarm](https://github.com/eclarke/ggbeeswarm) package.

```python
import plotnine as p9
from plotnine.data import penguins
from p9beeswarm import geom_quasirandom

penguins_plot = (
    p9.ggplot(penguins.dropna(subset=["species", "body_mass_g"]),
              p9.aes("species", "body_mass_g", color="species"))
    + geom_quasirandom(width=0.35)
    + p9.labs(
        title="Penguin body mass by species",
        x="Species",
        y="Body mass (g)",
    )
    + p9.theme_bw()
)
```

The package is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv run --extra test pytest
```

The [side-by-side ggbeeswarm examples](docs/ggbeeswarm-examples.qmd) recreate
the upstream README plots with both the Python and R implementations. Render
them with:

```sh
quarto render docs/ggbeeswarm-examples.qmd
```

The public Python compatibility module for the upstream
[vipor](https://github.com/sherrillmix/vipor) helpers is available as
`vipor`. It includes `offsetX`, `offsetSingleGroup`, the Tukey distribution
helpers, and the van der Corput utilities used by ggbeeswarm:

```python
from vipor import offsetX

offsets = offsetX(values, groups, method="quasirandom")
```

Tests mirroring vipor's upstream testthat suite are kept separately in
[tests/upstream_vipor](tests/upstream_vipor). The R-backed compatibility
checks use `rpy2` when the R vipor package is installed. Type-check the module
and those tests with:

```sh
uv run --extra test mypy src/vipor tests/upstream_vipor
```

To enable the optional R-backed comparisons locally, install the `r-test`
extra in an environment with R and the upstream vipor package:

```sh
uv run --extra test --extra r-test pytest tests/upstream_vipor
```
