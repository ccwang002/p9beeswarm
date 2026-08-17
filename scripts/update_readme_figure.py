"""Regenerate the penguins figure embedded in the README."""

from pathlib import Path

import plotnine as p9
from plotnine.data import penguins

from plotnine_beeswarm import geom_quasirandom

ROOT = Path(__file__).resolve().parents[1]


def main():
    penguins_plot = (
        p9.ggplot(
            penguins.dropna(subset=["species", "body_mass_g"]),
            p9.aes("species", "body_mass_g", color="species"),
        )
        + geom_quasirandom(width=0.35)
        + p9.labs(
            title="Penguin body mass by species",
            x="Species",
            y="Body mass (g)",
        )
        + p9.theme_bw()
    )
    output = ROOT / "docs" / "penguins-beeswarm.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    penguins_plot.save(output, verbose=False)


if __name__ == "__main__":
    main()
