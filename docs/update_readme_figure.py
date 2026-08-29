"""Regenerate the README figure at docs/penguins-beeswarm.png.

The plot is the rendered version of the example in README.md, so the script
builds the same ggplot object and saves it with plotnine. Run from anywhere:

    uv run python docs/update_readme_figure.py
"""

from __future__ import annotations

from pathlib import Path

import plotnine as p9
from plotnine.data import penguins

from p9beeswarm import geom_quasirandom

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "penguins-beeswarm.png"


def build_plot() -> p9.ggplot:
    return (
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


def main() -> None:
    plot = build_plot()
    plot.save(OUTPUT, width=6, height=4, dpi=150, verbose=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
