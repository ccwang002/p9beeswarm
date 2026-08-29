"""Data and plotting helpers for the README example tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from plotnine import aes, ggplot
from plotnine.data import mpg


def iris_data() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    dataset: Any = load_iris()
    return pd.DataFrame(
        dataset.data,
        columns=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    ).assign(Species=[dataset.target_names[index] for index in dataset.target])


def base_plot(data: pd.DataFrame, x: str, y: str):
    return ggplot(data, aes(x, y))


def sub_mpg() -> pd.DataFrame:
    return mpg[mpg["class"].isin(["midsize", "pickup", "suv"])]


def alternative_data() -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    return pd.DataFrame({"x": "A", "y": rng.choice(np.arange(1, 101), 200)})


def priority_data() -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    x = np.repeat([1, 2, 3], [20, 40, 80])
    return pd.DataFrame({"x": x, "y": rng.normal(x, 1)})
