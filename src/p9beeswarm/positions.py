"""Plotnine position adjustments for beeswarm-style plots.

The position adjustments do the data transformation; the corresponding geoms
are intentionally thin wrappers around :class:`plotnine.geoms.geom_point`.
Keeping the transformation here means that a swarm can be shared by several
geoms and can be composed with plotnine's normal layer machinery.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from plotnine.positions.position import position
from plotnine.scales.scale_continuous import scale_continuous
from plotnine.scales.scale_discrete import scale_discrete

from .beeswarm import beeswarm, quasirandom, sina


def _resolution(values: np.ndarray) -> float:
    """Return the smallest positive spacing in *values*."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    unique = np.unique(finite)
    if len(unique) < 2:
        return 1.0
    return float(np.min(np.diff(unique)))


def _as_numeric_axis(data: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Convert a discrete axis to plotnine's numeric representation.

    Plotnine normally performs this conversion before a position is called.
    Handling object and categorical columns as well makes the position classes
    useful in focused tests and when called directly.
    """
    if pd.api.types.is_numeric_dtype(data[axis]):
        data[axis] = data[axis].astype(float)
    else:
        data[axis] = pd.Series(
            pd.factorize(data[axis], sort=False)[0] + 1,
            index=data.index,
            dtype=float,
        )
    return data


def _dodge(data: pd.DataFrame, axis: str, width: float) -> pd.DataFrame:
    """Place groups side-by-side around each discrete-axis value."""
    result = data.copy()
    if width == 0:
        return result

    for axis_value, category in data.groupby(
        axis, sort=False, observed=True, dropna=False
    ):
        groups = list(pd.unique(category["group"]))
        if len(groups) < 2:
            continue
        try:
            groups.sort()
        except TypeError:
            # Group identifiers are normally integers, but a stable order is
            # preferable to failing for a user-supplied object column.
            groups.sort(key=str)
        step = width / len(groups)
        centers = {
            group: axis_value - width / 2 + step / 2 + index * step
            for index, group in enumerate(groups)
        }
        for group, center in centers.items():
            mask = category["group"] == group
            result.loc[category.index[mask], axis] = center
    return result


def _is_dodgeable(data: pd.DataFrame, dodge_width: float | None) -> bool:
    """Whether a layer has a repeated grouping aesthetic to dodge."""
    if dodge_width is None or "group" not in data:
        return False
    groups = data["group"]
    return groups.nunique(dropna=False) > 1 and groups.duplicated().any()


def _orientation(data: pd.DataFrame, params: dict[str, Any]) -> bool:
    """Return ``True`` when x is the discrete (swarm) axis."""
    value = params.get("orientation")
    if value not in (None, "x", "y"):
        raise ValueError("orientation must be 'x', 'y', or None")
    if value == "x":
        return True
    if value == "y":
        return False
    group_on_x = params.get("group_on_x")
    if group_on_x is not None:
        return bool(group_on_x)
    return data["x"].nunique(dropna=False) <= data["y"].nunique(dropna=False)


def _random_generator(random_state: Any) -> Any:
    """Create one generator for all groups in a panel."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def get_range(scale: Any) -> float:
    """Return a position scale's upstream-compatible range.

    This mirrors ggbeeswarm's ``get_range`` helper: continuous scales use the
    difference between their limits, while discrete scales use the number of
    unique limits. Explicit limits take precedence over trained values, and a
    zero-length range is normalized to one.
    """
    if isinstance(scale, scale_discrete):
        limits = scale.final_limits
        result = len(pd.unique(np.asarray(limits, dtype=object)))
    elif isinstance(scale, scale_continuous):
        limits = scale.final_limits
        result = abs(float(limits[1]) - float(limits[0]))
    else:
        raise TypeError(f"unknown position scale type: {type(scale).__name__}")
    return float(result) if result else 1.0


def _data_range(values: pd.Series) -> float:
    """Provide the position-scale range when called outside plotnine's build."""
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    span = float(np.ptp(finite))
    return span if span else 1.0


def _beeswarm_sizes(
    data: pd.DataFrame,
    params: dict[str, Any],
    swarm_axis: str,
    value_axis: str,
    scales: Any,
) -> tuple[float, float]:
    """Match ggbeeswarm's one-hundredth-of-scale collision circle sizes."""
    if scales is None:
        swarm_range = _data_range(data[swarm_axis])
        value_range = _data_range(data[value_axis])
    else:
        swarm_range = get_range(getattr(scales, swarm_axis))
        value_range = get_range(getattr(scales, value_axis))
    width = params["width"]
    x_size = float(width) if width is not None else swarm_range / 100
    return x_size, value_range / 100


def _position_swarm(
    data: pd.DataFrame,
    params: dict[str, Any],
    algorithm: str,
    scales: Any,
) -> pd.DataFrame:
    """Shared panel pipeline for all swarm positions."""
    if data.empty:
        return data

    result = data.copy()
    x_is_discrete = _orientation(result, params)
    swarm_axis = "x" if x_is_discrete else "y"
    value_axis = "y" if x_is_discrete else "x"
    result = _as_numeric_axis(result, swarm_axis)

    counts = result.groupby(swarm_axis, observed=True, dropna=False).size()
    inferred_max_length = int(counts.max()) if len(counts) else 0
    dodge_width = params.get("dodge_width")
    dodged = _is_dodgeable(result, dodge_width)
    if dodged:
        result = _dodge(result, swarm_axis, float(cast(float, dodge_width)))

    if dodged:
        grouping = ["group", swarm_axis]
    else:
        grouping = [swarm_axis]
    max_length = params.get("max_length")
    if max_length is None:
        max_length = inferred_max_length

    width = params["width"]
    if algorithm != "beeswarm" and width is None:
        # ggbeeswarm derives the default width after collision/dodging. The
        # distance between dodged group centers, rather than the original
        # categorical spacing, is the scale of quasirandom offsets.
        width = 0.4 * _resolution(result[swarm_axis].to_numpy(dtype=float))
    if algorithm == "beeswarm":
        x_size, y_size = _beeswarm_sizes(
            result, params, swarm_axis, value_axis, scales
        )
    rng = _random_generator(params.get("random_state"))
    for _, group in result.groupby(grouping, sort=False, observed=True, dropna=False):
        indices = group.index
        values = result.loc[indices, value_axis].to_numpy(dtype=float)
        if algorithm == "beeswarm":
            offsets = beeswarm(
                values,
                width=width,
                cex=params["cex"],
                method=params["method"],
                priority=params["priority"],
                side=params["side"],
                corral=params["corral"],
                corral_width=params["corral_width"],
                random_state=rng,
                x_size=x_size,
                y_size=y_size,
            )
        elif algorithm == "sina":
            offsets = sina(
                values,
                width=width,
                maxwidth=params["maxwidth"],
                random_state=rng,
            )
        else:
            offsets = quasirandom(
                values,
                width=width,
                bandwidth=params["bandwidth"],
                nbins=params["nbins"],
                method=params["method"],
                varwidth=params["varwidth"],
                max_length=max_length,
                random_state=rng,
            )
        result.loc[indices, swarm_axis] += offsets
    return result


class _SwarmPosition(position):
    """Base class implementing the common position pipeline."""

    algorithm: ClassVar[str] = "quasirandom"

    def setup_params(self, data: pd.DataFrame) -> dict[str, Any]:
        return deepcopy(self.params)

    @classmethod
    def compute_panel(cls, data, scales, params):
        return _position_swarm(data, params, cls.algorithm, scales)


class position_beeswarm(_SwarmPosition):
    """Pack points into a beeswarm around their discrete position."""

    algorithm: ClassVar[str] = "beeswarm"

    def __init__(
        self,
        width: float | None = None,
        cex: float = 1.0,
        method: str = "swarm",
        priority: str = "ascending",
        dodge_width: float | None = None,
        group_on_x: bool | None = None,
        orientation: str | None = None,
        side: int = 0,
        corral: str = "none",
        corral_width: float = 0.9,
        random_state: Any = None,
    ):
        self.params = {
            "width": width,
            "cex": cex,
            "method": method,
            "priority": priority,
            "dodge_width": dodge_width,
            "group_on_x": group_on_x,
            "orientation": orientation,
            "side": side,
            "corral": corral,
            "corral_width": corral_width,
            "random_state": random_state,
        }


class position_quasirandom(_SwarmPosition):
    """Distribute points using vipor's quasi-random offsets."""

    algorithm: ClassVar[str] = "quasirandom"

    def __init__(
        self,
        width: float | None = None,
        varwidth: bool = False,
        bandwidth: float = 0.5,
        nbins: int | None = None,
        method: str = "quasirandom",
        dodge_width: float | None = None,
        group_on_x: bool | None = None,
        orientation: str | None = None,
        max_length: float | None = None,
        random_state: Any = None,
    ):
        self.params = {
            "width": width,
            "varwidth": varwidth,
            "bandwidth": bandwidth,
            "nbins": nbins,
            "method": method,
            "dodge_width": dodge_width,
            "group_on_x": group_on_x,
            "orientation": orientation,
            "max_length": max_length,
            "random_state": random_state,
        }


class _SinaPosition(_SwarmPosition):
    """Private position used by geom_sina with the shared pipeline."""

    algorithm: ClassVar[str] = "sina"

    def __init__(
        self,
        width: float | None = 0.4,
        maxwidth: float = 1.0,
        method: str = "density",
        dodge_width: float | None = 0.0,
        group_on_x: bool | None = None,
        orientation: str | None = None,
        random_state: Any = None,
    ):
        self.params = {
            "width": width,
            "maxwidth": maxwidth,
            "method": method,
            "dodge_width": dodge_width,
            "group_on_x": group_on_x,
            "orientation": orientation,
            "random_state": random_state,
        }


__all__ = ["position_beeswarm", "position_quasirandom"]
