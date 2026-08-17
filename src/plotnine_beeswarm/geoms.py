"""plotnine geom implementations."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from plotnine.geoms.geom_point import geom_point

from .swarm import beeswarm, quasirandom, sina


class _SwarmGeom(geom_point):
    DEFAULT_PARAMS: ClassVar = {
        **geom_point.DEFAULT_PARAMS,
        "width": 0.4,
        "cex": 1.0,
        "priority": "ascending",
        "maxwidth": 1.0,
        "varwidth": False,
        "method": "quasirandom",
        "bandwidth": 0.5,
        "nbins": None,
        "dodge_width": None,
        "group_on_x": None,
        "orientation": None,
        "side": 0,
        "corral": "none",
        "corral_width": 0.9,
        "random_state": None,
        "swarm_function": beeswarm,
    }

    def setup_data(self, data):
        data = super().setup_data(data).copy()
        function = self.params["swarm_function"]
        orientation = self.params.get("orientation")
        if orientation == "x":
            x_is_discrete = True
        elif orientation == "y":
            x_is_discrete = False
        elif self.params["group_on_x"] is None:
            x_is_discrete = data["x"].nunique() <= data["y"].nunique()
        else:
            x_is_discrete = self.params["group_on_x"]
        swarm_axis = "x" if x_is_discrete else "y"
        value_axis = "y" if x_is_discrete else "x"
        data[swarm_axis] = data[swarm_axis].astype(float)
        dodge_width = self.params["dodge_width"]
        if dodge_width is not None and not data["group"].duplicated().any():
            dodge_width = None
        category_counts = data.groupby(
            ["PANEL", swarm_axis], observed=True, sort=False
        ).size()
        max_lengths = category_counts.groupby(level=0).max()
        if dodge_width is not None:
            data = _dodge(data, swarm_axis, float(dodge_width))
            grouping = ["PANEL", "group", swarm_axis]
        else:
            grouping = ["PANEL", swarm_axis]
        width = self.params["width"]
        if width is None:
            width = 0.4 * _resolution(data[swarm_axis].to_numpy())
        random_state = np.random.default_rng(self.params["random_state"])
        for keys, group in data.groupby(grouping, sort=False, observed=True):
            indices = group.index
            values = data.loc[indices, value_axis].to_numpy()
            panel = keys[0] if isinstance(keys, tuple) else keys
            if function is beeswarm:
                offsets = function(
                    values,
                    width=width,
                    cex=self.params["cex"],
                    priority=self.params["priority"],
                    side=self.params["side"],
                    corral=self.params["corral"],
                    corral_width=self.params["corral_width"],
                    random_state=random_state,
                )
            elif function is sina:
                offsets = function(
                    values,
                    width=width,
                    maxwidth=self.params["maxwidth"],
                    random_state=random_state,
                )
            else:
                offsets = function(
                    values,
                    width=width,
                    bandwidth=self.params["bandwidth"],
                    nbins=self.params["nbins"],
                    method=self.params["method"],
                    varwidth=self.params["varwidth"],
                    max_length=max_lengths.loc[panel],
                    random_state=random_state,
                )
            data.loc[indices, swarm_axis] += offsets
        return data


def _dodge(data, axis, width):
    """Apply the group-centering part of ggplot2's position dodge."""
    if width == 0:
        return data
    result = data.copy()
    for (_, axis_value), category in data.groupby(
        ["PANEL", axis], sort=False, observed=True
    ):
        groups = sorted(category["group"].unique())
        count = len(groups)
        step = width / count
        centers = {
            group: axis_value - width / 2 + step / 2 + index * step
            for index, group in enumerate(groups)
        }
        for group, center in centers.items():
            result.loc[category.index[category["group"] == group], axis] = center
    return result


def _resolution(values):
    """Return the smallest non-zero spacing used by plotnine's resolution."""
    unique = sorted(set(values))
    if len(unique) < 2:
        return 1.0
    differences = [right - left for left, right in zip(unique, unique[1:])]  # noqa: RUF007
    positive = [difference for difference in differences if difference > 0]
    return min(positive, default=1.0)


def geom_beeswarm(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    na_rm=False,
    cex=1.0,
    priority="ascending",
    width=0.4,
    dodge_width=None,
    group_on_x=None,
    orientation=None,
    side=0,
    corral="none",
    corral_width=0.9,
    method="compactswarm",
    random_state=None,
    **kwargs,
):
    """Draw points packed into a beeswarm."""
    return _SwarmGeom(
        mapping=mapping,
        data=data,
        stat=stat,
        position=position,
        na_rm=na_rm,
        cex=cex,
        priority=priority,
        width=width,
        dodge_width=dodge_width,
        group_on_x=group_on_x,
        orientation=orientation,
        side=side,
        corral=corral,
        corral_width=corral_width,
        method=method,
        random_state=random_state,
        swarm_function=beeswarm,
        **kwargs,
    )


def geom_quasirandom(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    na_rm=False,
    width=None,
    varwidth=False,
    bandwidth=0.5,
    nbins=None,
    method="quasirandom",
    dodge_width=None,
    group_on_x=None,
    orientation=None,
    random_state=None,
    **kwargs,
):
    """Draw points with a quasi-random horizontal distribution."""
    return _SwarmGeom(
        mapping=mapping,
        data=data,
        stat=stat,
        position=position,
        na_rm=na_rm,
        width=width,
        varwidth=varwidth,
        random_state=random_state,
        bandwidth=bandwidth,
        nbins=nbins,
        method=method,
        dodge_width=dodge_width,
        group_on_x=group_on_x,
        orientation=orientation,
        swarm_function=quasirandom,
        **kwargs,
    )


def geom_sina(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    na_rm=False,
    width=0.4,
    maxwidth=1.0,
    method="density",
    dodge_width=0.0,
    random_state=None,
    **kwargs,
):
    """Draw points with offsets scaled by the local distribution density."""
    return _SwarmGeom(
        mapping=mapping,
        data=data,
        stat=stat,
        position=position,
        na_rm=na_rm,
        width=width,
        maxwidth=maxwidth,
        method=method,
        dodge_width=dodge_width,
        random_state=random_state,
        swarm_function=sina,
        **kwargs,
    )
