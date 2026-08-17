"""plotnine geom implementations."""

from __future__ import annotations

from typing import ClassVar

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
        "bandwidth": 0.0,
        "nbins": None,
        "dodge_width": 0.0,
        "group_on_x": None,
        "side": "both",
        "corral": "none",
        "corral_width": 0.9,
        "random_state": None,
        "swarm_function": beeswarm,
    }

    def setup_data(self, data):
        data = super().setup_data(data).copy()
        function = self.params["swarm_function"]
        if self.params["group_on_x"] is None:
            x_is_discrete = data["x"].nunique() <= data["y"].nunique()
        else:
            x_is_discrete = self.params["group_on_x"]
        swarm_axis = "x" if x_is_discrete else "y"
        value_axis = "y" if x_is_discrete else "x"
        data[swarm_axis] = data[swarm_axis].astype(float)
        grouping = ["PANEL", "group", swarm_axis]
        for _, group in data.groupby(grouping, sort=False, observed=True):
            indices = group.index
            values = data.loc[indices, value_axis].to_numpy()
            if function is beeswarm:
                offsets = function(
                    values,
                    width=self.params["width"],
                    cex=self.params["cex"],
                    priority=self.params["priority"],
                    side=self.params["side"],
                    corral=self.params["corral"],
                    corral_width=self.params["corral_width"],
                    random_state=self.params["random_state"],
                )
            elif function is sina:
                offsets = function(
                    values,
                    width=self.params["width"],
                    maxwidth=self.params["maxwidth"],
                    random_state=self.params["random_state"],
                )
            else:
                offsets = function(
                    values,
                    width=self.params["width"],
                    bandwidth=self.params["bandwidth"],
                    method=self.params["method"],
                    random_state=self.params["random_state"],
                )
            data.loc[indices, swarm_axis] += offsets
        return data


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
    dodge_width=0.0,
    group_on_x=None,
    side="both",
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
    width=0.4,
    bandwidth=0.0,
    nbins=None,
    method="quasirandom",
    dodge_width=0.0,
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
        random_state=random_state,
        bandwidth=bandwidth,
        nbins=nbins,
        method=method,
        dodge_width=dodge_width,
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
