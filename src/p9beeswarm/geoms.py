"""Beeswarm-style geoms for plotnine."""

from __future__ import annotations

from typing import Any

from plotnine.geoms.geom_point import geom_point

from .positions import (
    _SinaPosition,
    position_beeswarm,
    position_quasirandom,
)


def _use_default_position(value: Any) -> bool:
    return value is None or value == "identity"


class geom_beeswarm(geom_point):
    """Draw points packed into a beeswarm."""

    def __init__(
        self,
        mapping=None,
        data=None,
        *,
        stat: Any = "identity",
        position: Any = "identity",
        na_rm: bool = False,
        cex: float = 1.0,
        method: str = "swarm",
        priority: str = "ascending",
        width: float | None = None,
        dodge_width: float | None = None,
        group_on_x: bool | None = None,
        orientation: str | None = None,
        side: int = 0,
        corral: str = "none",
        corral_width: float = 0.9,
        random_state: Any = None,
        **kwargs: Any,
    ):
        if _use_default_position(position):
            position = position_beeswarm(
                width=width,
                cex=cex,
                method=method,
                priority=priority,
                dodge_width=dodge_width,
                group_on_x=group_on_x,
                orientation=orientation,
                side=side,
                corral=corral,
                corral_width=corral_width,
                random_state=random_state,
            )
        super().__init__(
            mapping=mapping,
            data=data,
            stat=stat,
            position=position,
            na_rm=na_rm,
            **kwargs,
        )


class geom_quasirandom(geom_point):
    """Draw points with vipor's quasi-random distribution."""

    def __init__(
        self,
        mapping=None,
        data=None,
        *,
        stat: Any = "identity",
        position: Any = "identity",
        na_rm: bool = False,
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
        **kwargs: Any,
    ):
        if _use_default_position(position):
            position = position_quasirandom(
                width=width,
                varwidth=varwidth,
                bandwidth=bandwidth,
                nbins=nbins,
                method=method,
                dodge_width=dodge_width,
                group_on_x=group_on_x,
                orientation=orientation,
                max_length=max_length,
                random_state=random_state,
            )
        super().__init__(
            mapping=mapping,
            data=data,
            stat=stat,
            position=position,
            na_rm=na_rm,
            **kwargs,
        )


class geom_sina(geom_point):
    """Draw points with offsets scaled by their local distribution density."""

    def __init__(
        self,
        mapping=None,
        data=None,
        *,
        stat: Any = "identity",
        position: Any = "identity",
        na_rm: bool = False,
        width: float | None = 0.4,
        maxwidth: float = 1.0,
        method: str = "density",
        dodge_width: float | None = 0.0,
        group_on_x: bool | None = None,
        orientation: str | None = None,
        random_state: Any = None,
        **kwargs: Any,
    ):
        if _use_default_position(position):
            position = _SinaPosition(
                width=width,
                maxwidth=maxwidth,
                method=method,
                dodge_width=dodge_width,
                group_on_x=group_on_x,
                orientation=orientation,
                random_state=random_state,
            )
        super().__init__(
            mapping=mapping,
            data=data,
            stat=stat,
            position=position,
            na_rm=na_rm,
            **kwargs,
        )


__all__ = ["geom_beeswarm", "geom_quasirandom", "geom_sina"]
