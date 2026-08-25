"""Beeswarm placement algorithms.

The implementation in this module follows :func:`beeswarm::swarmx` rather
than the (visually similar) jittering algorithms commonly used by plotting
libraries.  In particular, points are circles with a diameter of one in
normalised coordinates and each candidate position is checked against every
already placed circle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vipor import offsetSingleGroup

FloatArray = NDArray[np.float64]
RandomState = int | np.random.Generator | None


@dataclass(frozen=True)
class SwarmResult:
    """The two columns returned by upstream ``swarmx``.

    Attribute access is the idiomatic Python spelling, while string indexing
    is provided for code which mirrors R's ``result$x``/``result$y`` columns.
    """

    x: FloatArray
    y: FloatArray

    def __getitem__(self, key: str | int) -> FloatArray:
        if key == "x" or key == 0:
            return self.x
        if key == "y" or key == 1:
            return self.y
        raise KeyError(key)


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _vector(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=float).reshape(-1)


def _recycle(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Apply R's scalar recycling, while rejecting accidental truncation."""
    if x.size == y.size:
        return x, y
    if x.size == 1:
        return np.repeat(x, y.size), y
    if y.size == 1:
        return x, np.repeat(y, x.size)
    raise ValueError("x and y must have the same length (or one must be scalar)")


def _priority_order(values: FloatArray, priority: str, rng: np.random.Generator) -> NDArray[np.int_]:
    if priority not in {"ascending", "descending", "density", "random", "none"}:
        raise ValueError(
            "priority must be one of 'ascending', 'descending', 'density', 'random', or 'none'"
        )
    if priority == "none":
        return np.arange(values.size, dtype=int)
    if priority == "random":
        return rng.permutation(values.size)
    if priority == "ascending":
        return np.argsort(values, kind="stable")
    if priority == "descending":
        return np.argsort(-values, kind="stable")
    density = _density_at(values)
    return np.argsort(-density, kind="stable")


def _density_at(values: FloatArray) -> FloatArray:
    """A small Gaussian KDE used for the ``density`` priority.

    ``stats::density`` is deliberately not required: the exact bandwidth is
    not important to the placement algorithm, but stable interpolation at the
    observations is.
    """
    if values.size < 2 or np.ptp(values) == 0:
        return np.ones(values.size)
    span = float(np.ptp(values))
    sd = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    scale = min(sd, iqr / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        scale = sd if sd > 0 else span
    bandwidth = max(0.9 * scale * values.size ** -0.2, span / 1000)
    distances = (values[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * distances * distances).sum(axis=1)


def _place_swarm(
    values: FloatArray,
    *,
    compact: bool,
    side: int,
    priority: str,
    rng: np.random.Generator,
) -> FloatArray:
    """Place normalised circles using beeswarm's R implementation."""
    result = np.full(values.size, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result

    finite_values = values[finite]
    order = _priority_order(finite_values, priority, rng)
    ordered_values = finite_values[order]
    if ordered_values.size == 1:
        result[finite[order[0]]] = 0.0
        return result

    if not compact:
        placed_points: list[tuple[float, float]] = []
        for local_index, value in zip(order, ordered_values):
            candidates = [0.0]
            nearby = [
                (other_value, other_offset)
                for other_value, other_offset in placed_points
                if abs(value - other_value) < 1
            ]
            for other_value, other_offset in nearby:
                offset = float(np.sqrt(max(0.0, 1 - (value - other_value) ** 2)))
                if side != -1:
                    candidates.append(other_offset + offset)
                if side != 1:
                    candidates.append(other_offset - offset)

            chosen = np.inf
            for candidate in candidates:
                if abs(candidate) >= abs(chosen):
                    continue
                if all(
                    (value - other_value) ** 2 + (candidate - other_offset) ** 2
                    >= 0.999
                    for other_value, other_offset in nearby
                ):
                    chosen = candidate
            # A potential candidate always exists (the last candidate can be
            # moved farther from the finite set), but retain a safe fallback
            # for extreme floating-point inputs.
            if not np.isfinite(chosen):
                chosen = max(candidates, key=abs) + np.finfo(float).eps
            result[finite[local_index]] = chosen
            placed_points.append((value, chosen))
        return result

    # Compact swarm: at each iteration place the unplaced point whose best
    # available position is nearest the non-data axis.
    n = ordered_values.size
    offsets = np.zeros(n, dtype=float)
    low = np.zeros(n, dtype=float)
    high = np.zeros(n, dtype=float)
    best = np.zeros(n, dtype=float)
    placed = np.zeros(n, dtype=bool)
    for _ in range(n):
        unplaced_indices = np.flatnonzero(~placed)
        # argmin is stable, which is the priority tie-break after sorting.
        selected = unplaced_indices[np.argmin(np.abs(best[unplaced_indices]))]
        value = ordered_values[selected]
        offset = best[selected]
        offsets[selected] = offset
        placed[selected] = True
        for index in np.flatnonzero(~placed):
            difference = abs(value - ordered_values[index])
            if difference >= 1:
                continue
            clearance = float(np.sqrt(max(0.0, 1 - difference * difference)))
            high[index] = max(high[index], offset + clearance)
            if side == 0:
                low[index] = min(low[index], offset - clearance)
                best[index] = low[index] if -low[index] < high[index] else high[index]
            elif side == 1:
                best[index] = high[index]
            else:
                low[index] = min(low[index], offset - clearance)
                best[index] = low[index]

    result[finite[order]] = offsets
    return result


def _apply_corral(
    offsets: FloatArray,
    corral: str,
    corral_width: float,
    side: int,
    rng: np.random.Generator,
) -> FloatArray:
    if corral not in {"none", "gutter", "wrap", "random", "omit"}:
        raise ValueError("corral must be one of 'none', 'gutter', 'wrap', 'random', or 'omit'")
    if corral == "none":
        return offsets
    if not np.isfinite(corral_width) or corral_width <= 0:
        raise ValueError("corral_width must be positive")
    low = (side - 1) * corral_width / 2
    high = (side + 1) * corral_width / 2
    result = offsets.copy()
    finite = np.isfinite(result)
    if corral == "gutter":
        result[finite] = np.clip(result[finite], low, high)
    elif corral == "wrap":
        if side == -1:
            result[finite] = high - np.mod(high - result[finite], corral_width)
        else:
            result[finite] = np.mod(result[finite] - low, corral_width) + low
    elif corral == "random":
        runaway = finite & ((result < low) | (result > high))
        result[runaway] = rng.uniform(low, high, size=int(runaway.sum()))
    else:  # omit
        result[finite & ((result < low) | (result > high))] = np.nan
    return result


def swarmx(
    x: ArrayLike,
    y: ArrayLike,
    *,
    x_size: float = 0.08,
    y_size: float = 0.08,
    cex: float = 1.0,
    side: int = 0,
    priority: str = "ascending",
    fast: bool = True,
    compact: bool = False,
    log: str | None = None,
    random_state: RandomState = None,
    corral: str = "none",
    corral_width: float = 0.9,
    **aliases: Any,
) -> SwarmResult:
    """Return an upstream ``swarmx``-style ``(x, y)`` placement.

    ``x_size`` and ``y_size`` are the point diameters on the group and data
    axes respectively.  The aliases ``xsize`` and ``ysize`` are accepted for
    direct translation of R calls.  ``fast`` is retained for API
    compatibility; both paths use the numerically equivalent Python layout.
    """
    if "xsize" in aliases:
        x_size = aliases.pop("xsize")
    if "ysize" in aliases:
        y_size = aliases.pop("ysize")
    if aliases:
        unknown = next(iter(aliases))
        raise TypeError(f"unexpected keyword argument {unknown!r}")
    if side not in (-1, 0, 1):
        raise ValueError("side must be -1, 0, or 1")
    if not np.isfinite(cex) or cex <= 0:
        raise ValueError("cex must be positive")
    if not np.isfinite(x_size) or x_size <= 0 or not np.isfinite(y_size) or y_size <= 0:
        raise ValueError("x_size and y_size must be positive")

    x_values, y_values = _recycle(_vector(x), _vector(y))
    if x_values.size and np.any(~np.isfinite(x_values)):
        raise ValueError("x must contain finite values")
    if x_values.size and np.ptp(x_values) > 0:
        raise ValueError("all x values must be equal in swarmx")
    transformed = y_values.copy()
    log_value = "" if log is None else str(log)
    if any(axis not in "xy" for axis in log_value) or len(set(log_value)) != len(log_value):
        raise ValueError("log must be None, 'x', 'y', or 'xy'")
    if "y" in log_value:
        if np.any(np.isfinite(transformed) & (transformed <= 0)):
            raise ValueError("log-scaled y values must be positive")
        transformed[np.isfinite(transformed)] = np.log10(transformed[np.isfinite(transformed)])
    base_x = x_values.copy()
    if "x" in log_value:
        if np.any(x_values <= 0):
            raise ValueError("log-scaled x values must be positive")
        base_x = np.log10(x_values)

    generator = _rng(random_state)
    offsets = _place_swarm(
        transformed / (float(y_size) * float(cex)),
        compact=compact,
        side=side,
        priority=priority,
        rng=generator,
    )
    offsets *= float(x_size) * float(cex)
    offsets = _apply_corral(offsets, corral, float(corral_width), side, generator)
    result_x = base_x + offsets
    if "x" in log_value:
        result_x = 10 ** result_x
    return SwarmResult(result_x, y_values)


def _grid_offsets(
    values: FloatArray,
    *,
    method: str,
    x_size: float,
    y_size: float,
    cex: float,
    side: int,
) -> FloatArray:
    """Implement ggbeeswarm's square, hex, and centre layouts."""
    result = np.full(values.size, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return result
    width = y_size * cex
    if method == "hex":
        width *= np.sqrt(3) / 2
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if maximum == minimum:
        bins = np.zeros(finite.size, dtype=int)
    else:
        step = max(width, np.finfo(float).eps)
        bins = np.floor((values[finite] - minimum) / step).astype(int)
    for row in np.unique(bins):
        members = np.flatnonzero(bins == row)
        positions = np.arange(1, members.size + 1, dtype=float)
        if method in {"center", "centre"}:
            if side == -1:
                positions -= members.size
            elif side == 1:
                positions -= 1
            else:
                positions -= positions.mean()
        elif method == "square":
            if side == -1:
                positions -= members.size
            elif side == 1:
                positions -= 1
            else:
                positions -= np.floor(positions.mean())
        else:  # hex
            # R's ``cut`` labels rows from one, whereas our bins start at
            # zero; the first row is therefore the odd row.
            odd = bool((row + 1) % 2)
            if side == -1:
                positions -= members.size
                if not odd:
                    positions -= 0.5
            elif side == 1:
                positions -= 1
                if not odd:
                    positions -= 0.5
            elif odd:
                positions -= np.floor(positions.mean()) + 0.25
            else:
                positions -= np.ceil(positions.mean()) - 0.25
        result[finite[members]] = positions * x_size * cex
    return result


def beeswarm(
    values: ArrayLike,
    *,
    width: float = 0.4,
    cex: float = 1.0,
    method: str = "swarm",
    priority: str = "ascending",
    side: int = 0,
    corral: str = "none",
    corral_width: float = 0.9,
    random_state: RandomState = None,
) -> FloatArray:
    """Compatibility wrapper returning offsets for one group of values."""
    values_array = _vector(values)
    if side not in (-1, 0, 1):
        raise ValueError("side must be -1, 0, or 1")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be positive")
    if not np.isfinite(cex) or cex <= 0:
        raise ValueError("cex must be positive")
    finite_values = values_array[np.isfinite(values_array)]
    data_span = float(np.ptp(finite_values)) if finite_values.size else 0.0
    y_size = max(data_span, 1.0) / 100
    if method == "compactswarm":
        compact = True
    elif method == "swarm":
        compact = False
    elif method in {"center", "centre", "hex", "square"}:
        offsets = _grid_offsets(
            values_array,
            method=method,
            x_size=float(width),
            y_size=y_size,
            cex=float(cex),
            side=side,
        )
        return _apply_corral(
            offsets,
            corral,
            float(corral_width),
            side,
            _rng(random_state),
        )
    else:
        raise ValueError("method must be swarm, compactswarm, center, centre, hex, or square")
    return swarmx(
        np.zeros(values_array.size),
        values_array,
        x_size=float(width),
        y_size=y_size,
        cex=cex,
        side=side,
        priority=priority,
        compact=compact,
        random_state=random_state,
        corral=corral,
        corral_width=corral_width,
    ).x


def quasirandom(
    values: ArrayLike,
    *,
    width: float = 0.4,
    bandwidth: float = 0.5,
    nbins: int | None = None,
    method: str = "quasirandom",
    varwidth: bool = False,
    max_length: float | None = None,
    random_state: RandomState = None,
) -> FloatArray:
    """Return vipor-backed quasi-random offsets."""
    values_array = _vector(values)
    result = np.full(values_array.size, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values_array))
    if finite.size:
        result[finite] = offsetSingleGroup(
            values_array[finite],
            maxLength=max_length if varwidth else None,
            method=method,
            nbins=nbins,
            adjust=bandwidth,
            random_state=random_state,
        )
        result[finite] *= width
    return result


def sina(values: ArrayLike, *, width: float = 0.4, maxwidth: float = 1.0, random_state: RandomState = None) -> FloatArray:
    """Return density-scaled offsets for a sina plot."""
    values_array = _vector(values)
    result = quasirandom(values_array, width=width, random_state=random_state)
    finite = np.isfinite(values_array)
    if finite.any():
        ranks = np.argsort(np.argsort(values_array[finite], kind="stable"), kind="stable")
        density = 1 - np.abs(2 * ranks / max(len(ranks) - 1, 1) - 1)
        result[finite] *= maxwidth * (0.25 + 0.75 * density)
    return result


__all__ = ["SwarmResult", "beeswarm", "quasirandom", "sina", "swarmx"]
