"""Pure data transformations used by the plotnine geoms."""

from __future__ import annotations

import numpy as np

from vipor import offsetSingleGroup


def _as_array(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values


def beeswarm(
    values,
    *,
    width=0.4,
    cex=1.0,
    priority="ascending",
    side=0,
    corral="none",
    corral_width=0.9,
    random_state=None,
):
    """Return horizontal offsets that pack points without overlap.

    This is the same bottom-up packing strategy used by the default
    ``compactswarm`` method in ggbeeswarm.  The returned values are in data
    units relative to the centre of the category.
    """
    y = _as_array(values)
    out = np.full(y.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(y))
    if not len(valid):
        return out
    rng = np.random.default_rng(random_state)
    if priority == "random":
        order = rng.permutation(valid)
    elif priority == "descending":
        order = valid[np.argsort(-y[valid], kind="stable")]
    elif priority == "density":
        density = _density_at(y[valid])
        order = valid[np.argsort(-density, kind="stable")]
    elif priority == "none":
        order = valid[np.argsort(y[valid], kind="stable")]
    else:
        order = valid[np.argsort(y[valid], kind="stable")]
    placed = []
    span = np.ptp(y[valid])
    if span == 0:
        span = 1.0
    # cex is expressed in point-size units; convert it to a small fraction
    # of the data range so continuous samples retain the dense swarm shape.
    minimum = max(float(cex), 1e-12) * span / 135.0
    for index in order:
        candidates = [0.0]
        for other, other_y, other_x in placed:
            distance = abs(y[index] - other_y)
            if distance < minimum:
                horizontal = np.sqrt(max(0.0, minimum * minimum - distance * distance))
                candidates.extend((other_x - horizontal, other_x + horizontal))
        candidates.sort(key=lambda x: (abs(x), x))
        valid_candidates = [
            candidate
            for candidate in candidates
            if all(
                (y[index] - other_y) ** 2 + (candidate - other_x) ** 2
                >= minimum * minimum - 1e-15
                for _, other_y, other_x in placed
            )
        ]
        out[index] = valid_candidates[0] if valid_candidates else candidates[-1]
        placed.append((index, y[index], out[index]))
    if side == -1:
        out = np.minimum(out, 0)
    elif side in (1, 2):
        out = np.maximum(out, 0)
    if corral != "none":
        low = (side - 1) * corral_width / 2
        high = (side + 1) * corral_width / 2
        if corral == "gutter":
            out = np.clip(out, low, high)
        elif corral == "wrap":
            if side == -1:
                out = high - ((high - out) % corral_width)
            else:
                out = ((out - low) % corral_width) + low
        elif corral == "random":
            out = np.where(
                (out >= low) & (out <= high),
                out,
                rng.uniform(low, high, size=out.shape),
            )
        elif corral == "omit":
            out[(out < low) | (out > high)] = np.nan
    return out


def quasirandom(
    values,
    *,
    width=0.4,
    bandwidth=0.5,
    nbins=None,
    method="quasirandom",
    varwidth=False,
    max_length=None,
    random_state=None,
):
    """Return offsets using vipor's ``offsetSingleGroup`` implementation."""
    y = _as_array(values)
    out = np.full(y.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(y))
    if not len(valid):
        return out
    out[valid] = (
        offsetSingleGroup(
            y[valid],
            maxLength=max_length if varwidth else None,
            method=method,
            nbins=nbins,
            adjust=bandwidth,
            random_state=random_state,
        )
        * width
    )
    return out


def sina(values, *, width=0.4, maxwidth=1.0, random_state=None):
    """Return density-scaled offsets for a sina plot."""
    y = _as_array(values)
    result = quasirandom(y, width=width, random_state=random_state)
    valid = np.isfinite(y)
    if valid.any():
        ranks = np.argsort(np.argsort(y[valid], kind="stable"), kind="stable")
        density = 1 - np.abs(2 * ranks / max(len(ranks) - 1, 1) - 1)
        result[valid] *= maxwidth * (0.25 + 0.75 * density)
    return result


def _van_der_corput(n):
    sequence = np.empty(n, dtype=float)
    for index in range(1, n + 1):
        value = index
        denominator = 1.0
        result = 0.0
        while value:
            value, remainder = divmod(value, 2)
            denominator *= 2
            result += remainder / denominator
        sequence[index - 1] = result
    return sequence


def _density_at(values, *, bandwidth=0.0):
    grid, density = _density_estimate(values, bandwidth=bandwidth, points=1024)
    return np.interp(values, grid, density)


def _density_estimate(values, *, bandwidth=0.0, points=1024):
    span = np.ptp(values)
    if span == 0:
        return np.array([values[0] - 0.5, values[0] + 0.5]), np.ones(2)
    standard_deviation = np.std(values, ddof=1)
    interquartile_range = np.subtract(*np.percentile(values, [75, 25]))
    scale = min(standard_deviation, interquartile_range / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        scale = standard_deviation if standard_deviation > 0 else span
    if bandwidth and bandwidth > 0:
        kernel_width = float(bandwidth)
    else:
        kernel_width = 0.9 * scale * len(values) ** (-0.2)
    kernel_width = max(kernel_width, span / 1000)
    grid = np.linspace(
        values.min() - 3 * kernel_width,
        values.max() + 3 * kernel_width,
        max(2, points),
    )
    distances = (grid[:, None] - values[None, :]) / kernel_width
    density = np.exp(-0.5 * distances**2).sum(axis=1)
    maximum = density.max()
    return grid, density / maximum if maximum else np.ones(len(grid))


def _top_bottom_sequence(values, *, frowney, bins=None):
    """Port vipor's topBottomDistribute within empirical density bins."""
    n = len(values)
    if n < 2:
        return np.full(n, 0.5)
    if bins is None:
        bins = np.linspace(values.min(), values.max(), max(2, int(np.ceil(n / 5))))
    groups = np.clip(np.searchsorted(bins, values, side="right") - 1, 0, len(bins) - 1)
    sequence = np.empty(n, dtype=float)
    for group in range(len(bins)):
        members = np.flatnonzero(groups == group)
        if not len(members):
            continue
        ranked_values = -values[members] if frowney else values[members]
        ranks = np.argsort(np.argsort(ranked_values, kind="stable"), kind="stable") + 1
        ranks[ranks % 2 == 1] *= -1
        ranks = np.argsort(np.argsort(ranks, kind="stable"), kind="stable") + 1
        sequence[members] = (ranks - 1.0) / max(len(members) - 1, 1)
    return sequence
