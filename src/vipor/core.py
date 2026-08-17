"""Typed Python implementation of the public functions in R's vipor package."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import permutations
from math import isfinite, pi, sqrt
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
T = TypeVar("T")


def _as_float_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=float).reshape(-1)


def offsetX(
    y: ArrayLike,
    x: Sequence[Hashable] | None = None,
    width: float = 0.4,
    varwidth: bool = False,
    *,
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> FloatArray:
    """Return vipor offsets for one or more groups of observations."""
    values = _as_float_array(y)
    if x is None:
        groups: list[Hashable] = [0] * len(values)
    else:
        if len(x) != len(values):
            raise ValueError("x and y not the same length in offsetX")
        groups = list(x)
    if not values.size:
        return np.array([], dtype=float)

    group_indices: dict[Hashable, list[int]] = {}
    for index, group in enumerate(groups):
        group_indices.setdefault(group, []).append(index)
    max_length = max(len(indices) for indices in group_indices.values())
    output = np.zeros(len(values), dtype=float)
    rng = _make_rng(random_state)
    for indices in group_indices.values():
        subgroup = values[indices]
        output[indices] = offsetSingleGroup(
            subgroup,
            maxLength=max_length if varwidth else None,
            random_state=rng,
            **kwargs,
        )
    return output * float(width)


def offsetSingleGroup(
    y: ArrayLike,
    maxLength: float | None = None,
    method: str = "quasirandom",
    nbins: int | None = None,
    adjust: float = 1,
    *,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Return vipor offsets for one group of numeric observations."""
    values = _as_float_array(y)
    if not values.size:
        return np.array([], dtype=float)
    if values.size == 1:
        return np.array([0.0])
    if method == "smiley":
        method = "maxout"
    elif method == "frowney":
        method = "minout"
    methods = {
        "quasirandom",
        "pseudorandom",
        "maxout",
        "minout",
        "tukey",
        "tukeyDense",
    }
    if method not in methods:
        raise ValueError(f"Unrecognized method in offsetSingleGroup: {method}")
    if nbins is None:
        nbins = 2**10 if method in {"quasirandom", "pseudorandom"} else max(
            2, int(np.ceil(values.size / 5))
        )
    if nbins < 1:
        raise ValueError("nbins must be positive")

    subgroup_width = 1.0
    if maxLength is not None and maxLength > 0:
        subgroup_width = sqrt(values.size / float(maxLength))

    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("y must contain only finite values")
    density_x, density_y = _density(values, int(nbins), float(adjust))
    if method == "quasirandom":
        ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        offset = vanDerCorput(values.size)[ranks]
    elif method == "pseudorandom":
        offset = _make_rng(random_state).random(values.size)
    elif method in {"maxout", "minout"}:
        bins = np.searchsorted(density_x, values, side="left")
        bins = np.clip(bins, 1, len(density_x) - 1)
        offset = np.empty(values.size, dtype=float)
        for group in np.unique(bins):
            members = np.flatnonzero(bins == group)
            offset[members] = topBottomDistribute(
                values[members], frowney=method == "minout"
            )
    else:
        offset = tukeyTexture(
            values,
            jitter=method == "tukey",
            thin=method == "tukey",
            random_state=random_state,
        ) / 100.0

    point_density = (
        np.ones(values.size)
        if method == "tukey"
        else np.interp(values, density_x, density_y)
    )
    return (offset - 0.5) * 2 * point_density * subgroup_width


def _density(values: FloatArray, nbins: int, adjust: float) -> tuple[FloatArray, FloatArray]:
    """Approximate ``stats::density(..., kernel='gaussian')`` without R."""
    if adjust <= 0:
        raise ValueError("adjust must be positive")
    n = values.size
    standard_deviation = float(np.std(values, ddof=1))
    interquartile_range = float(np.percentile(values, 75) - np.percentile(values, 25))
    scale = min(standard_deviation, interquartile_range / 1.34)
    if not isfinite(scale) or scale <= 0:
        scale = standard_deviation if standard_deviation > 0 else 1.0
    bandwidth = 0.9 * scale * n ** (-0.2) * adjust
    lower = float(values.min() - 3 * bandwidth)
    upper = float(values.max() + 3 * bandwidth)
    if lower == upper:
        lower -= 0.5
        upper += 0.5

    # R rounds density's internal FFT grid up to a power of two, then returns
    # the requested number of points between min(x)-3*bw and max(x)+3*bw.
    grid_size = max(512, 1 << (int(nbins - 1).bit_length()))
    grid = np.linspace(lower - 4 * bandwidth, upper + 4 * bandwidth, grid_size)
    density_grid = np.zeros(grid_size, dtype=float)
    chunk_size = 4096
    for start in range(0, n, chunk_size):
        chunk = values[start : start + chunk_size]
        distances = (grid[:, None] - chunk[None, :]) / bandwidth
        density_grid += np.exp(-0.5 * distances**2).sum(axis=1)
    density_grid /= n * bandwidth * sqrt(2 * pi)
    result_x = np.linspace(lower, upper, nbins)
    result_y = np.interp(result_x, grid, density_grid)
    maximum = float(result_y.max())
    if maximum:
        result_y /= maximum
    return result_x, result_y


def topBottomDistribute(
    x: ArrayLike, frowney: bool = False, prop: bool = True
) -> FloatArray:
    """Arrange values with extremes alternating between the two sides."""
    values = _as_float_array(x)
    if values.size == 1:
        return np.array([0.5]) if prop else np.array([1.0])
    ranked = np.argsort(
        np.argsort(values if not frowney else -values, kind="stable"), kind="stable"
    ) + 1
    ranked[ranked % 2 == 1] *= -1
    order = np.argsort(np.argsort(ranked, kind="stable"), kind="stable") + 1
    if prop:
        return (order - 1) / (values.size - 1)
    return order.astype(float)


def vanDerCorput(n: int, base: int = 2, start: int = 1) -> FloatArray:
    """Generate the first ``n`` values of a van der Corput sequence."""
    if n < 0:
        raise ValueError("n < 0 in vanDerCorput")
    if base <= 1:
        raise ValueError("base <=1 in vanDerCorput")
    if start < 1:
        raise ValueError("start < 1 in vanDerCorput")
    output = np.empty(n, dtype=float)
    for index in range(n):
        number = index + start
        value = 0.0
        denominator = float(base)
        while number:
            number, remainder = divmod(number, base)
            value += remainder / denominator
            denominator *= base
        output[index] = value
    return output


def number2digits(n: int, base: int = 2) -> FloatArray:
    """Convert an integer to least-significant-first arbitrary-base digits."""
    if n < 0:
        raise ValueError("negative number in number2digits")
    if base <= 1:
        raise ValueError("base <=1 in number2digits")
    if n == 0:
        return np.array([], dtype=float)
    digits: list[float] = []
    value = n
    while value:
        value, remainder = divmod(value, base)
        digits.append(float(remainder))
    return np.asarray(digits)


def digits2number(
    digits: Iterable[int | float] | float,
    base: int = 2,
    fractional: bool = False,
) -> float:
    """Convert least-significant-first arbitrary-base digits to a number."""
    if base <= 1:
        raise ValueError("base <= 1 in digits2number")
    if isinstance(digits, (int, float)):
        values = np.asarray([digits], dtype=float)
        is_scalar = True
    else:
        values = np.asarray(list(digits), dtype=float)
        is_scalar = False
    if values.size and np.any(values < 0):
        raise ValueError("digit < 0 in digits2number")
    if not is_scalar and values.size and np.any(values >= base):
        raise ValueError("digit >= base in digits2number")
    result = float(np.sum(values * base ** np.arange(values.size)))
    return result / base ** values.size if fractional else result


def permute(vals: Sequence[T]) -> list[tuple[T, ...]] | None:  # noqa: UP047
    """Return all permutations of ``vals`` in vipor's recursive order."""
    if len(vals) == 0:
        return None
    return list(permutations(vals))


def tukeyPermutes(n: int = 5, limit: int = 2) -> list[tuple[int, ...]]:
    """Return permutations without runs of ``limit`` increases/decreases."""
    if n < 1:
        return []
    output: list[tuple[int, ...]] = []
    for candidate in permutations(range(1, n + 1)):
        directions = np.diff(candidate) > 0
        if directions.size:
            run_lengths = np.diff(
                np.r_[
                    [0],
                    np.flatnonzero(directions[1:] != directions[:-1]) + 1,
                    [len(directions)],
                ]
            )
            if int(run_lengths.max()) >= limit:
                continue
        output.append(candidate)
    return output


def generatePermuteString(
    nReps: int = 20, n: int = 5, *, random_state: int | np.random.Generator | None = None
) -> NDArray[np.int_]:
    """Generate a random concatenation of Tukey-valid permutations."""
    if nReps < 0 or n < 1:
        raise ValueError("nReps and n must be positive")
    valid = tukeyPermutes(n)
    if not valid or nReps == 0:
        return np.array([], dtype=int)
    rng = _make_rng(random_state)
    by_first: dict[int, list[tuple[int, ...]]] = {}
    for candidate in valid:
        by_first.setdefault(candidate[0], []).append(candidate)
    selected: list[int] = []
    previous: tuple[int, ...] | None = None
    for _ in range(nReps):
        if previous is None:
            candidate = valid[int(rng.integers(len(valid)))]
        else:
            increasing = previous[-1] > previous[-2]
            allowed = (
                [item for value in range(1, previous[-1]) for item in by_first.get(value, [])]
                if increasing
                else [
                    item
                    for value in range(previous[-1] + 1, n + 1)
                    for item in by_first.get(value, [])
                ]
            )
            candidate = allowed[int(rng.integers(len(allowed)))] if allowed else valid[0]
        selected.extend(candidate)
        previous = candidate
    return np.asarray(selected, dtype=int)


def tukeyT(
    nReps: int = 10,
    base: int = 5,
    *,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """Combine Tukey permutation strings into lateral positions."""
    if nReps < 0 or base < 1:
        raise ValueError("nReps and base must be positive")
    rng = _make_rng(random_state)
    values = generatePermuteString(nReps, base, random_state=rng)
    textures = [
        generatePermuteString(nReps, base, random_state=rng) for _ in range(base)
    ]
    if not values.size:
        return np.array([], dtype=int)
    texture = np.asarray(
        [textures[value - 1][index // base] for index, value in enumerate(values)],
        dtype=int,
    )
    return 1 + 4 * (texture - 1) + 20 * (values - 1)


def tukeyTexture(
    x: ArrayLike,
    jitter: bool = True,
    thin: bool = False,
    hollow: bool = False,
    delta: float | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Generate Tukey textured-strip displacements."""
    values = _as_float_array(x)
    if not values.size:
        return np.array([], dtype=float)
    if delta is None:
        delta = float(np.percentile(values, 75) - np.percentile(values, 25)) * 0.03
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    spread = np.resize(tukeyT(random_state=random_state), values.size).astype(float)
    spread[25:50] += 2
    rng = _make_rng(random_state)
    if jitter:
        spread += rng.uniform(-1, 1, size=values.size)
    if thin:
        left = np.r_[np.inf, np.diff(sorted_values)]
        right = np.r_[np.diff(sorted_values), np.inf]
        spread[(left > delta) & (right > delta)] = 50
    if hollow:
        starts = np.arange(0, values.size, 5)
        ends = np.minimum(starts + 4, values.size - 1)
        for start, end in zip(starts, ends):
            candidates = starts[sorted_values[ends] - sorted_values[start] > delta * 10]
            if not candidates.size:
                break
            rightmost = min(int(candidates[0]) + 4, values.size - 1)
            section = spread[start : rightmost + 1]
            spread[start : rightmost + 1] = (
                (section - section.min()) / np.ptp(section) * 100
                if np.ptp(section)
                else 0
            )
    output = np.empty_like(spread)
    output[order] = spread
    return output


def aveWithArgs(
    x: ArrayLike,
    y: Sequence[Hashable] | None = None,
    FUN: Callable[..., Any] = np.mean,
    *args: Any,
    **kwargs: Any,
) -> FloatArray:
    """Apply a function to groups, broadcasting each group's result."""
    values = _as_float_array(x)
    if y is None:
        result = FUN(values, *args, **kwargs)
        return np.full(values.shape, result, dtype=float)
    if len(y) != len(values):
        raise ValueError("x and y not the same length")
    groups: dict[Hashable, list[int]] = {}
    for index, group in enumerate(y):
        groups.setdefault(group, []).append(index)
    output = np.empty(values.size, dtype=float)
    for indices in groups.values():
        result = np.asarray(FUN(values[indices], *args, **kwargs), dtype=float)
        if result.size == 1:
            output[indices] = float(result.flat[0])
        elif result.size == len(indices):
            output[indices] = result
        else:
            raise ValueError("FUN returned a result with the wrong length")
    return output


def vpPlot(
    x: Sequence[Hashable] | None = None,
    y: ArrayLike | None = None,
    *,
    offsetXArgs: dict[str, Any] | None = None,
    **_: Any,
) -> FloatArray:
    """Return categorical x positions for a violin-point plot."""
    if y is None:
        raise TypeError("vpPlot() missing required argument: 'y'")
    values = _as_float_array(y)
    groups = [0] * len(values) if x is None else list(x)
    if len(groups) != len(values):
        raise ValueError("x and y not the same length")
    try:
        levels = sorted(set(groups), key=lambda group: cast(Any, group))
    except TypeError:
        levels = list(dict.fromkeys(groups))
    level_map = {level: index + 1 for index, level in enumerate(levels)}
    ids = np.asarray([level_map[group] for group in groups], dtype=float)
    return ids + offsetX(values, groups, **(offsetXArgs or {}))


def _make_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


__all__ = [
    "aveWithArgs",
    "digits2number",
    "generatePermuteString",
    "number2digits",
    "offsetSingleGroup",
    "offsetX",
    "permute",
    "topBottomDistribute",
    "tukeyPermutes",
    "tukeyT",
    "tukeyTexture",
    "vanDerCorput",
    "vpPlot",
]
