"""Python compatibility module for the public API of R's vipor package."""

from .core import (
    aveWithArgs,
    digits2number,
    generatePermuteString,
    number2digits,
    offsetSingleGroup,
    offsetX,
    permute,
    topBottomDistribute,
    tukeyPermutes,
    tukeyT,
    tukeyTexture,
    vanDerCorput,
    vpPlot,
)

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
