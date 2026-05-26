"""Exact numeric helpers shared across geometry modules."""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["as_fraction"]


def as_fraction(value: float) -> Fraction:
    """Convert a finite float coordinate to the exact rational represented by that float."""
    if not math.isfinite(value):
        raise ValueError(f"Point coordinate must be finite, got {value!r}")

    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)
