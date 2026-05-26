"""Exact canonical keys for 3D planes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from domain.entities import Point3D
from geometry.exact_numbers import as_fraction

__all__ = [
    "PlaneKey",
    "canonical_plane_key",
    "point_lies_on_plane",
]


@dataclass(frozen=True, order=True)
class PlaneKey:
    """Canonical exact key for a 3D plane.

    Represents:
        a*x + b*y + c*z + d = 0
    """

    a: Fraction
    b: Fraction
    c: Fraction
    d: Fraction


def _normalize_plane_coefficients(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    d: Fraction,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Normalize plane coefficients to a canonical sign and scale."""
    coefficients = (a, b, c, d)
    first_nonzero = next((coefficient for coefficient in coefficients if coefficient != 0), None)
    if first_nonzero is None:
        raise ValueError("Cannot build a plane from collinear points.")

    scale = abs(first_nonzero)
    normalized_a = a / scale
    normalized_b = b / scale
    normalized_c = c / scale
    normalized_d = d / scale
    normalized = (normalized_a, normalized_b, normalized_c, normalized_d)
    if normalized[next(index for index, value in enumerate(normalized) if value != 0)] < 0:
        return (-normalized_a, -normalized_b, -normalized_c, -normalized_d)
    return normalized_a, normalized_b, normalized_c, normalized_d


def canonical_plane_key(a: Point3D, b: Point3D, c: Point3D) -> PlaneKey:
    """Build a canonical exact key for the plane through three non-collinear points."""
    ax = as_fraction(a.x)
    ay = as_fraction(a.y)
    az = as_fraction(a.z)

    abx = as_fraction(b.x) - ax
    aby = as_fraction(b.y) - ay
    abz = as_fraction(b.z) - az

    acx = as_fraction(c.x) - ax
    acy = as_fraction(c.y) - ay
    acz = as_fraction(c.z) - az

    coefficient_a = aby * acz - abz * acy
    coefficient_b = abz * acx - abx * acz
    coefficient_c = abx * acy - aby * acx
    if coefficient_a == 0 and coefficient_b == 0 and coefficient_c == 0:
        raise ValueError("Cannot build a plane from collinear points.")

    coefficient_d = -(coefficient_a * ax + coefficient_b * ay + coefficient_c * az)
    normalized = _normalize_plane_coefficients(
        coefficient_a,
        coefficient_b,
        coefficient_c,
        coefficient_d,
    )
    return PlaneKey(*normalized)


def point_lies_on_plane(point: Point3D, plane: PlaneKey) -> bool:
    """Return True if point satisfies the plane equation exactly."""
    x = as_fraction(point.x)
    y = as_fraction(point.y)
    z = as_fraction(point.z)
    return plane.a * x + plane.b * y + plane.c * z + plane.d == 0
