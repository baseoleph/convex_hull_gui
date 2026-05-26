"""Shared random-number source for reproducible algorithm runs."""

from __future__ import annotations

import random


def make_random_seed() -> int:
    """Return a fresh non-deterministic seed value for one operation.

    The returned value fits into the GUI seed spinbox range and can be stored
    in metadata or status messages for reproducibility.

    Returns:
        A non-deterministically chosen 31-bit integer seed.
    """
    return random.SystemRandom().randrange(0, 2_147_483_648)
