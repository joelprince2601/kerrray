"""Random-seed control for reproducible runs (PROJECT.md section 35).

The photon physics in KerrRay is deterministic; randomness only enters through
auxiliary sampling in later phases (for example random impact-parameter
samples or perturbation studies). Every run records its seed in the run
manifest. :data:`DEFAULT_SEED` (``0``) is used whenever a configuration or
command does not specify a seed.
"""

from __future__ import annotations

import random
from typing import Final

import numpy as np

DEFAULT_SEED: Final[int] = 0
"""Seed used when none is given. Documented default for reproducibility."""

_LEGACY_SEED_MODULUS: Final[int] = 2**32
"""``numpy.random.seed`` only accepts seeds below 2**32."""


def set_seed(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed the random sources KerrRay may use and return a NumPy generator.

    The returned :class:`numpy.random.Generator` is the preferred source of
    randomness and should be passed explicitly to any code that needs it. For
    third-party code that still relies on the legacy global generators,
    :func:`random.seed` and :func:`numpy.random.seed` are seeded as well (the
    legacy NumPy seed is ``seed`` modulo ``2**32``).

    Args:
        seed: Non-negative integer seed. Booleans are rejected.

    Returns:
        A fresh ``Generator`` created by :func:`numpy.random.default_rng`.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
    random.seed(value)
    np.random.seed(value % _LEGACY_SEED_MODULUS)
    return np.random.default_rng(value)
