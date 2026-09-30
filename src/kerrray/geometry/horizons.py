"""Kerr horizons and ergosphere (PROJECT.md sections 6 and 7).

The horizons are the roots of ``Delta = r^2 - 2 M r + a^2 = 0`` where
``g_rr = Sigma / Delta`` diverges (MTW 1973 section 33.2; BPT 1972 section
II)::

    r_pm = M +- sqrt(M^2 - a^2)

The ergosphere (outer stationary-limit surface) is where the Killing vector
``d/dt`` becomes null, ``g_tt = 0``, i.e. ``Sigma = 2 M r`` (MTW 1973 chapter
33; Visser 2007 arXiv:0706.0622)::

    r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))

Both are verified in ``tests/test_horizon.py``: ``Delta(r_pm) = 0``,
``g_tt(r_E, theta) = 0``, ``r_E(0) = r_plus``, ``r_E(pi/2) = 2 M``, the
Schwarzschild limit ``a -> 0`` and the extremal limit ``a -> M``. Photon
capture uses the outer horizon ``r_plus`` (PROJECT.md section 6).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geometry.metric import Array, Spacetime

__all__ = ["ergosphere_radius", "horizon_radii", "inside_horizon", "outer_horizon"]


def horizon_radii(st: Spacetime) -> tuple[float, float]:
    """Return ``(r_plus, r_minus) = M +- sqrt(M^2 - a^2)`` (units of ``M``)."""
    m = st.mass
    root = math.sqrt(m * m - st.a * st.a)
    return m + root, m - root


def outer_horizon(st: Spacetime) -> float:
    """The outer event horizon ``r_plus``, the capture radius (section 6)."""
    return horizon_radii(st)[0]


def ergosphere_radius(st: Spacetime, theta: ArrayLike) -> Array:
    """Outer stationary-limit radius ``r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))``."""
    th = np.asarray(theta, dtype=np.float64)
    m = st.mass
    return m + np.sqrt(m * m - st.a**2 * np.cos(th) ** 2)


def inside_horizon(st: Spacetime, r: ArrayLike, epsilon: float = 0.0) -> NDArray[np.bool_]:
    """Boolean mask ``r <= r_plus + epsilon`` (the CAPTURED rule of architecture.md).

    Args:
        st: The spacetime.
        r: Boyer-Lindquist radius (scalar or array).
        epsilon: Non-negative safety margin added to ``r_plus`` (``horizon_epsilon``
            in the termination configuration).
    """
    if not (math.isfinite(epsilon) and epsilon >= 0.0):
        raise ValueError(f"epsilon must be a finite number >= 0, got {epsilon!r}")
    r_ = np.asarray(r, dtype=np.float64)
    return r_ <= outer_horizon(st) + epsilon
