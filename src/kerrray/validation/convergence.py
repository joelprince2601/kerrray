"""Generic convergence-order estimates (PROJECT.md sections 18 to 20, 41).

Two estimators of the order ``p`` in ``e(h) ~ C h^p`` for a discretisation
parameter ``h`` (an RK4 step size, an adaptive ``rtol``, a pixel size):

* :func:`loglog_order`: least-squares slope of ``log e`` against ``log h``
  over every usable pair (``h > 0``, ``e > 0``, both finite). It needs an
  error ``e`` measured against a reference, so it answers "how fast does the
  error fall" when a reference exists.
* :func:`richardson_order`: the reference-free estimate from three solutions
  ``q(h)``, ``q(h/k)``, ``q(h/k^2)`` at a constant refinement ratio ``k``
  (Richardson 1911; Roache 1998, *Verification and Validation in
  Computational Science and Engineering*, section 5.4)::

      p = log(|q(h) - q(h/k)| / |q(h/k) - q(h/k^2)|) / log(k)

  and the Richardson-extrapolated value ``q* = q(h/k^2) + (q(h/k^2) -
  q(h/k)) / (k^p - 1)``.

Both return ``nan`` (never a made-up number) when the data cannot support an
estimate, e.g. differences at round-off level or fewer than two points.
:func:`convergence_study` bundles the per-level errors, successive orders and
the fitted order into one serialisable record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ConvergenceStudy",
    "RichardsonEstimate",
    "convergence_study",
    "loglog_order",
    "richardson_order",
    "successive_orders",
]


def _usable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)


def loglog_order(h: ArrayLike, err: ArrayLike) -> float:
    """Least-squares exponent ``p`` in ``err ~ C h^p`` (``nan`` with fewer than two usable points)."""
    x = np.asarray(h, dtype=np.float64).ravel()
    y = np.asarray(err, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"h and err must have the same length, got {x.size} and {y.size}")
    ok = _usable(x, y)
    if np.count_nonzero(ok) < 2 or np.unique(x[ok]).size < 2:
        return math.nan
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def successive_orders(h: ArrayLike, err: ArrayLike) -> list[float]:
    """Local orders ``log(e_i / e_{i+1}) / log(h_i / h_{i+1})`` between neighbouring levels."""
    x = np.asarray(h, dtype=np.float64).ravel()
    y = np.asarray(err, dtype=np.float64).ravel()
    out = []
    for i in range(x.size - 1):
        pair_x, pair_y = x[i : i + 2], y[i : i + 2]
        if np.all(_usable(pair_x, pair_y)) and pair_x[0] != pair_x[1]:
            out.append(float(math.log(pair_y[0] / pair_y[1]) / math.log(pair_x[0] / pair_x[1])))
        else:
            out.append(math.nan)
    return out


@dataclass(frozen=True)
class RichardsonEstimate:
    """Order and extrapolated value from three solutions at a constant refinement ratio."""

    order: float
    extrapolated: float
    ratio: float

    def as_dict(self) -> dict[str, float]:
        """Plain mapping."""
        return {"order": self.order, "extrapolated": self.extrapolated, "ratio": self.ratio}


def richardson_order(q_coarse: float, q_medium: float, q_fine: float, ratio: float) -> RichardsonEstimate:
    """Richardson order and extrapolation from ``q(h)``, ``q(h/ratio)``, ``q(h/ratio^2)``.

    ``order`` is ``nan`` when either difference is zero or the differences do
    not decrease (no asymptotic convergence), and ``extrapolated`` is then
    ``nan`` too.
    """
    if not ratio > 1.0:
        raise ValueError(f"ratio must be > 1, got {ratio!r}")
    d1 = abs(q_coarse - q_medium)
    d2 = abs(q_medium - q_fine)
    if not (d1 > 0.0 and d2 > 0.0 and d2 < d1 and math.isfinite(d1) and math.isfinite(d2)):
        return RichardsonEstimate(math.nan, math.nan, ratio)
    p = math.log(d1 / d2) / math.log(ratio)
    extrapolated = q_fine + (q_fine - q_medium) / (ratio**p - 1.0)
    return RichardsonEstimate(p, extrapolated, ratio)


@dataclass(frozen=True)
class ConvergenceStudy:
    """Values ``q(h)`` of a quantity at several levels ``h`` and their error analysis.

    Attributes:
        parameter: Name of ``h`` (``"rtol"``, ``"step_size"``, ...).
        levels: The ``h`` values, in the order given.
        values: ``q(h)``.
        reference: Value the errors are measured against (``nan`` if none).
        errors: ``|q(h) - reference|``.
        fitted_order: :func:`loglog_order` of ``errors`` against ``levels``,
            restricted to errors above ``noise_floor``.
        local_orders: :func:`successive_orders`.
        richardson: Estimate from the three finest levels when they are in a
            constant ratio, else ``None``.
        noise_floor: Errors at or below this (e.g. the bisection half-width)
            are excluded from the fit because they are not resolved.
        extra: Per-level metadata (runtime, steps, ...).
    """

    parameter: str
    levels: tuple[float, ...]
    values: tuple[float, ...]
    reference: float
    errors: tuple[float, ...]
    fitted_order: float
    local_orders: tuple[float, ...]
    richardson: RichardsonEstimate | None
    noise_floor: float
    extra: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly mapping."""
        return {
            "parameter": self.parameter,
            "levels": list(self.levels),
            "values": list(self.values),
            "reference": self.reference,
            "errors": list(self.errors),
            "fitted_order": self.fitted_order,
            "local_orders": list(self.local_orders),
            "richardson": self.richardson.as_dict() if self.richardson is not None else None,
            "noise_floor": self.noise_floor,
            "levels_detail": list(self.extra),
        }


def convergence_study(
    parameter: str,
    levels: Sequence[float],
    values: Sequence[float],
    reference: float,
    *,
    noise_floor: float = 0.0,
    extra: Sequence[dict[str, Any]] = (),
) -> ConvergenceStudy:
    """Assemble a :class:`ConvergenceStudy` (errors, fitted and local orders, Richardson).

    The Richardson estimate uses the last three levels when their ratios agree
    to 1e-9 relative (levels ordered coarse to fine).
    """
    h = np.asarray(levels, dtype=np.float64)
    q = np.asarray(values, dtype=np.float64)
    if h.shape != q.shape:
        raise ValueError("levels and values must have the same length")
    err = np.abs(q - reference) if math.isfinite(reference) else np.full_like(q, math.nan)
    resolved = np.where(err > noise_floor, err, math.nan)
    richardson = None
    if h.size >= 3:
        k1, k2 = h[-3] / h[-2], h[-2] / h[-1]
        if k1 > 1.0 and abs(k1 - k2) <= 1e-9 * k1:
            richardson = richardson_order(float(q[-3]), float(q[-2]), float(q[-1]), float(k1))
    return ConvergenceStudy(
        parameter=parameter,
        levels=tuple(float(v) for v in h),
        values=tuple(float(v) for v in q),
        reference=float(reference),
        errors=tuple(float(v) for v in err),
        fitted_order=loglog_order(h, resolved),
        local_orders=tuple(successive_orders(h, resolved)),
        richardson=richardson,
        noise_floor=float(noise_floor),
        extra=tuple(dict(e) for e in extra),
    )
