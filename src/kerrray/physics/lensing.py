"""Strong-field gravitational lensing: deflection angles of equatorial photons
(PROJECT.md section 22; docs/lensing.md).

Three ways to obtain the deflection of a photon with impact parameter ``b``:

* :func:`deflection_from_trajectory`: from an integrated equatorial ray
  (numerical, Schwarzschild or Kerr). The asymptotic direction of each leg is
  taken from the coordinate velocity at the end points (docs/lensing.md
  section 2), so the finite launch radius contributes only
  ``1.5 M b^3 / r_0^4`` in Schwarzschild (derived there and verified in
  ``tests/test_lensing.py``) plus ``2 M |a| / r_0^2`` of frame dragging in
  Kerr.
* :func:`schwarzschild_deflection_exact`: numerical quadrature of the exact
  Schwarzschild deflection integral (Darwin 1959, Proc. R. Soc. A 249, 180;
  Weinberg 1972, *Gravitation and Cosmology*, section 8.5). Reference only.
* :func:`weak_field_deflection`: the Einstein value ``4 M / b`` (labelled
  approximation).

:func:`strong_field_fit` fits the logarithmic divergence
``alpha = -a_bar log(b / b_c - 1) + b_bar`` of Bozza 2002, Phys. Rev. D 66,
103001, near the critical impact parameter; for Schwarzschild ``a_bar = 1``
and ``b_bar = log[216 (7 - 4 sqrt 3)] - pi`` (Darwin 1959, as quoted by
Bozza 2002), returned by :func:`darwin_strong_field_coefficients` as the
comparison value and re-derived numerically from the exact integral in the
tests (docs/lensing.md section 4).

Conventions: ``G = c = 1``, Boyer-Lindquist coordinates, ``b = |L_z| / E``,
deflection in radians, windings included (a ray that circles the hole ``n``
times has ``alpha > 2 pi n``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

from kerrray.geodesics.equations import geodesic_rhs
from kerrray.geodesics.state import IDX_PH, IDX_R, IDX_TH
from kerrray.geometry import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.photons.trajectories import EQUATORIAL_TOLERANCE, Trajectory

__all__ = [
    "QUAD_LIMIT",
    "QUAD_RTOL",
    "StrongFieldFit",
    "darwin_strong_field_coefficients",
    "deflection_from_trajectory",
    "schwarzschild_closest_approach",
    "schwarzschild_critical_impact_parameter",
    "schwarzschild_deflection_exact",
    "strong_field_fit",
    "weak_field_deflection",
]

FloatArray = NDArray[np.float64]

QUAD_RTOL: Final[float] = 1e-12
"""Relative tolerance of the adaptive quadrature in :func:`schwarzschild_deflection_exact`.

Measured (docs/lensing.md section 3): the integrand is smooth after the
end-point substitution, the reported error estimate of the deflection is at
most ``1.6e-11`` rad for ``b / b_c - 1`` between ``1e-8`` and ``1e3``, and tighter tolerances trigger
round-off warnings from QUADPACK without gaining accuracy.
"""

QUAD_LIMIT: Final[int] = 500
"""Maximum number of QUADPACK sub-intervals (the near-critical integrand is
sharply peaked at ``t = 0``, docs/lensing.md section 3)."""

_ROOT_XTOL: Final[float] = 1e-14
_ROOT_RTOL: Final[float] = 1e-14


def _require_schwarzschild(st: Spacetime) -> float:
    if not st.is_schwarzschild:
        raise ValueError("the exact deflection integral is implemented for Schwarzschild (spin = 0) only")
    return float(st.mass)


def schwarzschild_critical_impact_parameter(st: Spacetime) -> float:
    """``b_c = |L_z| / E`` of the circular photon orbit, computed by root finding.

    Delegates to :func:`kerrray.photons.orbits.critical_impact_parameters`;
    the reference value ``3 sqrt(3) M`` is used only to *check* it in the
    tests, never to produce it.
    """
    _require_schwarzschild(st)
    return float(critical_impact_parameters(st)[0])


def schwarzschild_closest_approach(st: Spacetime, b: float) -> float:
    """Closest approach ``r_0 > 3M`` of a Schwarzschild photon with ``b > b_c``.

    ``r_0`` is the largest root of the turning-point condition ``dr/dphi = 0``
    of the orbit equation (Weinberg 1972 section 8.5; docs/lensing.md
    section 3), ``b^2 = r_0^3 / (r_0 - 2M)``, i.e. of ``f(r) = r^3 - b^2 (r -
    2M)``. ``f(3M) = M (27 M^2 - b^2) < 0`` for ``b > 3 sqrt(3) M`` and
    ``f(b) = 2 M b^2 > 0``, so the root is bracketed by ``[3M, b]`` and is
    found with ``brentq``. The tests compare it with the trigonometric closed
    form ``r_0 = (2 b / sqrt 3) cos[(1/3) arccos(-3 sqrt(3) M / b)]``.

    Raises:
        ValueError: If ``b <= b_c`` (the photon is captured; no turning point).
    """
    mass = _require_schwarzschild(st)
    b_c = schwarzschild_critical_impact_parameter(st)
    b = float(b)
    if not (math.isfinite(b) and b > b_c):
        raise ValueError(f"need b > b_c = {b_c:.12g} M for a scattering orbit, got b = {b!r}")

    def f(r: float) -> float:
        return r * r * r - b * b * (r - 2.0 * mass)

    return float(brentq(f, 3.0 * mass, b, xtol=_ROOT_XTOL * mass, rtol=_ROOT_RTOL))


def _deflection_scalar(mass: float, b: float, r_0: float) -> float:
    """Quadrature of the exact deflection integral after the end-point substitution.

    With ``u = 1/r`` the integral ``2 int_{r_0}^inf dr / (r^2 sqrt(1/b^2 - (1 -
    2M/r)/r^2))`` becomes ``2 int_0^{u_0} du / sqrt(P(u))`` with ``P(u) = 1/b^2
    - u^2 + 2 M u^3`` and ``P(u_0) = 0``. Factoring the root, ``P(u) = (u_0 -
    u) Q(u)`` with ``Q(u) = (u + u_0) - 2M (u^2 + u u_0 + u_0^2)`` (polynomial
    division, docs/lensing.md section 3), and the substitution ``u = u_0 (1 -
    t^2)`` removes the inverse-square-root end-point singularity::

        alpha = 4 sqrt(u_0) int_0^1 dt / sqrt(Q(u_0 (1 - t^2))) - pi
    """
    u_0 = 1.0 / r_0

    def integrand(t: float) -> float:
        u = u_0 * (1.0 - t * t)
        q = (u + u_0) - 2.0 * mass * (u * u + u * u_0 + u_0 * u_0)
        return 1.0 / math.sqrt(q)

    value, _err = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=QUAD_RTOL, limit=QUAD_LIMIT)
    return 4.0 * math.sqrt(u_0) * value - math.pi


def schwarzschild_deflection_exact(st: Spacetime, b: ArrayLike) -> float | FloatArray:
    """Exact Schwarzschild deflection angle ``alpha(b)`` by numerical quadrature.

    ``alpha = 2 int_{r_0}^inf dr / (r^2 sqrt(1/b^2 - (1 - 2M/r)/r^2)) - pi``
    with ``r_0`` from :func:`schwarzschild_closest_approach` (Darwin 1959;
    Weinberg 1972 section 8.5; docs/lensing.md section 3). The end-point
    singularity is removed analytically (:func:`_deflection_scalar`), so the
    quadrature is smooth; ``b`` may be a scalar (a ``float`` is returned) or
    an array (evaluated element-wise). Every ``b`` must exceed ``b_c``.

    The value includes all windings: it diverges as ``-log(b / b_c - 1)``
    when ``b -> b_c`` (Bozza 2002) and tends to ``4 M / b`` for ``b >> M``
    (both verified in ``tests/test_lensing.py``).
    """
    mass = _require_schwarzschild(st)
    b_arr = np.asarray(b, dtype=np.float64)
    flat = b_arr.ravel()
    out = np.empty(flat.shape, dtype=np.float64)
    for k, bk in enumerate(flat):
        r_0 = schwarzschild_closest_approach(st, float(bk))
        out[k] = _deflection_scalar(mass, float(bk), r_0)
    if b_arr.ndim == 0:
        return float(out[0])
    return out.reshape(b_arr.shape)


def weak_field_deflection(st: Spacetime, b: ArrayLike) -> FloatArray:
    """Weak-field deflection ``alpha = 4 M / b`` (approximation).

    * WHAT: the leading term of the expansion of the exact deflection in
      ``M / b`` (Weinberg 1972 section 8.5; the next term is ``15 pi M^2 /
      (4 b^2)``).
    * WHY: the textbook reference against which the strong-field results are
      contrasted; it is exact only as ``b -> infinity``.
    * LIMITATION: the relative error is ``~ (15 pi / 16) M / b`` (``3e-3`` at
      ``b = 1000 M``, ``0.3`` at ``b = 10 M``) and the formula is meaningless
      near ``b_c`` where the exact deflection diverges. It does not depend on
      the spin: frame dragging enters at order ``a M / b^2``.
    """
    b_arr = np.asarray(b, dtype=np.float64)
    return 4.0 * float(st.mass) / b_arr


def deflection_from_trajectory(traj: Trajectory) -> float:
    """Deflection of an equatorial escaped ray from its end-point directions.

    At an end point ``(r, phi)`` with coordinate velocity ``(dr/dlambda,
    dphi/dlambda)`` the tangent has azimuth ``chi = phi + atan2(r dphi/dlambda,
    dr/dlambda)`` in the asymptotically flat plane; the deflection is
    ``|chi_exit - chi_launch|`` with ``phi`` continuous (windings included).
    Derivation, residual and comparison with the straight-line correction of
    :func:`kerrray.photons.trajectories.deflection_angle` in docs/lensing.md
    section 2: in Schwarzschild the value is short of the ``r_0 -> infinity``
    limit by ``1.5 M b^3 / r_0^4`` (``1.2e-8`` for ``b = 20``, ``r_0 = 1000``),
    in Kerr an additional ``2 M |a| / r_0^2`` of frame dragging remains.

    Raises:
        ValueError: Unless the ray is ``ESCAPED`` and stays within
            :data:`kerrray.photons.trajectories.EQUATORIAL_TOLERANCE` of the
            equatorial plane.
    """
    if traj.state != TerminationState.ESCAPED:
        raise ValueError(f"deflection needs an ESCAPED ray, got {traj.state.name}")
    theta = traj.y[:, IDX_TH]
    if np.max(np.abs(theta - 0.5 * np.pi)) > EQUATORIAL_TOLERANCE:
        raise ValueError("deflection_from_trajectory is defined for equatorial rays only")
    st = traj.spacetime
    y0, y1 = traj.y[0], traj.y[-1]
    f0, f1 = geodesic_rhs(st, y0), geodesic_rhs(st, y1)
    chi0 = float(y0[IDX_PH]) + math.atan2(float(y0[IDX_R] * f0[IDX_PH]), float(f0[IDX_R]))
    chi1 = float(y1[IDX_PH]) + math.atan2(float(y1[IDX_R] * f1[IDX_PH]), float(f1[IDX_R]))
    return abs(chi1 - chi0)


@dataclass(frozen=True)
class StrongFieldFit:
    """Least-squares coefficients of ``alpha = -a_bar log(b / b_c - 1) + b_bar``.

    Attributes:
        a_bar, b_bar: Fitted coefficients (Bozza 2002 notation).
        b_critical: The ``b_c`` used for the abscissa.
        n_points: Number of points with ``b > b_c`` and finite ``alpha`` used.
        rms_residual: Root-mean-square residual of the fit (radians).
    """

    a_bar: float
    b_bar: float
    b_critical: float
    n_points: int
    rms_residual: float

    def evaluate(self, b: ArrayLike) -> FloatArray:
        """The fitted ``alpha(b)`` (NaN where ``b <= b_c``)."""
        b_arr = np.asarray(b, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.log(b_arr / self.b_critical - 1.0)
        return -self.a_bar * x + self.b_bar


def strong_field_fit(b: ArrayLike, alpha: ArrayLike, *, b_c: float) -> StrongFieldFit:
    """Fit ``alpha = -a_bar log(b / b_c - 1) + b_bar`` to deflection data near ``b_c``.

    Bozza 2002 shows that the deflection of any spherically symmetric black
    hole diverges logarithmically at the photon sphere with this form; for
    Schwarzschild ``a_bar = 1``. Points with ``b <= b_c`` or non-finite
    ``alpha`` are ignored; at least two usable points are required. The fit
    is linear in ``(a_bar, b_bar)`` and is solved by least squares. The
    coefficients depend on how close to ``b_c`` the data lie (the expansion
    has corrections of order ``(b/b_c - 1) log(b/b_c - 1)``, Bozza 2002), so
    they are reported, not asserted tightly.
    """
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    a_arr = np.asarray(alpha, dtype=np.float64).ravel()
    if b_arr.shape != a_arr.shape:
        raise ValueError("b and alpha must have the same length")
    if not (math.isfinite(b_c) and b_c > 0.0):
        raise ValueError(f"b_c must be a finite positive number, got {b_c!r}")
    usable = (b_arr > b_c) & np.isfinite(a_arr)
    if np.count_nonzero(usable) < 2:
        raise ValueError("strong_field_fit needs at least two points with b > b_c and finite alpha")
    x = np.log(b_arr[usable] / b_c - 1.0)
    design = np.column_stack([-x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, a_arr[usable], rcond=None)
    residual = a_arr[usable] - design @ coef
    return StrongFieldFit(
        a_bar=float(coef[0]),
        b_bar=float(coef[1]),
        b_critical=float(b_c),
        n_points=int(np.count_nonzero(usable)),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
    )


def darwin_strong_field_coefficients() -> tuple[float, float]:
    """Schwarzschild strong-deflection coefficients ``(a_bar, b_bar)`` of Darwin 1959.

    ``a_bar = 1`` and ``b_bar = log[216 (7 - 4 sqrt 3)] - pi`` (approximately
    ``-0.4002``), the Schwarzschild entry of Bozza 2002 (Phys. Rev. D 66,
    103001). These are *comparison* values: ``tests/test_lensing.py``
    re-derives them from the exact integral of
    :func:`schwarzschild_deflection_exact` (fit over ``b / b_c - 1`` between
    ``1e-6`` and ``1e-3``: ``a_bar = 0.9997``, ``b_bar = -0.396``, the
    difference being the neglected ``(b/b_c - 1) log(b/b_c - 1)`` terms).
    """
    return 1.0, math.log(216.0 * (7.0 - 4.0 * math.sqrt(3.0))) - math.pi
