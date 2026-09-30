"""Kerr metric in Boyer-Lindquist coordinates (PROJECT.md sections 4 to 5).

Conventions (docs/architecture.md section 1 and docs/equations.md):

* geometric units ``G = c = 1``; lengths and times in units of the mass ``M``;
* signature ``(-, +, +, +)``;
* Boyer-Lindquist coordinates ``(t, r, theta, phi)`` with index order
  ``0, 1, 2, 3``;
* ``a = spin * M`` with the dimensionless spin ``|spin| < 1``; ``a > 0`` means
  the hole rotates towards ``+phi``.

With ``Sigma = r^2 + a^2 cos^2(theta)`` and ``Delta = r^2 - 2 M r + a^2`` the
line element is (Misner, Thorne and Wheeler 1973, *Gravitation*, section
33.2, eq. 33.2 with the Kerr-Newman charge set to zero)::

    ds^2 = -(1 - 2 M r / Sigma) dt^2 - (4 M a r sin^2(theta) / Sigma) dt dphi
           + (Sigma / Delta) dr^2 + Sigma dtheta^2
           + (r^2 + a^2 + 2 M a^2 r sin^2(theta) / Sigma) sin^2(theta) dphi^2

The same metric is written by Bardeen, Press and Teukolsky 1972, ApJ 178,
347, eq. 2.1 as ``ds^2 = -e^{2 nu} dt^2 + e^{2 psi} (dphi - omega dt)^2 +
e^{2 mu_1} dr^2 + e^{2 mu_2} dtheta^2`` with ``e^{2 nu} = Sigma Delta / A``,
``e^{2 psi} = A sin^2(theta) / Sigma``, ``omega = 2 M a r / A`` and
``A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)``; the tests show symbolically
that the two forms agree (``tests/test_metric.py``). Chandrasekhar 1983, *The
Mathematical Theory of Black Holes*, chapter 6, and Visser 2007,
arXiv:0706.0622, section on Boyer-Lindquist coordinates, present the same
line element. The decisive check that the implemented components are the
vacuum Kerr solution is the numerical vanishing of the symbolic Ricci tensor
(``tests/test_christoffel.py``).

Everything here is exact general relativity; no approximation is made. All
functions are NumPy-vectorised over broadcastable ``r`` and ``theta`` arrays
(scalars included) and return float64 arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

__all__ = [
    "Array",
    "InverseMetricComponents",
    "MetricComponents",
    "Spacetime",
    "delta",
    "inverse_metric",
    "inverse_metric_components",
    "inverse_metric_derivatives",
    "kerr",
    "metric",
    "metric_components",
    "metric_derivatives",
    "schwarzschild",
    "sigma",
]


@dataclass(frozen=True)
class Spacetime:
    """A Kerr (or Schwarzschild) black hole in geometric units.

    Attributes:
        mass: The black-hole mass ``M > 0`` (sets the length and time scale).
        spin: The dimensionless spin ``a* = a / M`` with ``|a*| < 1``
            (PROJECT.md section 4.1). Negative values are allowed and denote
            rotation towards ``-phi``.
    """

    mass: float = 1.0
    spin: float = 0.0

    def __post_init__(self) -> None:
        mass = float(self.mass)
        spin = float(self.spin)
        if not (math.isfinite(mass) and mass > 0.0):
            raise ValueError(f"mass must be a finite number > 0, got {self.mass!r}")
        if not (math.isfinite(spin) and abs(spin) < 1.0):
            raise ValueError(f"spin must satisfy abs(spin) < 1, got {self.spin!r}")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "spin", spin)

    @property
    def a(self) -> float:
        """Kerr parameter ``a = spin * mass`` (angular momentum per unit mass)."""
        return self.spin * self.mass

    @property
    def is_schwarzschild(self) -> bool:
        """True when ``spin == 0`` exactly."""
        return self.spin == 0.0


class MetricComponents(NamedTuple):
    """Non-zero covariant components ``g_{mu nu}`` (``g_tphi = g_phit``)."""

    g_tt: Array
    g_tphi: Array
    g_rr: Array
    g_thth: Array
    g_phph: Array


class InverseMetricComponents(NamedTuple):
    """Non-zero contravariant components ``g^{mu nu}`` (``gtphi = gphit``)."""

    gtt: Array
    gtphi: Array
    grr: Array
    gthth: Array
    gphph: Array


def schwarzschild(mass: float = 1.0) -> Spacetime:
    """Return the non-rotating (``spin = 0``) spacetime of the given mass."""
    return Spacetime(mass=mass, spin=0.0)


def kerr(mass: float, spin: float) -> Spacetime:
    """Return the Kerr spacetime with mass ``mass`` and dimensionless ``spin``."""
    return Spacetime(mass=mass, spin=spin)


def _f64(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def sigma(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> Array:
    """``Sigma = r^2 + a^2 cos^2(theta)`` (MTW eq. 33.2; BPT 1972 eq. 2.1)."""
    r_, th = _f64(r), _f64(theta)
    return r_**2 + st.a**2 * np.cos(th) ** 2


def delta(st: Spacetime, r: ArrayLike) -> Array:
    """``Delta = r^2 - 2 M r + a^2`` (MTW eq. 33.2; BPT 1972 eq. 2.1)."""
    r_ = _f64(r)
    return r_**2 - 2.0 * st.mass * r_ + st.a**2


def metric_components(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> MetricComponents:
    """Covariant Kerr metric components in Boyer-Lindquist coordinates.

    Source: MTW 1973 eq. 33.2 (cross-checked symbolically against BPT 1972
    eq. 2.1 in ``tests/test_metric.py``). All other components vanish.
    """
    r_, th = _f64(r), _f64(theta)
    m, a = st.mass, st.a
    s2 = np.sin(th) ** 2
    sig = r_**2 + a**2 * np.cos(th) ** 2
    dlt = r_**2 - 2.0 * m * r_ + a**2
    g_tt = -(1.0 - 2.0 * m * r_ / sig)
    g_tphi = -2.0 * m * a * r_ * s2 / sig
    g_rr = sig / dlt
    g_thth = sig + 0.0 * dlt  # broadcast to the common shape of r and theta
    g_phph = (r_**2 + a**2 + 2.0 * m * a**2 * r_ * s2 / sig) * s2
    return MetricComponents(g_tt, g_tphi, g_rr, g_thth, g_phph)


def inverse_metric_components(
    st: Spacetime, r: ArrayLike, theta: ArrayLike
) -> InverseMetricComponents:
    """Contravariant Kerr metric components ``g^{mu nu}``.

    With ``A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)`` (BPT 1972, eq. 2.1
    definitions)::

        g^tt   = -A / (Sigma Delta)
        g^tphi = -2 M a r / (Sigma Delta)
        g^rr   = Delta / Sigma
        g^thth = 1 / Sigma
        g^phph = (Delta - a^2 sin^2(theta)) / (Sigma Delta sin^2(theta))

    Verified symbolically (``g . g^-1 = 1`` with ``sympy.simplify``) and
    numerically in ``tests/test_metric.py``.
    """
    r_, th = _f64(r), _f64(theta)
    m, a = st.mass, st.a
    s2 = np.sin(th) ** 2
    sig = r_**2 + a**2 * np.cos(th) ** 2
    dlt = r_**2 - 2.0 * m * r_ + a**2
    big_a = (r_**2 + a**2) ** 2 - a**2 * dlt * s2
    sd = sig * dlt
    gtt = -big_a / sd
    gtphi = -2.0 * m * a * r_ / sd
    grr = dlt / sig
    gthth = 1.0 / sig + 0.0 * dlt
    gphph = (dlt - a**2 * s2) / (sd * s2)
    return InverseMetricComponents(gtt, gtphi, grr, gthth, gphph)


def _assemble(c: MetricComponents | InverseMetricComponents) -> Array:
    """Place five components into a symmetric ``(..., 4, 4)`` array."""
    tt, tphi, rr, thth, phph = (np.asarray(x, dtype=np.float64) for x in c)
    shape = np.broadcast(tt, tphi, rr, thth, phph).shape
    out = np.zeros(shape + (4, 4), dtype=np.float64)
    out[..., 0, 0] = tt
    out[..., 0, 3] = tphi
    out[..., 3, 0] = tphi
    out[..., 1, 1] = rr
    out[..., 2, 2] = thth
    out[..., 3, 3] = phph
    return out


def metric(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> Array:
    """Covariant metric ``g_{mu nu}`` as an array of shape ``(..., 4, 4)``."""
    return _assemble(metric_components(st, r, theta))


def inverse_metric(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> Array:
    """Contravariant metric ``g^{mu nu}`` as an array of shape ``(..., 4, 4)``."""
    return _assemble(inverse_metric_components(st, r, theta))


def metric_derivatives(
    st: Spacetime, r: ArrayLike, theta: ArrayLike
) -> tuple[MetricComponents, MetricComponents]:
    """Closed-form ``d/dr`` and ``d/dtheta`` of the five covariant components.

    Obtained by differentiating MTW eq. 33.2 by hand with ``dSigma/dr = 2 r``,
    ``dSigma/dtheta = -a^2 sin(2 theta)`` and ``dDelta/dr = 2 (r - M)``;
    verified against ``sympy.diff`` in ``tests/test_christoffel.py``. Used to
    assemble the Christoffel symbols. The ``t`` and ``phi`` derivatives vanish
    (stationarity and axisymmetry).

    Returns:
        ``(d_r g, d_theta g)`` as two ``MetricComponents`` tuples.
    """
    r_, th = _f64(r), _f64(theta)
    m, a = st.mass, st.a
    s2 = np.sin(th) ** 2
    sin2th = np.sin(2.0 * th)
    sig = r_**2 + a**2 * np.cos(th) ** 2
    dlt = r_**2 - 2.0 * m * r_ + a**2
    sig_r = 2.0 * r_ + 0.0 * th
    sig_th = -(a**2) * sin2th + 0.0 * r_
    dlt_r = 2.0 * (r_ - m)
    sig2 = sig**2
    q_r = (sig - r_ * sig_r) / sig2  # d/dr of r / Sigma
    d_r = MetricComponents(
        g_tt=2.0 * m * q_r,
        g_tphi=-2.0 * m * a * s2 * q_r,
        g_rr=(sig_r * dlt - sig * dlt_r) / dlt**2,
        g_thth=sig_r,
        g_phph=2.0 * r_ * s2 + 2.0 * m * a**2 * s2**2 * q_r,
    )
    d_th = MetricComponents(
        g_tt=-2.0 * m * r_ * sig_th / sig2,
        g_tphi=-2.0 * m * a * r_ * (sin2th * sig - s2 * sig_th) / sig2,
        g_rr=sig_th / dlt,
        g_thth=sig_th,
        g_phph=(r_**2 + a**2) * sin2th
        + 2.0 * m * a**2 * r_ * (2.0 * s2 * sin2th * sig - s2**2 * sig_th) / sig2,
    )
    return d_r, d_th


def inverse_metric_derivatives(
    st: Spacetime, r: ArrayLike, theta: ArrayLike
) -> tuple[InverseMetricComponents, InverseMetricComponents]:
    """Closed-form ``d/dr`` and ``d/dtheta`` of the five contravariant components.

    These drive the Hamiltonian geodesic equations ``dp_mu/dlambda =
    -(1/2) (d_mu g^{alpha beta}) p_alpha p_beta`` (docs/architecture.md
    section 1). They are obtained from the closed forms of
    :func:`inverse_metric_components` by the quotient rule with
    ``A_r = 4 r (r^2 + a^2) - 2 a^2 (r - M) sin^2(theta)``,
    ``A_theta = -a^2 Delta sin(2 theta)`` and ``g^phph = 1 / (Sigma sin^2(theta))
    - a^2 / (Sigma Delta)``; verified against ``sympy.diff`` of the symbolic
    inverse metric to 1e-10 relative in ``tests/test_metric.py``.

    Returns:
        ``(d_r g^-1, d_theta g^-1)`` as two ``InverseMetricComponents`` tuples.
    """
    r_, th = _f64(r), _f64(theta)
    m, a = st.mass, st.a
    s2 = np.sin(th) ** 2
    sin2th = np.sin(2.0 * th)
    sig = r_**2 + a**2 * np.cos(th) ** 2
    dlt = r_**2 - 2.0 * m * r_ + a**2
    sig_r = 2.0 * r_ + 0.0 * th
    sig_th = -(a**2) * sin2th + 0.0 * r_
    dlt_r = 2.0 * (r_ - m)
    big_a = (r_**2 + a**2) ** 2 - a**2 * dlt * s2
    a_r = 4.0 * r_ * (r_**2 + a**2) - 2.0 * a**2 * (r_ - m) * s2
    a_th = -(a**2) * dlt * sin2th
    sd = sig * dlt
    sd_r = sig_r * dlt + sig * dlt_r
    sd_th = sig_th * dlt
    sd2 = sd**2
    sig2 = sig**2
    d_r = InverseMetricComponents(
        gtt=-(a_r * sd - big_a * sd_r) / sd2,
        gtphi=-2.0 * m * a * (sd - r_ * sd_r) / sd2,
        grr=(dlt_r * sig - dlt * sig_r) / sig2,
        gthth=-sig_r / sig2,
        gphph=-sig_r / (sig2 * s2) + a**2 * sd_r / sd2,
    )
    d_th = InverseMetricComponents(
        gtt=-(a_th * sd - big_a * sd_th) / sd2,
        gtphi=2.0 * m * a * r_ * sd_th / sd2,
        grr=-dlt * sig_th / sig2,
        gthth=-sig_th / sig2,
        gphph=-(sig_th * s2 + sig * sin2th) / (sig2 * s2**2) + a**2 * sd_th / sd2,
    )
    return d_r, d_th
