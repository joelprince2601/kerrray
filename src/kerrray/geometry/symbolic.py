"""Symbolic Kerr geometry (SymPy): the authoritative reference layer.

Everything in this module is derived from the closed-form Kerr metric in
Boyer-Lindquist coordinates by symbolic differentiation and algebra; nothing
downstream is typed in from memory. The numerical geometry implementation
(``kerrray.geometry.metric`` and ``kerrray.geometry.christoffel``) is
cross-checked against the callables returned here (PROJECT.md section 32,
``docs/architecture.md`` section 1.1, ``docs/derivations.md`` section 1).

Conventions (``docs/architecture.md`` section 1): geometric units G = c = 1,
signature (-, +, +, +), coordinates (t, r, theta, phi) with index order
0, 1, 2, 3, Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2 M r + a^2, and
``a`` the dimensional spin (a = spin * M).

References for the metric and inverse metric:

* Bardeen, Press & Teukolsky 1972, ApJ 178, 347, eq. (2.1) (line element;
  the cross term -4 M a r sin^2(theta) / Sigma dt dphi gives
  g_tphi = -2 M a r sin^2(theta) / Sigma).
* Misner, Thorne & Wheeler 1973, *Gravitation*, eq. (33.2) and (33.3).
* Chandrasekhar 1983, *The Mathematical Theory of Black Holes*, ch. 6,
  eq. (57) for the metric and eq. (60) for the contravariant form.
* Visser 2007, arXiv:0706.0622, section 2 (Boyer-Lindquist form).

References for the definitions of the connection and the Ricci tensor:

* Christoffel symbols of the Levi-Civita connection,
  Gamma^mu_{alpha beta} = (1/2) g^{mu nu} (d_alpha g_{nu beta}
  + d_beta g_{nu alpha} - d_nu g_{alpha beta}): Carroll 2004, *Spacetime and
  Geometry*, eq. (3.27); MTW eq. (8.24b).
* Ricci tensor R_{mu nu} = d_lambda Gamma^lambda_{mu nu}
  - d_nu Gamma^lambda_{mu lambda} + Gamma^lambda_{lambda sigma}
  Gamma^sigma_{mu nu} - Gamma^lambda_{nu sigma} Gamma^sigma_{mu lambda}:
  Carroll 2004 eq. (3.113) contracted on the first and third index;
  MTW eq. (8.51). Kerr is a vacuum solution, so R_{mu nu} must vanish
  identically; this is convention independent.

SymPy is imported lazily inside the functions so that importing this module
never pulls SymPy onto a numerical hot path (architecture section 8). All
results are cached with :func:`functools.lru_cache`; the symbolic objects
returned are immutable.

Full simplification of the Christoffel symbols is deliberately avoided:
``sympy.simplify`` on all 64 components was measured at about 550 s, while
``sympy.factor`` takes under 1 s and halves the operation count, and the
Ricci tensor is left unsimplified and evaluated numerically as
``docs/architecture.md`` section 1.1 prescribes.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

# Type alias for a lambdified callable evaluated at one point (M, a, r, theta).
PointCallable = Callable[[float, float, float, float], Any]


class KerrSymbols(NamedTuple):
    """The SymPy symbols shared by every symbolic object of this module.

    All six symbols are declared real (and nothing more): declaring ``M``
    and ``r`` positive was measured to slow ``sympy.factor`` on the
    Christoffel symbols from under 1 s to about 250 s. ``a`` is the
    dimensional spin parameter (a = spin * M).
    """

    t: Any
    r: Any
    theta: Any
    phi: Any
    mass: Any
    a: Any


@lru_cache(maxsize=None)
def kerr_symbols() -> KerrSymbols:
    """Return the shared coordinate and parameter symbols (t, r, theta, phi, M, a)."""
    import sympy as sp

    t, r, theta, phi, mass, a = sp.symbols("t r theta phi M a", real=True)
    return KerrSymbols(t=t, r=r, theta=theta, phi=phi, mass=mass, a=a)


def coordinates() -> tuple[Any, Any, Any, Any]:
    """Return the coordinate symbols (t, r, theta, phi) in index order 0..3."""
    s = kerr_symbols()
    return (s.t, s.r, s.theta, s.phi)


@lru_cache(maxsize=None)
def symbolic_metric() -> tuple[Any, KerrSymbols]:
    """Covariant Kerr metric g_{mu nu} in Boyer-Lindquist coordinates.

    Returns an immutable 4x4 SymPy matrix and the symbols it uses. The
    components are the closed form of BPT 1972 eq. (2.1) (see the module
    docstring for the full reference list); they are verified, not trusted:
    ``tests/test_symbolic_geometry.py`` checks g times g^-1 = I, the
    Schwarzschild limit a -> 0, det g = -Sigma^2 sin^2(theta) and the vacuum
    condition R_{mu nu} = 0 numerically.
    """
    import sympy as sp

    s = kerr_symbols()
    r, th, M, a = s.r, s.theta, s.mass, s.a
    sigma = r**2 + a**2 * sp.cos(th) ** 2
    delta = r**2 - 2 * M * r + a**2
    sin2 = sp.sin(th) ** 2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * M * r / sigma)
    g[0, 3] = -2 * M * a * r * sin2 / sigma
    g[3, 0] = g[0, 3]
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = (r**2 + a**2 + 2 * M * a**2 * r * sin2 / sigma) * sin2
    return sp.ImmutableMatrix(g), s


@lru_cache(maxsize=None)
def symbolic_inverse_metric() -> Any:
    """Contravariant Kerr metric g^{mu nu} (simplified closed form).

    The closed form (architecture section 1.1; Chandrasekhar 1983 ch. 6
    eq. (60)) is written with A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta) and
    then *verified* against the covariant matrix: this function raises
    ``AssertionError`` unless ``simplify(g * g_inv) == I`` holds symbolically,
    so a wrong closed form can never be returned.
    """
    import sympy as sp

    g, s = symbolic_metric()
    r, th, M, a = s.r, s.theta, s.mass, s.a
    sigma = r**2 + a**2 * sp.cos(th) ** 2
    delta = r**2 - 2 * M * r + a**2
    sin2 = sp.sin(th) ** 2
    big_a = (r**2 + a**2) ** 2 - a**2 * delta * sin2
    ginv = sp.zeros(4, 4)
    ginv[0, 0] = -big_a / (sigma * delta)
    ginv[0, 3] = -2 * M * a * r / (sigma * delta)
    ginv[3, 0] = ginv[0, 3]
    ginv[1, 1] = delta / sigma
    ginv[2, 2] = 1 / sigma
    ginv[3, 3] = (delta - a**2 * sin2) / (sigma * delta * sin2)
    product = (g * ginv).applyfunc(sp.simplify)
    if product != sp.eye(4):
        raise AssertionError("closed-form inverse metric does not invert g")
    return sp.ImmutableMatrix(ginv)


@lru_cache(maxsize=None)
def symbolic_christoffel() -> Any:
    """Christoffel symbols Gamma^mu_{alpha beta} from the metric definition.

    Returns an immutable SymPy array of shape (4, 4, 4) indexed
    ``[mu, alpha, beta]`` (architecture section 3). Each component is
    computed from Carroll 2004 eq. (3.27) with the symbolic inverse metric
    and symbolic derivatives of the covariant metric, then lightly
    simplified with ``sympy.factor`` (measured: under 1 s for all 64
    components; ``sympy.simplify`` would take about 550 s).
    """
    import sympy as sp

    g, _ = symbolic_metric()
    ginv = symbolic_inverse_metric()
    x = coordinates()
    dg = [[[sp.diff(g[i, j], x[k]) for j in range(4)] for i in range(4)] for k in range(4)]
    half = sp.Rational(1, 2)
    gamma = [[[sp.Integer(0)] * 4 for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for al in range(4):
            for be in range(4):
                expr = sp.Integer(0)
                for nu in range(4):
                    if ginv[mu, nu] != 0:
                        expr += ginv[mu, nu] * (dg[al][nu][be] + dg[be][nu][al] - dg[nu][al][be])
                gamma[mu][al][be] = sp.factor(half * expr)
    return sp.ImmutableDenseNDimArray(gamma)


@lru_cache(maxsize=None)
def symbolic_ricci() -> Any:
    """Ricci tensor R_{mu nu} built from the Christoffel symbols, unsimplified.

    Uses Carroll 2004 eq. (3.113) contracted, R_{mu nu} = R^lambda_{mu lambda nu}.
    The expression is left unsimplified on purpose (architecture section
    1.1); use :func:`ricci_tensor_numeric` to evaluate it at a point.
    """
    import sympy as sp

    gamma = symbolic_christoffel()
    x = coordinates()
    ricci = [[sp.Integer(0)] * 4 for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            expr = sp.Integer(0)
            for lam in range(4):
                expr += sp.diff(gamma[lam, mu, nu], x[lam]) - sp.diff(gamma[lam, mu, lam], x[nu])
                for sig in range(4):
                    expr += gamma[lam, lam, sig] * gamma[sig, mu, nu]
                    expr -= gamma[lam, nu, sig] * gamma[sig, mu, lam]
            ricci[mu][nu] = expr
    return sp.ImmutableMatrix(ricci)


def _lambdify(expr: Any) -> PointCallable:
    """Lambdify ``expr`` over (M, a, r, theta) with NumPy and common-subexpression elimination."""
    import sympy as sp

    s = kerr_symbols()
    return sp.lambdify((s.mass, s.a, s.r, s.theta), expr, modules="numpy", cse=True)


@lru_cache(maxsize=None)
def lambdified_metric() -> PointCallable:
    """Callable ``(M, a, r, theta) -> ndarray(4, 4)`` for the covariant metric at a point."""
    g, _ = symbolic_metric()
    return _as_array_callable(_lambdify(g))


@lru_cache(maxsize=None)
def lambdified_inverse_metric() -> PointCallable:
    """Callable ``(M, a, r, theta) -> ndarray(4, 4)`` for the contravariant metric at a point."""
    return _as_array_callable(_lambdify(symbolic_inverse_metric()))


@lru_cache(maxsize=None)
def lambdified_christoffel() -> PointCallable:
    """Callable ``(M, a, r, theta) -> ndarray(4, 4, 4)``, indexed ``[mu, alpha, beta]``.

    Scalar arguments only (one spacetime point per call); the caller loops
    over points. The result is a float64 NumPy array.
    """
    return _as_array_callable(_lambdify(symbolic_christoffel()))


@lru_cache(maxsize=None)
def lambdified_inverse_metric_derivatives() -> Callable[[float, float, float, float], tuple[Any, Any]]:
    """Callable ``(M, a, r, theta) -> (d_r g^{mu nu}, d_theta g^{mu nu})``.

    Both outputs are ``ndarray(4, 4)`` float64 arrays of the symbolic partial
    derivatives of the contravariant metric (the only two non-vanishing
    derivatives; d_t and d_phi are zero by stationarity and axisymmetry).
    Used to verify the closed-form ``inverse_metric_derivatives`` of the
    numerical geometry layer.
    """
    import sympy as sp

    s = kerr_symbols()
    ginv = symbolic_inverse_metric()
    d_r = ginv.applyfunc(lambda e: sp.diff(e, s.r))
    d_theta = ginv.applyfunc(lambda e: sp.diff(e, s.theta))
    f = _lambdify((d_r, d_theta))

    def evaluate(mass: float, a: float, r: float, theta: float) -> tuple[Any, Any]:
        out_r, out_theta = f(mass, a, r, theta)
        return (
            np.asarray(out_r, dtype=np.float64),
            np.asarray(out_theta, dtype=np.float64),
        )

    return evaluate


@lru_cache(maxsize=None)
def lambdified_ricci() -> PointCallable:
    """Callable ``(M, a, r, theta) -> ndarray(4, 4)`` for the unsimplified Ricci tensor."""
    return _as_array_callable(_lambdify(symbolic_ricci()))


def ricci_tensor_numeric(mass: float, a: float, r: float, theta: float) -> Any:
    """Evaluate the symbolic Ricci tensor R_{mu nu} at one point.

    Parameters are the mass ``M``, dimensional spin ``a``, and the
    Boyer-Lindquist ``r`` and ``theta`` (radians). Returns a float64 array of
    shape (4, 4). For the vacuum Kerr solution every component must vanish
    up to floating-point cancellation, whose natural scale is the square of
    the Christoffel magnitudes (see :func:`christoffel_scale`).
    """
    return lambdified_ricci()(float(mass), float(a), float(r), float(theta))


def christoffel_scale(mass: float, a: float, r: float, theta: float) -> float:
    """Return (sum over all |Gamma^mu_{alpha beta}|)^2 at a point.

    This is the magnitude of the Gamma-Gamma products that cancel in the
    Ricci tensor and hence the natural reference scale for a numerical
    ``R_{mu nu} = 0`` check (round-off in R_{mu nu} is a small multiple of
    machine epsilon times this scale, plus the derivative terms of the same
    order).
    """
    gamma = lambdified_christoffel()(float(mass), float(a), float(r), float(theta))
    return float(np.abs(gamma).sum() ** 2)


def _as_array_callable(f: PointCallable) -> PointCallable:
    """Wrap a lambdified callable so that it always returns a float64 ndarray."""

    def evaluate(mass: float, a: float, r: float, theta: float) -> Any:
        return np.asarray(f(mass, a, r, theta), dtype=np.float64)

    return evaluate


__all__ = [
    "KerrSymbols",
    "christoffel_scale",
    "coordinates",
    "kerr_symbols",
    "lambdified_christoffel",
    "lambdified_inverse_metric",
    "lambdified_inverse_metric_derivatives",
    "lambdified_metric",
    "lambdified_ricci",
    "ricci_tensor_numeric",
    "symbolic_christoffel",
    "symbolic_inverse_metric",
    "symbolic_metric",
    "symbolic_ricci",
]
