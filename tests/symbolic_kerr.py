"""Independent SymPy construction of the Kerr metric for the geometry tests.

Nothing here is imported by the package. It exists so that the closed-form
NumPy implementation in ``kerrray.geometry`` is checked against symbolic
expressions built from two published forms of the line element (Misner,
Thorne and Wheeler 1973 eq. 33.2 and Bardeen, Press and Teukolsky 1972 eq.
2.1), against the closed-form inverse of docs/architecture.md section 1.1,
and against the *definitions* of the Christoffel symbols and of the Ricci
tensor. Expressions are lambdified without simplification (PROJECT.md
section 32: symbolic verification, numerical implementation).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import sympy as sp

from kerrray.geometry import Spacetime, outer_horizon


class KerrSymbols(NamedTuple):
    """Coordinate and parameter symbols shared by the symbolic builders."""

    t: sp.Symbol
    r: sp.Symbol
    theta: sp.Symbol
    phi: sp.Symbol
    M: sp.Symbol
    a: sp.Symbol


@lru_cache(maxsize=None)
def symbols() -> KerrSymbols:
    t, r, theta, phi = sp.symbols("t r theta phi", real=True)
    mass = sp.symbols("M", positive=True)
    a = sp.symbols("a", real=True)
    return KerrSymbols(t, r, theta, phi, mass, a)


def sigma_delta(s: KerrSymbols) -> tuple[sp.Expr, sp.Expr]:
    sigma = s.r**2 + s.a**2 * sp.cos(s.theta) ** 2
    delta = s.r**2 - 2 * s.M * s.r + s.a**2
    return sigma, delta


def mtw_metric(s: KerrSymbols) -> sp.Matrix:
    """Covariant Kerr metric, MTW 1973 eq. 33.2 (index order t, r, theta, phi)."""
    sigma, delta = sigma_delta(s)
    sin2 = sp.sin(s.theta) ** 2
    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * s.M * s.r / sigma)
    g[0, 3] = g[3, 0] = -2 * s.M * s.a * s.r * sin2 / sigma
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = (s.r**2 + s.a**2 + 2 * s.M * s.a**2 * s.r * sin2 / sigma) * sin2
    return g


def bpt_metric(s: KerrSymbols) -> sp.Matrix:
    """Covariant Kerr metric, BPT 1972 eq. 2.1 form.

    ``ds^2 = -e^{2nu} dt^2 + e^{2psi} (dphi - omega dt)^2 + e^{2mu1} dr^2
    + e^{2mu2} dtheta^2`` with ``e^{2nu} = Sigma Delta / A``, ``e^{2psi} =
    A sin^2(theta) / Sigma``, ``e^{2mu1} = Sigma / Delta``, ``e^{2mu2} =
    Sigma``, ``omega = 2 M a r / A``, ``A = (r^2 + a^2)^2 - a^2 Delta sin^2``.
    """
    sigma, delta = sigma_delta(s)
    sin2 = sp.sin(s.theta) ** 2
    big_a = (s.r**2 + s.a**2) ** 2 - s.a**2 * delta * sin2
    e2nu = sigma * delta / big_a
    e2psi = big_a * sin2 / sigma
    omega = 2 * s.M * s.a * s.r / big_a
    g = sp.zeros(4, 4)
    g[0, 0] = -e2nu + e2psi * omega**2
    g[0, 3] = g[3, 0] = -e2psi * omega
    g[1, 1] = sigma / delta
    g[2, 2] = sigma
    g[3, 3] = e2psi
    return g


def closed_form_inverse(s: KerrSymbols) -> sp.Matrix:
    """Contravariant Kerr metric, closed form of docs/architecture.md 1.1."""
    sigma, delta = sigma_delta(s)
    sin2 = sp.sin(s.theta) ** 2
    big_a = (s.r**2 + s.a**2) ** 2 - s.a**2 * delta * sin2
    gi = sp.zeros(4, 4)
    gi[0, 0] = -big_a / (sigma * delta)
    gi[0, 3] = gi[3, 0] = -2 * s.M * s.a * s.r / (sigma * delta)
    gi[1, 1] = delta / sigma
    gi[2, 2] = 1 / sigma
    gi[3, 3] = (delta - s.a**2 * sin2) / (sigma * delta * sin2)
    return gi


def christoffel_from_definition(
    s: KerrSymbols, g: sp.Matrix, ginv: sp.Matrix
) -> list[list[list[sp.Expr]]]:
    """``Gamma^mu_{ab} = (1/2) g^{mu nu} (d_a g_{nu b} + d_b g_{nu a} - d_nu g_{ab})``.

    Returned as ``Gamma[mu][a][b]``; no simplification is applied.
    """
    x = [s.t, s.r, s.theta, s.phi]
    return [
        [
            [
                sum(
                    ginv[mu, nu]
                    * (
                        sp.diff(g[nu, b], x[a])
                        + sp.diff(g[nu, a], x[b])
                        - sp.diff(g[a, b], x[nu])
                    )
                    for nu in range(4)
                )
                / 2
                for b in range(4)
            ]
            for a in range(4)
        ]
        for mu in range(4)
    ]


def ricci_from_christoffel(s: KerrSymbols, gam: list[list[list[sp.Expr]]]) -> sp.Matrix:
    """``R_{ab} = d_c Gamma^c_{ab} - d_b Gamma^c_{ac} + Gamma^c_{cd} Gamma^d_{ab}
    - Gamma^c_{bd} Gamma^d_{ac}`` (the Riemann tensor in terms of the
    connection coefficients, MTW 1973 chapter 8, contracted on the first and
    third indices); no simplification."""
    x = [s.t, s.r, s.theta, s.phi]
    ricci = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            expr: sp.Expr = sp.Integer(0)
            for c in range(4):
                expr += sp.diff(gam[c][a][b], x[c]) - sp.diff(gam[c][a][c], x[b])
                for d in range(4):
                    expr += gam[c][c][d] * gam[d][a][b] - gam[c][b][d] * gam[d][a][c]
            ricci[a, b] = expr
    return ricci


PointFunction = Callable[[float, float, float, float], np.ndarray]


def _lambdify(expr: sp.Basic) -> PointFunction:
    s = symbols()
    fn = sp.lambdify((s.M, s.a, s.r, s.theta), expr, modules="numpy", cse=True)
    return lambda m, a, r, theta: np.asarray(fn(m, a, r, theta), dtype=np.float64)


@lru_cache(maxsize=None)
def lambdified_metric() -> PointFunction:
    return _lambdify(mtw_metric(symbols()))


@lru_cache(maxsize=None)
def lambdified_inverse() -> PointFunction:
    return _lambdify(closed_form_inverse(symbols()))


@lru_cache(maxsize=None)
def lambdified_metric_derivatives() -> tuple[PointFunction, PointFunction]:
    s = symbols()
    g = mtw_metric(s)
    return _lambdify(g.diff(s.r)), _lambdify(g.diff(s.theta))


@lru_cache(maxsize=None)
def lambdified_inverse_derivatives() -> tuple[PointFunction, PointFunction]:
    s = symbols()
    gi = closed_form_inverse(s)
    return _lambdify(gi.diff(s.r)), _lambdify(gi.diff(s.theta))


@lru_cache(maxsize=None)
def symbolic_christoffel() -> list[list[list[sp.Expr]]]:
    s = symbols()
    return christoffel_from_definition(s, mtw_metric(s), closed_form_inverse(s))


@lru_cache(maxsize=None)
def lambdified_christoffel() -> PointFunction:
    return _lambdify(sp.Array(symbolic_christoffel()))


@lru_cache(maxsize=None)
def lambdified_ricci() -> PointFunction:
    return _lambdify(ricci_from_christoffel(symbols(), symbolic_christoffel()))


def sample_points(
    st: Spacetime, rng: np.random.Generator, n: int, *, r_max: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Random ``(r, theta)`` outside the horizon and away from the axis.

    ``r`` is uniform in ``[r_plus + 0.5 M, r_max M]`` (the closed forms have
    ``Delta`` in denominators and the comparison is a relative one, so the
    horizon itself is excluded) and ``theta`` uniform in ``[0.05, pi - 0.05]``
    (``sin^2(theta)`` appears in ``g^phph``).
    """
    r_lo = outer_horizon(st) + 0.5 * st.mass
    r = r_lo + (r_max * st.mass - r_lo) * rng.random(n)
    eps = 0.05
    theta = eps + (np.pi - 2 * eps) * rng.random(n)
    return r, theta


def assert_relative_close(actual: np.ndarray, expected: np.ndarray, rtol: float) -> None:
    """``|actual - expected| <= rtol * max(|expected|, max|expected|)`` elementwise.

    Structurally zero components are compared on the scale of the largest
    component of ``expected`` so that round-off in a vanishing entry does not
    turn into an infinite relative error.
    """
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)
