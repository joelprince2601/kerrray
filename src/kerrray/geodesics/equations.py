"""Null geodesic equations in Kerr spacetime (PROJECT.md sections 8 and 9).

Primary formulation: the first-order Hamiltonian form (docs/architecture.md
section 1; docs/equations_geodesics.md; Carter 1968, Phys. Rev. 174, 1559;
Misner, Thorne and Wheeler 1973, section 25.2 for the super-Hamiltonian of
geodesic motion and section 33.5 for Kerr; Chandrasekhar 1983, chapter 7).
With the super-Hamiltonian ::

    H = (1/2) g^{mu nu}(x) p_mu p_nu

Hamilton's equations for the affine parameter ``lambda`` are ::

    dx^mu / dlambda = dH/dp_mu = g^{mu nu} p_nu
    dp_mu / dlambda = -dH/dx^mu = -(1/2) (d_mu g^{alpha beta}) p_alpha p_beta

and ``H = 0`` on null geodesics. The Boyer-Lindquist metric depends on ``r``
and ``theta`` only, so ``dp_t/dlambda = dp_phi/dlambda = 0`` identically:
``E = -p_t`` and ``L_z = p_phi`` are constants of the ODE itself, not merely
of its exact solution. The five non-zero inverse-metric components and their
``r`` and ``theta`` derivatives come from :mod:`kerrray.geometry`
(closed forms, verified against SymPy there); the right-hand side is verified
against a SymPy differentiation of ``H`` in ``tests/test_geodesics.py``.

Secondary formulation (cross-check only): the second-order geodesic equation
``d^2 x^mu/dlambda^2 = -Gamma^mu_{alpha beta} u^alpha u^beta`` (MTW 1973
chapter 10; Carroll 2004 section 3.3; PROJECT.md section 8) written as the
first-order system for
``z = [x^mu, u^mu]`` with ``u^mu = dx^mu/dlambda``. The two forms are related
by ``p_mu = g_{mu nu} u^nu`` and are checked against each other in
``tests/test_geodesics.py``.

Polar axis. ``g^{phi phi} = 1/(Sigma sin^2 theta) - a^2/(Sigma Delta)`` and
its derivatives diverge on the axis ``sin(theta) = 0``. Every occurrence of
``g^{phi phi}`` (and of ``d_r g^{phi phi}``, ``d_theta g^{phi phi}``) in the
equations is multiplied by ``p_phi = L_z``. For rays with exactly ``L_z = 0``
those products vanish identically, so the coefficients are replaced by zero
on such rays and the finite axis limit is obtained instead of ``0 * inf``.
Rays with ``L_z != 0`` cannot reach the axis (the polar potential ``Theta``
becomes negative there, docs/equations_geodesics.md), so if they do so
numerically the classifier marks them ``OUT_OF_DOMAIN`` (see
:mod:`kerrray.photons.classification`); this module does not raise.

Precision. The geometry functions evaluate in float64. For a float32 state
the metric terms are cast to float32 *before* they are combined with the
momenta, so all subsequent arithmetic is float32 (docs/numerical_methods.md,
section on precision).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.state import (
    IDX_PH,
    IDX_PPH,
    IDX_PR,
    IDX_PT,
    IDX_PTH,
    IDX_R,
    IDX_T,
    IDX_TH,
    STATE_SIZE,
)
from kerrray.geometry import (
    Spacetime,
    christoffel,
    inverse_metric,
    inverse_metric_components,
    inverse_metric_derivatives,
    metric,
)

__all__ = [
    "christoffel_to_hamiltonian_state",
    "geodesic_rhs",
    "geodesic_rhs_christoffel",
    "hamiltonian",
    "hamiltonian_to_christoffel_state",
    "null_momentum_pt",
]

FloatArray = NDArray[np.floating]


def _state_dtype(y: np.ndarray) -> np.dtype:
    """float32 stays float32; anything else is computed in float64."""
    return np.dtype(np.float32) if y.dtype == np.float32 else np.dtype(np.float64)


def _as_state(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y)
    if arr.shape[-1:] != (STATE_SIZE,):
        raise ValueError(f"state must end in a dimension of size {STATE_SIZE}, got {arr.shape}")
    return arr.astype(_state_dtype(arr), copy=False)


def _metric_terms(st: Spacetime, r: np.ndarray, theta: np.ndarray, dtype: np.dtype) -> tuple:
    """Inverse metric and its r/theta derivatives, cast to ``dtype``.

    Evaluated with ``divide`` and ``invalid`` floating-point errors silenced:
    the only locus where the geometry produces non-finite values for a state
    outside the horizon is the polar axis, which the caller guards.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        g = inverse_metric_components(st, r, theta)
        d_r, d_th = inverse_metric_derivatives(st, r, theta)
    cast = [np.asarray(c, dtype=dtype) for c in (*g, *d_r, *d_th)]
    return tuple(cast)


def hamiltonian(st: Spacetime, y: ArrayLike) -> FloatArray:
    """Super-Hamiltonian ``H = (1/2) g^{mu nu} p_mu p_nu`` at each state (shape ``(...)``).

    Zero on null geodesics; used as the null-constraint monitor
    (:func:`kerrray.photons.constants.null_constraint`). The polar-axis
    guard for ``L_z = 0`` applies (module docstring).
    """
    y_arr = _as_state(y)
    dtype = y_arr.dtype
    r, th = y_arr[..., IDX_R], y_arr[..., IDX_TH]
    p_t, p_r, p_th, p_ph = (y_arr[..., i] for i in (IDX_PT, IDX_PR, IDX_PTH, IDX_PPH))
    gtt, gtph, grr, gthth, gphph = _metric_terms(st, r, th, dtype)[:5]
    gphph = np.where(p_ph == 0, dtype.type(0), gphph)
    return 0.5 * (gtt * p_t**2 + 2.0 * gtph * p_t * p_ph + grr * p_r**2 + gthth * p_th**2 + gphph * p_ph**2)


def geodesic_rhs(st: Spacetime, y: ArrayLike) -> FloatArray:
    """Hamiltonian right-hand side ``dy/dlambda`` for states ``y`` of shape ``(..., 8)``.

    Returns an array of the same shape and dtype (float32 in, float32 out)::

        dt/dlambda      = g^{tt} p_t + g^{tphi} p_phi
        dr/dlambda      = g^{rr} p_r
        dtheta/dlambda  = g^{thth} p_theta
        dphi/dlambda    = g^{tphi} p_t + g^{phph} p_phi
        dp_t/dlambda    = 0
        dp_r/dlambda    = -(1/2) [d_r g^{tt} p_t^2 + 2 d_r g^{tphi} p_t p_phi + d_r g^{rr} p_r^2
                                  + d_r g^{thth} p_theta^2 + d_r g^{phph} p_phi^2]
        dp_theta/dlambda= -(1/2) [same with d_theta]
        dp_phi/dlambda  = 0
    """
    y_arr = _as_state(y)
    dtype = y_arr.dtype
    r, th = y_arr[..., IDX_R], y_arr[..., IDX_TH]
    p_t, p_r, p_th, p_ph = (y_arr[..., i] for i in (IDX_PT, IDX_PR, IDX_PTH, IDX_PPH))
    (
        gtt, gtph, grr, gthth, gphph,
        r_gtt, r_gtph, r_grr, r_gthth, r_gphph,
        t_gtt, t_gtph, t_grr, t_gthth, t_gphph,
    ) = _metric_terms(st, r, th, dtype)
    # Polar-axis guard: the g^{phph} coefficients only ever multiply p_phi.
    no_lz = p_ph == 0
    zero = dtype.type(0)
    gphph = np.where(no_lz, zero, gphph)
    r_gphph = np.where(no_lz, zero, r_gphph)
    t_gphph = np.where(no_lz, zero, t_gphph)

    out = np.empty(y_arr.shape, dtype=dtype)
    # Overflow / invalid values (a ray that has blown through the horizon, or
    # float32 near it) propagate as inf/nan and are reported by the classifier
    # as NUMERICAL_FAILURE; the warnings would only repeat that signal.
    with np.errstate(over="ignore", invalid="ignore"):
        out[..., IDX_T] = gtt * p_t + gtph * p_ph
        out[..., IDX_R] = grr * p_r
        out[..., IDX_TH] = gthth * p_th
        out[..., IDX_PH] = gtph * p_t + gphph * p_ph
        out[..., IDX_PT] = 0
        pt2, ptpph, pr2, pth2, pph2 = p_t * p_t, p_t * p_ph, p_r * p_r, p_th * p_th, p_ph * p_ph
        out[..., IDX_PR] = -0.5 * (
            r_gtt * pt2 + 2.0 * r_gtph * ptpph + r_grr * pr2 + r_gthth * pth2 + r_gphph * pph2
        )
        out[..., IDX_PTH] = -0.5 * (
            t_gtt * pt2 + 2.0 * t_gtph * ptpph + t_grr * pr2 + t_gthth * pth2 + t_gphph * pph2
        )
        out[..., IDX_PPH] = 0
    return out


def geodesic_rhs_christoffel(st: Spacetime, z: ArrayLike) -> FloatArray:
    """Second-order (Christoffel) form for ``z = [x^mu, u^mu]`` of shape ``(..., 8)``.

    ``dx^mu/dlambda = u^mu`` and ``du^mu/dlambda = -Gamma^mu_{alpha beta} u^alpha
    u^beta`` (the geodesic equation, MTW 1973 chapter 10; Carroll 2004 section
    3.3; PROJECT.md section 8) with the Christoffel
    symbols from :func:`kerrray.geometry.christoffel`. Cross-check only: it has
    no axis guard and evaluates all 64 Christoffel components. Always float64.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.shape[-1:] != (STATE_SIZE,):
        raise ValueError(f"state must end in a dimension of size {STATE_SIZE}, got {z_arr.shape}")
    u = z_arr[..., 4:]
    gam = christoffel(st, z_arr[..., IDX_R], z_arr[..., IDX_TH])
    out = np.empty_like(z_arr)
    out[..., :4] = u
    out[..., 4:] = -np.einsum("...mab,...a,...b->...m", gam, u, u)
    return out


def hamiltonian_to_christoffel_state(st: Spacetime, y: ArrayLike) -> FloatArray:
    """Convert ``y = [x, p_mu]`` to ``z = [x, u^mu]`` with ``u^mu = g^{mu nu} p_nu``."""
    y_arr = np.asarray(y, dtype=np.float64)
    ginv = inverse_metric(st, y_arr[..., IDX_R], y_arr[..., IDX_TH])
    out = y_arr.copy()
    out[..., 4:] = np.einsum("...mn,...n->...m", ginv, y_arr[..., 4:])
    return out


def christoffel_to_hamiltonian_state(st: Spacetime, z: ArrayLike) -> FloatArray:
    """Convert ``z = [x, u^mu]`` to ``y = [x, p_mu]`` with ``p_mu = g_{mu nu} u^nu``."""
    z_arr = np.asarray(z, dtype=np.float64)
    g = metric(st, z_arr[..., IDX_R], z_arr[..., IDX_TH])
    out = z_arr.copy()
    out[..., 4:] = np.einsum("...mn,...n->...m", g, z_arr[..., 4:])
    return out


def null_momentum_pt(
    st: Spacetime,
    x: ArrayLike,
    p_r: ArrayLike,
    p_theta: ArrayLike,
    p_phi: ArrayLike,
    *,
    future_directed: bool = True,
) -> FloatArray:
    """Solve the null condition for ``p_t`` given the spatial covariant momenta.

    The null condition ``g^{mu nu} p_mu p_nu = 0`` is the quadratic ::

        g^{tt} p_t^2 + 2 g^{tphi} p_t p_phi + C = 0,
        C = g^{rr} p_r^2 + g^{thth} p_theta^2 + g^{phph} p_phi^2

    with roots ``p_t = [-g^{tphi} p_phi -+ sqrt(D)] / g^{tt}`` and
    ``D = (g^{tphi} p_phi)^2 - g^{tt} C``. For either root ``dt/dlambda = g^{tt}
    p_t + g^{tphi} p_phi = +-sqrt(D)``, so the root with the upper sign is the
    future-directed one (``dt/dlambda > 0``) and the other is past-directed.
    Outside the ergosphere this coincides with ``E = -p_t > 0``
    (docs/architecture.md section 1); inside the ergosphere a future-directed
    photon may have ``E < 0`` (Penrose 1969), which is why the sign of
    ``dt/dlambda`` rather than of ``E`` is used. Raises ``ValueError`` if
    ``D < 0`` or ``g^{tt} >= 0`` anywhere (no null solution: at or inside the
    horizon).

    Args:
        st: The spacetime.
        x: Positions ``(t, r, theta, phi)`` of shape ``(..., 4)``.
        p_r, p_theta, p_phi: Spatial covariant momenta, broadcastable with
            the leading shape of ``x``.
        future_directed: Select the root with ``dt/dlambda > 0`` (default) or
            ``dt/dlambda < 0``.

    Returns:
        ``p_t`` with the broadcast shape (float64).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.shape[-1:] != (4,):
        raise ValueError(f"x must end in a dimension of size 4, got {x_arr.shape}")
    r, th = x_arr[..., 1], x_arr[..., 2]
    pr = np.asarray(p_r, dtype=np.float64)
    pth = np.asarray(p_theta, dtype=np.float64)
    pph = np.asarray(p_phi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = inverse_metric_components(st, r, th)
    gphph = np.where(pph == 0, 0.0, g.gphph)
    c = g.grr * pr**2 + g.gthth * pth**2 + gphph * pph**2
    b = g.gtphi * pph
    disc = b * b - g.gtt * c
    if np.any(~np.isfinite(disc)) or np.any(disc < 0.0) or np.any(g.gtt >= 0.0):
        raise ValueError("no real null momentum p_t exists (point at or inside the horizon?)")
    sign = 1.0 if future_directed else -1.0
    return (-b + sign * np.sqrt(disc)) / g.gtt
