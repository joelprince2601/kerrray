"""Scalar Numba kernels of the ``numba`` backend: one ray per parallel iteration.

Every function here is written in the Numba-compilable subset of Python and
mirrors the NumPy reference (:mod:`kerrray.geodesics.integrators_batch`,
:mod:`kerrray.geodesics.tableaus`, :mod:`kerrray.geodesics.equations`,
:mod:`kerrray.photons.classification`) *operation by operation*, in the same
floating-point order, so that the two backends agree to round-off
(docs/architecture.md section 9; docs/performance.md section 2):

* :func:`inverse_metric_terms` evaluates the five contravariant Kerr metric
  components and their ``r`` and ``theta`` derivatives in float64 with the
  closed forms of :func:`kerrray.geometry.inverse_metric_components` and
  :func:`kerrray.geometry.inverse_metric_derivatives` (BPT 1972 eq. 2.1
  definitions; verified against SymPy there and against those functions in
  ``tests/test_numba_backend.py``), once per right-hand-side call instead of
  twice as the NumPy path does.
* :func:`geodesic_rhs_scalar` is the Hamiltonian right-hand side with the
  ``L_z = 0`` polar-axis guard; the metric terms are cast to the working
  dtype *before* they meet the momenta, exactly as the NumPy path does, so
  the float32 variant is float32 everywhere except inside the metric closed
  forms.
* :func:`dormand_prince_step` / :func:`rk4_step` use the tableau arrays
  passed in from :mod:`kerrray.geodesics.tableaus` (never retyped), the
  error norm is the Hairer-Norsett-Wanner scaled RMS with NumPy's pairwise
  (tree) summation order over the eight components, and the step factor,
  termination rules and their priority are those of the reference.

dtype genericity: Numba specialises on the array dtype. Every constant that
meets a working-dtype value is read from the ``consts`` array of that dtype
(a Python literal would promote float32 to float64), and float64 metric
values are cast through a working-dtype scratch array (``g``). The affine
parameter, the conservation diagnostics and the disk-crossing interpolant
are float64, as in the reference.

Numba is optional: when it cannot be imported ``njit`` is an identity
decorator and ``prange`` is ``range``, so the same source runs interpreted
(slowly) for verification; :mod:`kerrray.raytracing.backends.numba_backend`
reports the backend unavailable in that case. ``fastmath`` is off so that no
reassociation or contraction breaks the agreement with NumPy.
"""

from __future__ import annotations

import math

import numpy as np

from kerrray.geodesics.state import IDX_PH, IDX_PPH, IDX_PR, IDX_PT, IDX_PTH, IDX_R, IDX_T, IDX_TH

try:  # pragma: no cover - depends on the environment
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """Identity decorator used when Numba is not installed (interpreted mode)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap


__all__ = [
    "C_LAYOUT",
    "G_LAYOUT",
    "NUMBA_AVAILABLE",
    "P_LAYOUT",
    "STATE_SIZE",
    "dormand_prince_step",
    "geodesic_rhs_scalar",
    "integrate_rays_kernel",
    "inverse_metric_terms",
    "rk4_step",
]

STATE_SIZE = 8
N_TERMS = 15
"""Metric terms per point: g^tt, g^tph, g^rr, g^thth, g^phph and their d_r and d_theta."""

# Termination codes (kerrray.photons.classification.TerminationState values).
RUNNING, ESCAPED, CAPTURED, MAX_AFFINE, FAILURE, OUT_OF_DOMAIN, DISK_HIT = 0, 1, 2, 3, 4, 5, 6

# Layout of the float64 geometry constants (evaluated in Python once per batch).
G_LAYOUT = ("mass", "a", "a2", "neg_2ma", "pos_2ma", "two_m", "two_a2")
G_MASS, G_A, G_A2, G_NEG_2MA, G_POS_2MA, G_TWO_M, G_TWO_A2 = range(7)

# Layout of the working-dtype constants.
C_LAYOUT = (
    "zero", "one", "two", "neg_half", "eight", "safety", "factor_min", "factor_max",
    "error_exponent", "rtol", "atol", "h_min", "r_capture", "r_escape", "theta_low",
    "theta_high", "axis_sin_tol", "h_initial", "r_in", "r_out", "half_pi",
)
(
    C_ZERO, C_ONE, C_TWO, C_NEG_HALF, C_EIGHT, C_SAFETY, C_FMIN, C_FMAX, C_EXP, C_RTOL,
    C_ATOL, C_HMIN, C_RCAP, C_RESC, C_THLO, C_THHI, C_AXIS, C_H0, C_RIN, C_ROUT, C_HALFPI,
) = range(21)

# Layout of the float64 parameters and of the integer parameters.
P_LAYOUT = ("lambda_max", "half_pi", "h_max")
P_LAMMAX, P_HALFPI, P_HMAX = 0, 1, 2
I_MAX_STEPS, I_GUARD, I_ADAPTIVE, I_DISK = 0, 1, 2, 3


@njit(cache=True)
def inverse_metric_terms(geom, r, th, out):
    """Fill ``out[0:15]`` (float64) with the inverse Kerr metric and its derivatives at ``(r, th)``.

    Order: ``g^tt, g^tphi, g^rr, g^thth, g^phph``, then ``d_r`` of the five,
    then ``d_theta`` of the five. Same closed forms and floating-point
    operation order as :func:`kerrray.geometry.inverse_metric_components` and
    :func:`kerrray.geometry.inverse_metric_derivatives` (module docstring).
    """
    m = geom[G_MASS]
    a2 = geom[G_A2]
    s = math.sin(th)
    c = math.cos(th)
    s2 = s * s
    sin2th = math.sin(2.0 * th)
    r2 = r * r
    sig = r2 + a2 * (c * c)
    dlt = r2 - geom[G_TWO_M] * r + a2
    r2a2 = r2 + a2
    big_a = r2a2 * r2a2 - a2 * dlt * s2
    sd = sig * dlt
    out[0] = -big_a / sd
    out[1] = geom[G_NEG_2MA] * r / sd
    out[2] = dlt / sig
    out[3] = 1.0 / sig
    out[4] = (dlt - a2 * s2) / (sd * s2)
    sig_r = 2.0 * r
    sig_th = (-a2) * sin2th
    dlt_r = 2.0 * (r - m)
    a_r = 4.0 * r * r2a2 - geom[G_TWO_A2] * (r - m) * s2
    a_th = (-a2) * dlt * sin2th
    sd_r = sig_r * dlt + sig * dlt_r
    sd_th = sig_th * dlt
    sd2 = sd * sd
    sig2 = sig * sig
    out[5] = -(a_r * sd - big_a * sd_r) / sd2
    out[6] = geom[G_NEG_2MA] * (sd - r * sd_r) / sd2
    out[7] = (dlt_r * sig - dlt * sig_r) / sig2
    out[8] = -sig_r / sig2
    out[9] = -sig_r / (sig2 * s2) + a2 * sd_r / sd2
    out[10] = -(a_th * sd - big_a * sd_th) / sd2
    out[11] = geom[G_POS_2MA] * r * sd_th / sd2
    out[12] = -dlt * sig_th / sig2
    out[13] = -sig_th / sig2
    out[14] = -(sig_th * s2 + sig * sin2th) / (sig2 * (s2 * s2)) + a2 * sd_th / sd2


@njit(cache=True)
def geodesic_rhs_scalar(geom, consts, y, f, g64, g):
    """Hamiltonian right-hand side ``f = dy/dlambda`` for one state ``y`` (working dtype).

    ``g64`` (15, float64) and ``g`` (15, working dtype) are scratch arrays;
    the metric is evaluated in float64 and cast into ``g`` before it meets
    the momenta (:func:`kerrray.geodesics.equations.geodesic_rhs`).
    """
    inverse_metric_terms(geom, np.float64(y[IDX_R]), np.float64(y[IDX_TH]), g64)
    for i in range(N_TERMS):
        g[i] = g64[i]
    zero = consts[C_ZERO]
    two = consts[C_TWO]
    p_t = y[IDX_PT]
    p_r = y[IDX_PR]
    p_th = y[IDX_PTH]
    p_ph = y[IDX_PPH]
    gphph = g[4]
    r_gphph = g[9]
    t_gphph = g[14]
    if p_ph == zero:  # polar-axis guard: these coefficients only ever multiply p_phi
        gphph = zero
        r_gphph = zero
        t_gphph = zero
    f[IDX_T] = g[0] * p_t + g[1] * p_ph
    f[IDX_R] = g[2] * p_r
    f[IDX_TH] = g[3] * p_th
    f[IDX_PH] = g[1] * p_t + gphph * p_ph
    f[IDX_PT] = zero
    pt2 = p_t * p_t
    ptpph = p_t * p_ph
    pr2 = p_r * p_r
    pth2 = p_th * p_th
    pph2 = p_ph * p_ph
    f[IDX_PR] = consts[C_NEG_HALF] * (
        g[5] * pt2 + two * g[6] * ptpph + g[7] * pr2 + g[8] * pth2 + r_gphph * pph2
    )
    f[IDX_PTH] = consts[C_NEG_HALF] * (
        g[10] * pt2 + two * g[11] * ptpph + g[12] * pr2 + g[13] * pth2 + t_gphph * pph2
    )
    f[IDX_PPH] = zero


@njit(cache=True)
def dormand_prince_step(geom, consts, tab_a, tab_e, y, h, k, ys, y_new, err, g64, g):
    """One Dormand-Prince 5(4) step of size ``h``; ``k[0]`` holds ``f(y)`` on entry (FSAL).

    Fills ``k[1:7]``, ``y_new`` and the embedded error estimate ``err``;
    ``k[6] = f(y_new)`` is the next step's first stage. Same stage sums (and
    the same skipped zero coefficients) as :func:`kerrray.geodesics.tableaus.make_step_rk45`.
    """
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (tab_a[1, 0] * k[0, i])
    geodesic_rhs_scalar(geom, consts, ys, k[1], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (tab_a[2, 0] * k[0, i] + tab_a[2, 1] * k[1, i])
    geodesic_rhs_scalar(geom, consts, ys, k[2], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (tab_a[3, 0] * k[0, i] + tab_a[3, 1] * k[1, i] + tab_a[3, 2] * k[2, i])
    geodesic_rhs_scalar(geom, consts, ys, k[3], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (
            tab_a[4, 0] * k[0, i] + tab_a[4, 1] * k[1, i] + tab_a[4, 2] * k[2, i] + tab_a[4, 3] * k[3, i]
        )
    geodesic_rhs_scalar(geom, consts, ys, k[4], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (
            tab_a[5, 0] * k[0, i] + tab_a[5, 1] * k[1, i] + tab_a[5, 2] * k[2, i]
            + tab_a[5, 3] * k[3, i] + tab_a[5, 4] * k[4, i]
        )
    geodesic_rhs_scalar(geom, consts, ys, k[5], g64, g)
    for i in range(STATE_SIZE):
        y_new[i] = y[i] + h * (
            tab_a[6, 0] * k[0, i] + tab_a[6, 2] * k[2, i] + tab_a[6, 3] * k[3, i]
            + tab_a[6, 4] * k[4, i] + tab_a[6, 5] * k[5, i]
        )
    geodesic_rhs_scalar(geom, consts, y_new, k[6], g64, g)
    for i in range(STATE_SIZE):
        err[i] = h * (
            tab_e[0] * k[0, i] + tab_e[2] * k[2, i] + tab_e[3] * k[3, i]
            + tab_e[4] * k[4, i] + tab_e[5] * k[5, i] + tab_e[6] * k[6, i]
        )


@njit(cache=True)
def rk4_step(geom, consts, rk4_b, rk4_c, y, h, k, ys, y_new, g64, g):
    """One classical RK4 step; ``k[0] = f(y)`` on entry, ``k[6] = f(y_new)`` on exit."""
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (rk4_c[1] * k[0, i])
    geodesic_rhs_scalar(geom, consts, ys, k[1], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (rk4_c[2] * k[1, i])
    geodesic_rhs_scalar(geom, consts, ys, k[2], g64, g)
    for i in range(STATE_SIZE):
        ys[i] = y[i] + h * (rk4_c[3] * k[2, i])
    geodesic_rhs_scalar(geom, consts, ys, k[3], g64, g)
    for i in range(STATE_SIZE):
        y_new[i] = y[i] + h * (
            rk4_b[0] * k[0, i] + rk4_b[1] * k[1, i] + rk4_b[2] * k[2, i] + rk4_b[3] * k[3, i]
        )
    geodesic_rhs_scalar(geom, consts, y_new, k[6], g64, g)


@njit(cache=True)
def _error_norm(consts, err, y, y_new, q):
    """HNW scaled RMS norm with NumPy's pairwise summation order over the eight components."""
    for i in range(STATE_SIZE):
        scale = consts[C_ATOL] + consts[C_RTOL] * max(abs(y[i]), abs(y_new[i]))
        v = err[i] / scale
        q[i] = v * v
    total = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]))
    return np.sqrt(total / consts[C_EIGHT])


@njit(cache=True)
def _step_factor(consts, en, rejected_before):
    """``min(facmax, max(facmin, 0.9 en^(-1/5)))``, ``facmax`` for ``en = 0``, capped at 1 after a rejection."""
    if en == consts[C_ZERO]:
        factor = consts[C_FMAX]
    else:
        raw = consts[C_SAFETY] * en ** consts[C_EXP]
        factor = min(consts[C_FMAX], max(consts[C_FMIN], raw))
    if rejected_before:
        factor = min(factor, consts[C_ONE])
    return factor


@njit(cache=True)
def _all_finite(y, f):
    for i in range(STATE_SIZE):
        if not (math.isfinite(y[i]) and math.isfinite(f[i])):
            return False
    return True


@njit(cache=True)
def _classify(consts, params, ipar, y, f, lam, n_steps, h, check_h):
    """Termination rules in the priority order of :mod:`kerrray.photons.classification`."""
    if not _all_finite(y, f):
        return FAILURE
    if check_h and h < consts[C_HMIN]:
        return FAILURE
    r = y[IDX_R]
    th = y[IDX_TH]
    if r < consts[C_ZERO] or th < consts[C_THLO] or th > consts[C_THHI]:
        return OUT_OF_DOMAIN
    if y[IDX_PPH] != consts[C_ZERO] and abs(np.sin(th)) < consts[C_AXIS]:
        return OUT_OF_DOMAIN
    if r <= consts[C_RCAP]:
        return CAPTURED
    if r >= consts[C_RESC] and f[IDX_R] > consts[C_ZERO]:
        return ESCAPED
    if lam >= params[P_LAMMAX] or n_steps >= ipar[I_MAX_STEPS]:
        return MAX_AFFINE
    return RUNNING


@njit(cache=True)
def _null_error(y, f):
    """``|(1/2) p_mu dx^mu/dlambda|`` in float64, summed in NumPy's sequential order."""
    total = np.float64(y[IDX_PT]) * np.float64(f[IDX_T])
    total = total + np.float64(y[IDX_PR]) * np.float64(f[IDX_R])
    total = total + np.float64(y[IDX_PTH]) * np.float64(f[IDX_TH])
    total = total + np.float64(y[IDX_PPH]) * np.float64(f[IDX_PH])
    return abs(0.5 * total)


@njit(cache=True)
def _carter(a2, y):
    """Null Carter constant in float64 (:func:`kerrray.photons.constants.carter_constant`)."""
    th = np.float64(y[IDX_TH])
    p_th = np.float64(y[IDX_PTH])
    lz = np.float64(y[IDX_PPH])
    e = -np.float64(y[IDX_PT])
    s = math.sin(th)
    c = math.cos(th)
    lz2_over_s2 = 0.0
    if lz != 0.0:
        lz2_over_s2 = (lz * lz) / (s * s)
    return p_th * p_th + (c * c) * (lz2_over_s2 - a2 * e * e)


@njit(cache=True)
def _integrate_ray(n, geom, consts, params, ipar, tab_a, tab_e, rk4_b, rk4_c, Y, F, lam_out,
                   n_steps_out, n_rej_out, state_out, max_null, max_e, max_lz, max_q,
                   e0, lz0, q0, fe, fl, fq, fe2, event_Y, event_hit):
    """Integrate ray ``n`` in place (``Y[n]``, ``F[n]``) and write its summaries."""
    y = Y[n]
    f = F[n]
    k = np.empty((7, STATE_SIZE), Y.dtype)
    ys = np.empty(STATE_SIZE, Y.dtype)
    y_new = np.empty(STATE_SIZE, Y.dtype)
    y_prev = np.empty(STATE_SIZE, Y.dtype)
    err = np.empty(STATE_SIZE, Y.dtype)
    q = np.empty(STATE_SIZE, Y.dtype)
    g = np.empty(N_TERMS, Y.dtype)
    cast = np.empty(1, Y.dtype)
    g64 = np.empty(N_TERMS, np.float64)
    a2 = geom[G_A2]
    adaptive = ipar[I_ADAPTIVE] != 0
    disk = ipar[I_DISK] != 0
    lambda_max = params[P_LAMMAX]
    half_pi = params[P_HALFPI]
    geodesic_rhs_scalar(geom, consts, y, f, g64, g)
    m_null = _null_error(y, f) / fe2[n]
    m_e = 0.0
    m_lz = 0.0
    m_q = 0.0
    lam = 0.0
    h = consts[C_H0]
    h_cap = params[P_HMAX]
    if adaptive and np.float64(h) > h_cap:
        cast[0] = h_cap
        h = cast[0]
    n_steps = 0
    n_rej = 0
    attempts = 0
    rejected_before = False
    state = _classify(consts, params, ipar, y, f, lam, n_steps, h, False)
    while state == RUNNING:
        attempts += 1
        if attempts > ipar[I_GUARD]:  # cannot happen by the bound of max_attempts; defensive
            state = FAILURE
            break
        remaining = lambda_max - lam
        last = np.float64(h) >= remaining
        h_eff = h
        if last:
            cast[0] = remaining
            h_eff = cast[0]
        for i in range(STATE_SIZE):
            k[0, i] = f[i]
            y_prev[i] = y[i]
        if adaptive:
            dormand_prince_step(geom, consts, tab_a, tab_e, y, h_eff, k, ys, y_new, err, g64, g)
        else:
            rk4_step(geom, consts, rk4_b, rk4_c, y, h_eff, k, ys, y_new, g64, g)
        finite = _all_finite(y_new, k[6])
        accept = finite
        h_new = h
        if adaptive and finite:
            en = _error_norm(consts, err, y, y_new, q)
            if not math.isfinite(en):
                finite = False
                accept = False
            else:
                accept = en <= consts[C_ONE]
                factor = _step_factor(consts, en, rejected_before)
                if not (accept and last):
                    h_new = h_eff * factor
                    if np.float64(h_new) > h_cap:
                        cast[0] = h_cap
                        h_new = cast[0]
        if not finite:
            state = FAILURE
            break
        rejected = not accept
        if rejected:
            n_rej += 1
        rejected_before = rejected
        h = h_new
        if rejected:
            if h_new < consts[C_HMIN]:
                state = FAILURE
                break
            continue
        for i in range(STATE_SIZE):
            y[i] = y_new[i]
            f[i] = k[6, i]
        lam_old = lam
        if last:
            lam = lambda_max
        else:
            lam = lam + np.float64(h_eff)
        n_steps += 1
        e = -np.float64(y[IDX_PT])
        lz = np.float64(y[IDX_PPH])
        m_e = max(m_e, abs(e - e0[n]) / max(abs(e0[n]), fe[n]))
        m_lz = max(m_lz, abs(lz - lz0[n]) / max(abs(lz0[n]), fl[n]))
        m_q = max(m_q, abs(_carter(a2, y) - q0[n]) / max(abs(q0[n]), fq[n]))
        m_null = max(m_null, _null_error(y, f) / fe2[n])
        code = _classify(consts, params, ipar, y, f, lam, n_steps, h, True)
        if disk:
            th_old = np.float64(y_prev[IDX_TH])
            th_new = np.float64(y[IDX_TH])
            if (th_old - half_pi) * (th_new - half_pi) < 0.0:
                s = (half_pi - th_old) / (th_new - th_old)
                for i in range(STATE_SIZE):
                    cast[0] = s * np.float64(y[i] - y_prev[i])
                    ys[i] = y_prev[i] + cast[0]
                ys[IDX_TH] = half_pi
                if ys[IDX_R] >= consts[C_RIN] and ys[IDX_R] <= consts[C_ROUT]:
                    for i in range(STATE_SIZE):
                        event_Y[n, i] = ys[i]
                        y[i] = ys[i]
                    event_hit[n] = True
                    lam = lam_old + s * np.float64(h_eff)
                    code = DISK_HIT
        state = code
    state_out[n] = state
    n_steps_out[n] = n_steps
    n_rej_out[n] = n_rej
    lam_out[n] = lam
    max_null[n] = m_null
    max_e[n] = m_e
    max_lz[n] = m_lz
    max_q[n] = m_q


@njit(parallel=True, fastmath=False, cache=True)
def integrate_rays_kernel(geom, consts, params, ipar, tab_a, tab_e, rk4_b, rk4_c, Y, F, lam_out,
                          n_steps_out, n_rej_out, state_out, max_null, max_e, max_lz, max_q,
                          e0, lz0, q0, fe, fl, fq, fe2, event_Y, event_hit):
    """Integrate every row of ``Y`` (N, 8) in parallel; see :func:`_integrate_ray` for the outputs."""
    for n in prange(Y.shape[0]):
        _integrate_ray(n, geom, consts, params, ipar, tab_a, tab_e, rk4_b, rk4_c, Y, F, lam_out,
                       n_steps_out, n_rej_out, state_out, max_null, max_e, max_lz, max_q,
                       e0, lz0, q0, fe, fl, fq, fe2, event_Y, event_hit)
