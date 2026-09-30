"""Computation endpoints behind KerrRay Desktop.

Each public function takes a plain ``dict`` of request parameters, runs the
engine, and returns a JSON-serialisable ``dict``. Parameters are validated
and clamped to ranges that keep an interactive request under about a minute
on a laptop CPU. Reference values (3 M, 3 sqrt(3) M) appear only as
comparison targets for errors, never as results.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from kerrray import __version__
from kerrray.geodesics.initial_conditions import equatorial_photon, tangential_photon
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate
from kerrray.geodesics.integrators_batch import integrate_batch
from kerrray.geodesics.state import IDX_PH, IDX_R, IDX_TH
from kerrray.geometry.coordinates import bl_to_cartesian
from kerrray.geometry.horizons import ergosphere_radius, horizon_radii
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.photons.orbits import (
    critical_impact_parameters,
    equatorial_photon_orbit_radius,
    isco_radius,
    shadow_curve,
)
from kerrray.photons.trajectories import azimuthal_winding, closest_approach, to_cartesian
from kerrray.raytracing.backends import BackendUnavailable, get_backend
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.utils.manifest import collect_environment
from kerrray.web.jobs import JobManager

REF_PHOTON_SPHERE = 3.0
"""Schwarzschild photon-sphere radius in units of M; comparison reference only."""
REF_CRITICAL_B = 3.0 * math.sqrt(3.0)
"""Schwarzschild critical impact parameter in units of M; comparison reference only."""
MAX_RESOLUTION = 160
"""Largest interactive shadow resolution (pixels per side)."""
MAX_TRAJECTORY_POINTS = 4000
"""Trajectories are thinned to at most this many points for plotting."""


class ApiError(ValueError):
    """A request parameter is missing or out of range."""


def _num(p: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(p.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{key} must be a number") from exc
    if not math.isfinite(value) or value < lo or value > hi:
        raise ApiError(f"{key} must lie in [{lo}, {hi}], got {value}")
    return value


def _spacetime(p: dict[str, Any]) -> Spacetime:
    return Spacetime(mass=1.0, spin=_num(p, "spin", 0.9, -0.9999, 0.9999))


def _integrator(p: dict[str, Any], rtol_default: float = 1e-8) -> IntegratorOptions:
    method = str(p.get("method", "rk45"))
    if method not in ("rk4", "rk45"):
        raise ApiError("method must be rk4 or rk45")
    rtol = _num(p, "rtol", rtol_default, 1e-12, 1e-4)
    step = _num(p, "step_size", 0.05, 1e-3, 1.0)
    return IntegratorOptions(method=method, rtol=rtol, atol=rtol * 1e-2, step_size=step)


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def info(_: dict[str, Any]) -> dict[str, Any]:
    """Engine version and host environment."""
    env = collect_environment()
    return {
        "version": __version__,
        "python": env.python_version,
        "platform": env.platform,
        "processor": env.processor,
        "cpu_count": env.cpu_count,
        "git_commit": env.git_commit[:10],
        "packages": env.package_versions,
    }


def blackhole(p: dict[str, Any]) -> dict[str, Any]:
    """Horizons, ergosphere, photon orbits, ISCO and critical impact parameters."""
    st = _spacetime(p)
    r_plus, r_minus = horizon_radii(st)
    b_pro, b_retro = critical_impact_parameters(st)
    return {
        "spin": st.spin,
        "mass": st.mass,
        "a": st.a,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "ergosphere_equator": float(ergosphere_radius(st, 0.5 * math.pi)),
        "ergosphere_pole": float(ergosphere_radius(st, 0.0)),
        "photon_orbit_prograde": equatorial_photon_orbit_radius(st, prograde=True),
        "photon_orbit_retrograde": equatorial_photon_orbit_radius(st, prograde=False),
        "isco_prograde": isco_radius(st, prograde=True),
        "isco_retrograde": isco_radius(st, prograde=False),
        "b_crit_prograde": b_pro,
        "b_crit_retrograde": b_retro,
    }


def _ergosphere_curve(st: Spacetime, n: int = 181) -> dict[str, list[float]]:
    th = np.linspace(0.0, math.pi, n)
    r_e = ergosphere_radius(st, th)
    rho = np.sqrt(r_e**2 + st.a**2) * np.sin(th)
    return {"x": np.round(rho, 5).tolist(), "z": np.round(r_e * np.cos(th), 5).tolist()}


def geodesic(p: dict[str, Any]) -> dict[str, Any]:
    """Integrate one equatorial photon and return its path and diagnostics."""
    st = _spacetime(p)
    b = _num(p, "b", 5.0, 0.0, 200.0)
    r0 = _num(p, "r0", 50.0, 5.0, 5000.0)
    prograde = bool(p.get("prograde", True))
    integ = _integrator(p, 1e-10)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=max(1.2 * r0, 60.0))
    y0 = equatorial_photon(st, r0, b, prograde=prograde)
    traj = integrate(st, y0, integ, term)
    x, y, _ = to_cartesian(traj)
    stride = max(1, len(x) // MAX_TRAJECTORY_POINTS)
    d = traj.diagnostics
    b_pro, b_retro = critical_impact_parameters(st)
    return {
        "spin": st.spin,
        "b": b,
        "prograde": prograde,
        "r0": r0,
        "state": traj.state.name,
        "x": np.round(x[::stride], 5).tolist(),
        "y": np.round(y[::stride], 5).tolist(),
        "lam": np.round(traj.lam[::stride], 5).tolist(),
        "null_error": [float(v) for v in np.maximum(d.null_error[::stride], 1e-18)],
        "closest_approach": closest_approach(traj),
        "delta_phi": azimuthal_winding(traj),
        "turns": azimuthal_winding(traj) / (2.0 * math.pi),
        "steps": traj.n_steps,
        "rejected": traj.n_rejected,
        "runtime_s": traj.runtime_s,
        "max_null_error": float(np.max(d.null_error)),
        "max_energy_drift": float(np.max(d.energy_drift)),
        "max_lz_drift": float(np.max(d.lz_drift)),
        "r_plus": horizon_radii(st)[0],
        "b_crit": b_pro if prograde else b_retro,
        "ergosphere": _ergosphere_curve(st),
        "photon_orbit": equatorial_photon_orbit_radius(st, prograde=prograde),
        "method": integ.method,
        "rtol": integ.rtol,
    }


def shadow(p: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Trace a camera image and return per-pixel outcomes and sky coordinates.

    ``progress``, when given, receives ``{"done", "total", "state"}`` after
    every integrator sweep, where ``state`` holds the termination code of
    each finished ray and 0 for rays still running. It may raise to cancel.
    """
    st = _spacetime(p)
    incl = _num(p, "inclination", 60.0, 0.0, 180.0)
    res = int(_num(p, "resolution", 64, 8, MAX_RESOLUTION))
    res += res % 2
    fov = _num(p, "fov", 8.0, 2.0, 40.0)
    r_obs = _num(p, "observer_radius", 1000.0, 50.0, 1.0e5)
    integ = _integrator(p, 1e-8)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=r_obs)
    cam = Camera(radius=r_obs, inclination_deg=incl, fov=fov, resolution=res)
    y0 = initial_states(cam, st)
    # Optional band of image rows [row_start, row_end) so the UI can stream the image.
    row_start = int(_num(p, "row_start", 0, 0, res))
    row_end = int(_num(p, "row_end", res, row_start + 1, res)) if row_start < res else res
    y0 = y0[row_start * res : row_end * res]
    backend = str(p.get("backend", "auto"))
    if backend not in ("auto", "numpy", "numba"):
        raise ApiError("backend must be auto, numpy or numba")
    if backend in ("auto", "numba"):
        try:
            compiled = get_backend("numba")
            backend = "numba"
        except BackendUnavailable:
            if backend == "numba":
                raise ApiError("the numba backend is not available (pip install numba)") from None
            backend = "numpy"
    t0 = time.perf_counter()
    if backend == "numba":
        # The compiled kernel has no per-step hook, so it traces in chunks and
        # reports each finished chunk; results are identical to the numpy path.
        n_rays = y0.shape[0]
        n_chunks = max(1, min(16, n_rays // 256))
        snap = np.zeros(n_rays, dtype=np.int64)
        parts = []
        for k, idx in enumerate(np.array_split(np.arange(n_rays), n_chunks)):
            part = compiled.integrate_batch(st, y0[idx], integ, term)
            parts.append(part)
            snap[idx] = part.state
            if progress is not None:
                progress({"done": int(np.count_nonzero(snap)), "total": n_rays, "state": snap, "fraction": (k + 1) / n_chunks})
        result = _concat_batches(parts)
        runtime = time.perf_counter() - t0
        return _shadow_payload(st, p, incl, res, fov, r_obs, integ, result, runtime, row_start, row_end, y0.shape[0], backend)
    on_step = None
    if progress is not None:
        snap = np.zeros(y0.shape[0], dtype=np.int64)
        r_now = y0[:, IDX_R].astype(float).copy()
        outward = np.zeros(y0.shape[0], dtype=bool)

        def on_step(idx: np.ndarray, y: np.ndarray, codes: np.ndarray) -> None:
            fin = codes != int(TerminationState.RUNNING)
            if fin.any():
                snap[idx[fin]] = codes[fin]
            r_now[idx] = y[:, IDX_R]
            outward[idx] = y[:, 5] > 0.0  # sign of p_r equals the sign of dr/dlambda (g^rr > 0 outside the horizon)
            # Progress estimate from photon positions: inbound leg 0..0.5, outbound leg 0.5..1, finished 1.
            frac = np.where(outward, 0.5 + 0.5 * np.clip(r_now / r_obs, 0, 1), 0.5 * np.clip(1 - r_now / r_obs, 0, 1))
            frac[snap != 0] = 1.0
            progress({"done": int(np.count_nonzero(snap)), "total": int(snap.size), "state": snap,
                      "fraction": float(frac.mean()), "r": np.round(r_now, 1), "out": outward.astype(np.int8)})

    result = integrate_batch(st, y0, integ, term, on_step=on_step)
    runtime = time.perf_counter() - t0
    return _shadow_payload(st, p, incl, res, fov, r_obs, integ, result, runtime, row_start, row_end, y0.shape[0], "numpy")


def _concat_batches(parts: list[Any]) -> Any:
    """Join chunked BatchResults (only the fields the shadow payload uses)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        Y=np.concatenate([q.Y for q in parts]),
        state=np.concatenate([q.state for q in parts]),
        n_steps=np.concatenate([q.n_steps for q in parts]),
        max_null_error=np.concatenate([q.max_null_error for q in parts]),
        max_carter_drift=np.concatenate([q.max_carter_drift for q in parts]),
    )


def _shadow_payload(st: Spacetime, p: dict[str, Any], incl: float, res: int, fov: float, r_obs: float, integ: IntegratorOptions,
                    result: Any, runtime: float, row_start: int, row_end: int, n_traced: int, backend: str) -> dict[str, Any]:
    """Assemble the JSON result of a shadow trace (shared by both backends)."""
    state = result.state.astype(int)
    captured = state == int(TerminationState.CAPTURED)
    escaped = state == int(TerminationState.ESCAPED)
    theta_end = np.where(escaped, result.Y[:, IDX_TH], np.nan)
    phi_end = np.where(escaped, np.mod(result.Y[:, IDX_PH], 2.0 * math.pi), np.nan)
    alpha_c, beta_c = shadow_curve(st, incl)
    pix = 2.0 * fov / res
    area_num = float(captured.sum()) * pix * pix  # of this band; the full image when no band is given
    area_ref = float(0.5 * abs(np.sum(alpha_c * np.roll(beta_c, -1) - np.roll(alpha_c, -1) * beta_c)))
    counts = {s.name: int(np.sum(state == int(s))) for s in TerminationState if np.any(state == int(s))}
    null_err = np.maximum(result.max_null_error.astype(float), 1e-18)
    return {
        "spin": st.spin,
        "inclination": incl,
        "resolution": res,
        "fov": fov,
        "observer_radius": r_obs,
        "state": state.tolist(),
        "steps": result.n_steps.astype(int).tolist(),
        "log_null_error": np.round(np.log10(null_err), 3).tolist(),
        "theta_end": [None if not math.isfinite(v) else round(float(v), 6) for v in theta_end],
        "phi_end": [None if not math.isfinite(v) else round(float(v), 6) for v in phi_end],
        "analytic_alpha": np.round(alpha_c, 5).tolist(),
        "analytic_beta": np.round(beta_c, 5).tolist(),
        "counts": counts,
        "row_start": row_start,
        "row_end": row_end,
        "n_rays": int(n_traced),
        "runtime_s": runtime,
        "rays_per_s": n_traced / runtime if runtime > 0 else None,
        "backend": backend,
        "shadow_area_numerical": area_num,
        "shadow_area_analytic": area_ref,
        "area_rel_diff": abs(area_num - area_ref) / area_ref if area_ref > 0 else None,
        "max_null_error_escaped": _finite(float(np.max(null_err[escaped]))) if escaped.any() else None,
        "max_null_error_captured": _finite(float(np.max(null_err[captured]))) if captured.any() else None,
        "max_carter_drift_escaped": _finite(float(np.max(result.max_carter_drift[escaped]))) if escaped.any() else None,
        "method": integ.method,
        "rtol": integ.rtol,
    }


def _bisect(
    is_captured: Any, lo: float, hi: float, tol: float, on_iter: Callable[[int, float, float], None] | None = None
) -> tuple[float, float, int]:
    n = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_captured(mid):
            lo = mid
        else:
            hi = mid
        n += 1
        if on_iter is not None:
            on_iter(n, lo, hi)
    return 0.5 * (lo + hi), hi - lo, n


def validate(p: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Recover the Schwarzschild photon sphere and critical impact parameter by bisection.

    ``progress``, when given, receives the current check, iteration and
    bracket after every bisection step. It may raise to cancel.
    """
    st = Spacetime(mass=1.0, spin=0.0)
    tol = _num(p, "tol", 1e-6, 1e-9, 1e-2)
    pass_rel = _num(p, "pass_rel", 1e-5, 1e-12, 1e-1)
    integ = _integrator(p, 1e-9)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)

    def captured(y0: np.ndarray) -> bool:
        return integrate(st, y0, integ, term, record=False).state == TerminationState.CAPTURED

    which = str(p.get("which", "both"))
    if which not in ("both", "photon_sphere", "critical_b"):
        raise ApiError("which must be both, photon_sphere or critical_b")
    t0 = time.perf_counter()
    def reporter(check: str) -> Callable[[int, float, float], None] | None:
        if progress is None:
            return None
        return lambda k, lo, hi: progress({"check": check, "iteration": k, "lo": lo, "hi": hi})

    r_ph, r_w, n1 = _bisect(lambda r: captured(tangential_photon(st, r)), 2.5, 3.5, tol, reporter("photon_sphere")) if which != "critical_b" else (math.nan, math.nan, 0)
    t1 = time.perf_counter()
    b_c, b_w, n2 = _bisect(lambda b: captured(equatorial_photon(st, 1000.0, b)), 4.0, 6.0, tol, reporter("critical_b")) if which != "photon_sphere" else (math.nan, math.nan, 0)
    t2 = time.perf_counter()
    checks = [c for c, keep in zip([
            {
                "name": "Photon sphere radius",
                "reference": REF_PHOTON_SPHERE,
                "computed": r_ph,
                "bracket": r_w,
                "rel_error": abs(r_ph - REF_PHOTON_SPHERE) / REF_PHOTON_SPHERE,
                "iterations": n1,
                "runtime_s": t1 - t0,
                "passed": abs(r_ph - REF_PHOTON_SPHERE) / REF_PHOTON_SPHERE <= pass_rel,
            },
            {
                "name": "Critical impact parameter",
                "reference": REF_CRITICAL_B,
                "computed": b_c,
                "bracket": b_w,
                "rel_error": abs(b_c - REF_CRITICAL_B) / REF_CRITICAL_B,
                "iterations": n2,
                "runtime_s": t2 - t1,
                "passed": abs(b_c - REF_CRITICAL_B) / REF_CRITICAL_B <= pass_rel,
            },
        ], (which != "critical_b", which != "photon_sphere")) if keep]
    return {
        "checks": checks,
        "method": integ.method,
        "rtol": integ.rtol,
        "tol": tol,
        "pass_rel": pass_rel,
        "runtime_s": t2 - t0,
    }


def spin_sweep(p: dict[str, Any]) -> dict[str, Any]:
    """Horizon, photon orbits, ISCO and critical impact parameters across spin."""
    n = int(_num(p, "n", 99, 5, 400))
    spins = np.linspace(0.0, 0.998, n)
    rows = {k: [] for k in ("r_plus", "ph_pro", "ph_retro", "isco_pro", "isco_retro", "b_pro", "b_retro")}
    for a in spins:
        st = Spacetime(mass=1.0, spin=float(a))
        rows["r_plus"].append(horizon_radii(st)[0])
        rows["ph_pro"].append(equatorial_photon_orbit_radius(st, prograde=True))
        rows["ph_retro"].append(equatorial_photon_orbit_radius(st, prograde=False))
        rows["isco_pro"].append(isco_radius(st, prograde=True))
        rows["isco_retro"].append(isco_radius(st, prograde=False))
        b_pro, b_retro = critical_impact_parameters(st)
        rows["b_pro"].append(b_pro)
        rows["b_retro"].append(b_retro)
    return {"spin": spins.round(6).tolist(), **{k: np.round(v, 6).tolist() for k, v in rows.items()}}


MAX_BEAM_PHOTONS = 160
"""Largest photon count for one animated beam."""


def _beam_impact_parameters(b_max: float, n: int, b_pro: float, b_retro: float, sign_pro: float) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced impact parameters plus photons clustered at both critical values.

    ``sign_pro`` is the sign of ``b`` (the side of the beam) whose photons are
    prograde. Returns the signed impact parameters and a flag marking the
    near-critical ones.
    """
    base = np.linspace(-b_max, b_max, n)
    near = []
    for b_c, sign in ((b_pro, sign_pro), (b_retro, -sign_pro)):
        for f in (0.995, 1.002, 1.01, 1.04):
            if b_c * f <= b_max:
                near.append(sign * b_c * f)
    b = np.concatenate([base, np.array(near)])
    flag = np.concatenate([np.zeros(base.size, bool), np.ones(len(near), bool)])
    order = np.argsort(b)
    return b[order], flag[order]


def photon_beam(p: dict[str, Any]) -> dict[str, Any]:
    """Integrate a flat wavefront of equatorial photons and resample it in coordinate time.

    Photons start on the line ``x = x0`` at ``t = 0`` moving in the ``-x``
    direction, so a photon at height ``y = b`` has ``L_z = b E`` (``E = 1``).
    The batch integrator records every accepted step; each path is then
    linearly interpolated onto a common grid of Boyer-Lindquist coordinate
    time ``t``, the time of a distant static observer. Photons falling in
    therefore slow down near the horizon, as they do for such an observer.
    Plot positions use the same oblate-spheroidal embedding as the other
    trajectory plots (kerrray.geometry.bl_to_cartesian).
    """
    st = _spacetime(p)
    n = int(_num(p, "n", 48, 4, MAX_BEAM_PHOTONS - 8))
    x0 = _num(p, "x0", 30.0, 10.0, 200.0)
    b_max = _num(p, "b_max", 10.0, 1.0, 40.0)
    frames = int(_num(p, "frames", 360, 30, 1500))
    integ = IntegratorOptions(method="rk45", rtol=_num(p, "rtol", 1e-8, 1e-11, 1e-5), atol=1e-10, step_size=0.05)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1.5 * math.hypot(x0, b_max))
    b_pro, b_retro = critical_impact_parameters(st)
    sign_pro = 1.0 if st.a >= 0 else -1.0
    b, near = _beam_impact_parameters(b_max, n, b_pro, b_retro, sign_pro)
    y0 = np.empty((b.size, 8))
    for i, bi in enumerate(b):
        r0 = math.hypot(x0, bi)
        y0[i] = equatorial_photon(st, r0, abs(bi), prograde=bool((bi > 0) == (st.a >= 0)) if bi != 0 else True)
        y0[i, IDX_PH] = math.atan2(bi, x0)

    samples: list[tuple[np.ndarray, np.ndarray]] = [(np.arange(b.size), y0.copy())]
    result = integrate_batch(st, y0, integ, term, on_step=lambda idx, y, codes: samples.append((idx, y)))
    idx_all = np.concatenate([s[0] for s in samples])
    y_all = np.concatenate([s[1] for s in samples]).astype(float)
    xs, ys, _ = bl_to_cartesian(st, y_all[:, IDX_R], y_all[:, IDX_TH], y_all[:, IDX_PH])
    t_all = y_all[:, 0]

    t_end = np.array([t_all[idx_all == i].max() for i in range(b.size)])
    t_max = float(np.max(t_end))
    t_grid = np.linspace(0.0, t_max, frames)
    paths_x, paths_y = [], []
    for i in range(b.size):
        m = idx_all == i
        ti, order = t_all[m], np.argsort(t_all[m])
        xi = np.interp(t_grid, ti[order], xs[m][order], right=np.nan)
        yi = np.interp(t_grid, ti[order], ys[m][order], right=np.nan)
        paths_x.append([None if not math.isfinite(v) else round(float(v), 3) for v in xi])
        paths_y.append([None if not math.isfinite(v) else round(float(v), 3) for v in yi])
    state = result.state.astype(int)
    return {
        "spin": st.spin,
        "x0": x0,
        "t": np.round(t_grid, 4).tolist(),
        "b": np.round(b, 5).tolist(),
        "near_critical": near.tolist(),
        "state": [TerminationState(int(s)).name for s in state],
        "t_end": np.round(t_end, 4).tolist(),
        "x": paths_x,
        "y": paths_y,
        "r_plus": horizon_radii(st)[0],
        "photon_orbit_prograde": equatorial_photon_orbit_radius(st, prograde=True),
        "photon_orbit_retrograde": equatorial_photon_orbit_radius(st, prograde=False),
        "b_crit_prograde": b_pro,
        "b_crit_retrograde": b_retro,
        "n_photons": int(b.size),
        "captured": int(np.sum(state == int(TerminationState.CAPTURED))),
        "escaped": int(np.sum(state == int(TerminationState.ESCAPED))),
        "accepted_steps": int(result.n_steps.sum()),
        "max_null_error_escaped": _finite(float(np.max(result.max_null_error[state == 1]))) if np.any(state == 1) else None,
        "runtime_s": result.runtime_s,
    }


ENDPOINTS = {
    "info": info,
    "blackhole": blackhole,
    "geodesic": geodesic,
    "shadow": shadow,
    "validate": validate,
    "spin_sweep": spin_sweep,
    "photon_beam": photon_beam,
}


JOBS = JobManager({"shadow": shadow, "validate": validate})
"""Background jobs with live progress (see kerrray.web.jobs)."""


def _job_start(p: dict[str, Any]) -> dict[str, Any]:
    params = p.get("params", {})
    if not isinstance(params, dict):
        raise ApiError("params must be an object")
    try:
        return JOBS.start(str(p.get("kind", "")), params)
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


def _job_call(method: Callable[[str], dict[str, Any]], p: dict[str, Any]) -> dict[str, Any]:
    try:
        return method(str(p.get("id", "")))
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


ENDPOINTS.update({
    "job_start": _job_start,
    "job_poll": lambda p: _job_call(JOBS.poll, p),
    "job_cancel": lambda p: _job_call(JOBS.cancel, p),
})


def dispatch(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run endpoint ``name`` with RuntimeWarnings from horizon/axis loci silenced."""
    if name not in ENDPOINTS:
        raise ApiError(f"unknown endpoint {name!r}")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return ENDPOINTS[name](params)
