"""Does the integrator tolerance matter once the edge is located precisely?

The mask-based edge of research.md section 5.1 is limited by pixel sampling
to about a quarter of a pixel, which hides any tolerance effect. Accurate
shadow work instead locates the edge by bisection along individual image
directions (e.g. Younsi et al. 2016; Medeiros et al. 2020). This script does
that: for 16 directions about the centroid of the analytic curve it bisects
the image-plane radius between photons that escape and photons that do not,
to a bracket of 1e-9 M, at several tolerances, and compares the result with
the analytic Bardeen curve after the finite-distance correction. The
observer sits at r_o = 1e5 M so that the finite-distance residual is small.

Output: paper/data/edge_bisection.json.
Usage: ./.venv/Scripts/python.exe scripts/study_edge_bisection.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.raytracing.backends import get_backend
from kerrray.raytracing.boundary import analytic_boundary, finite_distance_scale, polygon_centroid, polygon_radii
from kerrray.raytracing.camera import initial_states
from kerrray.utils.manifest import collect_environment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "edge_bisection.json"


class PointCamera:
    """Minimal camera for initial_states: photons at arbitrary image points (alpha, beta)."""

    def __init__(self, radius: float, inclination_deg: float, alpha: np.ndarray, beta: np.ndarray) -> None:
        self.radius = radius
        self.inclination_rad = math.radians(inclination_deg)
        self.phi_rad = 0.0
        self.fov = float(max(np.max(np.abs(alpha)), np.max(np.abs(beta))))
        self._alpha, self._beta = np.asarray(alpha, float), np.asarray(beta, float)

    def pixel_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self._alpha.reshape(1, -1), self._beta.reshape(1, -1)


def bisect_edge(st, incl, r_o, rtol, centre, angles, r_lo, r_hi, iterations, backend):
    integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, max_steps=2_000_000, lambda_max=4.0 * r_o)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=r_o)
    lo, hi = r_lo.copy(), r_hi.copy()  # inside (shadow) at lo, outside at hi
    failures = 0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        a = centre[0] + mid * np.cos(angles)
        b = centre[1] + mid * np.sin(angles)
        y0 = initial_states(PointCamera(r_o, incl, a, b), st)
        res = backend.integrate_batch(st, y0, integ, term)
        escaped = res.state == int(TerminationState.ESCAPED)
        failures += int(np.sum((res.state != int(TerminationState.ESCAPED)) & (res.state != int(TerminationState.CAPTURED))))
        hi = np.where(escaped, mid, hi)
        lo = np.where(escaped, lo, mid)
    return 0.5 * (lo + hi), hi - lo, failures


def main() -> None:
    backend = get_backend("numba")
    incl, r_o, n_dirs, iterations = 60.0, 1.0e5, 16, 34
    angles = np.linspace(0.0, 2 * math.pi, n_dirs, endpoint=False) + 0.1
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
    rows = []
    t_all = time.perf_counter()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for spin in (0.0, 0.9, 0.99):
            st = Spacetime(1.0, spin)
            k = finite_distance_scale(st, r_o)
            ca, cb = analytic_boundary(st, incl, 40000)
            c_ana = polygon_centroid(ca, cb)
            r_ana = polygon_radii(ca, cb, c_ana, angles)  # asymptotic (Bardeen) radii
            centre_img = (c_ana[0] / k, c_ana[1] / k)
            r_img = r_ana / k
            for rtol in tolerances:
                t0 = time.perf_counter()
                r_num, width, failures = bisect_edge(st, incl, r_o, rtol, centre_img, angles, 0.98 * r_img, 1.02 * r_img, iterations, backend)
                err = r_num * k - r_ana
                row = {"spin": spin, "rtol": rtol, "observer_radius": r_o, "inclination_deg": incl, "directions": n_dirs,
                       "bracket_M": float(np.max(width)), "rms_error_M": float(np.sqrt(np.mean(err**2))),
                       "max_error_M": float(np.max(np.abs(err))), "mean_error_M": float(np.mean(err)),
                       "errors_M": err.tolist(), "failed_rays": failures, "runtime_s": time.perf_counter() - t0}
                rows.append(row)
                print(f"a={spin:<5} rtol={rtol:.0e}: rms {row['rms_error_M']:.2e} M  max {row['max_error_M']:.2e} M  "
                      f"mean {row['mean_error_M']:+.2e} M  bracket {row['bracket_M']:.1e}  failed {failures}  {row['runtime_s']:.1f}s", flush=True)
    env = collect_environment()
    OUT.write_text(json.dumps({"rows": rows, "angles": angles.tolist(), "git_commit": env.git_commit, "git_dirty": env.git_dirty,
                               "hardware": env.hardware(), "runtime_s": time.perf_counter() - t_all}, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
