"""Audit: why photons fail in the rtol = 1e-4 bisection of Table 4.

Repeats the bisection of scripts/study_edge_bisection.py at rtol = 1e-4 and,
for every photon that neither escaped nor was captured, records the rule that
fired (r < 0 or theta outside [0, pi]), its angular momentum, and its true
fate from a re-trace at rtol = 1e-10. A failure whose true fate is ESCAPED
moves the inner bracket outward and misdirects the search.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_bisection_failures.py
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_edge_bisection import PointCamera  # noqa: E402

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.backends import get_backend
from kerrray.raytracing.boundary import analytic_boundary, finite_distance_scale, polygon_centroid, polygon_radii
from kerrray.raytracing.camera import initial_states

warnings.simplefilter("ignore")
np.seterr(all="ignore")
backend = get_backend("numba")
incl, r_o, iterations = 60.0, 1.0e5, 34
angles = np.linspace(0.0, 2 * math.pi, 16, endpoint=False) + 0.1
loose = IntegratorOptions(method="rk45", rtol=1e-4, atol=1e-6, max_steps=2_000_000, lambda_max=4.0 * r_o)
tight = IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12, max_steps=2_000_000, lambda_max=4.0 * r_o)
term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=r_o)
for spin in (0.0, 0.9, 0.99):
    st = Spacetime(1.0, spin)
    k = finite_distance_scale(st, r_o)
    ca, cb = analytic_boundary(st, incl, 40000)
    c = polygon_centroid(ca, cb)
    r_ana = polygon_radii(ca, cb, c, angles)
    centre = (c[0] / k, c[1] / k)
    lo, hi = 0.98 * r_ana / k, 1.02 * r_ana / k
    n_fail = n_rneg = n_theta = n_other = n_true_esc = 0
    lz_fail = []
    steps_fail = []
    dirs_hit = set()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        y0 = initial_states(PointCamera(r_o, incl, centre[0] + mid * np.cos(angles), centre[1] + mid * np.sin(angles)), st)
        res = backend.integrate_batch(st, y0, loose, term)
        esc = res.state == int(TS.ESCAPED)
        fail = ~esc & (res.state != int(TS.CAPTURED))
        if fail.any():
            Y = res.Y[fail]
            rneg = Y[:, 1] < 0
            thout = ~rneg & ((Y[:, 2] < -1e-9) | (Y[:, 2] > math.pi + 1e-9))
            n_fail += int(fail.sum()); n_rneg += int(rneg.sum()); n_theta += int(thout.sum()); n_other += int((~rneg & ~thout).sum())
            lz_fail.extend(np.abs(y0[fail, 7]).tolist())
            steps_fail.extend(res.n_steps[fail].tolist())
            ref = backend.integrate_batch(st, y0[fail], tight, term)
            n_true_esc += int(np.sum(ref.state == int(TS.ESCAPED)))
            dirs_hit.update(np.flatnonzero(fail).tolist())
        hi = np.where(esc, mid, hi)
        lo = np.where(esc, lo, mid)
    print(f"a={spin}: failed {n_fail} of {16 * iterations}; rule: r<0 {n_rneg}, theta out {n_theta}, other {n_other}; "
          f"true fate escaped {n_true_esc}; |L_z| of failed: min {min(lz_fail):.3f} median {float(np.median(lz_fail)):.3f}; "
          f"directions affected {len(dirs_hit)} of 16; accepted steps of failed: median {int(np.median(steps_fail))}, max {int(max(steps_fail))}", flush=True)
