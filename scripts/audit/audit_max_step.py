"""Audit: does a maximum step size remove the rtol = 1e-4 failures?

research.md sections 5.3 and 5.5 attribute most failures at rtol = 1e-4 to
the unbounded adaptive step, which grows until one step carries a photon
through the hole. This script repeats (i) the N = 512 images of Table 6 and
(ii) the bisection searches of Table 4 at rtol = 1e-4 with the step capped at
h_max (IntegratorOptions.h_max; None is the unbounded controller of the paper)
and reports failures by the rule that fired, pixels that disagree with the
analytic (oracle) mask, the mask-edge error, the bisection-edge error and the
run time.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_max_step.py
"""
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_edge_bisection import bisect_edge  # noqa: E402

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.backends import get_backend
from kerrray.raytracing.boundary import (analytic_boundary, boundary_error, extract_boundary_from_mask,
                                         finite_distance_scale, mask_centroid, polygon_centroid, polygon_radii)
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import compute_shadow

warnings.simplefilter("ignore")
np.seterr(all="ignore")
H_MAX = (None, 100.0, 10.0)
SPINS = (0.0, 0.5, 0.9, 0.99)
N, RTOL = 512, 1e-4
print(f"Images: N = {N}, i = 60 deg, r_o = 1000 M, rtol = {RTOL:g}, atol = rtol * 1e-2")
for spin in SPINS:
    st = Spacetime(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=N)
    term = TerminationOptions(1e-6, 1000.0)
    k = finite_distance_scale(st, 1000.0)
    ca, cb = analytic_boundary(st, 60.0, 4000)
    for h_max in H_MAX:
        t0 = time.perf_counter()
        img = compute_shadow(st, cam, IntegratorOptions(method="rk45", rtol=RTOL, atol=RTOL * 1e-2, h_max=h_max), term, backend="numba")
        dt = time.perf_counter() - t0
        state = img.state.ravel()
        failed = (state != TS.ESCAPED) & (state != TS.CAPTURED)
        Y = img.result.Y
        r_neg = failed & (Y[:, 1] < 0)
        th_out = failed & ~(Y[:, 1] < 0) & ((Y[:, 2] < -1e-9) | (Y[:, 2] > math.pi + 1e-9))
        al, be = img.alpha.ravel(), img.beta.ravel()
        inside = MPath(np.stack([ca, cb], 1)).contains_points(np.stack([al * k, be * k], 1))
        shadow = state != TS.ESCAPED
        disagree = int(np.sum(shadow != inside))
        sh = shadow.reshape(img.alpha.shape)
        c = mask_centroid(img.alpha, img.beta, sh)
        ang, rad = extract_boundary_from_mask(img.alpha, img.beta, sh, n_angles=720, centre=c)
        e = boundary_error((ang, rad), (ca, cb), c, pixel_size=img.pixel_size, scale=k)
        print(f"a={spin:<5} h_max={str(h_max):>5}: failed {int(failed.sum()):4d} (r<0 {int(r_neg.sum()):4d}, theta out {int(th_out.sum()):3d}); "
              f"pixels disagreeing with oracle {disagree:3d}; edge rms {e['rms']:.3e} M max {e['max']:.3e} M; "
              f"median steps {int(np.median(img.result.n_steps))}; {dt:.1f} s", flush=True)

print("Bisection: 16 directions, 34 iterations, r_o = 1e5 M, i = 60 deg")
backend = get_backend("numba")
angles = np.linspace(0.0, 2 * math.pi, 16, endpoint=False) + 0.1
r_o = 1.0e5
for spin in (0.0, 0.9, 0.99):
    st = Spacetime(1.0, spin)
    k = finite_distance_scale(st, r_o)
    ca, cb = analytic_boundary(st, 60.0, 40000)
    c = polygon_centroid(ca, cb)
    r_ana = polygon_radii(ca, cb, c, angles)
    for rtol in (1e-4, 1e-5):
        for h_max in (None, 1000.0, 100.0):
            import study_edge_bisection as seb
            orig = seb.IntegratorOptions
            seb.IntegratorOptions = lambda **kw: orig(**kw, h_max=h_max)
            t0 = time.perf_counter()
            r_num, width, fails = bisect_edge(st, 60.0, r_o, rtol, (c[0] / k, c[1] / k), angles, 0.98 * r_ana / k, 1.02 * r_ana / k, 34, backend)
            dt = time.perf_counter() - t0
            seb.IntegratorOptions = orig
            err = r_num * k - r_ana
            print(f"a={spin:<5} rtol={rtol:.0e} h_max={str(h_max):>6}: rms {np.sqrt(np.mean(err**2)):.2e} M, max {np.max(np.abs(err)):.2e} M, "
                  f"failed photons {fails}, {dt:.1f} s", flush=True)
