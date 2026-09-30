"""Audit item 10: Dormand-Prince tableau, error norm, controller, NumPy vs Numba kernels.

Run: ./.venv/Scripts/python.exe scripts/audit/audit_integrator.py
"""
import math
import warnings
from fractions import Fraction

import numpy as np
from scipy.integrate._ivp.rk import RK45

from kerrray.geodesics import tableaus as T
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.rays import trace_rays

warnings.simplefilter("ignore")

# ---- tableau vs SciPy and vs exact rationals (Dormand & Prince 1980) -------
A = np.zeros((7, 7))
for i, row in enumerate(T.DP_A):
    A[i, : len(row)] = row
print("max |A - scipy A| (rows 0..5):", np.max(np.abs(A[:6, :5] - RK45.A)), " (scipy A shape", RK45.A.shape, ")")
print("max |B - scipy B|:", np.max(np.abs(np.array(T.DP_B[:6]) - RK45.B)), " (B[6] = 0:", T.DP_B[6] == 0, ")")
print("max |C - scipy C|:", np.max(np.abs(np.array(T.DP_C[:6]) - RK45.C)))
print("max |E - scipy E|:", np.max(np.abs(np.array(T.DP_E) - RK45.E)))
print("FSAL: last row of A == B:", T.DP_A[6] == tuple(T.DP_B[:6]))
# order conditions of the 5th-order weights (exact rationals from the floats are fine at 1e-15)
b = np.array(T.DP_B); bh = np.array(T.DP_B_HAT); c = np.array(T.DP_C)
print("row-sum condition max |sum_j a_ij - c_i|:", max(abs(sum(T.DP_A[i]) - c[i]) for i in range(1, 7)))
for name, w, order in (("b", b, 5), ("b_hat", bh, 4)):
    conds = [abs(np.dot(w, c**k) - 1 / (k + 1)) for k in range(order)]
    print(f"quadrature conditions sum {name}_i c_i^k = 1/(k+1), k<{order}: max dev {max(conds):.1e}")
print("b_hat fifth-order quadrature deviation (must be non-zero):", abs(np.dot(bh, c**4) - 1 / 5))

# ---- error norm and step factor vs HNW (4.11)-(4.13) --------------------
rng = np.random.default_rng(1)
e = rng.normal(size=(50, 8)) * 1e-6; y0 = rng.normal(size=(50, 8)); y1 = y0 + rng.normal(size=(50, 8)) * 1e-3
rtol, atol = 1e-6, 1e-8
ref = np.sqrt(np.mean((e / (atol + rtol * np.maximum(abs(y0), abs(y1)))) ** 2, axis=1))
print("error_norm vs HNW formula:", np.max(np.abs(T.error_norm(e, y0, y1, atol, rtol) - ref)))
errs = np.array([0.0, 1e-12, 0.01, 0.5, 1.0, 2.0, 1e6])
print("step_factor:", T.step_factor(errs), " after rejection:", T.step_factor(errs, True))

# ---- empirical order of the propagated DP5 solution (y' = cos(t) y) ------
def rhs(y):
    return np.stack([np.ones_like(y[..., 0]), np.cos(y[..., 0]) * y[..., 1]], -1)
step = T.make_step_rk45(rhs)
errs = []
for n in (20, 40, 80, 160):
    h = 2.0 / n; y = np.array([[0.0, 1.0]]); f = None
    for _ in range(n):
        y, _, f = step(y, np.array([h]), f)
    errs.append(abs(y[0, 1] - math.exp(math.sin(2.0))))
print("DP5 global order:", [round(math.log2(errs[i] / errs[i + 1]), 2) for i in range(3)])

# ---- NumPy vs Numba backends on camera rays -------------------------------
for method, extra in (("rk45", dict(rtol=1e-6, atol=1e-8)), ("rk45", dict(rtol=1e-4, atol=1e-6)), ("rk4", dict(step_size=0.1, max_steps=20000))):
    st = Spacetime(1.0, 0.9)
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=24)
    Y0 = initial_states(cam, st)
    integ = IntegratorOptions(method=method, **extra)
    term = TerminationOptions(1e-6, 1000.0)
    r1 = trace_rays(st, Y0, integ, term, backend="numpy")
    r2 = trace_rays(st, Y0, integ, term, backend="numba")
    same_state = np.array_equal(r1.state, r2.state)
    fin = np.isfinite(r1.Y).all(1) & np.isfinite(r2.Y).all(1)
    dy = np.max(np.abs(r1.Y[fin] - r2.Y[fin]) / (1 + np.abs(r1.Y[fin])))
    print(f"numpy vs numba {method} {extra}: identical states {same_state}, identical steps "
          f"{np.array_equal(r1.n_steps, r2.n_steps)}, max rel |dY| {dy:.1e}, states {np.bincount(r1.state, minlength=7)}")
