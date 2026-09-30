"""Audit item 11b (mechanism): step history of one failing photon near the alpha = 0 column at rtol = 1e-4 vs 1e-6.

The photon chosen (xi = -0.027) fails through the r < 0 rule: its last step jumps
from r = 406 M to r = -412 M (research.md section 5.5, first mechanism).

Run: ./.venv/Scripts/python.exe scripts/audit/audit_axis_mechanism.py
"""
import math
import warnings

import numpy as np

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geodesics.integrators_batch import integrate_batch
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.camera import Camera, initial_states

warnings.simplefilter("ignore")
st = Spacetime(1.0, 0.5)
cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=256)
Y0 = initial_states(cam, st)
al = cam.pixel_coordinates()[0].ravel()
cols = np.flatnonzero(np.abs(al) < 0.05)
term = TerminationOptions(1e-6, 1000.0)
res = integrate_batch(st, Y0[cols], IntegratorOptions(method="rk45", rtol=1e-4, atol=1e-6), term)
bad = cols[res.state == TS.OUT_OF_DOMAIN]
i = bad[len(bad) // 2]
y0 = Y0[i]
E, L = -y0[4], y0[7]
from kerrray.photons.constants import carter_constant
eta = float(carter_constant(st, y0)) / E**2
xi = L / E
th_min = math.atan(abs(xi) / math.sqrt(eta + st.a**2))  # approx turning angle from Theta = 0 (small-angle)
print(f"pixel {i}: xi = {xi:.4e}, eta = {eta:.3f}; exact polar turning point theta_min ~ {th_min:.3e} rad")
for rtol in (1e-4, 1e-6):
    rows = []
    integrate_batch(st, y0[None], IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2), term,
                    on_step=lambda idx, ys, c: rows.append((ys[0, 1], ys[0, 2], ys[0, 3], ys[0, 6], int(c[0]))))
    arr = np.array(rows)
    k = int(np.argmin(np.abs(np.sin(arr[:, 1]))))
    print(f" rtol={rtol:g}: {len(rows)} accepted steps, final {TS(rows[-1][4]).name}; closest approach to the axis at step {k}:")
    for rr in rows[max(0, k - 3): k + 3]:
        print(f"    r={rr[0]:9.4f} theta={rr[1]:+.6e} phi={rr[2]:+.4f} p_theta={rr[3]:+.4e} {TS(rr[4]).name}")
