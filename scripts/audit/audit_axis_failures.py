"""Audit item 11b: the OUT_OF_DOMAIN photons at rtol = 1e-4 near the alpha = 0 column.

For each failed photon: which rule fired, whether it lies inside the analytic
shadow, and its true fate from a re-trace at rtol = 1e-10. Then the edge error
with the failed photons (i) counted as shadow (paper) and (ii) replaced by
their true fate.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_axis_failures.py [N] [spin]
"""
import math
import sys
import warnings

import numpy as np
from matplotlib.path import Path

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.boundary import analytic_boundary, boundary_error, extract_boundary_from_mask, finite_distance_scale, mask_centroid
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.rays import trace_rays
from kerrray.raytracing.shadow import compute_shadow

warnings.simplefilter("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
spin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
st = Spacetime(1.0, spin)
cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=N)
term = TerminationOptions(1e-6, 1000.0)
img = compute_shadow(st, cam, IntegratorOptions(method="rk45", rtol=1e-4, atol=1e-6), term, backend="numba")
state = img.state.ravel()
failed = (state != TS.ESCAPED) & (state != TS.CAPTURED)
Yend = img.result.Y
th = Yend[failed, 2]
rf = Yend[failed, 1]
lz = Yend[failed, 7]
r_neg = rf < 0
th_out = ~r_neg & ((th < -1e-9) | (th > math.pi + 1e-9))
axis = ~r_neg & ~th_out & (np.abs(np.sin(th)) < 1e-12) & (lz != 0)
print(f"a={spin} N={N} rtol=1e-4: failed {failed.sum()} ({dict((TS(i).name, int(n)) for i, n in enumerate(np.bincount(state[failed], minlength=7)) if n)}); "
      f"rule that fired: r < 0: {int(r_neg.sum())}, theta outside [0, pi]: {int(th_out.sum())}, |sin theta| < 1e-12: {int(axis.sum())}")
nsteps = img.result.n_steps.ravel()[failed]
print(f"  accepted steps of failed photons: median {int(np.median(nsteps))}, max {int(nsteps.max())} (r < 0 group: median {int(np.median(nsteps[r_neg])) if r_neg.any() else -1})")
k = finite_distance_scale(st, 1000.0)
al, be = img.alpha.ravel(), img.beta.ravel()
ca, cb = analytic_boundary(st, 60.0, 4000)
inside = Path(np.stack([ca, cb], 1)).contains_points(np.stack([al * k, be * k], 1))
fi = inside[failed]
print(f"  r<0 group: inside shadow {int((r_neg & fi).sum())}, outside {int((r_neg & ~fi).sum())}; theta/axis group: inside {int((~r_neg & fi).sum())}, outside {int((~r_neg & ~fi).sum())}")
print(f"  failed photons inside the analytic shadow: {int((failed & inside).sum())}, outside: {int((failed & ~inside).sum())}; "
      f"max |alpha| of failed = {np.max(np.abs(al[failed])):.4f} M = {np.max(np.abs(al[failed]))/img.pixel_size:.2f} px")
Y0 = initial_states(cam, st)[failed]
ref = trace_rays(st, Y0, IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12), term, backend="numba")
true_cap = ref.state == TS.CAPTURED
print(f"  re-traced at rtol 1e-10: {dict((TS(i).name, int(n)) for i, n in enumerate(np.bincount(ref.state, minlength=7)) if n)}; "
      f"true fate agrees with the analytic inside/outside test for {int((true_cap == inside[failed]).sum())}/{failed.sum()}")

shadow_paper = (state != TS.ESCAPED).reshape(img.alpha.shape)
fixed = state.copy(); fixed[np.flatnonzero(failed)] = ref.state
shadow_true = (fixed != TS.ESCAPED).reshape(img.alpha.shape)
for name, sh in (("failed counted as shadow (paper)", shadow_paper), ("failed replaced by true fate", shadow_true)):
    c = mask_centroid(img.alpha, img.beta, sh)
    ang, rad = extract_boundary_from_mask(img.alpha, img.beta, sh, n_angles=720, centre=c)
    e = boundary_error((ang, rad), (ca, cb), c, pixel_size=img.pixel_size, scale=k)
    print(f"  {name}: rms {e['rms']:.3e} M, max {e['max']:.3e} M")
