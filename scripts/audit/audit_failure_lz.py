"""Angular momentum of the photons that fail at rtol = 1e-4 (research.md section 5.5).

For the N = 512 images of Table 6, prints |L_z| of the failed photons (max,
median) and the median |L_z| over the whole image. At large r_o,
L_z / E = -alpha sin(i) depends only on the pixel, not on the spin.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_failure_lz.py
"""
import warnings

import numpy as np

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.shadow import compute_shadow

warnings.simplefilter("ignore")
np.seterr(all="ignore")
for spin in (0.0, 0.5, 0.9, 0.99):
    st = Spacetime(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=512)
    img = compute_shadow(st, cam, IntegratorOptions(method="rk45", rtol=1e-4, atol=1e-6), TerminationOptions(1e-6, 1000.0), backend="numba")
    y0 = initial_states(cam, st)
    lz = np.abs(y0[:, 7] / -y0[:, 4])  # L_z / E
    s = img.state.ravel()
    failed = (s != TS.ESCAPED) & (s != TS.CAPTURED)
    print(f"a={spin}: failed {int(failed.sum())}; |L_z/E| of failed: max {lz[failed].max():.4f}, median {np.median(lz[failed]):.4f}; "
          f"median over image {np.median(lz):.3f}", flush=True)
