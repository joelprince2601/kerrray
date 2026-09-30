"""How large is the finite-observer-distance effect, and does the correction remove it?

The camera sits at r_o, not at infinity. The paper compares traced shadows
with Bardeen's asymptotic curve after scaling image coordinates by
k = 1/sqrt(1 - 2M/r_o), which is exact for a = 0 and leading order for Kerr.
This script traces one image per (spin, r_o) and counts the pixels whose
classification disagrees with the analytic curve (i) without any correction
and (ii) with the correction. It also reports the expected number of
disagreeing pixels for a coherent radial shift delta of the edge,
E[flips] ~ L delta / h^2 (L the curve length, h the pixel size), which turns
an observed count of zero into an upper bound on the residual shift.

Output: paper/data/finite_distance.json.
Usage: ./.venv/Scripts/python.exe scripts/study_finite_distance.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.raytracing.boundary import analytic_boundary, finite_distance_scale
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import compute_shadow
from kerrray.utils.manifest import collect_environment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "finite_distance.json"


def main() -> None:
    rows = []
    t0 = time.perf_counter()
    res, incl, fov, rtol = 512, 60.0, 8.0, 1e-6
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for spin in (0.0, 0.9, 0.99):
            for r_o in (1.0e3, 1.0e4, 1.0e5):
                st = Spacetime(1.0, spin)
                cam = Camera(radius=r_o, inclination_deg=incl, fov=fov, resolution=res)
                integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, max_steps=400_000, lambda_max=max(1e4, 4 * r_o))
                img = compute_shadow(st, cam, integ, TerminationOptions(horizon_epsilon=1e-6, escape_radius=r_o), backend="numba")
                traced = img.state != int(TerminationState.ESCAPED)
                ca, cb = analytic_boundary(st, incl, 4000)
                path = MplPath(np.column_stack([ca, cb]))
                length = float(np.sum(np.hypot(np.diff(ca), np.diff(cb))))
                k = finite_distance_scale(st, r_o)
                counts = {}
                for label, scale in (("uncorrected", 1.0), ("corrected", k)):
                    pts = np.column_stack([(img.alpha * scale).ravel(), (img.beta * scale).ravel()])
                    oracle = path.contains_points(pts).reshape(img.alpha.shape)
                    counts[label] = int((traced ^ oracle).sum())
                h = img.pixel_size
                shift_uncorr = (k - 1.0) * 5.0  # radial shift removed by the correction, for a ~5 M shadow
                row = {
                    "spin": spin, "observer_radius": r_o, "resolution": res, "rtol": rtol, "pixel_size": h,
                    "k_minus_1": k - 1.0, "curve_length_M": length,
                    "disagree_uncorrected": counts["uncorrected"], "disagree_corrected": counts["corrected"],
                    "expected_flips_for_uncorrected_shift": length * shift_uncorr / h**2,
                    # zero observed flips: P(0) = exp(-L delta / h^2) < 0.05  =>  delta < 3 h^2 / L
                    "residual_shift_bound_95_M": (3.0 * h**2 / length) if counts["corrected"] == 0 else None,
                    "failed": int(np.sum(traced & (img.state != int(TerminationState.CAPTURED)))),
                }
                rows.append(row)
                print(f"a={spin} r_o={r_o:.0e}: k-1={k - 1:.2e} disagree uncorrected={counts['uncorrected']} corrected={counts['corrected']} "
                      f"bound={row['residual_shift_bound_95_M']}", flush=True)
    env = collect_environment()
    OUT.write_text(json.dumps({"rows": rows, "git_commit": env.git_commit, "git_dirty": env.git_dirty,
                               "runtime_s": time.perf_counter() - t0}, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
