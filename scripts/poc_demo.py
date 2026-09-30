"""KerrRay proof-of-concept demo.

Runs a small set of real computations with the tested core engine and saves
figures plus a JSON summary to results/poc/. Every printed number is computed
here; reference values (3 M, 3 sqrt(3) M) are used only to report errors.

Usage:  ./.venv/Scripts/python.exe scripts/poc_demo.py [--resolution 48]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kerrray.geodesics.initial_conditions import equatorial_photon, tangential_photon
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate
from kerrray.geodesics.integrators_batch import integrate_batch
from kerrray.geometry.horizons import ergosphere_radius, horizon_radii
from kerrray.geometry.metric import kerr, schwarzschild
from kerrray.photons.classification import TerminationState
from kerrray.photons.orbits import critical_impact_parameters, isco_radius, shadow_curve
from kerrray.photons.trajectories import closest_approach, to_cartesian
from kerrray.raytracing.camera import Camera, initial_states

OUT = Path(__file__).resolve().parents[1] / "results" / "poc"
REF_R_PH = 3.0  # Schwarzschild photon sphere, comparison reference only
REF_B_C = 3.0 * math.sqrt(3.0)  # Schwarzschild critical impact parameter, comparison reference only


def outcome(st, y0, integ, term) -> TerminationState:
    return integrate(st, y0, integ, term, record=False).state


def bisect(is_captured, lo, hi, tol):
    """Bisection on a captured(x) predicate: captured at lo, escapes at hi."""
    n = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_captured(mid):
            lo = mid
        else:
            hi = mid
        n += 1
    return 0.5 * (lo + hi), hi - lo, n


def schwarzschild_validation(integ, term):
    st = schwarzschild(1.0)
    t0 = time.perf_counter()
    r_ph, r_w, n1 = bisect(
        lambda r: outcome(st, tangential_photon(st, r), integ, term) == TerminationState.CAPTURED,
        2.5, 3.5, 1e-7,
    )
    b_c, b_w, n2 = bisect(
        lambda b: outcome(st, equatorial_photon(st, 1000.0, b), integ, term) == TerminationState.CAPTURED,
        4.0, 6.0, 1e-7,
    )
    return {
        "photon_sphere_numerical": r_ph, "photon_sphere_bracket": r_w,
        "photon_sphere_reference": REF_R_PH, "photon_sphere_rel_error": abs(r_ph - REF_R_PH) / REF_R_PH,
        "critical_b_numerical": b_c, "critical_b_bracket": b_w,
        "critical_b_reference": REF_B_C, "critical_b_rel_error": abs(b_c - REF_B_C) / REF_B_C,
        "bisection_steps": n1 + n2, "runtime_s": time.perf_counter() - t0,
    }


def kerr_numbers(spin):
    st = kerr(1.0, spin)
    rp, rm = horizon_radii(st)
    bp, br = critical_impact_parameters(st)
    return {
        "spin": spin, "r_plus": rp, "r_minus": rm,
        "ergosphere_equator": float(ergosphere_radius(st, math.pi / 2)),
        "isco_prograde": isco_radius(st, prograde=True),
        "isco_retrograde": isco_radius(st, prograde=False),
        "b_crit_prograde": bp, "b_crit_retrograde": br,
    }


def trajectories_figure(integ, term, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    rows = []
    for ax, spin in zip(axes, (0.0, 0.9)):
        st = kerr(1.0, spin) if spin else schwarzschild(1.0)
        rp, _ = horizon_radii(st)
        bp, br = critical_impact_parameters(st)
        rays = [(bp * 0.97, True, "tab:red"), (bp * 1.003, True, "tab:orange"),
                (bp * 1.3, True, "tab:green"), (br * 1.003, False, "tab:blue")]
        for b, pro, col in rays:
            tr = integrate(st, equatorial_photon(st, 40.0, b, prograde=pro), integ,
                           TerminationOptions(horizon_epsilon=term.horizon_epsilon, escape_radius=60.0))
            x, y, _ = to_cartesian(tr)
            kind = "pro" if pro else "retro"
            ax.plot(x, y, color=col, lw=1.2, label=f"b={b:.3f} {kind} -> {tr.state.name}")
            rows.append({"spin": spin, "b": b, "prograde": pro, "state": tr.state.name,
                         "closest_approach": closest_approach(tr), "steps": tr.n_steps,
                         "runtime_s": tr.runtime_s})
        ax.add_patch(plt.Circle((0, 0), rp, color="black"))
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal")
        ax.set_title(f"Equatorial photon orbits, a* = {spin} (black disk = horizon)")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return rows


def shadow_figure(spin, incl, res, integ, term, path):
    st = kerr(1.0, spin) if spin else schwarzschild(1.0)
    cam = Camera(radius=1000.0, inclination_deg=incl, fov=8.0, resolution=res)
    Y0 = initial_states(cam, st)
    t0 = time.perf_counter()
    result = integrate_batch(st, Y0, integ, term)
    runtime = time.perf_counter() - t0
    state = result.state.reshape(res, res)
    captured = state == TerminationState.CAPTURED
    counts = {s.name: int(np.sum(result.state == s)) for s in TerminationState if np.sum(result.state == s)}
    a_c, b_c = shadow_curve(st, incl)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.where(captured, 0.0, 1.0), cmap="gray", extent=(-8, 8, -8, 8), vmin=0, vmax=1)
    ax.plot(a_c, b_c, "r-", lw=1.5, label="analytic shadow edge (Bardeen curve)")
    ax.set_xlabel("alpha [M]")
    ax.set_ylabel("beta [M]")
    ax.set_title(f"Numerical shadow: a*={spin}, i={incl} deg, {res}x{res} rays\n"
                 "black = CAPTURED rays, white = ESCAPED rays")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    pix = 16.0 / res
    area_num = captured.sum() * pix * pix
    area_ref = 0.5 * abs(np.sum(a_c * np.roll(b_c, -1) - np.roll(a_c, -1) * b_c))
    return {"spin": spin, "inclination_deg": incl, "resolution": res, "n_rays": res * res,
            "counts": counts, "runtime_s": runtime, "rays_per_s": res * res / runtime,
            "max_null_error": float(np.max(result.max_null_error)),
            "max_energy_drift": float(np.max(result.max_energy_drift)),
            "shadow_area_numerical_M2": float(area_num), "shadow_area_analytic_M2": float(area_ref),
            "area_rel_diff": float(abs(area_num - area_ref) / area_ref)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)
    summary = {}

    print("[1/4] Schwarzschild validation (photon sphere and critical impact parameter)...")
    v = schwarzschild_validation(integ, term)
    summary["schwarzschild_validation"] = v
    print(f"  Analytical photon sphere : {v['photon_sphere_reference']:.6f} M")
    print(f"  Numerical photon sphere  : {v['photon_sphere_numerical']:.6f} M  (bracket {v['photon_sphere_bracket']:.1e})")
    print(f"  Relative error           : {v['photon_sphere_rel_error']:.2e}")
    print(f"  Analytical b_c           : {v['critical_b_reference']:.6f} M")
    print(f"  Numerical b_c            : {v['critical_b_numerical']:.6f} M  (bracket {v['critical_b_bracket']:.1e})")
    print(f"  Relative error           : {v['critical_b_rel_error']:.2e}   runtime {v['runtime_s']:.1f} s")

    print("[2/4] Kerr black-hole properties for several spins...")
    summary["kerr_properties"] = [kerr_numbers(s) for s in (0.0, 0.5, 0.9, 0.99)]
    for k in summary["kerr_properties"]:
        print(f"  a*={k['spin']:<5} r+={k['r_plus']:.4f}  ISCO pro/retro={k['isco_prograde']:.4f}/"
              f"{k['isco_retrograde']:.4f}  b_c pro/retro={k['b_crit_prograde']:.4f}/{k['b_crit_retrograde']:.4f}")

    print("[3/4] Photon trajectories (Schwarzschild vs Kerr a*=0.9)...")
    summary["trajectories"] = trajectories_figure(integ, term, OUT / "trajectories.png")
    for r in summary["trajectories"]:
        kind = "pro  " if r["prograde"] else "retro"
        print(f"  a*={r['spin']} b={r['b']:.3f} {kind} -> {r['state']:<8} r_min={r['closest_approach']:.3f}")

    print(f"[4/4] Numerical shadows at {args.resolution}x{args.resolution} rays (slowest step)...")
    summary["shadows"] = []
    shadow_integ = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10)
    for spin in (0.0, 0.9):
        s = shadow_figure(spin, 60.0, args.resolution, shadow_integ, term, OUT / f"shadow_a{spin}_i60.png")
        summary["shadows"].append(s)
        print(f"  a*={spin}: {s['counts']}  area num/analytic = {s['shadow_area_numerical_M2']:.2f}/"
              f"{s['shadow_area_analytic_M2']:.2f} M^2 ({100 * s['area_rel_diff']:.1f}% diff)  "
              f"max null err {s['max_null_error']:.1e}  {s['runtime_s']:.1f} s")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"Saved figures and summary.json to {OUT}")


if __name__ == "__main__":
    main()
