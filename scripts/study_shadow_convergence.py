"""Shadow reconstruction accuracy versus ray density and integrator tolerance.

The research question of the project (docs/scientific_background.md): how do
integration tolerance and ray density control the accuracy of a numerically
reconstructed Kerr shadow on a CPU? For each spin, resolution and tolerance
this script traces a full image with the compiled Numba backend (identical
classifications to the NumPy reference), extracts the sub-pixel shadow
boundary and compares it with the analytic Bardeen curve after the
finite-observer-distance correction (kerrray.raytracing.boundary).

Outputs: paper/data/shadow_convergence.json and figures in paper/figures/.

Usage:
    ./.venv/Scripts/python.exe scripts/study_shadow_convergence.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.raytracing.boundary import (
    analytic_boundary,
    boundary_error,
    extract_boundary_from_mask,
    finite_distance_scale,
    mask_centroid,
)
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import compute_shadow
from kerrray.utils.manifest import collect_environment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGS = ROOT / "paper" / "figures"


def shoelace(x: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def run_one(spin: float, incl: float, res: int, rtol: float, fov: float, r_obs: float, backend: str) -> dict:
    st = Spacetime(mass=1.0, spin=spin)
    cam = Camera(radius=r_obs, inclination_deg=incl, fov=fov, resolution=res)
    integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=r_obs)
    t0 = time.perf_counter()
    img = compute_shadow(st, cam, integ, term, backend=backend)
    runtime = time.perf_counter() - t0
    # The shadow is the set of pixels whose photon did not escape. Photons that
    # end in a failure state (for example OUT_OF_DOMAIN near the polar axis at
    # loose tolerance) are therefore counted inside it, and reported separately.
    shadow = img.state != int(TerminationState.ESCAPED)
    failed_mask = shadow & (img.state != int(TerminationState.CAPTURED))
    centre = mask_centroid(img.alpha, img.beta, shadow)
    angles, radii = extract_boundary_from_mask(img.alpha, img.beta, shadow, n_angles=720, centre=centre)
    alpha_c, beta_c = analytic_boundary(st, incl, 2000)
    k = finite_distance_scale(st, r_obs)
    err = boundary_error((angles, radii), (alpha_c, beta_c), centre, pixel_size=img.pixel_size, scale=k)
    area_num = float(shadow.sum()) * img.pixel_size**2 * k**2
    area_ana = shoelace(np.asarray(alpha_c), np.asarray(beta_c))
    failed = int(failed_mask.sum())
    failed_alpha = np.abs(img.alpha[failed_mask]) if failed else np.array([np.nan])
    return {
        "spin": spin, "inclination_deg": incl, "resolution": res, "rtol": rtol, "fov": fov, "observer_radius": r_obs,
        "pixel_size": img.pixel_size, "runtime_s": runtime, "rays": res * res, "rays_per_s": res * res / runtime,
        "rms_M": err["rms"], "max_M": err["max"], "mean_M": err["mean"], "rms_px": err["rms_px"], "max_px": err["max_px"],
        "area_numeric": area_num, "area_analytic": area_ana, "area_rel_err": abs(area_num - area_ana) / area_ana,
        "centroid_shift_M": err.get("centroid_shift_norm"), "failed_rays": failed,
        "failed_max_abs_alpha_M": float(np.max(failed_alpha)),
        "captured": int(img.captured.sum()),
    }


def fit_power(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of log10 y against log10 x."""
    p = np.polyfit(np.log10(x), np.log10(y), 1)
    return float(p[0]), float(p[1])


def plot(out: dict) -> None:
    """Figures from the saved results (also available through --plot-only)."""
    rows = out["rows"]
    spins = sorted({r["spin"] for r in rows})
    tolerances = sorted({r["rtol"] for r in rows}, reverse=True)
    resolutions = sorted({r["resolution"] for r in rows})
    colors = {0.0: "#0072BD", 0.5: "#77AC30", 0.9: "#D95319", 0.99: "#7E2F8E"}
    styles = {1e-4: (0, (1, 3)), 1e-5: "-.", 1e-6: ":", 1e-8: "--", 1e-10: "-"}
    tight = min(tolerances)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for spin in spins:
        for rtol in tolerances:
            sub = sorted((r for r in rows if r["spin"] == spin and r["rtol"] == rtol), key=lambda r: r["resolution"])
            kw = dict(linestyle=styles.get(rtol, "-"), marker="o", ms=3, color=colors.get(spin, "k"))
            axes[0].loglog([r["resolution"] for r in sub], [r["rms_M"] for r in sub], label=f"a* = {spin}" if rtol == tight else None, **kw)
            axes[1].loglog([r["resolution"] for r in sub], [max(r["area_rel_err"], 1e-7) for r in sub], **kw)
    n = np.array(resolutions, float)
    first = min((r for r in rows if r["resolution"] == resolutions[0]), key=lambda r: r["rms_M"])
    axes[0].loglog(n, first["rms_M"] * (n / n[0]) ** -1, color="0.6", lw=1, label="slope -1 (guide)")
    axes[0].set(xlabel="resolution N (pixels per side)", ylabel="rms boundary error [M]", title="Shadow edge vs exact curve")
    axes[1].set(xlabel="resolution N (pixels per side)", ylabel="relative area error", title="Shadow area vs exact area")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.text(0.5, 0.005, "line style by rtol: sparse dots 1e-4, dash-dot 1e-5, dotted 1e-6, dashed 1e-8, solid 1e-10   "
             f"(i = {out['inclination_deg']:.0f} deg, r_o = {out['observer_radius']:.0f} M, RK45)", ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "shadow_convergence.png", dpi=150)
    plt.close(fig)

    # Figure 2: error at the finest resolution against tolerance
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    finest = max(resolutions)
    for spin in spins:
        sub = sorted((r for r in rows if r["spin"] == spin and r["resolution"] == finest), key=lambda r: r["rtol"])
        ax.loglog([r["rtol"] for r in sub], [r["rms_M"] for r in sub], marker="o", color=colors.get(spin, "k"), label=f"a* = {spin}")
    pix = next(r["pixel_size"] for r in rows if r["resolution"] == finest)
    ax.axhline(pix, color="0.6", lw=1, ls="--", label=f"one pixel ({pix:.3f} M)")
    ax.set(xlabel="relative tolerance rtol", ylabel="rms boundary error [M]", title=f"Tolerance at N = {finest}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "shadow_tolerance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid for a smoke test")
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from paper/data/shadow_convergence.json")
    args = ap.parse_args()
    if args.plot_only:
        FIGS.mkdir(parents=True, exist_ok=True)
        plot(json.loads((DATA / "shadow_convergence.json").read_text(encoding="utf-8")))
        return
    spins = [0.0, 0.5, 0.9, 0.99]
    resolutions = [64, 128, 256, 512] if not args.quick else [32, 64]
    tolerances = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10] if not args.quick else [1e-6, 1e-8]
    incl, fov, r_obs = 60.0, 8.0, 1000.0
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    rows = []
    t_all = time.perf_counter()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for spin in spins:
            for rtol in tolerances:
                for res in resolutions:
                    row = run_one(spin, incl, res, rtol, fov, r_obs, args.backend)
                    rows.append(row)
                    print(f"a={spin:<5} rtol={rtol:.0e} N={res:<4} rms={row['rms_M']:.3e} M ({row['rms_px']:.3f} px) "
                          f"max={row['max_M']:.3e} M  area err={row['area_rel_err']:.2e}  failed={row['failed_rays']}  {row['runtime_s']:.1f} s", flush=True)
    fits = {}
    for spin in spins:
        for rtol in tolerances:
            sub = [r for r in rows if r["spin"] == spin and r["rtol"] == rtol]
            fits[f"{spin}|{rtol}"] = {
                "rms_slope_vs_N": fit_power([r["resolution"] for r in sub], [r["rms_M"] for r in sub])[0],
                "max_slope_vs_N": fit_power([r["resolution"] for r in sub], [r["max_M"] for r in sub])[0],
                "area_slope_vs_N": fit_power([r["resolution"] for r in sub], [max(r["area_rel_err"], 1e-12) for r in sub])[0],
            }
    out = {
        "description": "Shadow boundary error vs resolution and rtol (RK45), corrected for finite observer distance",
        "inclination_deg": incl, "fov": fov, "observer_radius": r_obs, "backend": args.backend,
        "git_commit": env.git_commit, "git_dirty": env.git_dirty, "hardware": env.hardware(), "python": env.python_version,
        "total_runtime_s": time.perf_counter() - t_all, "rows": rows, "fits": fits,
    }
    (DATA / "shadow_convergence.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    plot(out)
    print(f"done in {time.perf_counter() - t_all:.0f} s; wrote {DATA / 'shadow_convergence.json'} and figures in {FIGS}")


if __name__ == "__main__":
    main()
