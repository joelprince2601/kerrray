"""Separate integration error from pixel-sampling error in Kerr shadow images.

For every traced image we build an *oracle mask*: a pixel is in the shadow
exactly when its centre (after the finite-observer-distance correction) lies
inside the analytic Bardeen curve. The oracle is what a perfect integrator
with the same camera, pixels and correction would produce. Comparing the
traced mask with the oracle pixel by pixel counts the photons the integrator
misclassified and measures how far they sit from the true edge; running the
same edge extraction on the oracle mask measures the error that comes from
pixel sampling and edge extraction alone.

Groups (all RK45, fov +-8 M):
  core        spins 0, 0.5, 0.9, 0.99; i = 60 deg; r_o = 1000 M; N = 128, 512;
              rtol 1e-4, 1e-6, 1e-10
  distance    spins 0.9, 0.99; i = 60 deg; N = 512; rtol 1e-6;
              r_o = 1000, 10000, 100000 M
  inclination spins 0.9, 0.99; i = 17, 85 deg; r_o = 1000 M; N = 128, 256, 512;
              rtol 1e-6, 1e-10
  precision   spins 0.9, 0.99; i = 60 deg; N = 256; rtol 1e-6; float32 and
              float64, both with horizon margin 1e-3 M (float32 cannot resolve
              1e-6 M next to r_+ ~ 1)

Outputs: paper/data/oracle.json, paper/figures/failure_map.png,
paper/figures/finite_distance.png.

Usage:
    ./.venv/Scripts/python.exe scripts/study_oracle.py [--groups core,distance,...] [--plot-only]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

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
DATA, FIGS = ROOT / "paper" / "data", ROOT / "paper" / "figures"
ESC, CAP = int(TerminationState.ESCAPED), int(TerminationState.CAPTURED)


def edge_errors(alpha, beta, mask, curve, k, pixel):
    centre = mask_centroid(alpha, beta, mask)
    ang, rad = extract_boundary_from_mask(alpha, beta, mask, n_angles=720, centre=centre)
    e = boundary_error((ang, rad), curve, centre, pixel_size=pixel, scale=k)
    return {"rms_M": e["rms"], "max_M": e["max"], "mean_M": e["mean"], "rms_px": e["rms_px"], "max_px": e["max_px"]}


def run_one(spin, incl, res, rtol, r_obs, dtype="float64", horizon_eps=1e-6, fov=8.0, keep_masks=False):
    st = Spacetime(1.0, spin)
    cam = Camera(radius=r_obs, inclination_deg=incl, fov=fov, resolution=res)
    # Photons travel in from r_o and back out to r_o: the affine budget must exceed ~2 r_o.
    integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, dtype=dtype, max_steps=400_000,
                              lambda_max=max(1.0e4, 4.0 * r_obs))
    term = TerminationOptions(horizon_epsilon=horizon_eps, escape_radius=r_obs)
    t0 = time.perf_counter()
    img = compute_shadow(st, cam, integ, term, backend="numba")
    runtime = time.perf_counter() - t0
    alpha, beta = img.alpha, img.beta
    k = finite_distance_scale(st, r_obs)
    # 4000 vertices: chord sagitta ~ (L/n)^2 / (8 R) ~ 2e-6 M, far below a pixel
    ca, cb = analytic_boundary(st, incl, 4000)
    curve_path = MplPath(np.column_stack([ca, cb]))
    pts = np.column_stack([(alpha * k).ravel(), (beta * k).ravel()])
    oracle = curve_path.contains_points(pts).reshape(alpha.shape)
    traced = img.state != ESC
    failed = traced & (img.state != CAP)
    diff = traced ^ oracle
    # distance (in pixels) of each disagreeing pixel centre from the analytic curve
    dense = np.column_stack([ca, cb])
    def dist_px(mask):
        if not mask.any():
            return np.array([])
        p = pts[mask.ravel()]
        d = np.concatenate([np.sqrt(((c[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(axis=1)
                            for c in np.array_split(p, max(1, len(p) // 2000))])
        return d / (img.pixel_size * k)
    d_diff = dist_px(diff)
    row = {
        "spin": spin, "inclination_deg": incl, "resolution": res, "rtol": rtol, "observer_radius": r_obs,
        "dtype": dtype, "horizon_epsilon": horizon_eps, "fov": fov, "pixel_size": img.pixel_size, "runtime_s": runtime,
        "disagree": int(diff.sum()), "disagree_traced_only": int((traced & ~oracle).sum()), "disagree_oracle_only": int((oracle & ~traced).sum()),
        "disagree_max_dist_px": float(d_diff.max()) if d_diff.size else 0.0,
        "failed": int(failed.sum()), "states": {TerminationState(int(v)).name: int(c) for v, c in zip(*np.unique(img.state, return_counts=True))}, "failed_inside_oracle": int((failed & oracle).sum()), "failed_outside_oracle": int((failed & ~oracle).sum()),
        "captured": int((img.state == CAP).sum()), "shadow_pixels": int(traced.sum()), "oracle_pixels": int(oracle.sum()),
        "traced_edge": edge_errors(alpha, beta, traced, (ca, cb), k, img.pixel_size),
        "oracle_edge": edge_errors(alpha, beta, oracle, (ca, cb), k, img.pixel_size),
    }
    masks = {"traced": traced, "oracle": oracle, "failed": failed, "alpha": alpha, "beta": beta} if keep_masks else None
    return row, masks


def groups_spec():
    core = [(s, 60.0, n, t, 1000.0, "float64", 1e-6) for s in (0.0, 0.5, 0.9, 0.99) for t in (1e-4, 1e-6, 1e-10) for n in (128, 512)]
    distance = [(s, 60.0, 512, 1e-6, r, "float64", 1e-6) for s in (0.9, 0.99) for r in (1000.0, 1.0e4, 1.0e5)]
    inclination = [(s, i, n, t, 1000.0, "float64", 1e-6) for s in (0.9, 0.99) for i in (17.0, 85.0) for t in (1e-6, 1e-10) for n in (128, 256, 512)]
    precision = [(s, 60.0, 256, 1e-6, 1000.0, d, 1e-3) for s in (0.9, 0.99) for d in ("float32", "float64")]
    return {"core": core, "distance": distance, "inclination": inclination, "precision": precision}


def plot(out):
    rows = out["rows"]
    # finite-distance figure: signed mean and rms edge error vs r_o
    dist = [r for r in rows if r["group"] == "distance"]
    if dist:
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for s, c in ((0.9, "#D95319"), (0.99, "#7E2F8E")):
            sub = sorted((r for r in dist if r["spin"] == s), key=lambda r: r["observer_radius"])
            x = [r["observer_radius"] for r in sub]
            ax.semilogx(x, [r["traced_edge"]["rms_M"] for r in sub], "o-", color=c, label=f"rms, a* = {s}")
            ax.semilogx(x, [r["traced_edge"]["mean_M"] for r in sub], "s--", color=c, label=f"signed mean, a* = {s}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(xlabel="observer radius r_o [M]", ylabel="edge error after correction [M]", title="Finite observer distance (N = 512, rtol 1e-6)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "finite_distance.png", dpi=150)
        plt.close(fig)


def failure_map(spin=0.9, res=512):
    """Traced minus oracle masks at rtol 1e-4 and 1e-6 (i = 60 deg, r_o = 1000 M)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, rtol in zip(axes, (1e-4, 1e-6)):
        row, m = run_one(spin, 60.0, res, rtol, 1000.0, keep_masks=True)
        a, b = m["alpha"], m["beta"]
        ext = [a.min(), a.max(), b.min(), b.max()]
        base = np.where(m["oracle"], 0.82, 1.0)
        ax.imshow(base, cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper", interpolation="nearest")
        yy, xx = np.nonzero(m["traced"] & ~m["oracle"])
        ax.scatter(a[yy, xx], b[yy, xx], s=6, color="#D95319", label=f"traced shadow, outside curve ({len(xx)})")
        yy, xx = np.nonzero(m["oracle"] & ~m["traced"])
        ax.scatter(a[yy, xx], b[yy, xx], s=6, color="#0072BD", label=f"escaped, inside curve ({len(xx)})")
        yy, xx = np.nonzero(m["failed"])
        ax.scatter(a[yy, xx], b[yy, xx], s=4, color="#A2142F", marker="x", label=f"failed (out of domain) ({len(xx)})")
        ca, cb = analytic_boundary(Spacetime(1.0, spin), 60.0, 4000)
        k = finite_distance_scale(Spacetime(1.0, spin), 1000.0)
        ax.plot(ca / k, cb / k, color="k", lw=0.6)
        ax.set(xlabel="alpha [M]", ylabel="beta [M]", title=f"a* = {spin}, N = {res}, rtol = {rtol:.0e}")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "failure_map.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="core,distance,inclination,precision")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    path = DATA / "oracle.json"
    if args.plot_only:
        plot(json.loads(path.read_text(encoding="utf-8")))
        return
    spec = groups_spec()
    env = collect_environment()
    rows, t_all = [], time.perf_counter()

    def save() -> None:
        out = {"rows": rows, "git_commit": env.git_commit, "git_dirty": env.git_dirty, "hardware": env.hardware(),
               "python": env.python_version, "packages": env.package_versions, "runtime_s": time.perf_counter() - t_all}
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in args.groups.split(","):
            for (s, i, n, t, r, d, eps) in spec[g]:
                row, _ = run_one(s, i, n, t, r, d, eps)
                row["group"] = g
                rows.append(row)
                save()
                te, oe = row["traced_edge"], row["oracle_edge"]
                print(f"[{g}] a={s} i={i:.0f} N={n} rtol={t:.0e} r_o={r:.0e} {d}: disagree={row['disagree']} "
                      f"(max {row['disagree_max_dist_px']:.2f} px) failed={row['failed']} (outside {row['failed_outside_oracle']}) "
                      f"edge rms traced {te['rms_px']:.4f} px / oracle {oe['rms_px']:.4f} px, mean {te['mean_M']:+.2e} M  {row['runtime_s']:.1f}s", flush=True)
        failure_map()
    plot(save())
    print(f"done in {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
