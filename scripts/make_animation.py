"""Render the photon-beam animation to an animated GIF.

A flat wavefront of photons is fired at a non-spinning and a spinning black
hole side by side. Every path is integrated by the KerrRay engine
(kerrray.web.api.photon_beam) and resampled in the coordinate time of a
distant observer, so both panels share one clock.

Usage:
    ./.venv/Scripts/python.exe scripts/make_animation.py [--spin 0.95] [--frames 120] [--out results/animation/photon_beam.gif]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from kerrray.web.api import dispatch

BG = "#07090d"
ESC, CAP, CRIT = "#4cc9f0", "#ff8a3d", "#ffe6a8"
TRAIL = 40


def beam(spin: float, n: int, frames: int) -> dict:
    return dispatch("photon_beam", {"spin": spin, "n": n, "frames": frames, "b_max": 10.0, "x0": 30.0})


def arrays(d: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[np.nan if v is None else v for v in row] for row in d["x"]], dtype=float)
    y = np.array([[np.nan if v is None else v for v in row] for row in d["y"]], dtype=float)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spin", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="results/animation/photon_beam.gif")
    args = ap.parse_args()

    t0 = time.perf_counter()
    runs = [beam(0.0, args.n, args.frames), beam(args.spin, args.n, args.frames)]
    # One shared clock: resample the shorter run onto the longer run's time grid.
    t_max = max(r["t"][-1] for r in runs)
    grid = np.linspace(0.0, t_max, args.frames)
    panels = []
    for r in runs:
        x, y = arrays(r)
        t = np.array(r["t"])
        xi = np.array([np.interp(grid, t, row, right=np.nan) for row in x])
        yi = np.array([np.interp(grid, t, row, right=np.nan) for row in y])
        panels.append((r, xi, yi))
    print(f"engine: {sum(r['runtime_s'] for r in runs):.2f} s for {sum(r['n_photons'] for r in runs)} photons")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), dpi=90, facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.04)
    title = fig.text(0.5, 0.94, "", color="#e8ecf2", ha="center", fontsize=13, family="monospace")
    artists = []
    for ax, (r, xi, yi) in zip(axes, panels):
        a = r["spin"]
        rho = lambda rr: math.sqrt(rr * rr + a * a)  # noqa: E731 - plotting embedding radius
        ax.set_facecolor(BG)
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_aspect("equal")
        ax.axis("off")
        for rr in (5, 10, 15):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="#96a5be", alpha=0.08, lw=0.8))
        ax.add_patch(plt.Circle((0, 0), rho(2.0), fill=False, color="#b388ff", ls="--", lw=0.9, alpha=0.6))
        for ro in (r["photon_orbit_prograde"], r["photon_orbit_retrograde"]):
            ax.add_patch(plt.Circle((0, 0), rho(ro), fill=False, color="#7bd88f", ls=":", lw=0.9, alpha=0.55))
        ax.add_patch(plt.Circle((0, 0), rho(r["r_plus"]) * 1.6, color=CAP, alpha=0.10, lw=0))
        ax.add_patch(plt.Circle((0, 0), rho(r["r_plus"]), color="black", ec="#ffb27a", lw=1.0, zorder=5))
        label = "Schwarzschild, a* = 0" if a == 0 else f"Kerr, a* = {a}"
        ax.text(0, -15.2, f"{label}   ·   captured {r['captured']} / escaped {r['escaped']}", color="#8d97a8", ha="center", fontsize=9)
        colors = [CRIT if nc else (CAP if s == "CAPTURED" else ESC) for nc, s in zip(r["near_critical"], r["state"])]
        lc = LineCollection([], linewidths=1.0, zorder=3)
        ax.add_collection(lc)
        dots = ax.scatter([], [], s=9, zorder=6)
        artists.append((lc, dots, xi, yi, colors))

    def update(k: int):
        out = []
        for lc, dots, xi, yi, colors in artists:
            segs, cols = [], []
            k0 = max(0, k - TRAIL)
            for i in range(xi.shape[0]):
                pts = np.column_stack([xi[i, k0 : k + 1], yi[i, k0 : k + 1]])
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) > 1:
                    segs.append(pts)
                    cols.append(colors[i])
            lc.set_segments(segs)
            lc.set_colors([matplotlib.colors.to_rgba(c, 0.55) for c in cols])
            live = np.isfinite(xi[:, k]) & np.isfinite(yi[:, k])
            dots.set_offsets(np.column_stack([xi[live, k], yi[live, k]]) if live.any() else np.empty((0, 2)))
            dots.set_color([colors[i] for i in np.flatnonzero(live)])
            out += [lc, dots]
        title.set_text(f"t = {grid[k]:6.1f} M   (coordinate time of a distant observer)")
        return out

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=args.frames, blit=False).save(out, writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": BG})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
