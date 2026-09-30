"""Composed report figures: shadow, trajectories, disk image, lensing curves
(PROJECT.md section 36; rendering role).

Built on the primitives and style constants of :mod:`kerrray.reporting.plots`
(headless Agg backend, fixed-order categorical colours, one axis per panel,
legend whenever two or more series share an axis). Image maps use a single-hue
sequential colour map for magnitudes (intensity) and a two-hue diverging map
with a neutral midpoint at ``g = 1`` for the redshift factor. Every function
saves the figure and returns the saved :class:`~pathlib.Path`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from kerrray.geometry import Spacetime, bl_to_cartesian, ergosphere_radius, outer_horizon
from kerrray.photons.trajectories import Trajectory, to_cartesian
from kerrray.reporting.plots import (
    DEFAULT_DPI,
    ERGOSPHERE_COLOR,
    HORIZON_COLOR,
    plot_lines,
    plot_shadow,
    series_style,
)
from kerrray.reporting.plots import _save, _style_axes, plt  # shared helpers of the plots module

__all__ = [
    "INTENSITY_CMAP",
    "LOG_STRETCH_FLOOR",
    "REDSHIFT_CMAP",
    "disk_figure",
    "lensing_figure",
    "shadow_figure",
    "trajectory_figure",
]

INTENSITY_CMAP = "Blues"
"""Single-hue sequential map (light = 0, dark = maximum) for intensity images."""

REDSHIFT_CMAP = "RdBu"
"""Diverging map for ``g``: blue (blue-shift, ``g > 1``), red (red-shift, ``g < 1``), white at ``g = 1``.

Matplotlib's ``RdBu`` runs from red at the low end to blue at the high end, so
with the limits ``1 -+ span`` of :func:`disk_figure` it gives the physical
colour convention (``RdBu_r`` would invert it)."""

LOG_STRETCH_FLOOR = 1e-4
"""Log stretch shows ``max(I, floor * I_max)``; pixels below the floor render as the floor."""


def shadow_figure(
    path: Path | str,
    img: Any,
    analytic_curve: tuple[ArrayLike, ArrayLike] | None = None,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Captured-pixel mask of a shadow image with an optional analytic boundary.

    ``img`` is any object with ``alpha``, ``beta`` (pixel-centre coordinates,
    ``(H, W)``) and ``captured`` (boolean ``(H, W)``) attributes, such as
    ``kerrray.raytracing.shadow.ShadowImage``. Thin wrapper over
    :func:`kerrray.reporting.plots.plot_shadow`.
    """
    return plot_shadow(path, img.alpha, img.beta, img.captured, analytic_curve, title=title, dpi=dpi)


def _horizon_and_ergosphere(st: Spacetime, n: int = 361) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Curves of the horizon and the ergosphere in the plotting embedding (top and side views)."""
    theta = np.linspace(0.0, np.pi, n)
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    r_plus = outer_horizon(st)
    r_e = ergosphere_radius(st, theta)
    hx, hy, _ = bl_to_cartesian(st, r_plus, 0.5 * np.pi, phi)
    ex, ey, _ = bl_to_cartesian(st, float(ergosphere_radius(st, 0.5 * np.pi)), 0.5 * np.pi, phi)
    sx, _, sz = bl_to_cartesian(st, r_plus, theta, 0.0)
    gx, _, gz = bl_to_cartesian(st, r_e, theta, 0.0)
    side_h = (np.concatenate([sx, -sx[::-1]]), np.concatenate([sz, sz[::-1]]))
    side_e = (np.concatenate([gx, -gx[::-1]]), np.concatenate([gz, gz[::-1]]))
    return {"top_horizon": (hx, hy), "top_ergosphere": (ex, ey), "side_horizon": side_h, "side_ergosphere": side_e}


def trajectory_figure(
    path: Path | str,
    trajectories: Sequence[Trajectory],
    st: Spacetime,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Photon paths in the top view ``(x, y)`` and the side view ``(x, z)``.

    Both panels draw the outer horizon ``r_+`` (filled) and the ergosphere
    ``r_E(theta)`` (dashed) as the closed curves of the oblate-spheroidal
    embedding of :func:`kerrray.geometry.bl_to_cartesian` (a plotting
    convention, not an isometric embedding: docs/equations.md). Each
    trajectory gets the next fixed-order colour; ``labels`` default to
    ``"ray k"``.
    """
    if len(trajectories) == 0:
        raise ValueError("trajectories must not be empty")
    names = list(labels) if labels is not None else [f"ray {k}" for k in range(len(trajectories))]
    if len(names) != len(trajectories):
        raise ValueError("labels must match the number of trajectories")
    curves = _horizon_and_ergosphere(st)
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(11.0, 5.5))
    for ax, view in ((ax_top, "top"), (ax_side, "side")):
        hx, hy = curves[f"{view}_horizon"]
        ex, ey = curves[f"{view}_ergosphere"]
        ax.fill(hx, hy, facecolor=HORIZON_COLOR, edgecolor="none", label="horizon r+")
        ax.plot(ex, ey, color=ERGOSPHERE_COLOR, linestyle="--", linewidth=1.0, label="ergosphere")
    for k, (traj, name) in enumerate(zip(trajectories, names)):
        x, y, z = to_cartesian(traj)
        style = series_style(k)
        ax_top.plot(x, y, label=name, **style)
        ax_side.plot(x, z, label=name, **style)
        ax_top.plot(x[0], y[0], marker="o", markersize=4.0, color=style["color"], linestyle="none")
        ax_side.plot(x[0], z[0], marker="o", markersize=4.0, color=style["color"], linestyle="none")
    for ax, ylabel in ((ax_top, "y [M]"), (ax_side, "z [M]")):
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        _style_axes(ax, "x [M]", ylabel, None)
    ax_top.set_title("top view (equatorial plane)")
    ax_side.set_title("side view (phi = 0 plane)")
    ax_top.legend(loc="best", frameon=False, fontsize=8)
    if title:
        fig.suptitle(title)
    return _save(fig, path, dpi)


def disk_figure(
    path: Path | str,
    image: Any,
    *,
    quantity: str = "intensity",
    stretch: str = "linear",
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Image of the rendered disk in the ``(alpha, beta)`` plane.

    ``image`` is a :class:`kerrray.raytracing.renderer.DiskImage` (or any
    object with ``intensity``, ``g``, ``hit``, ``alpha``, ``beta``).
    ``quantity`` is ``"intensity"`` (sequential single-hue map, ``stretch``
    ``"linear"`` or ``"log"``; the log stretch floors at
    :data:`LOG_STRETCH_FLOOR` times the maximum) or ``"redshift"`` (``g`` on
    disk pixels with a diverging map centred on ``g = 1``, other pixels
    blank). The extent is the outer pixel edges, row 0 at the top.
    """
    alpha = np.asarray(image.alpha, dtype=float)
    beta = np.asarray(image.beta, dtype=float)
    half = 0.5 * (alpha[0, 1] - alpha[0, 0]) if alpha.shape[1] > 1 else 0.5
    extent = (alpha.min() - half, alpha.max() + half, beta.min() - half, beta.max() + half)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    if quantity == "intensity":
        data = np.asarray(image.intensity, dtype=float)
        peak = float(np.max(data)) if data.size and np.max(data) > 0 else 1.0
        if stretch == "log":
            shown = np.log10(np.maximum(data, LOG_STRETCH_FLOOR * peak) / peak)
            label = "log10(I / I_max)"
        elif stretch == "linear":
            shown = data / peak
            label = "I / I_max"
        else:
            raise ValueError(f"stretch must be 'linear' or 'log', got {stretch!r}")
        cmap, vmin, vmax = INTENSITY_CMAP, float(np.min(shown)), float(np.max(shown))
    elif quantity == "redshift":
        g = np.asarray(image.g, dtype=float)
        shown = np.where(np.asarray(image.hit, dtype=bool), g, np.nan)
        finite = shown[np.isfinite(shown)]
        span = float(np.max(np.abs(finite - 1.0))) if finite.size else 0.5
        span = span if span > 0 else 0.5
        cmap, vmin, vmax = REDSHIFT_CMAP, 1.0 - span, 1.0 + span
        label = "g = nu_obs / nu_em"
    else:
        raise ValueError(f"quantity must be 'intensity' or 'redshift', got {quantity!r}")
    im = ax.imshow(shown, origin="upper", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.grid(False)
    ax.set_xlabel("alpha [M]")
    ax.set_ylabel("beta [M]")
    if title:
        ax.set_title(title)
    return _save(fig, path, dpi)


def lensing_figure(
    path: Path | str,
    b: ArrayLike,
    curves: Mapping[str, ArrayLike],
    *,
    xlabel: str = "impact parameter b [M]",
    ylabel: str = "deflection angle [rad]",
    logx: bool = False,
    logy: bool = False,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Deflection curves against a common abscissa ``b`` (NaN values leave gaps).

    ``curves`` maps a label to the ordinate array of the same length as ``b``;
    labels get colours in fixed order and a legend is drawn for two or more
    curves (:func:`kerrray.reporting.plots.plot_lines`). Use ``logx`` with
    ``b / b_c - 1`` as the abscissa to show the logarithmic divergence.
    """
    b_arr = np.asarray(b, dtype=float).ravel()
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, values in curves.items():
        y = np.asarray(values, dtype=float).ravel()
        if y.shape != b_arr.shape:
            raise ValueError(f"curve {label!r} has {y.size} values, b has {b_arr.size}")
        series.append((str(label), b_arr, y))
    if not series:
        raise ValueError("curves must contain at least one entry")
    return plot_lines(path, series, xlabel=xlabel, ylabel=ylabel, title=title, logx=logx, logy=logy, dpi=dpi)

