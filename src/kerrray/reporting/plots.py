"""Generic matplotlib figures for KerrRay reports (PROJECT.md section 36).

The Agg backend is selected at import time when no backend has been chosen
yet (``matplotlib.get_backend(auto_select=False)`` is ``None``), so figures
render on headless machines and nothing here ever calls ``plt.show``. Every
``plot_*`` function draws one generic figure, saves it and returns the saved
:class:`~pathlib.Path`; figure composition for the scientific report belongs
to :mod:`kerrray.reporting.figures` (rendering role).

Style rules: one axis per figure (never a dual y-axis), thin 1.5 pt lines,
recessive grid, a legend whenever two or more series share an axis, and
categorical colours assigned in the fixed order :data:`SERIES_COLORS`
(validated for colour-vision deficiency on adjacent pairs); beyond eight
series the line style changes as a secondary encoding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

if matplotlib.get_backend(auto_select=False) is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after the backend selection)
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

__all__ = [
    "DEFAULT_DPI",
    "SERIES_COLORS",
    "loglog_slope",
    "plot_conservation",
    "plot_convergence",
    "plot_lines",
    "plot_shadow",
    "plot_trajectory_xy",
    "series_style",
]

SERIES_COLORS: Final[tuple[str, ...]] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
"""Categorical colours in fixed assignment order (never re-ordered per figure)."""

LINE_STYLES: Final[tuple[str, ...]] = ("-", "--", ":", "-.")
HORIZON_COLOR: Final[str] = "#0b0b0b"
ERGOSPHERE_COLOR: Final[str] = SERIES_COLORS[1]
TRAJECTORY_COLOR: Final[str] = SERIES_COLORS[0]
ANALYTIC_COLOR: Final[str] = SERIES_COLORS[1]
LINE_WIDTH: Final[float] = 1.5
MARKER_SIZE: Final[float] = 4.0
GRID_ALPHA: Final[float] = 0.3
DEFAULT_DPI: Final[int] = 150
FIGURE_SIZE: Final[tuple[float, float]] = (6.0, 4.5)


def series_style(index: int) -> dict[str, Any]:
    """Colour and line style for the ``index``-th series of a figure."""
    return {
        "color": SERIES_COLORS[index % len(SERIES_COLORS)],
        "linestyle": LINE_STYLES[(index // len(SERIES_COLORS)) % len(LINE_STYLES)],
        "linewidth": LINE_WIDTH,
    }


def _style_axes(ax: Axes, xlabel: str, ylabel: str, title: str | None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: Any, path: Path | str, dpi: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def plot_trajectory_xy(
    target: Axes | Path | str,
    x: ArrayLike,
    y: ArrayLike,
    horizon_radius: float,
    ergosphere_curve: tuple[ArrayLike, ArrayLike] | None = None,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path | Axes:
    """Draw a photon path in the projected ``(x, y)`` plane with the horizon disc.

    Args:
        target: An :class:`~matplotlib.axes.Axes` to draw into (returned
            unchanged, nothing is saved) or a file path (figure saved, path
            returned).
        x, y: Trajectory coordinates in units of ``M``.
        horizon_radius: Outer horizon ``r_+`` drawn as a filled disc.
        ergosphere_curve: Optional ``(x_e, y_e)`` closed curve of the ergosphere
            in the same projection, drawn dashed.
        title: Optional axes title.
        dpi: Raster resolution when saving.
    """
    xs, ys = _as_1d(x, "x"), _as_1d(y, "y")
    if isinstance(target, Axes):
        ax, fig = target, None
    else:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.add_patch(
        Circle((0.0, 0.0), horizon_radius, facecolor=HORIZON_COLOR, edgecolor="none",
               label="horizon r+")
    )
    if ergosphere_curve is not None:
        ex, ey = _as_1d(ergosphere_curve[0], "x_e"), _as_1d(ergosphere_curve[1], "y_e")
        ax.plot(ex, ey, color=ERGOSPHERE_COLOR, linestyle="--", linewidth=1.0,
                label="ergosphere")
    ax.plot(xs, ys, color=TRAJECTORY_COLOR, linewidth=LINE_WIDTH, label="photon")
    ax.plot(xs[0], ys[0], marker="o", markersize=MARKER_SIZE, color=TRAJECTORY_COLOR,
            linestyle="none", label="launch")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    _style_axes(ax, "x [M]", "y [M]", title)
    ax.legend(loc="best", frameon=False)
    if fig is None:
        return ax
    return _save(fig, target, dpi)  # type: ignore[arg-type]


def plot_conservation(
    path: Path | str,
    lam: ArrayLike,
    drifts: Mapping[str, ArrayLike],
    *,
    floor: float = 1e-18,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Plot relative drifts of conserved quantities against the affine parameter (log y).

    Values below ``floor`` (including exact zeros, which a logarithmic axis
    cannot show) are drawn at ``floor``; a series that never exceeds the floor
    is labelled ``"<name> (<= floor)"`` so the reader is not misled. Non-finite
    values leave gaps.
    """
    if not drifts:
        raise ValueError("drifts must contain at least one series")
    if not floor > 0:
        raise ValueError(f"floor must be > 0, got {floor!r}")
    lam_arr = _as_1d(lam, "lam")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for index, (name, values) in enumerate(drifts.items()):
        arr = _as_1d(values, name)
        if arr.shape != lam_arr.shape:
            raise ValueError(f"drift {name!r} has {arr.size} values, lam has {lam_arr.size}")
        finite = np.isfinite(arr)
        clipped = np.where(finite, np.maximum(arr, floor), np.nan)
        label = name if np.any(finite & (arr > floor)) else f"{name} (<= floor)"
        ax.plot(lam_arr, clipped, label=label, **series_style(index))
    ax.set_yscale("log")
    _style_axes(ax, "affine parameter lambda [M]", f"relative drift (floor {floor:g})", title)
    ax.legend(loc="best", frameon=False)
    return _save(fig, path, dpi)


def loglog_slope(x: ArrayLike, y: ArrayLike) -> float:
    """Least-squares slope of ``log10(y)`` against ``log10(x)`` over positive finite pairs.

    This is the empirical convergence order of an error ``y`` against a
    resolution or tolerance ``x``. At least two usable points are required.
    """
    xs, ys = _as_1d(x, "x"), _as_1d(y, "y")
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length")
    usable = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if np.count_nonzero(usable) < 2:
        raise ValueError("need at least two positive finite (x, y) pairs for a log-log fit")
    slope, _ = np.polyfit(np.log10(xs[usable]), np.log10(ys[usable]), 1)
    return float(slope)


def plot_convergence(
    path: Path | str,
    x: ArrayLike,
    y: ArrayLike,
    xlabel: str,
    ylabel: str,
    *,
    fit_slope: bool = True,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Log-log convergence plot of an error ``y`` against ``x``, with a fitted slope.

    When ``fit_slope`` is true and at least two positive finite points exist,
    the least-squares line from :func:`loglog_slope` is drawn dashed and the
    slope annotated on the axes; the fit is computed from the data, never
    assumed.
    """
    xs, ys = _as_1d(x, "x"), _as_1d(y, "y")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(xs, ys, marker="o", markersize=MARKER_SIZE, label="measured", **series_style(0))
    usable = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if fit_slope and np.count_nonzero(usable) >= 2:
        slope = loglog_slope(xs, ys)
        log_x = np.log10(xs[usable])
        intercept = np.mean(np.log10(ys[usable]) - slope * log_x)
        grid = np.linspace(log_x.min(), log_x.max(), 50)
        ax.plot(10.0**grid, 10.0 ** (slope * grid + intercept), label=f"fit, slope {slope:.3f}",
                **{**series_style(1), "linestyle": "--"})
        ax.annotate(
            f"slope = {slope:.3f}", xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9
        )
        ax.legend(loc="best", frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(ax, xlabel, ylabel, title)
    return _save(fig, path, dpi)


def plot_shadow(
    path: Path | str,
    alpha: ArrayLike,
    beta: ArrayLike,
    captured_mask: ArrayLike,
    analytic_curve: tuple[ArrayLike, ArrayLike] | None = None,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Draw the captured-pixel mask in the celestial ``(alpha, beta)`` plane.

    ``alpha`` and ``beta`` are the pixel-centre coordinates (2-D arrays of the
    image shape, or 1-D axes of lengths ``W`` and ``H``); ``captured_mask`` is
    the boolean image. ``analytic_curve`` is an optional ``(alpha_c, beta_c)``
    reference boundary drawn as a line.
    """
    mask = np.asarray(captured_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"captured_mask must be 2-D, got shape {mask.shape}")
    a_arr, b_arr = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pcolormesh(
        a_arr, b_arr, mask.astype(float), cmap=ListedColormap(["#ffffff", HORIZON_COLOR]),
        vmin=0.0, vmax=1.0, shading="nearest", rasterized=True,
    )
    handles: list[Any] = [Patch(facecolor=HORIZON_COLOR, label="captured")]
    if analytic_curve is not None:
        ac, bc = _as_1d(analytic_curve[0], "alpha_c"), _as_1d(analytic_curve[1], "beta_c")
        ax.plot(ac, bc, color=ANALYTIC_COLOR, linewidth=LINE_WIDTH, label="analytic boundary")
        handles.append(Line2D([], [], color=ANALYTIC_COLOR, linewidth=LINE_WIDTH,
                              label="analytic boundary"))
    ax.set_aspect("equal", adjustable="box")
    _style_axes(ax, "alpha [M]", "beta [M]", title)
    ax.legend(handles=handles, loc="upper right", frameon=False)
    return _save(fig, path, dpi)


def plot_lines(
    path: Path | str,
    series: Mapping[str, tuple[ArrayLike, ArrayLike]] | Sequence[tuple[str, ArrayLike, ArrayLike]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    logx: bool = False,
    logy: bool = False,
    markers: bool = True,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Plot one or more ``(x, y)`` series on a single axis.

    ``series`` maps a label to ``(x, y)`` (or is a sequence of
    ``(label, x, y)`` triples); labels are assigned colours in order. A legend
    is drawn when there are at least two series.
    """
    items = (
        [(label, xy[0], xy[1]) for label, xy in series.items()]
        if isinstance(series, Mapping)
        else [tuple(item) for item in series]
    )
    if not items:
        raise ValueError("series must contain at least one entry")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for index, (label, x, y) in enumerate(items):
        ax.plot(
            _as_1d(x, f"{label} x"), _as_1d(y, f"{label} y"), label=str(label),
            marker="o" if markers else None, markersize=MARKER_SIZE, **series_style(index),
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style_axes(ax, xlabel, ylabel, title)
    if len(items) >= 2:
        ax.legend(loc="best", frameon=False)
    return _save(fig, path, dpi)
