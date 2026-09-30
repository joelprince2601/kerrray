"""Shadow-boundary extraction and comparison with the analytic curve.

Extraction (:func:`extract_boundary`). The captured mask is a 0/1 field
sampled at the pixel centres. It is interpolated bilinearly on that grid
(Press et al., *Numerical Recipes*, 3rd ed., section 3.6), with the value 0
(escaped) outside the outermost pixel centres, and the boundary is defined
as the 0.5 level set of the interpolant. From a centre (by default the
centroid of the captured pixels) a ray is marched outward along every polar
direction ``phi_k = 2 pi k / n_angles`` in steps of ``STEP_FRACTION`` pixels;
the first sample below 0.5 brackets the crossing, which is then located by
linear interpolation of the sampled field between the two bracketing
samples. The result is a sub-pixel radius ``r(phi_k)`` in units of ``M``
whose error is bounded by the pixel size (the level set of a bilinear
interpolant of a 0/1 mask lies within half a pixel of the true edge).

Comparison (:func:`boundary_error`). The analytic curve of
:func:`kerrray.photons.orbits.shadow_curve` is a closed polygon in the
``(alpha, beta)`` plane; its radius in the same polar angles about the same
centre is obtained by exact ray-segment intersection
(:func:`polygon_radii`), which requires the curve to be star-shaped about
the centre (true for the convex Kerr shadow). Radial differences are
reported as maximum, root-mean-square and signed mean, in units of ``M`` and
optionally in pixels, together with the mean radius, the radius range
(asymmetry) and the area centroid of each curve (shoelace formula:
``A = 1/2 sum (x_i y_{i+1} - x_{i+1} y_i)``, ``C_x = 1/(6A) sum (x_i +
x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)`` and likewise ``C_y``).

Finite observer distance (approximation, PROJECT.md section 43). WHAT: the
analytic curve is Bardeen's ``r_o -> infinity`` limit while the image is seen
by a ZAMO at finite ``r_o``, whose pixel coordinates are smaller by the
factor ``sqrt(1 - 2M/r_o)`` (exact for ``a = 0``, leading order ``1 - M/r_o``
for any spin; docs/raytracing.md section 5). WHY: rays must start at a
finite radius. LIMITATION: the raw comparison carries a systematic relative
offset ``M/r_o`` (``1e-3`` at ``r_o = 1000``); :func:`finite_distance_scale`
gives the factor that :func:`boundary_error` can apply through ``scale`` to
remove it to leading order, and docs/shadow.md quantifies the residual.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geometry import Spacetime
from kerrray.photons.orbits import shadow_curve
from kerrray.raytracing.shadow import ShadowImage

__all__ = [
    "STEP_FRACTION",
    "analytic_boundary",
    "boundary_error",
    "curve_summary",
    "extract_boundary",
    "extract_boundary_from_mask",
    "finite_distance_scale",
    "mask_centroid",
    "polygon_centroid",
    "polygon_radii",
    "summarise_shadow",
]

Float = NDArray[np.float64]

STEP_FRACTION = 0.125
"""Marching step along each polar ray as a fraction of the pixel size."""

EDGE_TOLERANCE = 1e-9
"""Round-off allowance on the edge parameter ``s`` of :func:`polygon_radii`."""


def mask_centroid(alpha: ArrayLike, beta: ArrayLike, captured: ArrayLike) -> tuple[float, float]:
    """Mean ``(alpha, beta)`` of the captured pixel centres."""
    mask = np.asarray(captured, dtype=bool)
    if not np.any(mask):
        raise ValueError("the captured mask is empty: no shadow to extract")
    a, b = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    return float(a[mask].mean()), float(b[mask].mean())


def _grid_geometry(alpha: Float, beta: Float) -> tuple[float, float, float]:
    """``(alpha_0, beta_0, d)``: first pixel centre and pixel size of a regular camera grid."""
    if alpha.ndim != 2 or alpha.shape != beta.shape or min(alpha.shape) < 2:
        raise ValueError("alpha and beta must be 2-D arrays of the same shape with >= 2 pixels per side")
    d = float(alpha[0, 1] - alpha[0, 0])
    if not d > 0.0:
        raise ValueError("alpha must increase along the columns")
    if not np.allclose(np.diff(alpha, axis=1), d) or not np.allclose(np.diff(beta, axis=0), -d):
        raise ValueError("the pixel grid must be regular with alpha increasing rightward and beta downward")
    return float(alpha[0, 0]), float(beta[0, 0]), d


def _bilinear(field: Float, u: Float, v: Float) -> Float:
    """Bilinear interpolation of ``field[i, j]`` at fractional ``(row v, column u)``; 0 outside."""
    n_rows, n_cols = field.shape
    padded = np.pad(field, 1)  # zero ring: the value outside the outermost centres decays to 0
    up, vp = u + 1.0, v + 1.0
    inside = (up >= 0.0) & (up <= n_cols) & (vp >= 0.0) & (vp <= n_rows)
    j0 = np.clip(np.floor(up).astype(np.intp), 0, n_cols)
    i0 = np.clip(np.floor(vp).astype(np.intp), 0, n_rows)
    fu = np.clip(up - j0, 0.0, 1.0)
    fv = np.clip(vp - i0, 0.0, 1.0)
    j1 = np.minimum(j0 + 1, n_cols + 1)
    i1 = np.minimum(i0 + 1, n_rows + 1)
    value = (
        padded[i0, j0] * (1.0 - fu) * (1.0 - fv)
        + padded[i0, j1] * fu * (1.0 - fv)
        + padded[i1, j0] * (1.0 - fu) * fv
        + padded[i1, j1] * fu * fv
    )
    return np.where(inside, value, 0.0)


def extract_boundary_from_mask(
    alpha: ArrayLike,
    beta: ArrayLike,
    captured: ArrayLike,
    *,
    n_angles: int = 360,
    centre: tuple[float, float] | None = None,
) -> tuple[Float, Float]:
    """Sub-pixel boundary radii ``r(phi)`` of a captured mask (module docstring).

    Args:
        alpha, beta: Pixel-centre coordinates, shape ``(H, W)``, regular grid.
        captured: Boolean mask of the same shape.
        n_angles: Number of polar directions ``phi_k = 2 pi k / n_angles``.
        centre: ``(alpha_c, beta_c)`` to march from; the captured centroid by default.

    Returns:
        ``(angles, radii)``: the angles in radians and the boundary distance
        from the centre in units of ``M`` along each of them.

    Raises:
        ValueError: If the mask is empty, touches the image border (the
            boundary would leave the image), or the centre is not inside the
            captured region.
    """
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    mask = np.asarray(captured, dtype=bool)
    if mask.shape != a.shape:
        raise ValueError(f"captured has shape {mask.shape}, expected {a.shape}")
    if n_angles < 1:
        raise ValueError("n_angles must be >= 1")
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("the captured region touches the image border; increase raytrace.fov")
    a0, b0, d = _grid_geometry(a, b)
    ca, cb = mask_centroid(a, b, mask) if centre is None else (float(centre[0]), float(centre[1]))
    field = mask.astype(np.float64)
    angles = 2.0 * np.pi * np.arange(n_angles, dtype=np.float64) / n_angles
    ds = STEP_FRACTION * d
    # Far enough to leave the image along every direction (image diagonal plus one pixel).
    reach = math.hypot(a[0, -1] - a[0, 0], b[0, 0] - b[-1, 0]) + 2.0 * d
    s = ds * np.arange(int(math.ceil(reach / ds)) + 1, dtype=np.float64)
    x = ca + np.cos(angles)[:, None] * s[None, :]
    y = cb + np.sin(angles)[:, None] * s[None, :]
    f = _bilinear(field, (x - a0) / d, (b0 - y) / d)
    if np.any(f[:, 0] < 0.5):
        raise ValueError("the centre lies outside the captured region")
    below = f < 0.5
    if not below.any(axis=1).all():  # cannot happen: f = 0 beyond the image border
        raise ValueError("no 0.5 crossing found along some direction")  # pragma: no cover
    k = np.argmax(below, axis=1)  # first sample below 0.5
    rows = np.arange(n_angles)
    f_in, f_out = f[rows, k - 1], f[rows, k]
    radii = s[k - 1] + ds * (f_in - 0.5) / (f_in - f_out)
    return angles, radii


def extract_boundary(
    img: ShadowImage, n_angles: int = 360, centre: tuple[float, float] | None = None
) -> tuple[Float, Float]:
    """Boundary ``(angles, radii)`` of the captured region of ``img``; see
    :func:`extract_boundary_from_mask`."""
    return extract_boundary_from_mask(img.alpha, img.beta, img.captured, n_angles=n_angles, centre=centre)


def analytic_boundary(st: Spacetime, inclination_deg: float, n: int = 720) -> tuple[Float, Float]:
    """Bardeen's ``r_o -> infinity`` shadow curve ``(alpha, beta)`` from
    :func:`kerrray.photons.orbits.shadow_curve` (``2 * (n // 2)`` points ordered by
    polar angle about the origin)."""
    return shadow_curve(st, inclination_deg, n)


def polygon_radii(alpha: ArrayLike, beta: ArrayLike, centre: tuple[float, float], angles: ArrayLike) -> Float:
    """Distance from ``centre`` to the closed polygon ``(alpha, beta)`` along each angle.

    Exact ray-segment intersection: for the ray ``c + t (cos phi, sin phi)``
    and the edge ``q_i + s (q_{i+1} - q_i)`` the 2x2 linear system in ``(t, s)``
    is solved for every (ray, edge) pair; the radius is the smallest ``t > 0``
    with ``0 <= s < 1``. Raises ``ValueError`` if some ray misses the polygon
    (the centre is outside it or the curve is not star-shaped about it).
    """
    px = np.asarray(alpha, dtype=np.float64).ravel()
    py = np.asarray(beta, dtype=np.float64).ravel()
    if px.size < 3 or px.shape != py.shape:
        raise ValueError("a polygon needs at least three (alpha, beta) points")
    phi = np.asarray(angles, dtype=np.float64).ravel()
    ca, cb = float(centre[0]), float(centre[1])
    qx, qy = px - ca, py - cb
    ex, ey = np.roll(qx, -1) - qx, np.roll(qy, -1) - qy  # edge vectors, last -> first closes it
    dx, dy = np.cos(phi)[:, None], np.sin(phi)[:, None]
    det = dx * (-ey[None, :]) - dy * (-ex[None, :])  # of [[dx, -ex], [dy, -ey]]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx[None, :] * (-ey[None, :]) - qy[None, :] * (-ex[None, :])) / det
        s = (dx * qy[None, :] - dy * qx[None, :]) / det
    # A ray through a vertex meets one edge at s = 1 and the next at s = 0; round-off
    # can push both just outside [0, 1], so the interval is widened by EDGE_TOLERANCE
    # (both edges then give the same t and the minimum is unaffected).
    valid = np.isfinite(t) & (t > 0.0) & (s >= -EDGE_TOLERANCE) & (s <= 1.0 + EDGE_TOLERANCE)
    t = np.where(valid, t, np.inf)
    radii = t.min(axis=1)
    if not np.all(np.isfinite(radii)):
        raise ValueError("a polar ray from the centre does not intersect the analytic curve")
    return radii


def polygon_centroid(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Area centroid of the closed polygon ``(x, y)`` (shoelace formula, module docstring)."""
    xs, ys = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cross = xs * yn - xn * ys
    area = 0.5 * cross.sum()
    if area == 0.0:
        raise ValueError("degenerate polygon (zero area)")
    return float(((xs + xn) * cross).sum() / (6.0 * area)), float(((ys + yn) * cross).sum() / (6.0 * area))


def curve_summary(angles: ArrayLike, radii: ArrayLike, centre: tuple[float, float]) -> dict[str, float]:
    """Mean/min/max radius, asymmetry ``max - min`` and area centroid of a polar curve."""
    phi, r = np.asarray(angles, dtype=np.float64), np.asarray(radii, dtype=np.float64)
    cx, cy = polygon_centroid(centre[0] + r * np.cos(phi), centre[1] + r * np.sin(phi))
    return {
        "radius_mean": float(r.mean()),
        "radius_min": float(r.min()),
        "radius_max": float(r.max()),
        "asymmetry": float(r.max() - r.min()),
        "centroid_alpha": cx,
        "centroid_beta": cy,
        "centroid_displacement": math.hypot(cx, cy),
    }


def finite_distance_scale(st: Spacetime, radius: float) -> float:
    """``1 / sqrt(1 - 2M / r_o)``: maps ZAMO pixel coordinates at ``r_o`` to Bardeen's
    (exact for ``a = 0``, leading order otherwise; docs/raytracing.md section 5)."""
    if not radius > 2.0 * st.mass:
        raise ValueError("the observer radius must exceed 2M for the finite-distance factor")
    return 1.0 / math.sqrt(1.0 - 2.0 * st.mass / radius)


def boundary_error(
    numeric: tuple[ArrayLike, ArrayLike],
    analytic: tuple[ArrayLike, ArrayLike],
    centre: tuple[float, float],
    *,
    pixel_size: float | None = None,
    scale: float = 1.0,
) -> dict[str, float]:
    """Radial differences between a numeric boundary and the analytic curve.

    Args:
        numeric: ``(angles, radii)`` from :func:`extract_boundary`, measured about ``centre``.
        analytic: ``(alpha, beta)`` closed curve (:func:`analytic_boundary`).
        centre: The centre the numeric radii refer to.
        pixel_size: If given, every length is also reported in pixels (``*_px``).
        scale: Factor applied to the numeric radii and centre before the
            comparison (:func:`finite_distance_scale`); ``1.0`` compares raw.

    Returns:
        ``max``, ``rms`` and signed ``mean`` of ``r_numeric - r_analytic`` (units of
        ``M``), ``mean_abs``, the :func:`curve_summary` of each curve with the
        prefixes ``numeric_`` and ``analytic_``, the centroid shift
        ``numeric - analytic`` (``centroid_shift_alpha/beta/norm``) and, with
        ``pixel_size``, the ``*_px`` counterparts of every length.
    """
    angles = np.asarray(numeric[0], dtype=np.float64)
    r_num = np.asarray(numeric[1], dtype=np.float64) * scale
    c = (float(centre[0]) * scale, float(centre[1]) * scale)
    r_ana = polygon_radii(analytic[0], analytic[1], c, angles)
    diff = r_num - r_ana
    out: dict[str, float] = {
        "max": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "mean": float(np.mean(diff)),
        "mean_abs": float(np.mean(np.abs(diff))),
        "scale": float(scale),
    }
    for prefix, radii in (("numeric_", r_num), ("analytic_", r_ana)):
        out.update({prefix + key: value for key, value in curve_summary(angles, radii, c).items()})
    # The centroid of the analytic curve from its own points (no resampling error).
    ax, ay = polygon_centroid(analytic[0], analytic[1])
    out["analytic_centroid_alpha"], out["analytic_centroid_beta"] = ax, ay
    out["analytic_centroid_displacement"] = math.hypot(ax, ay)
    out["centroid_shift_alpha"] = out["numeric_centroid_alpha"] - ax
    out["centroid_shift_beta"] = out["numeric_centroid_beta"] - ay
    out["centroid_shift_norm"] = math.hypot(out["centroid_shift_alpha"], out["centroid_shift_beta"])
    if pixel_size is not None:
        if not pixel_size > 0.0:
            raise ValueError("pixel_size must be > 0")
        lengths = [key for key in out if key != "scale" and not key.endswith("_px")]
        for key in lengths:
            out[key + "_px"] = out[key] / pixel_size
    return out


def summarise_shadow(
    img: ShadowImage,
    *,
    n_angles: int = 360,
    n_analytic: int = 720,
    centre: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Boundary, analytic curve and both raw and finite-distance-corrected errors of ``img``.

    Returns a mapping with ``centre``, ``angles``, ``radii``, ``analytic``
    (``(alpha, beta)``), ``finite_distance_scale`` and the two
    :func:`boundary_error` dictionaries ``error`` (raw) and
    ``error_corrected`` (numeric radii scaled by the finite-distance factor).
    """
    angles, radii = extract_boundary(img, n_angles=n_angles, centre=centre)
    c = mask_centroid(img.alpha, img.beta, img.captured) if centre is None else centre
    curve = analytic_boundary(img.spacetime, img.camera.inclination_deg, n_analytic)
    factor = finite_distance_scale(img.spacetime, img.camera.radius)
    return {
        "centre": (float(c[0]), float(c[1])),
        "angles": angles,
        "radii": radii,
        "analytic": curve,
        "finite_distance_scale": factor,
        "error": boundary_error((angles, radii), curve, c, pixel_size=img.pixel_size),
        "error_corrected": boundary_error(
            (angles, radii), curve, c, pixel_size=img.pixel_size, scale=factor
        ),
    }
