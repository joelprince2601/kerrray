"""Coordinate helpers: Boyer-Lindquist to Cartesian embedding and angle units.

Cartesian embedding used for plots (docs/architecture.md section 3)::

    x = sqrt(r^2 + a^2) sin(theta) cos(phi)
    y = sqrt(r^2 + a^2) sin(theta) sin(phi)
    z = r cos(theta)

Approximation label (PROJECT.md section 43):

* WHAT: the flat-space oblate-spheroidal relation between Boyer-Lindquist
  ``(r, theta, phi)`` and Cartesian ``(x, y, z)``; it is the ``r``-``theta``
  part of the Kerr-Schild transformation ``x + i y = (r + i a) sin(theta)
  exp(i phi_KS)`` (Visser 2007, arXiv:0706.0622, section on Kerr-Schild
  coordinates), without the azimuthal shift
  ``phi_KS - phi = int a / Delta dr`` that relates the two azimuths.
* WHY: it reduces to spherical polar coordinates for ``a = 0``, keeps the
  horizon an oblate spheroid rather than a sphere, and does not require the
  singular azimuthal integral near the horizon; it is only used to draw
  trajectories and surfaces.
* LIMITATION: not a physical (isometric) embedding of the curved geometry
  and not the Kerr-Schild Cartesian chart; distances measured in the plot
  are not proper distances and the azimuth is Boyer-Lindquist ``phi``. No
  physics is computed from these coordinates.

Angles are radians everywhere inside the engine; degrees appear only at the
configuration and CLI boundary (docs/architecture.md section 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from kerrray.geometry.metric import Array, Spacetime

__all__ = ["bl_to_cartesian", "deg_to_rad", "rad_to_deg"]


def bl_to_cartesian(
    st: Spacetime, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[Array, Array, Array]:
    """Map Boyer-Lindquist ``(r, theta, phi)`` to plotting Cartesian ``(x, y, z)``.

    The inputs broadcast against each other; the three outputs share the
    broadcast shape and are float64.
    """
    r_, th, ph = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    rho = np.sqrt(r_**2 + st.a**2) * np.sin(th)
    x = rho * np.cos(ph)
    y = rho * np.sin(ph)
    z = r_ * np.cos(th)
    return x, y, z


def deg_to_rad(x: ArrayLike) -> Array:
    """Degrees to radians (float64; scalars and arrays)."""
    return np.deg2rad(np.asarray(x, dtype=np.float64))


def rad_to_deg(x: ArrayLike) -> Array:
    """Radians to degrees (float64; scalars and arrays)."""
    return np.rad2deg(np.asarray(x, dtype=np.float64))
