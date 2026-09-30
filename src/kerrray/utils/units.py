"""Unit conventions and SI conversions for KerrRay (PROJECT.md section 33).

KerrRay works internally in **geometric units**, ``G = c = 1`` (PROJECT.md
sections 4.1 and 33). Lengths, times and the affine parameter are measured in
units of the black-hole mass ``M`` (``black_hole.mass``, normally ``1.0``),
so radii such as the observer radius and the horizon radius are all
expressed in units of ``M``. The spin parameter used in configurations is the
dimensionless ``a_* = a / M`` with ``|a_*| < 1``. The physics engine never
uses SI values; :class:`GeometricUnits` converts results to SI for reporting
only.

Constants (every value cited, none invented):

* ``G_SI = 6.67430e-11 m^3 kg^-1 s^-2``: CODATA 2018 recommended value of the
  Newtonian constant of gravitation (standard uncertainty ``0.00015e-11``).
  E. Tiesinga, P. J. Mohr, D. B. Newell and B. N. Taylor, "CODATA recommended
  values of the fundamental physical constants: 2018", Rev. Mod. Phys. 93,
  025010 (2021); https://physics.nist.gov/cuu/Constants/.
* ``C_SI = 299792458 m/s``: speed of light in vacuum, exact by definition of the
  SI metre (same CODATA 2018 tables).
* ``GM_SUN_SI = 1.3271244e20 m^3 s^-2``: IAU 2015 Resolution B3 nominal solar
  mass parameter ``(GM)_S^N`` (Prsa et al. 2016, AJ 152, 41, Table 1). The
  nominal value is exact by convention; the solar mass in kilograms is derived
  from it as ``GM_SUN_SI / G_SI`` and inherits the relative uncertainty of
  ``G`` (about 2.2e-5).

With these, one geometric length unit for a mass of ``m_sun`` solar masses is
``G M / c^2 = m_sun * GM_SUN_SI / C_SI**2`` metres and one time unit is
``G M / c^3`` seconds (Misner, Thorne and Wheeler 1973, section 36.1 and the
inside-cover conversion table).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["C_SI", "GM_SUN_SI", "G_SI", "GeometricUnits"]

G_SI: Final[float] = 6.67430e-11
"""Newtonian constant of gravitation, m^3 kg^-1 s^-2 (CODATA 2018)."""

C_SI: Final[float] = 299_792_458.0
"""Speed of light in vacuum, m/s (exact, SI definition)."""

GM_SUN_SI: Final[float] = 1.3271244e20
"""Nominal solar mass parameter GM_sun, m^3 s^-2 (IAU 2015 Resolution B3)."""


@dataclass(frozen=True)
class GeometricUnits:
    """Conversion between geometric units (``G = c = 1``) and SI for a given mass scale.

    Attributes:
        mass_solar: Physical mass, in solar masses, of one geometric mass unit
            (the unit in which ``black_hole.mass`` is expressed). For the
            default ``black_hole.mass = 1.0`` this is the black-hole mass in
            solar masses. Must be finite and strictly positive.

    Every conversion is NumPy-vectorised: scalars and arrays are accepted and
    an array input returns an array of the same shape.
    """

    mass_solar: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mass_solar) and self.mass_solar > 0.0):
            raise ValueError(f"mass_solar must be a finite number > 0, got {self.mass_solar!r}")

    @property
    def length_unit_m(self) -> float:
        """Metres per geometric length unit: ``G M / c^2``."""
        return self.mass_solar * GM_SUN_SI / C_SI**2

    @property
    def time_unit_s(self) -> float:
        """Seconds per geometric time unit: ``G M / c^3``."""
        return self.mass_solar * GM_SUN_SI / C_SI**3

    def mass_kg(self, mass_geometric: float = 1.0) -> float:
        """Kilograms for a mass of ``mass_geometric`` geometric units (default: one unit)."""
        return mass_geometric * self.mass_solar * GM_SUN_SI / G_SI

    def mass_geometric(self, mass_kg: float) -> float:
        """Inverse of :meth:`mass_kg`: geometric mass units for a mass in kilograms."""
        return mass_kg * G_SI / (self.mass_solar * GM_SUN_SI)

    def length_m(self, x: ArrayLike) -> NDArray[np.float64] | float:
        """Metres for a length ``x`` in geometric units."""
        return _scale(x, self.length_unit_m)

    def length_geometric(self, x_m: ArrayLike) -> NDArray[np.float64] | float:
        """Inverse of :meth:`length_m`: geometric length units for ``x_m`` metres."""
        return _scale(x_m, 1.0 / self.length_unit_m)

    def time_s(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Seconds for a time (or affine-parameter interval) ``t`` in geometric units."""
        return _scale(t, self.time_unit_s)

    def time_geometric(self, t_s: ArrayLike) -> NDArray[np.float64] | float:
        """Inverse of :meth:`time_s`: geometric time units for ``t_s`` seconds."""
        return _scale(t_s, 1.0 / self.time_unit_s)

    def frequency_hz(self, nu_geometric: ArrayLike) -> NDArray[np.float64] | float:
        """Hertz for a frequency in inverse geometric time units."""
        return _scale(nu_geometric, 1.0 / self.time_unit_s)

    def frequency_geometric(self, nu_hz: ArrayLike) -> NDArray[np.float64] | float:
        """Inverse of :meth:`frequency_hz`: inverse geometric time units for ``nu_hz`` hertz."""
        return _scale(nu_hz, self.time_unit_s)


def _scale(value: ArrayLike, factor: float) -> NDArray[np.float64] | float:
    """Multiply ``value`` by ``factor``; a scalar input gives a Python ``float``."""
    array = np.asarray(value, dtype=np.float64) * factor
    if array.ndim == 0:
        return float(array)
    return array
