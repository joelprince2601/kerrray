"""Simplified emitting accretion disk (PROJECT.md sections 23, 24 and 43;
docs/rendering.md section 3).

Approximation statement (PROJECT.md section 43):

* WHAT: an optically thin, geometrically thin, single-surface emitter in the
  equatorial plane ``theta = pi/2`` between ``r_in`` (default: the innermost
  stable circular orbit, BPT 1972 eq. 2.21 via
  :func:`kerrray.photons.orbits.isco_radius`, re-exported here) and
  ``r_out``, with the power-law emissivity ``I_em(r) = (r / r_in)^(-p)`` in
  the local rest frame of the gas, which moves on circular Keplerian orbits.
  The observed intensity is ``I_obs = g^3 I_em`` for the specific intensity
  at a fixed observed frequency (``I_nu / nu^3`` is invariant along a ray,
  Misner, Thorne and Wheeler 1973 chapter 22 (kinetic theory in curved
  spacetime); Cunningham 1975) or ``I_obs =
  g^4 I_em`` for the frequency-integrated intensity (one more power of ``g``
  from the transformation of the frequency interval, Luminet 1979), selected
  by ``intensity_law`` (``"g3"`` or ``"g4"``).
* WHY: it isolates the geodesic propagation (lensing and redshift) from
  radiative physics, which is the purpose of Phase 7 (PROJECT.md section 23:
  "a visualization/physics extension, not a full GRMHD simulation").
* LIMITATION: not equivalent to a GRMHD or radiative-transfer model. No
  radiative transfer (no absorption or emission along the ray, the disk is a
  surface hit once), no returning radiation (photons that re-cross the disk
  after the first hit are not followed), no disk thickness, no vertical
  structure, no spectral shape (``I_em`` is a frequency-independent number),
  no time dependence, and the power-law emissivity is a modelling choice with
  no physical derivation. The intensity is in arbitrary units normalised to
  ``I_em(r_in) = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.integrators import EventOptions
from kerrray.geometry import Spacetime
from kerrray.photons.orbits import equatorial_photon_orbit_radius, isco_radius
from kerrray.utils.config import ConfigError, experiment_block

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kerrray.utils.config import KerrRayConfig

__all__ = [
    "DISK_BLOCK",
    "INTENSITY_LAWS",
    "DiskModel",
    "DiskParams",
    "disk_params",
    "emissivity",
    "event_options",
    "isco_radius",
    "observed_intensity",
]

FloatArray = NDArray[np.float64]

DISK_BLOCK: Final[str] = "disk"
"""Name of the ``experiment.parameters`` sub-block (docs/architecture.md section 7, D-009)."""

INTENSITY_LAWS: Final[dict[str, int]] = {"g3": 3, "g4": 4}
"""Power of ``g`` multiplying the emitted intensity for each ``intensity_law`` (module docstring)."""


def _finite(value: float, key: str) -> None:
    if not math.isfinite(value):
        raise ConfigError(f"'{key}' must be finite, got {value!r}")


@dataclass(frozen=True)
class DiskParams:
    """The ``experiment.parameters.disk`` sub-block (docs/architecture.md section 7).

    Attributes:
        enabled: Whether drivers that *optionally* render a disk should do
            so; the ``render`` command always renders and only records it.
        r_in: Inner radius in units of ``M``; ``None`` (YAML ``null``) selects
            the ISCO of the spacetime for the chosen sense of rotation.
        r_out: Outer radius in units of ``M`` (``> r_in``).
        emissivity_index: The exponent ``p`` of ``I_em = (r / r_in)^(-p)``.
        intensity_law: ``"g3"`` or ``"g4"`` (module docstring).
        prograde: Gas orbits with ``a L_z > 0`` (``L_z > 0`` for ``a = 0``).
    """

    enabled: bool = False
    r_in: float | None = None
    r_out: float = 20.0
    emissivity_index: float = 3.0
    intensity_law: str = "g4"
    prograde: bool = True

    def __post_init__(self) -> None:
        _finite(self.r_out, "disk.r_out")
        if self.r_out <= 0.0:
            raise ConfigError(f"'disk.r_out' must be > 0, got {self.r_out!r}")
        if self.r_in is not None:
            _finite(self.r_in, "disk.r_in")
            if not (0.0 < self.r_in < self.r_out):
                raise ConfigError(f"'disk.r_in' must satisfy 0 < r_in < r_out, got {self.r_in!r}")
        _finite(self.emissivity_index, "disk.emissivity_index")
        if self.intensity_law not in INTENSITY_LAWS:
            raise ConfigError(
                f"'disk.intensity_law' must be one of {', '.join(sorted(INTENSITY_LAWS))}, "
                f"got {self.intensity_law!r}"
            )


def disk_params(cfg: KerrRayConfig) -> DiskParams:
    """Parse the ``disk`` sub-block of ``cfg`` (defaults when absent)."""
    return experiment_block(cfg, DISK_BLOCK, DiskParams)


@dataclass(frozen=True)
class DiskModel:
    """Resolved disk: ``r_in`` and ``r_out`` in units of ``M``, emissivity, intensity law.

    Attributes:
        r_in: Inner radius (``> 0``); use :meth:`from_params` to default it
            to the ISCO and to check that the gas can orbit there.
        r_out: Outer radius (``> r_in``).
        emissivity_index: Exponent ``p`` of :func:`emissivity`.
        intensity_law: ``"g3"`` or ``"g4"``.
        prograde: Sense of the gas orbits (``a L_z > 0``).
    """

    r_in: float
    r_out: float
    emissivity_index: float = 3.0
    intensity_law: str = "g4"
    prograde: bool = True

    def __post_init__(self) -> None:
        for name in ("r_in", "r_out", "emissivity_index"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not (0.0 < self.r_in < self.r_out):
            raise ValueError(f"need 0 < r_in < r_out, got r_in={self.r_in!r}, r_out={self.r_out!r}")
        if self.intensity_law not in INTENSITY_LAWS:
            raise ValueError(f"intensity_law must be one of {sorted(INTENSITY_LAWS)}, got {self.intensity_law!r}")

    @property
    def redshift_power(self) -> int:
        """The power ``n`` of ``g`` in ``I_obs = g^n I_em`` (3 or 4)."""
        return INTENSITY_LAWS[self.intensity_law]

    @classmethod
    def from_params(cls, st: Spacetime, params: DiskParams) -> DiskModel:
        """Resolve ``params`` for the spacetime ``st``.

        ``r_in = None`` becomes ``isco_radius(st, prograde)``. The inner radius
        must exceed the circular photon orbit radius, below which no circular
        (even unstable) gas orbit exists and the redshift factor is undefined.
        """
        r_in = params.r_in if params.r_in is not None else isco_radius(st, prograde=params.prograde)
        r_ph = equatorial_photon_orbit_radius(st, prograde=params.prograde)
        if not r_in > r_ph:
            raise ValueError(
                f"disk r_in = {r_in:.6g} must exceed the circular photon orbit radius "
                f"r_ph = {r_ph:.6g} (no circular gas orbit below it)"
            )
        return cls(
            r_in=float(r_in),
            r_out=float(params.r_out),
            emissivity_index=float(params.emissivity_index),
            intensity_law=params.intensity_law,
            prograde=bool(params.prograde),
        )

    @classmethod
    def from_config(cls, st: Spacetime, cfg: KerrRayConfig) -> DiskModel:
        """Build the disk from ``cfg.experiment.parameters.disk`` (defaults when absent)."""
        return cls.from_params(st, disk_params(cfg))


def emissivity(disk: DiskModel, r: ArrayLike) -> FloatArray:
    """Emitted intensity ``I_em(r) = (r / r_in)^(-p)`` on ``r_in <= r <= r_out``, ``0`` elsewhere.

    Normalised to ``1`` at ``r_in`` (arbitrary units, module docstring).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    inside = (r_arr >= disk.r_in) & (r_arr <= disk.r_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.power(r_arr / disk.r_in, -disk.emissivity_index)
    return np.where(inside, value, 0.0)


def observed_intensity(disk: DiskModel, r: ArrayLike, g: ArrayLike) -> FloatArray:
    """``I_obs = g^n I_em(r)`` with ``n = 3`` (``"g3"``) or ``4`` (``"g4"``); see the module docstring."""
    g_arr = np.asarray(g, dtype=np.float64)
    return emissivity(disk, r) * g_arr**disk.redshift_power


def event_options(disk: DiskModel) -> EventOptions:
    """The batched integrator's disk-plane event for this disk (``DISK_HIT`` on ``r_in <= r <= r_out``)."""
    return EventOptions(disk_plane=True, r_in=disk.r_in, r_out=disk.r_out)
