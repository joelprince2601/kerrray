"""Photon phase-space state layout (docs/architecture.md section 4).

A photon is described by its Boyer-Lindquist position ``x^mu = (t, r, theta,
phi)`` and the *covariant* momentum ``p_mu = (p_t, p_r, p_theta, p_phi)``
(docs/architecture.md section 1). The integrator state is the flat float
array ``y = [t, r, theta, phi, p_t, p_r, p_theta, p_phi]``; a batch of ``N``
rays is the array ``Y`` of shape ``(N, 8)``. Angles are radians. The affine
parameter ``lambda`` is in units of the mass ``M`` (geometric units, G = c =
1). Photon energy is ``E = -p_t`` (future-directed photons have ``E > 0``) and
the axial angular momentum is ``L_z = p_phi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "IDX_PH",
    "IDX_PPH",
    "IDX_PR",
    "IDX_PT",
    "IDX_PTH",
    "IDX_R",
    "IDX_T",
    "IDX_TH",
    "STATE_SIZE",
    "PhotonState",
    "pack",
    "unpack",
]

IDX_T, IDX_R, IDX_TH, IDX_PH, IDX_PT, IDX_PR, IDX_PTH, IDX_PPH = range(8)
"""Column indices of ``y = [t, r, theta, phi, p_t, p_r, p_theta, p_phi]``."""

STATE_SIZE = 8
"""Length of one photon state vector (four coordinates, four momenta)."""


@dataclass(frozen=True)
class PhotonState:
    """A single photon state: position ``x`` (4,) and covariant momentum ``p`` (4,).

    Attributes:
        x: Boyer-Lindquist coordinates ``(t, r, theta, phi)`` (float64).
        p: Covariant momentum components ``(p_t, p_r, p_theta, p_phi)``
            (float64).
    """

    x: NDArray[np.float64]
    p: NDArray[np.float64]

    def __post_init__(self) -> None:
        x = np.array(self.x, dtype=np.float64)
        p = np.array(self.p, dtype=np.float64)
        if x.shape != (4,) or p.shape != (4,):
            raise ValueError(f"x and p must have shape (4,), got {x.shape} and {p.shape}")
        x.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "p", p)

    def to_array(self) -> NDArray[np.float64]:
        """Return the flat state ``[t, r, theta, phi, p_t, p_r, p_theta, p_phi]``."""
        return np.concatenate([self.x, self.p])

    @classmethod
    def from_array(cls, y: ArrayLike) -> PhotonState:
        """Build a state from a flat array of shape ``(8,)``."""
        arr = np.asarray(y, dtype=np.float64)
        if arr.shape != (STATE_SIZE,):
            raise ValueError(f"y must have shape ({STATE_SIZE},), got {arr.shape}")
        return cls(arr[:4], arr[4:])

    @property
    def energy(self) -> float:
        """Photon energy at infinity ``E = -p_t``."""
        return float(-self.p[0])

    @property
    def angular_momentum(self) -> float:
        """Axial angular momentum ``L_z = p_phi``."""
        return float(self.p[3])


def pack(x: ArrayLike, p: ArrayLike) -> NDArray[np.floating]:
    """Concatenate positions ``x`` (..., 4) and momenta ``p`` (..., 4) into ``y`` (..., 8).

    The two inputs broadcast against each other over the leading dimensions;
    the result has the promoted floating dtype of the inputs.
    """
    x_arr = np.asarray(x)
    p_arr = np.asarray(p)
    if x_arr.shape[-1:] != (4,) or p_arr.shape[-1:] != (4,):
        raise ValueError(f"x and p must end in a dimension of size 4, got {x_arr.shape} and {p_arr.shape}")
    x_b, p_b = np.broadcast_arrays(x_arr, p_arr)
    return np.concatenate([x_b, p_b], axis=-1)


def unpack(y: ArrayLike) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Split ``y`` (..., 8) into ``(x, p)``, each of shape ``(..., 4)`` (views when possible)."""
    arr = np.asarray(y)
    if arr.shape[-1:] != (STATE_SIZE,):
        raise ValueError(f"y must end in a dimension of size {STATE_SIZE}, got {arr.shape}")
    return arr[..., :4], arr[..., 4:]
