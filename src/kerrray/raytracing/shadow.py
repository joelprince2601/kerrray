"""Shadow reconstruction from ray classification (PROJECT.md section 17).

The shadow is never drawn analytically: every pixel of the camera image is
traced backwards from the observer (:mod:`kerrray.raytracing.camera`) and
the pixel is a shadow pixel exactly when its ray ends ``CAPTURED``
(:class:`kerrray.photons.TerminationState`); ``ESCAPED`` pixels are the
background and every other outcome is kept in the state map for inspection.
:class:`ShadowImage` bundles the pixel coordinates, the state map, the
captured mask, the per-state counts and the full
:class:`~kerrray.photons.BatchResult` with the camera and spacetime that
produced them, and round-trips through a compressed ``.npz`` file. The
boundary of the captured region is extracted in
:mod:`kerrray.raytracing.boundary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult, TerminationState
from kerrray.raytracing.backends.base import Backend
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.rays import DEFAULT_CHUNK_SIZE, ProgressCallback, trace_rays

__all__ = [
    "FAILURE_STATES",
    "NPZ_FORMAT_VERSION",
    "ShadowImage",
    "compute_shadow",
    "state_counts",
]

NPZ_FORMAT_VERSION = 1
FAILURE_STATES = (TerminationState.NUMERICAL_FAILURE, TerminationState.OUT_OF_DOMAIN)
"""Outcomes counted as failures of the ray (geodesics role: an RK4 overshoot
through the horizon is OUT_OF_DOMAIN, not CAPTURED)."""

_CAMERA_FIELDS = ("radius", "inclination_deg", "phi_deg", "fov", "resolution")
_RESULT_ARRAYS = (
    "Y",
    "state",
    "n_steps",
    "lam",
    "max_null_error",
    "max_energy_drift",
    "max_lz_drift",
    "max_carter_drift",
)
_OPTIONAL_RESULT_ARRAYS = ("event_Y", "event_hit", "n_rejected")


def state_counts(state: NDArray[np.integer]) -> dict[str, int]:
    """Number of rays per :class:`TerminationState` name (every member, zeros included)."""
    codes = np.asarray(state)
    return {member.name: int(np.count_nonzero(codes == int(member))) for member in TerminationState}


@dataclass
class ShadowImage:
    """A traced image: pixel coordinates, termination map and captured mask.

    Attributes:
        alpha, beta: Celestial coordinates of the pixel centres, shape ``(H, W)``
            (units of ``M``; :meth:`Camera.pixel_coordinates`).
        state: :class:`TerminationState` code of each pixel, shape ``(H, W)``.
        captured: ``state == CAPTURED``, the shadow mask.
        counts: Pixels per termination-state name (sums to ``H * W``).
        result: The full batch result, rays in row-major pixel order.
        camera: The camera the image was traced with.
        spacetime: The spacetime the rays were integrated in.
    """

    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    state: NDArray[np.int64]
    captured: NDArray[np.bool_]
    counts: dict[str, int]
    result: BatchResult
    camera: Camera
    spacetime: Spacetime

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(H, W)``."""
        return self.camera.shape

    @property
    def n_rays(self) -> int:
        """Number of pixels (rays)."""
        return self.camera.n_rays

    @property
    def pixel_size(self) -> float:
        """Pixel side in units of ``M``."""
        return self.camera.pixel_size

    def count(self, state: TerminationState) -> int:
        """Pixels that ended in ``state``."""
        return self.counts[state.name]

    @property
    def n_failed(self) -> int:
        """Pixels whose ray failed (:data:`FAILURE_STATES`)."""
        return sum(self.count(s) for s in FAILURE_STATES)

    @property
    def n_other(self) -> int:
        """Pixels that neither ended CAPTURED/ESCAPED nor failed."""
        settled = self.count(TerminationState.CAPTURED) + self.count(TerminationState.ESCAPED)
        return self.n_rays - settled - self.n_failed

    def save(self, path: Path | str) -> Path:
        """Write a compressed ``.npz`` with the image, the batch result, camera and spacetime."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": np.int64(NPZ_FORMAT_VERSION),
            "alpha": self.alpha,
            "beta": self.beta,
            "image_state": self.state,
            "captured": self.captured,
            "mass": np.float64(self.spacetime.mass),
            "spin": np.float64(self.spacetime.spin),
            "runtime_s": np.float64(self.result.runtime_s),
        }
        for name in _CAMERA_FIELDS:
            payload[f"camera_{name}"] = np.asarray(getattr(self.camera, name))
        for name in _RESULT_ARRAYS:
            payload[f"result_{name}"] = np.asarray(getattr(self.result, name))
        for name in _OPTIONAL_RESULT_ARRAYS:
            value = getattr(self.result, name)
            if value is not None:
                payload[f"result_{name}"] = np.asarray(value)
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: Path | str) -> ShadowImage:
        """Read an image written by :meth:`save`."""
        with np.load(Path(path)) as data:
            version = int(data["format_version"])
            if version != NPZ_FORMAT_VERSION:
                raise ValueError(
                    f"unsupported shadow file version {version} (expected {NPZ_FORMAT_VERSION})"
                )
            camera = Camera(
                radius=float(data["camera_radius"]),
                inclination_deg=float(data["camera_inclination_deg"]),
                phi_deg=float(data["camera_phi_deg"]),
                fov=float(data["camera_fov"]),
                resolution=int(data["camera_resolution"]),
            )
            spacetime = Spacetime(mass=float(data["mass"]), spin=float(data["spin"]))
            fields: dict[str, Any] = {name: data[f"result_{name}"] for name in _RESULT_ARRAYS}
            for name in _OPTIONAL_RESULT_ARRAYS:
                key = f"result_{name}"
                fields[name] = data[key] if key in data.files else None
            result = BatchResult(runtime_s=float(data["runtime_s"]), **fields)
            state = np.asarray(data["image_state"], dtype=np.int64)
            return cls(
                alpha=np.asarray(data["alpha"], dtype=np.float64),
                beta=np.asarray(data["beta"], dtype=np.float64),
                state=state,
                captured=np.asarray(data["captured"], dtype=bool),
                counts=state_counts(state),
                result=result,
                camera=camera,
                spacetime=spacetime,
            )


def compute_shadow(
    st: Spacetime,
    cam: Camera,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    backend: str | Backend = "numpy",
    progress: ProgressCallback | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ShadowImage:
    """Trace every pixel of ``cam`` backwards in ``st`` and classify it.

    The initial states come from :func:`kerrray.raytracing.camera.initial_states`
    (row-major over the image), the integration from
    :func:`kerrray.raytracing.rays.trace_rays`; ``Y.reshape(H, W, 8)`` restores
    the image layout. See the module docstring for what the mask means.
    """
    Y0 = initial_states(cam, st)
    result = trace_rays(
        st, Y0, integ, term, backend=backend, progress=progress, chunk_size=chunk_size
    )
    alpha, beta = cam.pixel_coordinates()
    state = np.asarray(result.state, dtype=np.int64).reshape(cam.shape)
    return ShadowImage(
        alpha=alpha,
        beta=beta,
        state=state,
        captured=state == int(TerminationState.CAPTURED),
        counts=state_counts(state),
        result=result,
        camera=cam,
        spacetime=st,
    )
