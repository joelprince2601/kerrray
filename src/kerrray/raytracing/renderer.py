"""Accretion-disk renderer: camera rays, disk-plane event, redshift and intensity
(PROJECT.md sections 23 and 24; docs/rendering.md section 4).

:func:`render_disk` integrates the backward-traced camera rays of
:func:`kerrray.raytracing.camera.initial_states` with the batched integrator
and the disk-plane event (``DISK_HIT`` when the ray crosses ``theta = pi/2``
with ``r_in <= r <= r_out``), evaluates the redshift factor
:func:`kerrray.physics.redshift.redshift_factor` at every crossing state and
sets ``I_obs = g^n I_em(r_hit)`` there (``n = 3`` or ``4`` by the disk's
``intensity_law``) and ``0`` on every other pixel. The physics driving the
image is the geodesic propagation (lensing), the Keplerian emitter motion and
the gravitational plus Doppler shift; the emission model itself is the
simplified one of :mod:`kerrray.physics.accretion` (WHAT / WHY / LIMITATION
there).

Backends. The rays go through :func:`kerrray.raytracing.rays.trace_rays`
(chunked, so memory stays bounded at any resolution) on a backend resolved
by :func:`kerrray.raytracing.backends.get_backend`: ``numpy`` (the reference,
:func:`kerrray.geodesics.integrate_batch`) or ``numba`` (same scheme and
event, verified against ``numpy`` in ``tests/test_numba_backend.py``).
``cuda`` and an uninstalled ``numba`` raise
:class:`~kerrray.raytracing.backends.BackendUnavailable` (a ``RuntimeError``;
GPU acceleration is optional future work, docs/decisions.md D-007).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics.integrators import EventOptions, IntegratorOptions, TerminationOptions
from kerrray.geodesics.state import IDX_R, STATE_SIZE
from kerrray.geometry import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.photons.trajectories import BatchResult
from kerrray.physics.accretion import DiskModel, event_options, observed_intensity
from kerrray.physics.redshift import redshift_factor
from kerrray.raytracing.backends.base import Backend
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.rays import DEFAULT_CHUNK_SIZE, ProgressCallback, trace_rays
from kerrray.utils.config import KerrRayConfig

__all__ = [
    "DiskImage",
    "integrate_rays",
    "integrator_options_from_config",
    "load_disk_image_arrays",
    "render_disk",
    "save_disk_image",
    "termination_options_from_config",
]

FloatArray = NDArray[np.float64]


def integrator_options_from_config(cfg: KerrRayConfig) -> IntegratorOptions:
    """Map the ``integration`` (and ``raytrace.dtype``) blocks to :class:`IntegratorOptions`."""
    integ = cfg.integration
    return IntegratorOptions(
        method=integ.method,
        rtol=integ.rtol,
        atol=integ.atol,
        step_size=integ.step_size,
        max_steps=integ.max_steps,
        lambda_max=integ.lambda_max,
        dtype=cfg.raytrace.dtype,
    )


def termination_options_from_config(cfg: KerrRayConfig) -> TerminationOptions:
    """Map the ``termination`` block to :class:`TerminationOptions`."""
    return TerminationOptions(
        horizon_epsilon=cfg.termination.horizon_epsilon,
        escape_radius=cfg.termination.escape_radius,
    )


def integrate_rays(
    st: Spacetime,
    Y0: FloatArray,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    events: EventOptions | None = None,
    backend: str | Backend = "numpy",
    progress: ProgressCallback | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> BatchResult:
    """Integrate the batch ``Y0`` in chunks on ``backend`` (module docstring).

    A thin wrapper over :func:`kerrray.raytracing.rays.trace_rays`.

    Raises:
        kerrray.raytracing.backends.BackendUnavailable: For ``cuda`` (always)
            and for ``numba`` when Numba is not installed (a ``RuntimeError``).
    """
    return trace_rays(
        st, Y0, integ, term, backend=backend, events=events, progress=progress, chunk_size=chunk_size
    )


@dataclass
class DiskImage:
    """Rendered disk image and the per-pixel quantities behind it.

    All image arrays have the camera shape ``(H, W)`` in the layout of
    :meth:`kerrray.raytracing.camera.Camera.pixel_coordinates` (row 0 is the
    top, ``alpha`` increases to the right).

    Attributes:
        intensity: ``I_obs`` per pixel (``0`` where the ray missed the disk).
        g: Redshift factor ``nu_obs / nu_em`` per pixel (``NaN`` off the disk).
        hit: ``True`` where the ray ended as ``DISK_HIT``.
        r_hit: Boyer-Lindquist radius of the disk crossing (``NaN`` off the disk).
        state: Termination-state codes (:class:`TerminationState`).
        alpha, beta: Celestial coordinates of the pixel centres (units of ``M``).
        hit_states: The interpolated crossing states, shape ``(H, W, 8)``
            (``NaN`` off the disk); the momentum is the *reversed* one of the
            backward-traced ray (docs/raytracing.md section 6).
        result: The full :class:`BatchResult` of the integration.
        camera, spacetime, disk: The inputs of :func:`render_disk`.
        backend: Backend name used.
    """

    intensity: FloatArray
    g: FloatArray
    hit: NDArray[np.bool_]
    r_hit: FloatArray
    state: NDArray[np.int64]
    alpha: FloatArray
    beta: FloatArray
    hit_states: FloatArray
    result: BatchResult
    camera: Camera
    spacetime: Spacetime
    disk: DiskModel
    backend: str

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(H, W)``."""
        return self.camera.shape

    @property
    def n_hits(self) -> int:
        """Number of pixels whose ray hit the disk."""
        return int(np.count_nonzero(self.hit))

    def counts(self) -> dict[str, int]:
        """Number of pixels per termination state name (states with zero pixels omitted)."""
        out: dict[str, int] = {}
        for member in TerminationState:
            n = int(np.count_nonzero(self.state == int(member)))
            if n:
                out[member.name] = n
        return out


def render_disk(
    st: Spacetime,
    cam: Camera,
    integ: IntegratorOptions,
    term: TerminationOptions,
    disk: DiskModel,
    *,
    backend: str | Backend = "numpy",
    progress: ProgressCallback | None = None,
) -> DiskImage:
    """Render the simplified disk seen by ``cam`` in the spacetime ``st``.

    Steps: (1) ``initial_states(cam, st)`` (past-directed, null); (2)
    :func:`integrate_rays` with the disk's :func:`~kerrray.physics.accretion.event_options`;
    (3) on ``DISK_HIT`` pixels ``g = redshift_factor(st, y_cross, prograde=disk.prograde)``
    and ``I_obs = observed_intensity(disk, r_cross, g)``; other pixels get
    ``I_obs = 0`` and ``g = r_hit = NaN``. ``progress`` is called with the
    completed fraction after every chunk. Returns a :class:`DiskImage`.
    """
    y0 = initial_states(cam, st)
    result = integrate_rays(st, y0, integ, term, events=event_options(disk), backend=backend, progress=progress)
    if result.event_hit is None or result.event_Y is None:
        raise RuntimeError("the backend did not evaluate the disk-plane event")
    n = cam.n_rays
    hit_flat = np.asarray(result.event_hit, dtype=bool)
    states_flat = np.full((n, STATE_SIZE), np.nan, dtype=np.float64)
    states_flat[hit_flat] = np.asarray(result.event_Y, dtype=np.float64)[hit_flat]
    g_flat = np.full(n, np.nan, dtype=np.float64)
    r_flat = np.full(n, np.nan, dtype=np.float64)
    intensity_flat = np.zeros(n, dtype=np.float64)
    if np.any(hit_flat):
        y_hit = states_flat[hit_flat]
        g_hit = redshift_factor(st, y_hit, prograde=disk.prograde)
        r_hit = y_hit[:, IDX_R]
        g_flat[hit_flat] = g_hit
        r_flat[hit_flat] = r_hit
        intensity_flat[hit_flat] = observed_intensity(disk, r_hit, g_hit)
    shape = cam.shape
    alpha, beta = cam.pixel_coordinates()
    return DiskImage(
        intensity=intensity_flat.reshape(shape),
        g=g_flat.reshape(shape),
        hit=hit_flat.reshape(shape),
        r_hit=r_flat.reshape(shape),
        state=np.asarray(result.state, dtype=np.int64).reshape(shape),
        alpha=alpha,
        beta=beta,
        hit_states=states_flat.reshape(shape + (STATE_SIZE,)),
        result=result,
        camera=cam,
        spacetime=st,
        disk=disk,
        backend=backend if isinstance(backend, str) else backend.name,
    )


def save_disk_image(path: Path | str, image: DiskImage) -> Path:
    """Write the image arrays and the run parameters to a compressed ``.npz`` file.

    Arrays: ``intensity``, ``g``, ``hit``, ``r_hit``, ``state``, ``alpha``,
    ``beta``, ``hit_states``, plus the per-ray ``lam``, ``n_steps`` and
    ``max_null_error`` of the batch; scalars: ``mass``, ``spin``,
    ``inclination_deg``, ``camera_radius``, ``fov``, ``resolution``, ``r_in``,
    ``r_out``, ``emissivity_index``, ``redshift_power``, ``prograde``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cam, disk, st = image.camera, image.disk, image.spacetime
    np.savez_compressed(
        target,
        intensity=image.intensity,
        g=image.g,
        hit=image.hit,
        r_hit=image.r_hit,
        state=image.state,
        alpha=image.alpha,
        beta=image.beta,
        hit_states=image.hit_states,
        lam=np.asarray(image.result.lam, dtype=np.float64),
        n_steps=np.asarray(image.result.n_steps),
        max_null_error=np.asarray(image.result.max_null_error, dtype=np.float64),
        mass=st.mass,
        spin=st.spin,
        inclination_deg=cam.inclination_deg,
        camera_radius=cam.radius,
        fov=cam.fov,
        resolution=cam.resolution,
        r_in=disk.r_in,
        r_out=disk.r_out,
        emissivity_index=disk.emissivity_index,
        redshift_power=disk.redshift_power,
        prograde=disk.prograde,
    )
    return target


def load_disk_image_arrays(path: Path | str) -> dict[str, Any]:
    """Read a file written by :func:`save_disk_image` into a plain ``dict`` of arrays and scalars."""
    with np.load(Path(path)) as data:
        return {key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data.files}
