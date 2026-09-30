"""Photon state, null condition, geodesic equations, integrators and initial
conditions (PROJECT.md sections 8 to 10, 16; docs/architecture.md section 4).

Import order matters: ``state`` and ``equations`` are imported before
``integrators``, because the ``kerrray.photons`` modules that the integrators
use import those two submodules (and only those two) from this package.
"""

from kerrray.geodesics.state import (
    IDX_PH,
    IDX_PPH,
    IDX_PR,
    IDX_PT,
    IDX_PTH,
    IDX_R,
    IDX_T,
    IDX_TH,
    STATE_SIZE,
    PhotonState,
    pack,
    unpack,
)
from kerrray.geodesics.equations import (
    christoffel_to_hamiltonian_state,
    geodesic_rhs,
    geodesic_rhs_christoffel,
    hamiltonian,
    hamiltonian_to_christoffel_state,
    null_momentum_pt,
)
from kerrray.geodesics.tableaus import (
    error_norm,
    make_step_rk4,
    make_step_rk45,
    step_factor,
)
from kerrray.geodesics.initial_conditions import (
    carter_potentials,
    equatorial_photon,
    photon_from_constants,
    tangential_photon,
)
from kerrray.geodesics.integrators import (
    EventOptions,
    IntegratorOptions,
    TerminationOptions,
    integrate,
)
from kerrray.geodesics.integrators_batch import integrate_batch

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
    "EventOptions",
    "IntegratorOptions",
    "PhotonState",
    "TerminationOptions",
    "carter_potentials",
    "christoffel_to_hamiltonian_state",
    "equatorial_photon",
    "error_norm",
    "geodesic_rhs",
    "geodesic_rhs_christoffel",
    "hamiltonian",
    "hamiltonian_to_christoffel_state",
    "integrate",
    "integrate_batch",
    "make_step_rk4",
    "make_step_rk45",
    "null_momentum_pt",
    "pack",
    "photon_from_constants",
    "step_factor",
    "tangential_photon",
    "unpack",
]
