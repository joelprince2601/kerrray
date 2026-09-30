"""Photon classification, conserved quantities and trajectory containers
(PROJECT.md sections 10 and 11; docs/architecture.md section 4).

``kerrray.geodesics`` is imported first on purpose: its integrators import
the modules below, and importing it fully before them keeps the two packages'
mutual imports well ordered whichever package is imported first.
``kerrray.photons.orbits`` (spherical photon orbits and the analytic shadow
curve, symbolic role) is intentionally not imported here; import it directly.
"""

import kerrray.geodesics  # noqa: F401  (import-order guard, see module docstring)
from kerrray.photons.classification import (
    AXIS_SIN_TOLERANCE,
    THETA_DOMAIN_TOLERANCE,
    TerminationState,
    classify_batch,
    classify_state,
)
from kerrray.photons.constants import (
    ConservationDiagnostics,
    angular_momentum,
    carter_constant,
    conservation_diagnostics,
    energy,
    null_constraint,
    null_constraint_from_rhs,
    null_error,
    relative_drift,
)
from kerrray.photons.trajectories import (
    BatchResult,
    Trajectory,
    azimuthal_winding,
    closest_approach,
    deflection_angle,
    to_cartesian,
)

__all__ = [
    "AXIS_SIN_TOLERANCE",
    "THETA_DOMAIN_TOLERANCE",
    "BatchResult",
    "ConservationDiagnostics",
    "TerminationState",
    "Trajectory",
    "angular_momentum",
    "azimuthal_winding",
    "carter_constant",
    "classify_batch",
    "classify_state",
    "closest_approach",
    "conservation_diagnostics",
    "deflection_angle",
    "energy",
    "null_constraint",
    "null_constraint_from_rhs",
    "null_error",
    "relative_drift",
    "to_cartesian",
]
