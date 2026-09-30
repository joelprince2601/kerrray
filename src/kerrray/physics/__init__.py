"""Frame dragging, strong-field lensing, redshift and the simplified accretion
disk (PROJECT.md sections 14, 22 to 24).

Phase 4 (Kerr photon dynamics) provides :mod:`kerrray.physics.frame_dragging`
(EXP-005), whose public names are re-exported here. The Phase 7 rendering
modules ``lensing``, ``redshift`` and ``accretion`` are imported directly
(``from kerrray.physics import lensing``) so that this package stays
importable while they are being developed.
"""

from kerrray.physics.frame_dragging import (
    FrameDraggingResult,
    RaySummary,
    SpinPair,
    azimuth_quadrature,
    equatorial_ray,
    frame_dragging_experiment,
    leading_order_asymmetry,
    leading_order_polar_drag,
    polar_impact_parameter_bound,
    polar_ray,
    summarise_ray,
    turning_point_radius,
)

__all__ = [
    "FrameDraggingResult",
    "RaySummary",
    "SpinPair",
    "azimuth_quadrature",
    "equatorial_ray",
    "frame_dragging_experiment",
    "leading_order_asymmetry",
    "leading_order_polar_drag",
    "polar_impact_parameter_bound",
    "polar_ray",
    "summarise_ray",
    "turning_point_radius",
]
