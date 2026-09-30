"""Spacetime geometry: metric, inverse metric, Christoffel symbols, coordinates,
horizons and ergosphere (PROJECT.md sections 5 to 7, 32).

Every equation is stated with its source in ``docs/equations.md``. The
symbolic verification helpers (``kerrray.geometry.symbolic``) import SymPy
lazily and are deliberately not re-exported here so that this package stays
free of SymPy on the numerical path.
"""

from kerrray.geometry.christoffel import christoffel
from kerrray.geometry.coordinates import bl_to_cartesian, deg_to_rad, rad_to_deg
from kerrray.geometry.horizons import (
    ergosphere_radius,
    horizon_radii,
    inside_horizon,
    outer_horizon,
)
from kerrray.geometry.metric import (
    Array,
    InverseMetricComponents,
    MetricComponents,
    Spacetime,
    delta,
    inverse_metric,
    inverse_metric_components,
    inverse_metric_derivatives,
    kerr,
    metric,
    metric_components,
    metric_derivatives,
    schwarzschild,
    sigma,
)

__all__ = [
    "Array",
    "InverseMetricComponents",
    "MetricComponents",
    "Spacetime",
    "bl_to_cartesian",
    "christoffel",
    "deg_to_rad",
    "delta",
    "ergosphere_radius",
    "horizon_radii",
    "inside_horizon",
    "inverse_metric",
    "inverse_metric_components",
    "inverse_metric_derivatives",
    "kerr",
    "metric",
    "metric_components",
    "metric_derivatives",
    "outer_horizon",
    "rad_to_deg",
    "schwarzschild",
    "sigma",
]
