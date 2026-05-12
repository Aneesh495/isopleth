"""Numerical solvers, discretizations, and verification suites for Isopleth."""

from isopleth.numerics.burgers import (
    Burgers1DSolver,
    BurgersSolverConfig,
    FluxScheme,
    SlopeLimiter,
)
from isopleth.numerics.shallow_water import (
    ShallowWater2DSolver,
    ShallowWaterConfig,
    ShallowWaterFluxScheme,
)
from isopleth.numerics.limiters import (
    ConservativeDepthLimiter,
    LimiterDiagnostics,
)
from isopleth.numerics.sources import (
    BalanceAuditReport,
    BalanceAuditor,
    compute_spatial_integral,
    gray_scott_reaction_sources,
    integrate_source_rk4,
    shallow_water_bottom_friction,
    shallow_water_coriolis_source,
)
from isopleth.numerics.transforms import (
    conservative_prolongation_1d,
    conservative_prolongation_2d,
    conservative_restriction_1d,
    conservative_restriction_2d,
    discrete_divergence_1d,
    discrete_divergence_2d,
)
from isopleth.numerics.verification import (
    BurgersMMS,
    ConvergenceEntry,
    ConvergenceTable,
    ShallowWaterVerification,
)

__all__ = [
    "BalanceAuditReport",
    "BalanceAuditor",
    "Burgers1DSolver",
    "BurgersMMS",
    "BurgersSolverConfig",
    "ConvergenceEntry",
    "ConvergenceTable",
    "ConservativeDepthLimiter",
    "FluxScheme",
    "LimiterDiagnostics",
    "ShallowWater2DSolver",
    "ShallowWaterConfig",
    "ShallowWaterFluxScheme",
    "ShallowWaterVerification",
    "SlopeLimiter",
    "compute_spatial_integral",
    "conservative_prolongation_1d",
    "conservative_prolongation_2d",
    "conservative_restriction_1d",
    "conservative_restriction_2d",
    "discrete_divergence_1d",
    "discrete_divergence_2d",
    "gray_scott_reaction_sources",
    "integrate_source_rk4",
    "shallow_water_bottom_friction",
    "shallow_water_coriolis_source",
]
