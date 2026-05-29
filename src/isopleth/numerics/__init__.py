"""Numerical solvers, discretizations, Riemann solvers, and verification suites for Isopleth."""

from isopleth.numerics.burgers import (
    Burgers1DSolver,
    BurgersSolverConfig,
    FluxScheme,
    SlopeLimiter,
)
from isopleth.numerics.dimensional_analysis import (
    BurgersDimensionalScales,
    BurgersSimilitude,
    GrayScottDimensionalScales,
    GrayScottSimilitude,
    ShallowWaterDimensionalScales,
    ShallowWaterSimilitude,
)
from isopleth.numerics.limiters import (
    ConservativeDepthLimiter,
    LimiterDiagnostics,
)
from isopleth.numerics.reaction_diffusion import (
    REGIME_PARAMETERS,
    GrayScottRegime,
    MorphometricReport,
    ReactionDiffusionParameters,
    ReactionDiffusionSpectralSolver2D,
    TuringMorphometryAnalyzer,
)
from isopleth.numerics.riemann import (
    AudusseWellBalancedReconstruction,
    DesingularizedMomentumLimiter,
    ExactShallowWaterRiemannSolver,
    HLLCRiemannSolver2D,
    HLLRiemannSolver1D,
    RiemannFlux1D,
    RiemannState1D,
    RoeRiemannSolver1D,
)
from isopleth.numerics.shallow_water import (
    ShallowWater2DSolver,
    ShallowWaterConfig,
    ShallowWaterFluxScheme,
)
from isopleth.numerics.solvers_2d import (
    BenchmarkState2D,
    PartialBreachDambreakBenchmark,
    RadialDambreakBenchmark,
    StrangSplittingSolver2D,
    ThackerOscillatingBasin,
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
from isopleth.numerics.spectral_analysis import (
    AutocorrelationResult,
    EnstrophyDissipationAnalyzer2D,
    SpatialAutocorrelationAnalyzer,
    SpatialStructureFunctionAnalyzer,
    StructureFunctionResult,
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
    "AudusseWellBalancedReconstruction",
    "AutocorrelationResult",
    "BalanceAuditReport",
    "BalanceAuditor",
    "BenchmarkState2D",
    "Burgers1DSolver",
    "BurgersDimensionalScales",
    "BurgersMMS",
    "BurgersSimilitude",
    "BurgersSolverConfig",
    "ConservativeDepthLimiter",
    "ConvergenceEntry",
    "ConvergenceTable",
    "DesingularizedMomentumLimiter",
    "EnstrophyDissipationAnalyzer2D",
    "ExactShallowWaterRiemannSolver",
    "FluxScheme",
    "GrayScottDimensionalScales",
    "GrayScottRegime",
    "GrayScottSimilitude",
    "HLLCRiemannSolver2D",
    "HLLRiemannSolver1D",
    "LimiterDiagnostics",
    "MorphometricReport",
    "PartialBreachDambreakBenchmark",
    "REGIME_PARAMETERS",
    "RadialDambreakBenchmark",
    "ReactionDiffusionParameters",
    "ReactionDiffusionSpectralSolver2D",
    "RiemannFlux1D",
    "RiemannState1D",
    "RoeRiemannSolver1D",
    "ShallowWater2DSolver",
    "ShallowWaterConfig",
    "ShallowWaterDimensionalScales",
    "ShallowWaterFluxScheme",
    "ShallowWaterSimilitude",
    "ShallowWaterVerification",
    "SlopeLimiter",
    "SpatialAutocorrelationAnalyzer",
    "SpatialStructureFunctionAnalyzer",
    "StrangSplittingSolver2D",
    "StructureFunctionResult",
    "ThackerOscillatingBasin",
    "TuringMorphometryAnalyzer",
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
