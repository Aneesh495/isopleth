"""Inverse problems, sparse sensor observation operators, and gradient diagnostics."""

from isopleth.inverse.baselines import (
    BaselineComparisonMetrics,
    NumericalSolverInverseBaseline,
    PriorMeanBaseline,
    SpatialInterpolationBaseline,
    evaluate_reconstruction_accuracy,
)
from isopleth.inverse.gradient_checks import (
    FiniteDifferenceCheckResult,
    GradientVerificationSuite,
    SurrogateDistortionResult,
    SurrogateGradientDistortionAnalyzer,
)
from isopleth.inverse.observations import (
    ObservationRecord,
    SparseObservationOperator,
)
from isopleth.inverse.reconstruction import (
    DifferentiableInverseReconstructor,
    InverseConfig,
    InverseReconstructionResult,
)

__all__ = [
    "ObservationRecord",
    "SparseObservationOperator",
    "InverseConfig",
    "InverseReconstructionResult",
    "DifferentiableInverseReconstructor",
    "BaselineComparisonMetrics",
    "PriorMeanBaseline",
    "SpatialInterpolationBaseline",
    "NumericalSolverInverseBaseline",
    "evaluate_reconstruction_accuracy",
    "FiniteDifferenceCheckResult",
    "GradientVerificationSuite",
    "SurrogateDistortionResult",
    "SurrogateGradientDistortionAnalyzer",
]
